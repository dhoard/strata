// Real-artifact gate for causal batching: full logits and hidden vectors must be
// bitwise equal to the independently verified single-token path. No text-quality oracle.
#include "strata/qwen35/gpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace q = strata::qwen35;
static void equal(const std::vector<float> &a, const std::vector<float> &b, const char *what) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) {
        double max = 0, ss = 0, rr = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = double(a[i]) - b[i];
            max = std::max(max, std::abs(d));
            ss += d * d;
            rr += double(b[i]) * b[i];
        }
        std::fprintf(stderr, "%s max_abs=%.9g rms=%.9g relative_rms=%.9g\n", what, max,
                     std::sqrt(ss / a.size()), std::sqrt(ss / std::max(rr, 1e-30)));
        throw std::runtime_error(what);
    }
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : std::getenv("STRATA_ORNITH_TEST_MODEL");
    if (!path) {
        std::puts("set STRATA_ORNITH_TEST_MODEL to test real-model batching");
        return 77;
    }
    try {
        q::qwen35_enable_ggml();
        q::Qwen35Geometry g;
        q::TrunkWeights w;
        std::string err;
        if (!q::load_trunk(path, g, w, err))
            throw std::runtime_error(err);
        q::MtpWeights mtp;
        const char *draft = argc > 2 ? argv[2] : std::getenv("STRATA_ORNITH_TEST_MTP");
        if (draft && !q::load_mtp(draft, g, mtp, err))
            throw std::runtime_error(err);
        const int H = g.n_embd, V = g.n_vocab;
        const int64_t tokens[] = {198, 100, 198, 200, 42, 198, 300, 100, 76, 42, 198, 100};
        for (const auto &kv : {"f32", "f16"})
            for (int slots : {0, 256}) {
                q::GpuOptions o;
                o.kv = kv;
                o.expert_slots = slots;
                o.workers = 11;
                auto s = q::make_gpu_session(g, w, draft ? &mtp : nullptr, 512, o);
                // Freeze a fixed placement before either comparison arm executes.
                s->target_batch(tokens, 8, nullptr, nullptr);
                s->begin_decode();
                for (int n : {2, 3, 5, 8}) {
                    std::vector<float> ref(size_t(n) * V), rh(size_t(n) * H), got(ref.size()), gh(rh.size());
                    std::vector<float *> lp(n), hp(n);
                    s->reset();
                    s->begin_decode();
                    for (int t = 0; t < n; ++t)
                        s->target(tokens[t], ref.data() + size_t(t) * V, rh.data() + t * H);
                    // Keep the next-token logits too: this detects convolution/KV/state errors.
                    std::vector<float> after(V), bh(H), next(V), nh(H);
                    s->target(tokens[n], after.data(), bh.data());
                    s->reset();
                    s->begin_decode();
                    for (int t = 0; t < n; ++t) {
                        lp[t] = got.data() + size_t(t) * V;
                        hp[t] = gh.data() + t * H;
                    }
                    s->target_batch(tokens, n, lp.data(), hp.data());
                    equal(got, ref, "batch logits");
                    equal(gh, rh, "batch hidden");
                    s->target(tokens[n], next.data(), nh.data());
                    equal(next, after, "persistent state logits");
                    equal(nh, bh, "persistent state hidden");
                    std::printf("kv=%s slots=%d batch=%d logits/hidden/state bitwise equal\n", kv, slots, n);
                }
                if (draft)
                    for (int accepted = 0; accepted <= 5; ++accepted) {
                        std::vector<float> ref(V), rh(H), got(V), gh(H);
                        s->reset();
                        s->begin_decode();
                        s->target(tokens[0], nullptr, nullptr);
                        for (int t = 0; t < accepted; ++t)
                            s->target(tokens[t + 1], nullptr, nullptr);
                        s->target(tokens[accepted + 1], ref.data(), rh.data());
                        s->reset();
                        s->begin_decode();
                        s->target(tokens[0], nullptr, nullptr);
                        s->begin_verify();
                        s->target_batch(tokens + 1, 5, nullptr, nullptr);
                        s->commit_verify(accepted);
                        if (s->position() != 1 + accepted)
                            throw std::runtime_error("rollback cursor");
                        s->target(tokens[accepted + 1], got.data(), gh.data());
                        equal(got, ref, "rollback logits");
                        equal(gh, rh, "rollback hidden");
                        std::printf("kv=%s slots=%d accepted=%d rollback bitwise equal\n", kv, slots,
                                    accepted);
                    }
                // Identical requests cannot inherit the preceding request's placement or state.
                const std::vector<int64_t> prompt(tokens, tokens + 12);
                auto run = [&](int spec) {
                    std::vector<int64_t> output;
                    q::generate(
                        *s, g, w.eos_token, prompt, 12, spec,
                        [](const std::vector<float> &v) {
                            return std::max_element(v.begin(), v.end()) - v.begin();
                        },
                        [&](int64_t t) { output.push_back(t); }, [] { return false; });
                    return output;
                };
                const auto first = run(0);
                if (first.empty() || first != run(0))
                    throw std::runtime_error("sequential requests changed greedy tokens");
                if (draft && first != run(1))
                    throw std::runtime_error("speculation changed greedy tokens");
                int polls = 0;
                q::generate(
                    *s, g, w.eos_token, prompt, 12, 0,
                    [](const std::vector<float> &v) {
                        return std::max_element(v.begin(), v.end()) - v.begin();
                    },
                    [](int64_t) {}, [&] { return ++polls > 1; });
                if (first != run(0))
                    throw std::runtime_error("cancelled prefill poisoned next request");
                std::printf("kv=%s slots=%d repeated requests/spec/cancellation bitwise equal\n", kv, slots);
                // Reject bad batches before changing session state.
                const auto pos = s->position();
                const int64_t bad[] = {198, g.n_vocab};
                try {
                    s->target_batch(bad, 2, nullptr, nullptr);
                    throw std::runtime_error("bad token accepted");
                } catch (const std::out_of_range &) {
                }
                if (s->position() != pos)
                    throw std::runtime_error("invalid batch changed position");
            }
        std::puts("Qwen35 causal batch/rollback PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "batch session: %s\n", e.what());
        return 1;
    }
}
