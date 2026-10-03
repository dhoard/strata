#include "strata/qwen35/gpu.hpp"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <numeric>
namespace q = strata::qwen35;
static std::ifstream residuals;
static bool good = true;
static size_t token_index = 0;
static std::string oracle;
static void compare(const char *name, const float *a, const float *b, size_t n) {
    double max = 0, ss = 0, rr = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = double(a[i]) - b[i];
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            good = false;
        max = std::max(max, std::abs(e));
        ss += e * e;
        rr += double(b[i]) * b[i];
    }
    double rel = std::sqrt(ss / std::max(rr, 1e-30));
    std::printf("%s max_abs=%.9g rms=%.9g relative_rms=%.9g\n", name, max, std::sqrt(ss / n), rel);
    if (rel > 1e-3)
        good = false;
}
static void trace(int64_t layer, const char *name, const float *x, int64_t n) {
    if (std::string(name) != "l_out") {
        std::ifstream f(oracle + "/token-" + std::to_string(token_index) + "/" + name + "-" +
                            std::to_string(layer) + ".f32",
                        std::ios::binary);
        if (!f)
            return;
        std::vector<float> ref(n);
        f.read((char *)ref.data(), n * 4);
        if (!f || f.peek() != EOF)
            return;
        std::printf("stage layer=%lld ", (long long)layer);
        compare(name, x, ref.data(), n);
        return;
    }
    std::vector<float> ref(n);
    residuals.read((char *)ref.data(), n * 4);
    if (!residuals)
        throw std::runtime_error("short residual oracle");
    std::printf("layer=%lld ", (long long)layer);
    compare("residual", x, ref.data(), n);
}
int main(int argc, char **argv) {
    try {
        if (argc < 4) {
            std::fprintf(stderr, "usage: qwen35_gpu_parity MODEL TOKENS ORACLE [CACHE]\n");
            return 2;
        }
        q::qwen35_enable_ggml();
        q::Qwen35Geometry g;
        q::TrunkWeights w;
        std::string err;
        if (!q::load_trunk(argv[1], g, w, err))
            throw std::runtime_error(err);
        std::vector<int64_t> tokens;
        std::string ids = argv[2];
        for (size_t b = 0; b < ids.size();) {
            auto e = ids.find(',', b);
            tokens.push_back(std::stoll(ids.substr(b, e == std::string::npos ? e : e - b)));
            if (e == std::string::npos)
                break;
            b = e + 1;
        }
        std::string path = argv[3];
        oracle = path;
        residuals.open(path + "/residuals.f32", std::ios::binary);
        std::ifstream logits(path + "/logits.f32", std::ios::binary),
            hidden(path + "/hidden.f32", std::ios::binary);
        q::GpuOptions options;
        options.workers = 12;
        options.expert_slots = argc > 4 ? std::stoll(argv[4]) : 0;
        if (const char *kv = std::getenv("Q35_REFERENCE_KV"))
            options.kv = kv;
        q::MtpWeights mtp;
        const bool has_mtp = argc > 5;
        std::ifstream dl, dh;
        if (has_mtp) {
            if (!q::load_mtp(argv[5], g, mtp, err))
                throw std::runtime_error(err);
            dl.open(path + "/draft-logits.f32", std::ios::binary);
            dh.open(path + "/draft-hidden.f32", std::ios::binary);
        }
        auto session = q::make_gpu_session(g, w, has_mtp ? &mtp : nullptr,
                                           std::max<int64_t>(512, tokens.size() + 1), options);
        q::g_stage_trace = trace;
        std::vector<float> a(g.n_vocab), b(g.n_vocab), h(g.n_embd), hr(g.n_embd);
        std::vector<float> previous(g.n_embd, 0), da(g.n_vocab), db(g.n_vocab), dha(g.n_embd), dhb(g.n_embd);
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::printf("token=%zu id=%lld\n", i, (long long)tokens[i]);
            token_index = i;
            session->target(tokens[i], a.data(), h.data());
            logits.read((char *)b.data(), b.size() * 4);
            hidden.read((char *)hr.data(), hr.size() * 4);
            if (!logits || !hidden)
                throw std::runtime_error("short logits oracle");
            compare("hidden", h.data(), hr.data(), h.size());
            compare("logits", a.data(), b.data(), a.size());
            const float ma = *std::max_element(a.begin(), a.end()),
                        mb = *std::max_element(b.begin(), b.end());
            double sa = 0, sb = 0;
            for (size_t v = 0; v < a.size(); ++v) {
                sa += std::exp(double(a[v]) - ma);
                sb += std::exp(double(b[v]) - mb);
            }
            double kl = 0;
            for (size_t v = 0; v < a.size(); ++v) {
                const double p = std::exp(double(b[v]) - mb) / sb;
                kl += p * ((double(b[v]) - mb - std::log(sb)) - (double(a[v]) - ma - std::log(sa)));
            }
            std::vector<size_t> ia(a.size()), ib(b.size());
            std::iota(ia.begin(), ia.end(), 0);
            std::iota(ib.begin(), ib.end(), 0);
            std::partial_sort(ia.begin(), ia.begin() + 10, ia.end(),
                              [&](size_t x, size_t y) { return a[x] > a[y]; });
            std::partial_sort(ib.begin(), ib.begin() + 10, ib.end(),
                              [&](size_t x, size_t y) { return b[x] > b[y]; });
            int overlap = 0;
            for (int j = 0; j < 10; ++j)
                overlap += std::find(ib.begin(), ib.begin() + 10, ia[j]) != ib.begin() + 10;
            std::printf("distribution kl=%.9g top10_overlap=%d/10\n", kl, overlap);
            auto ta = std::max_element(a.begin(), a.end()) - a.begin(),
                 tb = std::max_element(b.begin(), b.end()) - b.begin();
            std::printf("top1=%lld/%lld\n", (long long)ta, (long long)tb);
            if (ta != tb)
                good = false;
            if (has_mtp) {
                q::g_stage_trace = nullptr;
                session->draft(tokens[i], previous.data(), da.data(), dha.data());
                q::g_stage_trace = trace;
                dl.read((char *)db.data(), db.size() * 4);
                dh.read((char *)dhb.data(), dhb.size() * 4);
                if (!dl || !dh)
                    throw std::runtime_error("short draft oracle");
                compare("draft hidden", dha.data(), dhb.data(), dha.size());
                compare("draft logits", da.data(), db.data(), da.size());
                if (std::max_element(da.begin(), da.end()) - da.begin() !=
                    std::max_element(db.begin(), db.end()) - db.begin())
                    good = false;
            }
            previous = has_mtp && std::getenv("Q35_MTP_CHAIN") ? dha : h;
            std::fflush(stdout);
        }
        q::g_stage_trace = nullptr;
        std::printf("GPU model parity: %s\n", good ? "PASS" : "FAIL");
        return good ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "GPU parity: %s\n", e.what());
        return 1;
    }
}
