// src/qwen35/qwen35_main.cpp - the Qwen35MoE (Ornith) entry point: a one-shot CLI and Strata's `--serve` protocol.
//
//   strata-qwen35 --model M.gguf --tokens "1,2,3" [--max-new N]      one greedy generation, prints ids
//   strata-qwen35 --model M.gguf --check                            load + validate only
//   strata-qwen35 --serve --model M.gguf [--max-context N]          the server's stdio protocol:
//
//     READY <ctx> stop
//     < GEN <max_new> [key=value ...] <id,id,...>
//     > PP <read> <total> <ms> <tok/s>   (while a prompt is read; the server's progress line)
//     > T <id>                  (one per generated token)
//     > DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <proposed> <reused>
//     < STOP | QUIT
//
// `reused` is the prompt cache: prompt tokens the session already held from the previous request, so the
// server reports them as cached_tokens.  --prompt-cache 0 turns the cache off (a full read every request).
//
// This is the same protocol `serve/server.py` speaks to `strata --serve`, so the Python server, the OpenAI API
// and run3.sh need no change to drive this binary.  Sampling keys the server sends are honoured where they are
// implemented and ignored otherwise.
#include "strata/qwen35/session.hpp"
#include "strata/qwen35/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace q = strata::qwen35;

namespace {

struct Sampling {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    uint64_t seed = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int64_t penalty_last_n = 64;
};

int64_t select_with_penalties(const std::vector<float>& logits, const Sampling& s,
                             std::mt19937_64& rng, const std::vector<int64_t>& history);

/// Greedy when temperature <= 0; otherwise temperature + top_k + top_p + min_p over the full vocabulary.
int64_t sample(const std::vector<float>& logits, const Sampling& s, std::mt19937_64& rng) {
    const int64_t V = (int64_t) logits.size();
    if (s.temperature <= 0.0f) {
        int64_t best = 0;
        for (int64_t v = 1; v < V; ++v) if (logits[(size_t) v] > logits[(size_t) best]) best = v;
        return best;
    }
    const float inv_t = 1.0f / s.temperature;
    std::vector<float> p((size_t) V);
    float mx = -INFINITY;
    for (int64_t v = 0; v < V; ++v) { p[(size_t) v] = logits[(size_t) v] * inv_t; mx = std::max(mx, p[(size_t) v]); }
    double sum = 0.0;
    for (int64_t v = 0; v < V; ++v) { p[(size_t) v] = std::exp(p[(size_t) v] - mx); sum += p[(size_t) v]; }
    for (int64_t v = 0; v < V; ++v) p[(size_t) v] = (float) (p[(size_t) v] / sum);
    std::vector<int64_t> idx((size_t) V);
    for (int64_t v = 0; v < V; ++v) idx[(size_t) v] = v;
    if (s.top_k > 0 && s.top_k < V) {
        std::partial_sort(idx.begin(), idx.begin() + s.top_k, idx.end(),
                          [&](int64_t a, int64_t b) { return p[(size_t) a] > p[(size_t) b]; });
        idx.resize((size_t) s.top_k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return p[(size_t) a] > p[(size_t) b]; });
    }
    std::vector<int64_t> keep;
    double acc = 0.0;
    const float pmax = p[(size_t) idx[0]];
    for (int64_t v : idx) {
        if (acc >= s.top_p) break;
        if (s.min_p > 0.0f && p[(size_t) v] < s.min_p * pmax) break;
        keep.push_back(v);
        acc += p[(size_t) v];
    }
    if (keep.empty()) return idx[0];
    double tot = 0.0;
    for (int64_t v : keep) tot += p[(size_t) v];
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng) * tot, c = 0.0;
    for (int64_t v : keep) { c += p[(size_t) v]; if (r <= c) return v; }
    return keep.back();
}

int64_t select_with_penalties(const std::vector<float>& logits, const Sampling& s,
                             std::mt19937_64& rng, const std::vector<int64_t>& history) {
    if (s.penalty_repeat == 1 && s.penalty_freq == 0 && s.penalty_present == 0) return sample(logits,s,rng);
    std::vector<float> adjusted = logits;
    std::vector<int> counts(logits.size());
    const size_t begin = history.size() > (size_t) s.penalty_last_n ? history.size()-(size_t) s.penalty_last_n : 0;
    for (size_t i=begin;i<history.size();++i) ++counts[(size_t) history[i]];
    for (size_t i=0;i<logits.size();++i) if (counts[i]) {
        adjusted[i] = adjusted[i] <= 0 ? adjusted[i]*s.penalty_repeat : adjusted[i]/s.penalty_repeat;
        adjusted[i] -= counts[i]*s.penalty_freq+s.penalty_present;
    }
    return sample(adjusted,s,rng);
}

int64_t integer(const std::string& s) {
    int64_t n = 0; const auto r = std::from_chars(s.data(),s.data()+s.size(),n);
    if (r.ec != std::errc() || r.ptr != s.data()+s.size()) throw std::invalid_argument("invalid integer: "+s);
    return n;
}
std::vector<int64_t> parse_ids(const std::string& s) {
    std::vector<int64_t> ids;
    for (size_t b=0;b<s.size();) {
        const size_t e = s.find(',',b);
        ids.push_back(integer(s.substr(b,e == std::string::npos ? e : e-b)));
        if (e == std::string::npos) break;
        b=e+1; if (b == s.size()) throw std::invalid_argument("empty token after comma");
    }
    return ids;
}

void done(const q::GenerationStats& r) {
    // Field 8 is the conversation cache's hit, which the server reports as cached_tokens (serve/server.py
    // parses DONE with the same layout as the Qwen3.8 engine).
    std::printf("DONE %lld %lld %.3f %.3f %s %lld %lld %lld\n",(long long) r.generated,
                (long long) r.prompt_tokens,r.prompt_ms,r.decode_ms,r.finish.c_str(),
                (long long) r.accepted,(long long) r.proposed,(long long) r.reused);
    std::fflush(stdout);
    std::fprintf(stderr,"qwen35 timing: prefill=%.3f ms decode=%.3f ms generated=%lld tok/s=%.3f accepted=%lld/%lld draft=%.3f ms verify=%.3f ms reused=%lld\n",
                 r.prompt_ms,r.decode_ms,(long long) r.generated,r.decode_ms ? 1000*r.generated/r.decode_ms : 0,
                 (long long) r.accepted,(long long) r.proposed,r.draft_ms,r.verify_ms,(long long) r.reused);
}

struct Command { std::string line; std::shared_ptr<std::atomic<bool>> stop; };
int run_serve(q::InferenceSession& session, const strata::core::Qwen35Geometry& g, int64_t eos, int spec) {
    std::mutex mutex; std::condition_variable ready; std::deque<Command> commands; bool eof = false;
    // Keep reading while the inference thread works. STOP belongs to the preceding GEN, including
    // when the reader receives both before inference starts. It cannot cancel the next request.
    std::thread reader([&] {
        std::string line; std::shared_ptr<std::atomic<bool>> current;
        while (std::getline(std::cin,line)) {
            if (line == "STOP") { if (current) current->store(true); continue; }
            Command command{line,{}};
            if (line.rfind("GEN ",0) == 0) command.stop = current = std::make_shared<std::atomic<bool>>(false);
            if (line == "QUIT" && current) current->store(true);
            { std::lock_guard<std::mutex> lock(mutex); commands.push_back(std::move(command)); }
            ready.notify_one();
            if (line == "QUIT") break;
        }
        { std::lock_guard<std::mutex> lock(mutex); eof = true; }
        ready.notify_one();
    });
    std::printf("READY %lld stop\n",(long long) session.context()); std::fflush(stdout);
    for (;;) {
        Command c;
        { std::unique_lock<std::mutex> lock(mutex); ready.wait(lock,[&]{return eof || !commands.empty();});
          if (commands.empty()) break;
          c = std::move(commands.front()); commands.pop_front(); }
        if (c.line == "QUIT") break;
        try {
            if (c.line.rfind("GEN ",0) != 0) throw std::invalid_argument("expected GEN, STOP or QUIT");
            std::vector<std::string> f;
            for (size_t b=0;b<c.line.size();) {
                const size_t e = c.line.find(' ',b); const auto field = c.line.substr(b,e == std::string::npos ? e : e-b);
                if (!field.empty()) f.push_back(field);
                if (e == std::string::npos) break; b=e+1;
            }
            if (f.size() < 3) throw std::invalid_argument("malformed GEN");
            const int64_t max_new = integer(f[1]); const auto ids = parse_ids(f.back()); Sampling sampling;
            for (size_t i=2;i+1<f.size();++i) {
                const size_t eq = f[i].find('='); if (eq == std::string::npos) throw std::invalid_argument("malformed sampling key");
                const auto key = f[i].substr(0,eq), val = f[i].substr(eq+1);
                if (key == "temperature") sampling.temperature = std::stof(val);
                else if (key == "top_p") sampling.top_p = std::stof(val);
                else if (key == "top_k") sampling.top_k = (int) integer(val);
                else if (key == "min_p") sampling.min_p = std::stof(val);
                else if (key == "seed") sampling.seed = (uint64_t) integer(val);
                else if (key == "penalty_repeat") sampling.penalty_repeat = std::stof(val);
                else if (key == "penalty_freq") sampling.penalty_freq = std::stof(val);
                else if (key == "penalty_present") sampling.penalty_present = std::stof(val);
                else if (key == "penalty_last_n") sampling.penalty_last_n = integer(val);
                else throw std::invalid_argument("unsupported sampling key: "+key);
            }
            if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
                !std::isfinite(sampling.top_p) || sampling.top_p <= 0 || sampling.top_p > 1 ||
                !std::isfinite(sampling.min_p) || sampling.min_p < 0 || sampling.min_p > 1 || sampling.top_k < 0 ||
                !std::isfinite(sampling.penalty_repeat) || sampling.penalty_repeat <= 0 ||
                !std::isfinite(sampling.penalty_freq) || !std::isfinite(sampling.penalty_present) || sampling.penalty_last_n < 0)
                throw std::invalid_argument("invalid sampling settings");
            std::mt19937_64 rng(sampling.seed);
            auto history = ids;
            const bool raw_greedy = sampling.temperature == 0 && sampling.penalty_repeat == 1 && sampling.penalty_freq == 0 && sampling.penalty_present == 0;
            // One PP line per half second of reading (the server turns it into "reading the prompt: N of M"
            // and a keep-alive), and always the last one.
            auto last_pp = std::chrono::steady_clock::now();
            const auto progress = [&](int64_t read, int64_t total) {
                const auto now = std::chrono::steady_clock::now();
                if (read < total && now - last_pp < std::chrono::milliseconds(500)) return;
                const double ms = std::chrono::duration<double,std::milli>(now - last_pp).count();
                last_pp = now;
                std::printf("PP %lld %lld %.0f %.1f\n",(long long) read,(long long) total, ms,
                            ms > 0 ? 1000.0 * (total ? double(total) : 1.0) / ms : 0.0);
                std::fflush(stdout);
            };
            const auto result = q::generate(session,g,eos,ids,max_new,raw_greedy ? spec : 0,
                [&](const auto& logits){return select_with_penalties(logits,sampling,rng,history);},
                [&](int64_t token){history.push_back(token); std::printf("T %lld\n",(long long) token); std::fflush(stdout);},
                [&]{return c.stop->load();},-1,progress);
            done(result);
        } catch (const std::exception& e) { std::printf("ERR %s\n",e.what()); std::fflush(stdout); }
    }
    reader.join(); return 0;
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try {
        std::string path, mtp_path; std::vector<int64_t> tokens;
        int64_t max_new = 8, context = 0; int spec = 0, workers = 0, force_reject = -1;
        bool check = false, serve = false, gpu = false, fit_report = false, prompt_cache = true; q::GpuOptions gpu_options;
        for (int i=1;i<argc;++i) {
            const std::string arg = argv[i];
            const auto next = [&]() -> std::string {
                if (i+1>=argc) throw std::invalid_argument(arg+" requires a value"); return argv[++i];
            };
            if (arg == "--model") path = next();
            else if (arg == "--mtp") mtp_path = next();
            else if (arg == "--tokens") tokens = parse_ids(next());
            else if (arg == "--max-new") max_new = integer(next());
            else if (arg == "--max-context") context = integer(next());
            else if (arg == "--spec") spec = (int) integer(next());
            else if (arg == "--pool-workers") workers = (int) integer(next());
            else if (arg == "--prefill") gpu_options.prefill = (int) integer(next());
            else if (arg == "--force-reject") force_reject = (int) integer(next());
            else if (arg == "--gpu") gpu = true;
            else if (arg == "--profile") gpu_options.profile = true;
            else if (arg == "--cpu") gpu = false;
            else if (arg == "--fit-report") fit_report = true;
            else if (arg == "--kv") { gpu_options.kv = next();
                if (gpu_options.kv != "f32" && gpu_options.kv != "f16")
                    throw std::invalid_argument("--kv must be f32 or f16 (the resident KV layouts this "
                                                "backend implements and has parity for)"); }
            else if (arg == "--expert-cache") { auto v=next(); gpu_options.expert_slots = v=="auto" ? -1 : integer(v); }
            else if (arg == "--prompt-cache") prompt_cache = integer(next()) != 0;
            else if (arg == "--prompt-cache-slots") gpu_options.prompt_cache_slots = (int) integer(next());
            else if (arg == "--serve") serve = true;
            else if (arg == "--check") check = true;
            else if (arg == "--capabilities") {
                // What this build can serve, and on which tiers. run3.sh refuses before the download when
                // `qwen35moe` is missing; `gpu` is absent in a build without a GPU backend.
                std::puts("qwen35moe mtp greedy-spec cancellation cpu gpu"); return 0;
            }
            else throw std::invalid_argument("unknown option: "+arg);
        }
        if (path.empty()) throw std::invalid_argument("--model is required");
        if (context < 0 || max_new < 0 || spec < 0 || spec > 4 || workers < 0 || workers > 256 || force_reject < -1 || force_reject > 4 || gpu_options.prefill < 1 || gpu_options.prefill > 4096)
            throw std::invalid_argument("invalid context/output/spec/worker/rejection setting");
        if (spec && mtp_path.empty()) throw std::invalid_argument("--spec requires a trained external --mtp GGUF");
#ifdef _OPENMP
        if (workers) omp_set_num_threads(workers);
#endif
        q::qwen35_enable_ggml(); strata::core::Qwen35Geometry g; q::TrunkWeights w; std::string err;
        if (!q::load_trunk(path,g,w,err)) throw std::runtime_error(err);
        if (!context) context = g.context_length;
        if (context > g.context_length) throw std::invalid_argument("requested context exceeds model maximum");
        q::MtpWeights mtp;
        if (!mtp_path.empty() && !q::load_mtp(mtp_path,g,mtp,err)) throw std::runtime_error(err);
        std::fprintf(stderr,"strata-qwen35: %lld layers, %lld wide, %lld experts top-%lld, context %lld, trained MTP %s spec %d\n",
                     (long long) g.n_layers,(long long) g.n_embd,(long long) g.n_expert,(long long) g.n_expert_used,
                     (long long) context,mtp_path.empty() ? "absent" : "loaded",spec);
        if (check) { std::puts("check ok"); return 0; }
        gpu_options.workers = workers;
        gpu_options.spec = spec;
        gpu_options.prompt_cache = prompt_cache;
        if (prompt_cache && gpu_options.prompt_cache_slots < 0) throw std::invalid_argument("--prompt-cache-slots must be >= 0");
        if (fit_report) {
            // Predicts the resident tier's VRAM before it is built, so the launcher can refuse a context/KV/slot
            // combination that cannot fit instead of failing after the model is loaded. Allocation-free.
            if (!gpu) throw std::invalid_argument("--fit-report describes the GPU tier; pass --gpu with it");
            const q::GpuMemoryPlan plan = q::qwen35_gpu_plan(g, w, spec ? &mtp : nullptr, context, gpu_options.kv, spec, prompt_cache, gpu_options.prompt_cache_slots);
            uint64_t free_bytes = 0;
            if (!q::gpu_free_bytes(free_bytes, err)) throw std::runtime_error("device memory query: " + err);
            const double GiB = 1073741824.0, MiB = 1048576.0;
            // The session sizes its expert cache from the bytes still free *after* its own allocations, so the
            // report takes the fixed tiers off the free bytes before asking what fits.
            const uint64_t free_for_cache = free_bytes > plan.fixed() ? free_bytes - plan.fixed() : 0;
            const int64_t capacity = q::gpu_expert_slot_capacity(plan, free_for_cache);
            const int64_t want = gpu_options.expert_slots < 0 ? capacity : gpu_options.expert_slots;
            std::printf("qwen35 fit: tier=gpu context=%lld kv=%s mtp=%s\n", (long long) context,
                        gpu_options.kv.c_str(), spec ? "yes" : "no");
            std::printf("qwen35 fit: dense=%.3f GiB gdn_state=%.3f GiB kv_cache=%.3f GiB scratch=%.3f GiB\n",
                        double(plan.dense)/GiB, double(plan.state)/GiB, double(plan.kv)/GiB,
                        double(plan.scratch)/GiB);
            std::printf("qwen35 fit: fixed=%.3f GiB; device free=%.3f GiB (budget, runtime reserve and slack "
                        "already removed)\n", double(plan.fixed())/GiB, double(free_bytes)/GiB);
            std::printf("qwen35 fit: expert blob=%.2f MiB, slots requested=%lld slots that fit=%lld of %lld "
                        "expert pairs\n", double(plan.expert_blob)/MiB, (long long) want, (long long) capacity,
                        (long long) plan.n_expert_pairs);
            if (plan.fixed() > free_bytes) {
                std::printf("qwen35 fit: DOES NOT FIT - the dense weights, state and KV alone need %.3f GiB "
                            "and %.3f GiB is free. Lower --max-context or use --kv f16; nothing is reduced "
                            "silently.\n", double(plan.fixed())/GiB, double(free_bytes)/GiB);
                return 1;
            }
            if (want > capacity) {
                std::printf("qwen35 fit: DOES NOT FIT - %lld expert slots need %.3f GiB of the %.3f GiB free "
                            "after the fixed %.3f GiB. Ask for <= %lld slots (--expert-cache) or lower the "
                            "context.\n", (long long) want,
                        double(uint64_t(want)*plan.expert_blob)/GiB, double(free_bytes-plan.fixed())/GiB,
                        double(plan.fixed())/GiB, (long long) capacity);
                return 1;
            }
            std::printf("qwen35 fit: FITS - total %.3f GiB of %.3f GiB with %lld expert slots\n",
                        double(plan.with_slots(want))/GiB, double(free_bytes)/GiB, (long long) want);
            return 0;
        }
        std::unique_ptr<q::InferenceSession> owned;
        std::fprintf(stderr, "qwen35 backend=%s kv=%s expert_cache=%s pool_workers=%s context=%lld\n",
                     gpu ? "gpu (dense+attention on card, experts split VRAM/CPU)" : "cpu (all layers on "
                     "CPU through ggml-cpu)", gpu_options.kv.c_str(),
                     gpu_options.expert_slots < 0 ? "auto" : std::to_string(gpu_options.expert_slots).c_str(),
                     workers == 0 ? "auto" : std::to_string(workers).c_str(), (long long) context);
        if (gpu) owned = q::make_gpu_session(g,w,spec ? &mtp : nullptr,context,gpu_options);
        else owned = std::make_unique<q::CpuSession>(g,w,spec ? &mtp : nullptr,context,prompt_cache,gpu_options.prompt_cache_slots);
        auto& session = *owned;
        if (serve) return run_serve(session,g,w.eos_token,spec);
        Sampling sampling; std::mt19937_64 rng(0);
        const auto result = q::generate(session,g,w.eos_token,tokens,max_new,spec,
            [&](const auto& logits){return sample(logits,sampling,rng);},
            [](int64_t token){std::printf("%lld,",(long long) token);},[]{return false;},force_reject);
        std::puts(""); done(result); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"strata-qwen35: %s\n",e.what()); return 1; }
}
