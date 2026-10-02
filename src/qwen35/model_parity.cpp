// Full-model numerical oracle. Runs the exact GGUF through independent upstream llama.cpp CPU graphs
// and Strata, one token at a time, and compares every layer residual and the full vocabulary.
// Usage: qwen35_model_parity MODEL.gguf TOKEN,TOKEN,... [greedy_steps] [threads]
#include "strata/qwen35/qwen35.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <map>
#include <vector>

namespace q = strata::qwen35;
using Ladder = std::vector<std::vector<float>>;
static std::map<std::string, std::vector<float>> stages;
static size_t token_index = 0;
static bool capture_stages = false;
struct Error { double max_abs = 0, rms = 0, rel = 0; bool finite = true; };
static Error compare(const float* x, const float* ref, size_t n);
static void compare_stage(int64_t layer, const char* name, const float* x, int64_t n) {
    const auto key = std::string(name)+"-"+std::to_string(layer);
    const auto it = stages.find(key);
    if (it == stages.end() || it->second.size() != (size_t) n) return;
    const auto e = compare(x, it->second.data(), (size_t) n);
    std::printf("stage token=%zu name=%s max_abs=%.8g rms=%.8g relative_rms=%.8g\n",
                token_index, key.c_str(), e.max_abs, e.rms, e.rel);
}

static bool capture(ggml_tensor* t, bool ask, void* user) {
    int layer = -1;
    const bool residual = std::sscanf(ggml_get_name(t), "l_out-%d", &layer) == 1 && layer >= 0;
    if (!residual && !capture_stages) return false;
    const bool state = std::strncmp(ggml_get_name(t),"new_state-",10)==0 || std::strncmp(ggml_get_name(t),"state_predelta-",15)==0;
    if (!residual && (t->type != GGML_TYPE_F32 || (!state && ggml_nelements(t) > 16384))) return false;
    if (ask) return true;
    if (!residual) {
        auto& v = stages[ggml_get_name(t)];
        v.resize((size_t) ggml_nelements(t));
        if (ggml_is_contiguous(t)) ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        else stages.erase(ggml_get_name(t));
        return true;
    }
    auto& ladder = *static_cast<Ladder*>(user);
    if ((size_t) layer >= ladder.size()) return false;
    auto& v = ladder[(size_t) layer];
    v.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return true;
}

static void trace(int64_t l, const float* x, int64_t n, void* user) {
    auto& v = (*static_cast<Ladder*>(user))[(size_t) l];
    v.assign(x, x + n);
}

static Error compare(const float* x, const float* ref, size_t n) {
    Error e;
    double ss = 0, rr = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) x[i] - ref[i];
        e.finite = e.finite && std::isfinite(x[i]) && std::isfinite(ref[i]);
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        ss += d*d; rr += (double) ref[i]*ref[i];
    }
    e.rms = std::sqrt(ss / n); e.rel = std::sqrt(ss / std::max(rr, 1e-30));
    return e;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if (argc < 3) { std::fprintf(stderr, "usage: %s MODEL TOKEN,TOKEN,... [greedy_steps] [threads]\n", argv[0]); return 2; }
    std::vector<llama_token> tokens;
    std::string ids = argv[2];
    for (size_t b = 0; b < ids.size();) {
        const size_t e = ids.find(',', b);
        tokens.push_back((llama_token) std::stoi(ids.substr(b, e == std::string::npos ? e : e-b)));
        if (e == std::string::npos) break;
        b = e+1;
    }
    const int steps = argc > 3 ? std::atoi(argv[3]) : 0;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 12;
    if (tokens.empty() || steps < 0 || threads < 1) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0; mp.load_mtp = false;
    mp.use_extra_bufts = false; // compare original quantized blocks, without CPU weight repacking
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    q::qwen35_enable_ggml();
    strata::core::Qwen35Geometry g;
    q::TrunkWeights w;
    std::string err;
    if (!q::load_trunk(argv[1], g, w, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Ladder ref((size_t) g.n_layers), got((size_t) g.n_layers);
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max<uint32_t>(512, (uint32_t) tokens.size() + steps + 1);
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = capture; cp.cb_eval_user_data = &ref;
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    llama_set_embeddings_nextn(ctx,true,false);
    llama_model* draft_model = nullptr;
    llama_context* draft_ctx = nullptr;
    q::MtpWeights mw;
    q::AttnState ms;
    std::vector<float> hidden((size_t) g.n_embd), previous_hidden((size_t) g.n_embd,0.0f);
    std::vector<float> draft_logits((size_t) g.n_vocab), draft_hidden((size_t) g.n_embd);
    if (argc > 5) {
        if (!q::load_mtp(argv[5],g,mw,err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 1; }
        mp.load_mtp = true;
        draft_model = llama_model_load_from_file(argv[5],mp);
        if (!draft_model) return 1;
        auto dp = cp; dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        draft_ctx = llama_init_from_model(draft_model,dp);
        if (!draft_ctx) return 1;
        llama_set_embeddings_nextn(draft_ctx,true,true);
        ms.resize(cp.n_ctx,g);
    }
    q::TrunkState st;
    st.reset(g, cp.n_ctx);
    std::vector<float> logits((size_t) g.n_vocab);
    bool pass = true;
    capture_stages = std::getenv("Q35_PARITY_STAGES") != nullptr;
    q::g_stage_trace = capture_stages ? compare_stage : nullptr;
    const size_t prompt = tokens.size();
    for (size_t i = 0; i < prompt + (size_t) steps; ++i) {
        token_index = i; stages.clear();
        llama_token token = tokens[i];
        auto batch = llama_batch_get_one(&token, 1);
        for (auto& v : ref) v.clear();
        if (llama_decode(ctx, batch) != 0) return 1;
        q::trunk_forward(g, w, st, token, logits.data(), trace, &got,hidden.data());
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (ref[(size_t) l].size() != (size_t) g.n_embd) {
                std::fprintf(stderr, "missing upstream l_out-%lld\n", (long long) l); return 1;
            }
            const auto e = compare(got[(size_t) l].data(), ref[(size_t) l].data(), (size_t) g.n_embd);
            std::printf("layer token=%zu layer=%lld max_abs=%.8g rms=%.8g relative_rms=%.8g\n",
                        i, (long long) l, e.max_abs, e.rms, e.rel);
            pass = pass && e.finite && e.rel < 1e-3;
        }
        const float* rl = llama_get_logits_ith(ctx, -1);
        const auto e = compare(logits.data(), rl, logits.size());
        const int best = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        const int rb = (int) (std::max_element(rl, rl + logits.size()) - rl);
        std::vector<int> gi(logits.size()), ri(logits.size());
        std::iota(gi.begin(), gi.end(), 0); std::iota(ri.begin(), ri.end(), 0);
        std::partial_sort(gi.begin(), gi.begin()+10, gi.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        std::partial_sort(ri.begin(), ri.begin()+10, ri.end(), [&](int a, int b) { return rl[a] > rl[b]; });
        int overlap = 0;
        for (int j = 0; j < 10; ++j) overlap += std::find(ri.begin(), ri.begin()+10, gi[j]) != ri.begin()+10;
        double zs = 0, zr = 0;
        for (size_t j = 0; j < logits.size(); ++j) { zs += std::exp((double) logits[j]-logits[best]); zr += std::exp((double) rl[j]-rl[rb]); }
        double kl = 0;
        for (size_t j = 0; j < logits.size(); ++j) {
            const double lr = rl[j]-rl[rb]-std::log(zr), ls = logits[j]-logits[best]-std::log(zs);
            kl += std::exp(lr)*(lr-ls);
        }
        std::printf("logits token=%zu id=%d max_abs=%.8g rms=%.8g relative_rms=%.8g kl=%.8g top1=%d/%d top10=%d/10\n",
                    i, token, e.max_abs, e.rms, e.rel, kl, best, rb, overlap);
        pass = pass && e.finite && e.rel < 1e-3 && best == rb;
        if (draft_ctx) {
            auto db = llama_batch_get_one(&token,1); db.embd = previous_hidden.data();
            stages.clear();
            if (llama_decode(draft_ctx,db) != 0) return 1;
            q::mtp_forward(g,mw,ms,token,previous_hidden.data(),draft_logits.data(),draft_hidden.data());
            const float* dl = llama_get_logits_ith(draft_ctx,-1);
            const auto de = compare(draft_logits.data(),dl,draft_logits.size());
            const auto dh = compare(draft_hidden.data(),llama_get_embeddings_nextn_ith(draft_ctx,0),draft_hidden.size());
            const int dbest = (int) (std::max_element(draft_logits.begin(),draft_logits.end())-draft_logits.begin());
            const int drbest = (int) (std::max_element(dl,dl+draft_logits.size())-dl);
            std::printf("mtp token=%zu logits_max_abs=%.8g logits_relative_rms=%.8g hidden_max_abs=%.8g hidden_relative_rms=%.8g top1=%d/%d\n",
                        i,de.max_abs,de.rel,dh.max_abs,dh.rel,dbest,drbest);
            pass = pass && de.finite && dh.finite && de.rel < 1e-3 && dh.rel < 1e-3 && dbest == drbest;
            const auto he = compare(hidden.data(),llama_get_embeddings_nextn_ith(ctx,0),hidden.size());
            pass = pass && he.finite && he.rel < 1e-3;
            const float* chain = std::getenv("Q35_MTP_CHAIN") ? llama_get_embeddings_nextn_ith(draft_ctx,0) : llama_get_embeddings_nextn_ith(ctx,0);
            std::copy_n(chain,g.n_embd,previous_hidden.data());
        }
        if (i+1 >= prompt) tokens.push_back(rb);
    }
    if (draft_ctx) llama_free(draft_ctx);
    if (draft_model) llama_model_free(draft_model);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    std::printf("model parity: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
