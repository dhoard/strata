// Independent upstream CPU/HIP graphs. Dumps per-layer residuals, target hidden states and
// full-vocabulary logits for the exact token sequence. No Strata arithmetic is linked here.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <map>

struct Capture {
    std::vector<std::vector<float>> layers;
    std::map<std::string, std::vector<float>> stages;
};
static bool capture(ggml_tensor *t, bool ask, void *opaque) {
    int l = -1;
    const bool residual = std::sscanf(ggml_get_name(t), "l_out-%d", &l) == 1 && l >= 0;
    const bool stages = std::getenv("Q35_REFERENCE_STAGES") != nullptr;
    if (!residual &&
        (!stages || t->type != GGML_TYPE_F32 || ggml_nelements(t) > 16384 || !ggml_is_contiguous(t)))
        return false;
    if (ask)
        return true;
    auto &c = *static_cast<Capture *>(opaque);
    if (residual) {
        if (size_t(l) >= c.layers.size())
            return false;
        c.layers[l].resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, c.layers[l].data(), 0, ggml_nbytes(t));
    } else {
        auto &v = c.stages[ggml_get_name(t)];
        v.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        std::printf("stage %s buffer=%s n=%zu\n", ggml_get_name(t), ggml_backend_buffer_name(t->buffer),
                    v.size());
    }
    return true;
}
static void write(std::ofstream &f, const float *x, size_t n) {
    f.write((const char *)x, (std::streamsize)(n * sizeof(float)));
    if (!f)
        throw std::runtime_error("oracle dump write failed");
}
int main(int argc, char **argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: ornith_reference MODEL TOKENS OUT GPU_LAYERS STEPS THREADS\n");
        return 2;
    }
    try {
        std::vector<llama_token> tokens;
        std::string ids = argv[2];
        for (size_t b = 0; b < ids.size();) {
            const auto e = ids.find(',', b);
            tokens.push_back(std::stoi(ids.substr(b, e == std::string::npos ? e : e - b)));
            if (e == std::string::npos)
                break;
            b = e + 1;
        }
        const int steps = std::stoi(argv[5]), threads = std::stoi(argv[6]);
        const size_t prompt = tokens.size();
        if (tokens.empty() || steps < 0 || threads < 1)
            return 2;
        const std::filesystem::path out = argv[3];
        std::filesystem::create_directories(out);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = std::stoi(argv[4]);
        mp.load_mtp = false;
        mp.use_extra_bufts = false;
        // Routed experts remain on the CPU, as in Strata's cache=0 parity arm. Dense, attention
        // and recurrent graphs use HIP; expert GPU arithmetic is checked separately per format.
        llama_model_tensor_buft_override overrides[] = {{".*ffn_.*_exps.*", ggml_backend_cpu_buffer_type()},
                                                        {nullptr, nullptr}};
        mp.tensor_buft_overrides = overrides;
        auto *model = llama_model_load_from_file(argv[1], mp);
        if (!model)
            return 1;
        const int H = llama_model_n_embd(model), L = llama_model_n_layer(model);
        const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
        Capture c;
        c.layers.resize((size_t)L);
        auto cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)std::max<size_t>(512, prompt + steps + 1);
        cp.n_batch = cp.n_ubatch = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (const char *kv = std::getenv("Q35_REFERENCE_KV")) {
            if (std::string(kv) != "f16")
                throw std::invalid_argument("reference KV must be f16");
            cp.type_k = cp.type_v = GGML_TYPE_F16;
        }
        cp.cb_eval = capture;
        cp.cb_eval_user_data = &c;
        auto *ctx = llama_init_from_model(model, cp);
        if (!ctx)
            return 1;
        llama_set_embeddings_nextn(ctx, true, false);
        llama_model *draft_model = nullptr;
        llama_context *draft_ctx = nullptr;
        std::vector<float> previous_hidden(H, 0);
        std::ofstream draft_logits, draft_hidden;
        if (argc > 7) {
            mp.load_mtp = true;
            draft_model = llama_model_load_from_file(argv[7], mp);
            if (!draft_model)
                return 1;
            auto dp = cp;
            dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            dp.cb_eval = nullptr;
            draft_ctx = llama_init_from_model(draft_model, dp);
            if (!draft_ctx)
                return 1;
            llama_set_embeddings_nextn(draft_ctx, true, true);
            draft_logits.open(out / "draft-logits.f32", std::ios::binary);
            draft_hidden.open(out / "draft-hidden.f32", std::ios::binary);
        }
        std::ofstream residuals(out / "residuals.f32", std::ios::binary),
            logits(out / "logits.f32", std::ios::binary), hidden(out / "hidden.f32", std::ios::binary);
        for (size_t i = 0; i < prompt + (size_t)steps; ++i) {
            auto token = tokens[i];
            auto batch = llama_batch_get_one(&token, 1);
            for (auto &x : c.layers)
                x.clear();
            c.stages.clear();
            if (llama_decode(ctx, batch))
                return 1;
            for (const auto &x : c.layers) {
                if (x.size() != (size_t)H)
                    throw std::runtime_error("missing upstream residual");
                write(residuals, x.data(), x.size());
            }
            if (!c.stages.empty()) {
                auto dir = out / ("token-" + std::to_string(i));
                std::filesystem::create_directories(dir);
                for (auto &v : c.stages) {
                    std::ofstream f(dir / (v.first + ".f32"), std::ios::binary);
                    write(f, v.second.data(), v.second.size());
                }
            }
            const float *x = llama_get_logits_ith(ctx, -1);
            write(logits, x, (size_t)V);
            write(hidden, llama_get_embeddings_nextn_ith(ctx, 0), (size_t)H);
            if (draft_ctx) {
                auto db = llama_batch_get_one(&token, 1);
                db.embd = previous_hidden.data();
                if (llama_decode(draft_ctx, db))
                    return 1;
                write(draft_logits, llama_get_logits_ith(draft_ctx, -1), V);
                write(draft_hidden, llama_get_embeddings_nextn_ith(draft_ctx, 0), H);
            }
            const float *next_hidden = draft_ctx && std::getenv("Q35_MTP_CHAIN")
                                           ? llama_get_embeddings_nextn_ith(draft_ctx, 0)
                                           : llama_get_embeddings_nextn_ith(ctx, 0);
            std::copy_n(next_hidden, H, previous_hidden.data());
            const auto best = (llama_token)(std::max_element(x, x + V) - x);
            if (i + 1 >= prompt)
                tokens.push_back(best);
            std::printf("reference token=%zu id=%d top1=%d\n", i, token, best);
            std::fflush(stdout);
        }
        tokens.resize(prompt + (size_t)steps);
        std::ofstream metadata(out / "metadata.json");
        metadata << "{\"layers\":" << L << ",\"hidden\":" << H << ",\"vocab\":" << V << ",\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); ++i)
            metadata << (i ? "," : "") << tokens[i];
        metadata << "]}\n";
        if (draft_ctx)
            llama_free(draft_ctx);
        if (draft_model)
            llama_model_free(draft_model);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ornith_reference: %s\n", e.what());
        return 1;
    }
}
