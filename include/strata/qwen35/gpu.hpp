#pragma once
#include "strata/qwen35/session.hpp"
#include <memory>
namespace strata::qwen35 {
struct GpuOptions {
    std::string kv = "f32";
    int64_t expert_slots = 0; // -1 = available budget after state/dense/checkpoints
    int workers = 0;
    bool profile = false;
    int prefill = 8; // Native MMVQ tiles, at most eight causal tokens per layer pass.
    int spec = 4; // Maximum draft window; allocate only spec+2 recurrent checkpoints with MTP.
    // Keep the conversation cache: a recurrent-state checkpoint at the end of each prompt plus a few
    // periodic ones inside it, so a request that extends the conversation reads only the new tail (a chat
    // template diverges at the previous prompt's very end, which is what the periodic points are for).
    // Each checkpoint costs one state copy of VRAM (61 MiB for Ornith's 30 GDN layers).
    bool prompt_cache = true;
    int prompt_cache_slots = 4;   // periodic resume points; a resumed turn re-reads <= 1/slots of the prompt
};
std::unique_ptr<InferenceSession> make_gpu_session(const Qwen35Geometry &, const TrunkWeights &,
                                                   const MtpWeights *, int64_t context, const GpuOptions &);

// What the resident tier will need before it is constructed, so a launcher can refuse a context/KV/slot
// combination that cannot fit the VRAM budget instead of dying after the model has been loaded. The plan is
// computed by the same formulas the session allocates with (see src/qwen35/gpu_session.cu); the session
// compares what it actually allocated against it and reports any drift.
struct GpuMemoryPlan {
    uint64_t dense = 0;          // uploaded dense projections, norms and the output head
    uint64_t state = 0;          // GDN convolution/recurrent state
    uint64_t kv = 0;             // the attention KV cache, for `context` (and the draft's own KV with MTP)
    uint64_t scratch = 0;        // activations, scores, expert plans and the MTP checkpoints
    uint64_t expert_blob = 0;    // one routed expert blob (gate/up + down)
    int64_t n_expert_pairs = 0;  // the whole routed set: n_layers * n_expert
    uint64_t fixed() const { return dense + state + kv + scratch; }
    uint64_t with_slots(int64_t slots) const { return fixed() + uint64_t(slots) * expert_blob; }
};
GpuMemoryPlan qwen35_gpu_plan(const Qwen35Geometry &, const TrunkWeights &, const MtpWeights *,
                              int64_t context, const std::string &kv, int spec = 4,
                              bool prompt_cache = true, int prompt_cache_slots = 4);
// Device bytes still available to Strata: the HIP allocation guard's own view (budget, runtime reserve and
// slack already removed), which is what an allocation is actually admitted against.
bool gpu_free_bytes(uint64_t &free_bytes, std::string &err);
// How many expert slots fit in `free_bytes`: the same arithmetic the resident session uses for
// `--expert-cache auto` (a 256 MiB reservation stays behind for the runtime and the next allocation), capped
// at the number of (layer, expert) pairs. One definition, so the fit report and the session never disagree.
int64_t gpu_expert_slot_capacity(const GpuMemoryPlan &plan, uint64_t free_bytes);
// Diagnostic parity hooks execute the production operators on caller-owned device buffers.
void gpu_moe_combine(const float* parts,const float* weights,const float* shared,float* output,
                     int hidden,int k,void* stream);
void gpu_router(const float *logits, int experts, int k, int *ids, float *weights, void *stream);
void gpu_dense_attention(const float *q, const float *gate, const void *key, const void *value, float *output,
                         float *scores, int cells, int heads, int kv_heads, int width, bool half_kv,
                         void *stream);
} // namespace strata::qwen35
