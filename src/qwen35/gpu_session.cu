// Resident Qwen35 execution: ordinary residual, native GDN, direct dense KV attention,
// native GGUF experts and the existing CPU pool/cache. No Qwen4Exp PLE/QSA state is used.
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/qwen35/gpu.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>
#include <cstdlib>
#include <unordered_map>
namespace strata::qwen35 {
namespace k = strata::kernels;
namespace {
constexpr int Batch = 8;
void checked(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
__device__ float sum32(float v) {
    for (int d = 16; d; d /= 2)
        v += __shfl_xor_sync(0xffffffffu, v, d, 32);
    return v;
}
// Pinned MMVF: paired input products, one 256-thread block per 2048-wide row.
__global__ void f32mv(const float *w, const float *x, float *y, int ni, int no) {
    const int r = blockIdx.x, t = threadIdx.x;
    x += size_t(blockIdx.y) * ni;
    y += size_t(blockIdx.y) * no;
    if (r >= no)
        return;
    float v = 0;
    for (int pair = t; pair < ni / 2; pair += blockDim.x) {
        int i = 2 * pair;
        v += w[size_t(r) * ni + i] * x[i];
        v += w[size_t(r) * ni + i + 1] * x[i + 1];
    }
    __shared__ float tmp[32];
    if (t < 32)
        tmp[t] = 0;
    __syncthreads();
    v = sum32(v);
    if (t % 32 == 0)
        tmp[t / 32] = v;
    __syncthreads();
    if (t < 32) {
        v = sum32(tmp[t]);
        if (t == 0)
            y[r] = v;
    }
}
__global__ void add(float *x, const float *y, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        x[i] = __fadd_rn(x[i], y[i]);
}
__global__ void swiglu(float *g, const float *u, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        g[i] = (g[i] / (1 + expf(-g[i]))) * u[i];
}
__global__ void shared_gate(float *y, const float *gate, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        y[size_t(blockIdx.y) * n + i] *= 1 / (1 + expf(-gate[blockIdx.y]));
}
__global__ void splitq(const float *qg, float *q, float *gate, int D, int heads) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < heads * D) {
        int h = i / D, d = i % D;
        q[i] = qg[h * 2 * D + d];
        gate[i] = qg[h * 2 * D + D + d];
    }
}
__global__ void fillpos(int *p, int rows, int pos) {
    int i = threadIdx.x;
    if (i < rows)
        p[i] = pos;
}
template <typename T> __global__ void storekv(T *cache, const float *x, int kv, int pos) {
    int i = threadIdx.x;
    if (i < kv)
        cache[size_t(pos) * kv + i] = T(x[i]);
}
// The activation arena's size, in floats. One definition, used both when the session allocates it and when
// the fit report predicts it, so the prediction cannot drift away from the allocation.
size_t scratch_floats(const Qwen35Geometry &g, int64_t ctx) {
    int H = g.n_embd, K = g.n_expert_used, Q = g.n_head, D = g.head_dim;
    return Batch * (2 * H + H + std::max(g.qkv_dim(), int64_t(2 * Q * D)) + g.value_dim() + g.ssm_dt_rank +
                    g.conv_channels() + g.ssm_dt_rank + g.ssm_dt_rank + g.value_dim() + g.conv_channels() +
                    2 * Q * D + 2 * g.n_head_kv * D + Q * D + K * H + 2 * g.n_ff_shexp + H + g.n_expert + K +
                    g.n_vocab) +
           size_t(Q) * ctx;
}
struct Allocation {
    void *p = nullptr;
    size_t bytes = 0;
    Allocation() = default;
    Allocation(const Allocation &) = delete;
    Allocation &operator=(const Allocation &) = delete;
    Allocation(Allocation &&o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
    Allocation &operator=(Allocation &&o) noexcept {
        if (p)
            (void)cudaFree(p);
        p = o.p;
        bytes = o.bytes;
        o.p = nullptr;
        return *this;
    }
    ~Allocation() {
        if (p)
            (void)cudaFree(p);
    }
    void alloc(size_t n) {
        if (p)
            throw std::logic_error("double allocate");
        checked(cudaMalloc(&p, n));
        bytes = n;
    }
    float *f() const { return static_cast<float *>(p); }
    void zero(cudaStream_t s) { checked(cudaMemsetAsync(p, 0, bytes, s)); }
};
struct Dense {
    GdnLayerWeights gdn;
    AttnLayerWeights attn;
    MoeLayerWeights moe;
    const float *post = nullptr;
};
struct State {
    Allocation conv, rec, key, value;
};
struct ExpertBank {
    k::NativeExpertLayout layout;
    k::cpu::NativeFmt fmt;
    std::vector<uint8_t> host;
};
struct EventPair {
    cudaEvent_t start = nullptr, end = nullptr;
    bool active = false;
    const char *name = "";
    EventPair() = default;
    EventPair(const EventPair &) = delete;
    EventPair &operator=(const EventPair &) = delete;
    ~EventPair() {
        if (start)
            (void)cudaEventDestroy(start);
        if (end)
            (void)cudaEventDestroy(end);
    }
};
struct Captured {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Captured() {
        if (exec)
            (void)cudaGraphExecDestroy(exec);
        if (graph)
            (void)cudaGraphDestroy(graph);
    }
};
class GpuSession final : public InferenceSession {
    const Qwen35Geometry &g_;
    TrunkWeights w_;
    const MtpWeights *mtp_;
    int64_t ctx_, pos_ = 0;
    GpuOptions opt_;
    cudaStream_t stream_ = nullptr;
    std::vector<Allocation> weights_;
    std::unordered_map<const void *, void *> uploaded_;
    std::vector<Dense> layers_;
    std::vector<State> state_;
    std::vector<ExpertBank> banks_;
    std::unique_ptr<k::cpu::ExpertPool> pool_;
    long long previous_affinity_ = -1;
    core::ExpertCache cache_;
    Allocation scratch_, q8_, expert_scratch_, plans_, snapshot_;
    float *x_, *xn_, *a_, *qkv_, *z_, *beta_, *alpha_, *gate_, *recout_, *normout_, *q_, *ag_, *kf_, *vf_,
        *attnout_, *parts_, *sg_, *su_, *shared_, *rlog_, *rweights_, *logits_, *scores_;
    int *ids_, *positions_;
    bool verifying_ = false, placement_frozen_ = false;
    std::vector<int64_t> checkpos_;
    size_t snapshot_bytes_ = 0;
    // Prompt cache: the tokens this session has consumed, the state at the end of the last prompt, and a
    // ring of periodic resume points.  The slots sit above the verification ones (0..spec+1).
    PromptCacheState prompt_cache_state_;
    std::vector<int64_t> periodic_;          // positions of the ring, oldest first
    std::vector<int64_t> periodic_slots_;    // their snapshot slots, in the same order
    int64_t prompt_end_slot_ = 0, periodic_writes_ = 0, next_periodic_ = 0;
    float *host_ = nullptr;
    int *host_ids_ = nullptr;
    float *cpu_parts_ = nullptr;
    cudaEvent_t handoff_ = nullptr;
    void *host_plan_ = nullptr;
    std::vector<uint8_t> cpu_act_;
    int64_t draft_pos_ = 0;
    Mat draft_eh_, draft_output_;
    const float *draft_enorm_ = nullptr, *draft_hnorm_ = nullptr, *draft_head_norm_ = nullptr;
    std::vector<uint64_t> prompt_routes_;
    uint64_t hits_ = 0, misses_ = 0;
    double cpu_ms_ = 0, wait_ms_ = 0;
    std::unique_ptr<EventPair[]> events_;
    std::map<std::string, double> gpu_ms_;
    uint64_t activation_d2h_bytes_ = 0, parts_h2d_bytes_ = 0, cache_h2d_bytes_ = 0;
    std::map<int, std::unique_ptr<Captured>> graphs_;
    bool graphs_enabled_ = false;
    bool profile_cache_ = true;
    template <class F> void graph(int layer, int part, int nt, F &&launch) {
        // Numerical tracing and event profiling need visible launches. Attention positions and
        // verification checkpoint offsets are deliberately excluded from captured subgraphs.
        if (!graphs_enabled_ || g_stage_trace || opt_.profile) {
            launch();
            return;
        }
        const int key = (layer * 4 + part) * Batch + nt - 1;
        auto &capture = graphs_[key];
        if (!capture) {
            checked(cudaStreamSynchronize(stream_));
            auto next = std::make_unique<Captured>();
            checked(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            try {
                launch();
            } catch (...) {
                (void)cudaStreamEndCapture(stream_, &next->graph);
                throw;
            }
            checked(cudaStreamEndCapture(stream_, &next->graph));
            checked(cudaGraphInstantiate(&next->exec, next->graph, nullptr, nullptr, 0));
            capture = std::move(next);
        }
        checked(cudaGraphLaunch(capture->exec, stream_));
    }
    void mark(int layer, int part, const char *name, bool end = false) {
        if (!events_)
            return;
        auto &e = events_[layer * 8 + part];
        e.name = name;
        checked(cudaEventRecord(end ? e.end : e.start, stream_));
        e.active = true;
    }
    void harvest() {
        if (!events_)
            return;
        for (int i = 0; i < (g_.n_layers + 1) * 8; ++i) {
            auto &e = events_[i];
            if (e.active) {
                float ms = 0;
                checked(cudaEventElapsedTime(&ms, e.start, e.end));
                gpu_ms_[e.name] += ms;
                e.active = false;
            }
        }
    }
    void *upload(const void *host, size_t n) {
        if (auto it = uploaded_.find(host); it != uploaded_.end())
            return it->second;
        Allocation d;
        d.alloc(n);
        checked(cudaMemcpy(d.p, host, n, cudaMemcpyHostToDevice));
        void *p = d.p;
        weights_.push_back(std::move(d));
        uploaded_[host] = p;
        return p;
    }
    const float *upnorm(const float *p, size_t n) { return static_cast<float *>(upload(p, n * 4)); }
    Mat upmat(Mat m) {
        if (m.type != 0 && m.type != 8 && m.type != 12 && m.type != 23)
            throw std::invalid_argument("unsupported resident matrix format " + std::to_string(m.type));
        m.dev = upload(m.data, m.row_bytes * size_t(m.n_out));
        return m;
    }
    const float *norm_repeated(const float *p, int cols, int rows) {
        std::vector<float> v(size_t(cols) * rows);
        for (int r = 0; r < rows; ++r)
            std::copy_n(p, cols, v.data() + r * cols);
        Allocation d;
        d.alloc(v.size() * 4);
        checked(cudaMemcpy(d.p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        auto *out = d.f();
        weights_.push_back(std::move(d));
        return out;
    }
    void quant(const float *x, int n, int nt = 1) { k::native_quantize_q8_1(x, q8_.p, n, nt, stream_); }
    void mv(const Mat &m, const float *x, float *y, bool already = false, int nt = 1) {
        if (m.type == 0)
            f32mv<<<dim3(m.n_out, nt), 256, 0, stream_>>>((const float *)m.dev, x, y, int(m.n_in),
                                                          int(m.n_out));
        else {
            if (!already)
                quant(x, int(m.n_in), nt);
            k::native_mmvq_rdna3(m.type, m.dev, q8_.p, y, int(m.n_in), int(m.n_out), nt, stream_);
        }
    }
    void norm(const float *x, const float *w, float *y, int n, int rows = 1) {
        k::native_gr_rms_norm_weighted(x, w, y, n, rows, g_.rms_eps, stream_);
    }
    void trace(int l, const char *name, const float *x, int n) {
        if (!g_stage_trace)
            return;
        std::vector<float> v(n);
        checked(cudaMemcpyAsync(v.data(), x, n * 4, cudaMemcpyDeviceToHost, stream_));
        checked(cudaStreamSynchronize(stream_));
        g_stage_trace(l, name, v.data(), n);
    }
    void gdn(int l) {
        mark(l, 0, "dense/GDN");
        auto &w = layers_[l].gdn;
        auto &s = state_[l];
        int H = g_.n_embd, S = g_.ssm_state, Hk = g_.ssm_groups, Hv = g_.ssm_dt_rank, C = g_.conv_channels();
        norm(x_, w.attn_norm, xn_, H);
        trace(l, "attn_norm", xn_, H);
        quant(xn_, H);
        mv(w.wqkv, xn_, qkv_, true);
        mv(w.wgate, xn_, z_, true);
        mv(w.ssm_beta, xn_, beta_, true);
        mv(w.ssm_alpha, xn_, alpha_, true);
        trace(l, "linear_attn_qkv_mixed", qkv_, C);
        trace(l, "z", z_, Hv * S);
        k::native_gdn_conv_silu(s.conv.f(), qkv_, w.ssm_conv, alpha_ + Hv, normout_, C, 4, stream_);
        trace(l, "conv_output_raw", alpha_ + Hv, C);
        k::native_gdn_l2_norm(normout_, Hk, S, g_.rms_eps, stream_);
        k::native_gdn_l2_norm(normout_ + Hk * S, Hk, S, g_.rms_eps, stream_);
        k::native_gdn_beta_gate(beta_, Hv, stream_);
        k::native_gdn_gate_qwen35(alpha_, w.ssm_dt, w.ssm_a, gate_, Hv, stream_);
        trace(l, "gate", gate_, Hv);
        trace(l, "beta_sigmoid", beta_, Hv);
        k::native_gdn_step(s.rec.f(), normout_, normout_ + Hk * S, normout_ + 2 * Hk * S, gate_, beta_,
                           recout_, {S, Hk, Hv}, stream_);
        trace(l, "attn_output", recout_, Hv * S);
        k::native_gdn_out_norm_silu(recout_, z_, w.ssm_norm, qkv_, Hv, S, g_.rms_eps, stream_);
        mv(w.ssm_out, qkv_, a_);
        mark(l, 0, "dense/GDN", true);
    }
    void attention(int l) {
        mark(l, 0, "dense/attention projections");
        auto &w = layers_[l].attn;
        auto &s = state_[l];
        int H = g_.n_embd, Q = g_.n_head, D = g_.head_dim, KV = g_.n_head_kv, N = KV * D;
        const int64_t position = l == g_.n_layers ? draft_pos_ : pos_;
        norm(x_, w.attn_norm, xn_, H);
        trace(l, "attn_norm", xn_, H);
        quant(xn_, H);
        mv(w.wq, xn_, qkv_, true);
        mv(w.wk, xn_, kf_, true);
        mv(w.wv, xn_, vf_, true);
        trace(l, "Qcur_full", qkv_, 2 * Q * D);
        trace(l, "Vcur", vf_, N);
        splitq<<<(Q * D + 255) / 256, 256, 0, stream_>>>(qkv_, q_, ag_, D, Q);
        norm(q_, w.q_norm, attnout_, D, Q);
        norm(kf_, w.k_norm, z_, D, KV);
        trace(l, "Qcur_normed", attnout_, Q * D);
        trace(l, "Kcur_normed", z_, N);
        k::RopeScaling rope;
        rope.freq_base = g_.rope_freq_base;
        k::native_rope_apply(attnout_, q_, Q, D, g_.rope_dim, rope, positions_, stream_);
        k::native_rope_apply(z_, kf_, KV, D, g_.rope_dim, rope, positions_, stream_);
        trace(l, "Qcur", q_, Q * D);
        trace(l, "Kcur", kf_, N);
        mark(l, 0, "dense/attention projections", true);
        mark(l, 1, "full attention");
        if (opt_.kv == "f16") {
            storekv<<<1, N, 0, stream_>>>((__half *)s.key.p, kf_, N, position);
            storekv<<<1, N, 0, stream_>>>((__half *)s.value.p, vf_, N, position);
        } else {
            storekv<<<1, N, 0, stream_>>>(s.key.f(), kf_, N, position);
            storekv<<<1, N, 0, stream_>>>(s.value.f(), vf_, N, position);
        }
        gpu_dense_attention(q_, ag_, s.key.p, s.value.p, attnout_, scores_, position + 1, Q, KV, D,
                            opt_.kv == "f16", stream_);
        trace(l, "attn_gated", attnout_, Q * D);
        mark(l, 1, "full attention", true);
        mark(l, 2, "attention output");
        mv(w.wo, attnout_, a_);
        trace(l, "attn_output", a_, H);
        mark(l, 2, "attention output", true);
    }
    void gdn_batch(int l, int nt, size_t snapshot_offset) {
        mark(l, 0, "dense/GDN");
        const auto &w = layers_[l].gdn;
        auto &s = state_[l];
        const int H = g_.n_embd, S = g_.ssm_state, Hk = g_.ssm_groups, Hv = g_.ssm_dt_rank;
        const int C = g_.conv_channels(), V = g_.value_dim();
        for (int t = 0; t < nt; ++t)
            norm(x_ + t * H, w.attn_norm, xn_ + t * H, H);
        quant(xn_, H, nt);
        mv(w.wqkv, xn_, qkv_, true, nt);
        mv(w.wgate, xn_, z_, true, nt);
        mv(w.ssm_beta, xn_, beta_, true, nt);
        mv(w.ssm_alpha, xn_, alpha_, true, nt);
        // The projections share weights across the tile; the recurrent update remains causal.
        // Assemble each accepted-prefix snapshot one layer at a time, without replaying tokens.
        for (int t = 0; t < nt; ++t) {
            auto *n = normout_ + t * C;
            k::native_gdn_conv_silu(s.conv.f(), qkv_ + t * C, w.ssm_conv, alpha_ + nt * Hv, n, C, 4, stream_);
            k::native_gdn_l2_norm(n, Hk, S, g_.rms_eps, stream_);
            k::native_gdn_l2_norm(n + Hk * S, Hk, S, g_.rms_eps, stream_);
            k::native_gdn_beta_gate(beta_ + t * Hv, Hv, stream_);
            k::native_gdn_gate_qwen35(alpha_ + t * Hv, w.ssm_dt, w.ssm_a, gate_ + t * Hv, Hv, stream_);
            k::native_gdn_step(s.rec.f(), n, n + Hk * S, n + 2 * Hk * S, gate_ + t * Hv, beta_ + t * Hv,
                               recout_ + t * V, {S, Hk, Hv}, stream_);
            if (verifying_) {
                const size_t slot = checkpos_.size() + t;
                auto *snap = (char *)snapshot_.p + slot * snapshot_bytes_ + snapshot_offset;
                checked(cudaMemcpyAsync(snap, s.conv.p, s.conv.bytes, cudaMemcpyDeviceToDevice, stream_));
                checked(cudaMemcpyAsync(snap + s.conv.bytes, s.rec.p, s.rec.bytes, cudaMemcpyDeviceToDevice,
                                        stream_));
            }
            // Output has a smaller column stride than QKV. All overwritten input rows have
            // already been consumed by this or earlier recurrence steps.
            k::native_gdn_out_norm_silu(recout_ + t * V, z_ + t * V, w.ssm_norm, qkv_ + t * V, Hv, S,
                                        g_.rms_eps, stream_);
        }
        mv(w.ssm_out, qkv_, a_, false, nt);
        mark(l, 0, "dense/GDN", true);
    }
    void attention_batch(int l, int nt) {
        mark(l, 0, "dense/attention projections");
        const auto &w = layers_[l].attn;
        auto &s = state_[l];
        const int H = g_.n_embd, Q = g_.n_head, D = g_.head_dim, KV = g_.n_head_kv, N = KV * D, A = Q * D;
        for (int t = 0; t < nt; ++t)
            norm(x_ + t * H, w.attn_norm, xn_ + t * H, H);
        quant(xn_, H, nt);
        mv(w.wq, xn_, qkv_, true, nt);
        mv(w.wk, xn_, kf_, true, nt);
        mv(w.wv, xn_, vf_, true, nt);
        splitq<<<(nt * A + 255) / 256, 256, 0, stream_>>>(qkv_, q_, ag_, D, nt * Q);
        k::RopeScaling rope;
        rope.freq_base = g_.rope_freq_base;
        for (int t = 0; t < nt; ++t) {
            norm(q_ + t * A, w.q_norm, attnout_ + t * A, D, Q);
            norm(kf_ + t * N, w.k_norm, z_ + t * N, D, KV);
            fillpos<<<1, 32, 0, stream_>>>(positions_, Q, pos_ + t);
            k::native_rope_apply(attnout_ + t * A, q_ + t * A, Q, D, g_.rope_dim, rope, positions_, stream_);
            k::native_rope_apply(z_ + t * N, kf_ + t * N, KV, D, g_.rope_dim, rope, positions_, stream_);
            if (opt_.kv == "f16") {
                storekv<<<1, N, 0, stream_>>>((__half *)s.key.p, kf_ + t * N, N, pos_ + t);
                storekv<<<1, N, 0, stream_>>>((__half *)s.value.p, vf_ + t * N, N, pos_ + t);
            } else {
                storekv<<<1, N, 0, stream_>>>(s.key.f(), kf_ + t * N, N, pos_ + t);
                storekv<<<1, N, 0, stream_>>>(s.value.f(), vf_ + t * N, N, pos_ + t);
            }
        }
        mark(l, 0, "dense/attention projections", true);
        mark(l, 1, "full attention");
        for (int t = 0; t < nt; ++t)
            gpu_dense_attention(q_ + t * A, ag_ + t * A, s.key.p, s.value.p, attnout_ + t * A, scores_,
                                pos_ + t + 1, Q, KV, D, opt_.kv == "f16", stream_);
        mark(l, 1, "full attention", true);
        mark(l, 2, "attention output");
        mv(w.wo, attnout_, a_, false, nt);
        mark(l, 2, "attention output", true);
    }
    void moe(int l, int nt = 1) {
        mark(l, 3, "post norm/router");
        auto &w = layers_[l].moe;
        auto &bank = banks_[l];
        const int H = g_.n_embd, F = g_.n_ff_shexp, K = g_.n_expert_used, cap = Batch * K;
        if (nt == 1)
            trace(l, "attn_residual", x_, H);
        graph(l, 1, nt, [&] {
            for (int t = 0; t < nt; ++t)
                norm(x_ + t * H, layers_[l].post, xn_ + t * H, H);
            if (nt == 1)
                trace(l, "attn_post_norm", xn_, H);
            f32mv<<<dim3(g_.n_expert, nt), 256, 0, stream_>>>(w.gate_inp, xn_, rlog_, H, g_.n_expert);
            if (nt == 1)
                trace(l, "ffn_moe_logits", rlog_, g_.n_expert);
            for (int t = 0; t < nt; ++t)
                gpu_router(rlog_ + t * g_.n_expert, g_.n_expert, K, ids_ + t * K, rweights_ + t * K, stream_);
            if (nt == 1)
                trace(l, "ffn_moe_weights_norm", rweights_, K);
        });
        mark(l, 3, "post norm/router", true);
        // The CPU only needs the router and activation. Let shared-expert work continue
        // on the GPU while the host prepares and computes the nonresident expert groups.
        checked(cudaMemcpyAsync(host_, xn_, nt * H * 4, cudaMemcpyDeviceToHost, stream_));
        checked(cudaMemcpyAsync(host_ids_, ids_, nt * K * 4, cudaMemcpyDeviceToHost, stream_));
        checked(cudaEventRecord(handoff_, stream_));
        activation_d2h_bytes_ += nt * (H * 4 + K * 4);
        mark(l, 4, "shared expert");
        graph(l, 2, nt, [&] {
            quant(xn_, H, nt);
            mv(w.gate_shexp, xn_, sg_, true, nt);
            mv(w.up_shexp, xn_, su_, true, nt);
            swiglu<<<(nt * F + 255) / 256, 256, 0, stream_>>>(sg_, su_, nt * F);
            mv(w.down_shexp, sg_, shared_, false, nt);
            f32mv<<<dim3(1, nt), 256, 0, stream_>>>(w.gate_inp_shexp, xn_, sg_, H, 1);
            shared_gate<<<dim3((H + 255) / 256, nt), 256, 0, stream_>>>(shared_, sg_, H);
        });
        mark(l, 4, "shared expert", true);
        auto wait0 = std::chrono::steady_clock::now();
        checked(cudaEventSynchronize(handoff_));
        wait_ms_ +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait0).count();
        std::vector<k::cpu::ExpertJobMulti> jobs;
        struct Group {
            unsigned long long ptr;
            std::vector<int> dst, tok;
        };
        std::vector<Group> groups;
        std::vector<int> group_of(g_.n_expert, -1), job_of(g_.n_expert, -1), missed;
        std::string err;
        for (int t = 0; t < nt; ++t) {
            k::cpu::native_quant_act(bank.fmt, host_ + t * H, cpu_act_.data() + t * k::cpu::kNativeActBytes);
            for (int j = 0; j < K; ++j) {
                int e = host_ids_[t * K + j], entry = t * K + j;
                if (e < 0 || e >= g_.n_expert)
                    throw std::runtime_error("nonfinite/invalid routing");
                if (l < g_.n_layers && !placement_frozen_)
                    ++prompt_routes_[size_t(l) * g_.n_expert + e];
                auto *blob = bank.host.data() + size_t(e) * bank.layout.bytes;
                // A prompt must not inherit the previous request's CPU/GPU expert rounding.
                // Build its frequency profile through one fixed (CPU) expert path, then use
                // the selected GPU placement for decode and every verification window.
                int slot = (!profile_cache_ || placement_frozen_) ? cache_.slot_of(l, e) : -1;
                if (!profile_cache_ && slot < 0 && !placement_frozen_) {
                    slot = cache_.admit(l, e);
                    if (slot >= 0 && !cache_.fill_slot(slot, blob, stream_, err, bank.layout.bytes))
                        throw std::runtime_error(err);
                    if (slot >= 0)
                        cache_h2d_bytes_ += bank.layout.bytes;
                }
                if (slot >= 0) {
                    ++hits_;
                    int &idx = group_of[e];
                    if (idx < 0) {
                        idx = int(groups.size());
                        groups.push_back({(unsigned long long)cache_.device_slot(slot), {}, {}});
                    }
                    groups[idx].dst.push_back(entry);
                    groups[idx].tok.push_back(t);
                } else {
                    ++misses_;
                    missed.push_back(entry);
                    int &idx = job_of[e];
                    if (idx < 0) {
                        idx = int(jobs.size());
                        jobs.emplace_back();
                        jobs.back().blob = blob;
                    }
                    auto &job = jobs[idx];
                    const int jt = job.nt++;
                    if (jt >= Batch)
                        throw std::logic_error("duplicate routed expert in one token");
                    job.out[jt] = cpu_parts_ + entry * H;
                    job.nact[jt] = cpu_act_.data() + t * k::cpu::kNativeActBytes;
                }
            }
        }
        if (!groups.empty()) {
            mark(l, 5, "GPU experts");
            std::vector<unsigned long long> ptr;
            std::vector<int> start{0}, dst, tok;
            for (auto &group : groups) {
                ptr.push_back(group.ptr);
                dst.insert(dst.end(), group.dst.begin(), group.dst.end());
                tok.insert(tok.end(), group.tok.begin(), group.tok.end());
                start.push_back(int(dst.size()));
            }
            int count = int(ptr.size());
            auto *hp = (unsigned long long *)host_plan_;
            auto *hs = (int *)((char *)host_plan_ + 8 * cap);
            auto *hd = hs + cap + 1, *ht = hd + cap, *hn = ht + cap;
            std::copy(ptr.begin(), ptr.end(), hp);
            std::copy(start.begin(), start.end(), hs);
            std::copy(dst.begin(), dst.end(), hd);
            std::copy(tok.begin(), tok.end(), ht);
            *hn = count;
            checked(cudaMemcpyAsync(plans_.p, host_plan_, plans_.bytes, cudaMemcpyHostToDevice, stream_));
            auto *gp = (unsigned long long *)plans_.p;
            auto *st = (int *)((char *)plans_.p + 8 * cap);
            auto *ed = st + cap + 1, *et = ed + cap, *ng = et + cap;
            // Shared down changed Q8 scratch; restore the whole input tile once.
            quant(xn_, H, nt);
            k::native_expert_grouped(bank.layout, gp, st, ng, ed, et, nt * K, nt * K, q8_.p,
                                     expert_scratch_.p, parts_, stream_);
            mark(l, 5, "GPU experts", true);
        }
        if (!jobs.empty()) {
            auto cpu0 = std::chrono::steady_clock::now();
            pool_->run_split_multi_native(bank.fmt, jobs.data(), int(jobs.size()));
            cpu_ms_ +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu0).count();
            // Pinned, contiguous runs of misses avoid a pageable-copy synchronization per expert.
            for (size_t b = 0; b < missed.size();) {
                size_t e = b + 1;
                while (e < missed.size() && missed[e] == missed[e - 1] + 1)
                    ++e;
                size_t bytes = (e - b) * H * 4;
                checked(cudaMemcpyAsync(parts_ + missed[b] * H, cpu_parts_ + missed[b] * H, bytes,
                                        cudaMemcpyHostToDevice, stream_));
                parts_h2d_bytes_ += bytes;
                b = e;
            }
        }
        if (nt == 1) {
            trace(l, "ffn_moe_down", parts_, K * H);
            trace(l, "ffn_shexp_gated", shared_, H);
        }
        mark(l, 6, "MoE combine/residual");
        graph(l, 3, nt, [&] {
            for (int t = 0; t < nt; ++t)
                gpu_moe_combine(parts_ + t * K * H, rweights_ + t * K, nullptr, a_ + t * H, H, K, stream_);
            if (nt == 1)
                trace(l, "ffn_moe_out", a_, H);
            add<<<(nt * H + 255) / 256, 256, 0, stream_>>>(a_, shared_, nt * H);
            if (nt == 1)
                trace(l, "ffn_out", a_, H);
            add<<<(nt * H + 255) / 256, 256, 0, stream_>>>(x_, a_, nt * H);
        });
        mark(l, 6, "MoE combine/residual", true);
    }
    void checkpoint(int slot, bool restore = false) {
        size_t off = 0;
        for (auto &s : state_)
            for (auto *d : {&s.conv, &s.rec})
                if (d->p) {
                    void *snap = (char *)snapshot_.p + size_t(slot) * snapshot_bytes_ + off;
                    checked(cudaMemcpyAsync(restore ? d->p : snap, restore ? snap : d->p, d->bytes,
                                            cudaMemcpyDeviceToDevice, stream_));
                    off += d->bytes;
                }
    }

  public:
    GpuSession(const Qwen35Geometry &g, const TrunkWeights &w, const MtpWeights *mtp, int64_t ctx,
               GpuOptions opt)
        : g_(g), w_(w), mtp_(mtp), ctx_(ctx), opt_(std::move(opt)) {
        if (ctx <= 0 || ctx > g.context_length)
            throw std::invalid_argument("invalid GPU context");
        if (mtp && (opt_.spec < 1 || opt_.spec > 4))
            throw std::invalid_argument("GPU MTP spec must be 1..4");
        if (opt_.prefill < 1 || opt_.prefill > 4096)
            throw std::invalid_argument("GPU prefill must be 1..4096");
        if (opt_.kv != "f32" && opt_.kv != "f16")
            throw std::invalid_argument("GPU KV must be f32 or f16");
        if (g_.ssm_state != 128 || g_.ssm_conv_kernel != 4 || g_.n_expert != 256)
            throw std::invalid_argument(
                "resident GPU geometry requires S=128, convolution=4 and 256 experts");
        if (const char *v = std::getenv("STRATA_QWEN35_GRAPHS"))
            graphs_enabled_ = std::atoi(v) != 0;
        if (const char *v = std::getenv("STRATA_QWEN35_CACHE_PROFILE"))
            profile_cache_ = std::atoi(v) != 0;
        checked(cudaStreamCreate(&stream_));
        if (opt_.profile) {
            events_ = std::make_unique<EventPair[]>((g_.n_layers + 1) * 8);
            for (int i = 0; i < (g_.n_layers + 1) * 8; ++i) {
                checked(cudaEventCreate(&events_[i].start));
                checked(cudaEventCreate(&events_[i].end));
            }
        }
        pool_ = std::make_unique<k::cpu::ExpertPool>(opt_.workers);
        int H = g.n_embd, K = g.n_expert_used, Q = g.n_head, D = g.head_dim;
        scratch_.alloc(scratch_floats(g, ctx) * 4 + Batch * K * 4 + Q * 4);
        float *p = scratch_.f();
        auto take = [&](size_t n) {
            float *out = p;
            p += Batch * n;
            return out;
        };
        x_ = take(H);
        xn_ = take(H);
        a_ = take(H);
        qkv_ = take(std::max(g.qkv_dim(), int64_t(2 * Q * D)));
        z_ = take(g.value_dim());
        beta_ = take(g.ssm_dt_rank);
        alpha_ = take(g.ssm_dt_rank + g.conv_channels());
        gate_ = take(g.ssm_dt_rank);
        recout_ = take(g.value_dim());
        normout_ = take(g.conv_channels());
        q_ = take(Q * D);
        ag_ = take(Q * D);
        kf_ = take(g.n_head_kv * D);
        vf_ = take(g.n_head_kv * D);
        attnout_ = take(Q * D);
        parts_ = take(K * H);
        sg_ = take(g.n_ff_shexp);
        su_ = take(g.n_ff_shexp);
        shared_ = take(H);
        rlog_ = take(g.n_expert);
        rweights_ = take(K);
        logits_ = take(g.n_vocab);
        scores_ = p;
        p += size_t(Q) * ctx;
        ids_ = (int *)p;
        p += Batch * K;
        positions_ = (int *)p;
        if (size_t((char *)(positions_ + Q) - (char *)scratch_.p) > scratch_.bytes)
            throw std::logic_error("GPU scratch accounting");
        q8_.alloc(k::native_q8_1_bytes(g.conv_channels(), Batch));
        expert_scratch_.alloc(k::native_expert_scratch_bytes(Batch * K, g.n_ff_exp));
        plans_.alloc(8 * Batch * K + 4 * (4 * Batch * K + 2));
        checked(cudaHostAlloc(&host_plan_, plans_.bytes, cudaHostAllocDefault));
        checked(cudaHostAlloc((void **)&host_, Batch * H * 4, cudaHostAllocDefault));
        checked(cudaHostAlloc((void **)&host_ids_, Batch * K * 4, cudaHostAllocDefault));
        checked(cudaHostAlloc((void **)&cpu_parts_, Batch * K * H * 4, cudaHostAllocDefault));
        checked(cudaEventCreateWithFlags(&handoff_, cudaEventDisableTiming));
        cpu_act_.resize(Batch * k::cpu::kNativeActBytes);
        layers_.resize(g.n_layers + (mtp ? 1 : 0));
        state_.resize(layers_.size());
        banks_.resize(layers_.size());
        prompt_routes_.resize(size_t(g_.n_layers) * g_.n_expert);
        std::string err;
        for (int l = 0; l < g.n_layers; ++l) {
            auto &dst = layers_[l];
            auto &st = state_[l];
            dst.post = upnorm(w.post_attn_norm[l], H);
            if (g.is_recurrent(l)) {
                dst.gdn = w.gdn[l];
                auto &d = dst.gdn;
                d.attn_norm = upnorm(d.attn_norm, H);
                d.wqkv = upmat(d.wqkv);
                d.wgate = upmat(d.wgate);
                d.ssm_beta = upmat(d.ssm_beta);
                d.ssm_alpha = upmat(d.ssm_alpha);
                d.ssm_out = upmat(d.ssm_out);
                d.ssm_conv = upnorm(d.ssm_conv, g.conv_channels() * 4);
                d.ssm_dt = upnorm(d.ssm_dt, g.ssm_dt_rank);
                d.ssm_a = upnorm(d.ssm_a, g.ssm_dt_rank);
                d.ssm_norm = upnorm(d.ssm_norm, g.ssm_state);
                st.conv.alloc(g.conv_channels() * 3 * 4);
                st.rec.alloc(g.ssm_state * g.ssm_state * g.ssm_dt_rank * 4);
                snapshot_bytes_ += st.conv.bytes + st.rec.bytes;
            } else {
                dst.attn = w.attn[l];
                auto &d = dst.attn;
                d.attn_norm = upnorm(d.attn_norm, H);
                d.wq = upmat(d.wq);
                d.wk = upmat(d.wk);
                d.wv = upmat(d.wv);
                d.wo = upmat(d.wo);
                d.q_norm = norm_repeated(d.q_norm, D, Q);
                d.k_norm = norm_repeated(d.k_norm, D, g.n_head_kv);
                size_t bytes = size_t(ctx) * g.n_head_kv * D * (opt_.kv == "f16" ? 2 : 4);
                st.key.alloc(bytes);
                st.value.alloc(bytes);
            }
            dst.moe = w.moe[l];
            auto &m = dst.moe;
            m.gate_inp = upnorm(m.gate_inp, H * g.n_expert);
            m.gate_inp_shexp = upnorm(m.gate_inp_shexp, H);
            m.gate_shexp = upmat(m.gate_shexp);
            m.up_shexp = upmat(m.up_shexp);
            m.down_shexp = upmat(m.down_shexp);
            auto &bank = banks_[l];
            auto &e = m.experts[0];
            bank.layout = k::native_expert_layout(e.gate.type, e.down.type, H, g.n_ff_exp);
            if (!k::native_expert_supported(e.gate.type, e.down.type, H, g.n_ff_exp) ||
                !k::cpu::native_fmt(e.gate.type, e.down.type, H, g.n_ff_exp, bank.fmt, err))
                throw std::runtime_error("expert format: " + err);
            bank.fmt.swiglu = g_swiglu_vec;
            bank.host.resize(bank.layout.bytes * g.n_expert);
            for (int j = 0; j < g.n_expert; ++j) {
                auto &ew = m.experts[j];
                auto *b = bank.host.data() + j * bank.layout.bytes;
                std::memcpy(b, ew.gate.data, bank.layout.up_off);
                std::memcpy(b + bank.layout.up_off, ew.up.data, bank.layout.down_off - bank.layout.up_off);
                std::memcpy(b + bank.layout.down_off, ew.down.data, bank.layout.bytes - bank.layout.down_off);
            }
        }
        w_.output = upmat(w.output);
        w_.output_norm = upnorm(w.output_norm, H);
        if (mtp) {
            auto &dst = layers_[g.n_layers];
            auto &st = state_[g.n_layers];
            dst.post = upnorm(mtp->post_norm, H);
            dst.attn = mtp->attn;
            auto &d = dst.attn;
            d.attn_norm = upnorm(d.attn_norm, H);
            d.wq = upmat(d.wq);
            d.wk = upmat(d.wk);
            d.wv = upmat(d.wv);
            d.wo = upmat(d.wo);
            d.q_norm = norm_repeated(d.q_norm, D, Q);
            d.k_norm = norm_repeated(d.k_norm, D, g.n_head_kv);
            size_t bytes = size_t(ctx) * g.n_head_kv * D * (opt_.kv == "f16" ? 2 : 4);
            st.key.alloc(bytes);
            st.value.alloc(bytes);
            dst.moe = mtp->moe;
            auto &m = dst.moe;
            m.gate_inp = upnorm(m.gate_inp, H * g.n_expert);
            m.gate_inp_shexp = upnorm(m.gate_inp_shexp, H);
            m.gate_shexp = upmat(m.gate_shexp);
            m.up_shexp = upmat(m.up_shexp);
            m.down_shexp = upmat(m.down_shexp);
            auto &bank = banks_[g.n_layers];
            auto &e = m.experts[0];
            bank.layout = k::native_expert_layout(e.gate.type, e.down.type, H, g.n_ff_exp);
            if (!k::cpu::native_fmt(e.gate.type, e.down.type, H, g.n_ff_exp, bank.fmt, err))
                throw std::runtime_error("draft expert format: " + err);
            bank.fmt.swiglu = g_swiglu_vec;
            bank.host.resize(bank.layout.bytes * g.n_expert);
            for (int j = 0; j < g.n_expert; ++j) {
                const auto &ew = m.experts[j];
                auto *b = bank.host.data() + j * bank.layout.bytes;
                std::memcpy(b, ew.gate.data, bank.layout.up_off);
                std::memcpy(b + bank.layout.up_off, ew.up.data, bank.layout.down_off - bank.layout.up_off);
                std::memcpy(b + bank.layout.down_off, ew.down.data, bank.layout.bytes - bank.layout.down_off);
            }
            draft_eh_ = upmat(mtp->eh_proj);
            draft_output_ = upmat(mtp->output);
            draft_enorm_ = upnorm(mtp->enorm, H);
            draft_hnorm_ = upnorm(mtp->hnorm, H);
            draft_head_norm_ = upnorm(mtp->head_norm, H);
        }
        // State checkpoints: spec+2 for verification when a draft is loaded, one for the end of the last
        // prompt, and `prompt_cache_slots` periodic ones (a chat template diverges at the previous
        // prompt's end, so periodic points are what make a resume possible).  The slot indices and the
        // allocation are both derived from the *effective* verification count, so a session built without
        // a draft cannot reserve draft-sized snapshot space; qwen35_gpu_plan() adds the same number of
        // state copies, which is what the drift check below compares.
        const int64_t verification_slots = mtp ? opt_.spec + 2 : 0;
        prompt_end_slot_ = verification_slots;
        const int64_t periodic_slots = opt_.prompt_cache ? std::max(0, opt_.prompt_cache_slots) : 0;
        if (snapshot_bytes_)
            snapshot_.alloc(snapshot_bytes_ *
                            size_t(verification_slots + (opt_.prompt_cache ? 1 + periodic_slots : 0)));
        const GpuMemoryPlan plan = qwen35_gpu_plan(g, w, mtp, ctx, opt_.kv, opt_.spec, opt_.prompt_cache, opt_.prompt_cache_slots);
        int64_t slots = opt_.expert_slots;
        if (slots < 0) {
            uint64_t free_bytes = 0;
            if (!gpu_free_bytes(free_bytes, err))
                throw std::runtime_error("device memory query: " + err);
            slots = gpu_expert_slot_capacity(plan, free_bytes);
        }
        slots = std::min(slots, plan.n_expert_pairs);
        if (slots > 0) {
            if (!cache_.open(slots, g.n_layers, g.n_expert, banks_[0].layout.bytes, err))
                throw std::runtime_error(err);
            cache_.set_per_layer_admission(true);
        }
        // The plan is computed by the same formulas this constructor allocates with; comparing it against the
        // bytes actually taken is what says the fit report a launcher refuses on is still telling the truth.
        uint64_t allocated = cache_.valid() ? uint64_t(cache_.bytes()) : 0;
        for (const auto &a : weights_)
            allocated += a.bytes;
        for (const auto &s : state_)
            allocated += s.conv.bytes + s.rec.bytes + s.key.bytes + s.value.bytes;
        allocated += scratch_.bytes + q8_.bytes + expert_scratch_.bytes + plans_.bytes + snapshot_.bytes;
        const uint64_t planned = plan.with_slots(slots);
        std::fprintf(stderr,
                     "qwen35 resident GPU: KV=%s slots=%lld workers=%d dense/state ready; CPU expert backing "
                     "%.3f GiB\n",
                     opt_.kv.c_str(), (long long)slots, pool_->workers(),
                     double(banks_[0].host.size() * g.n_layers) / (1ull << 30));
        std::fprintf(stderr,
                     "qwen35 GPU plan: dense=%.3f GiB gdn=%.3f GiB kv=%.3f GiB scratch=%.3f GiB; "
                     "planned %.3f GiB, allocated %.3f GiB\n",
                     double(plan.dense) / (1ull << 30), double(plan.state) / (1ull << 30),
                     double(plan.kv) / (1ull << 30), double(plan.scratch) / (1ull << 30),
                     double(planned) / (1ull << 30), double(allocated) / (1ull << 30));
        if (planned && (allocated > planned + planned / 100 || planned > allocated + allocated / 100))
            std::fprintf(stderr,
                         "qwen35 GPU plan: WARNING plan and allocation differ by %.1f%%; report this\n",
                         100.0 * double(allocated > planned ? allocated - planned : planned - allocated) /
                             double(planned));
        reset();
        // The pool reserves a physical core for its submitting thread. Honor that
        // reservation while waiting on HIP and while the host drains expert rows.
        if (!std::getenv("STRATA_QWEN35_NO_HOST_PIN"))
            previous_affinity_ = k::cpu::pin_current_thread(k::cpu::detect_cpu_topology(true).host_core);
    }
    ~GpuSession() override {
        if (stream_)
            (void)cudaStreamSynchronize(stream_);
        if (host_)
            (void)cudaFreeHost(host_);
        if (host_ids_)
            (void)cudaFreeHost(host_ids_);
        if (cpu_parts_)
            (void)cudaFreeHost(cpu_parts_);
        if (handoff_)
            (void)cudaEventDestroy(handoff_);
        if (host_plan_)
            (void)cudaFreeHost(host_plan_);
        if (stream_)
            (void)cudaStreamDestroy(stream_);
        k::cpu::restore_thread_affinity(previous_affinity_);
        std::fprintf(stderr, "qwen35 GPU totals: hits=%llu misses=%llu CPU=%.3f ms GPU-wait=%.3f ms\n",
                     (unsigned long long)hits_, (unsigned long long)misses_, cpu_ms_, wait_ms_);
        std::fprintf(stderr, "qwen35 transfers: activation D2H=%llu parts H2D=%llu cache H2D=%llu bytes\n",
                     (unsigned long long)activation_d2h_bytes_, (unsigned long long)parts_h2d_bytes_,
                     (unsigned long long)cache_h2d_bytes_);
        for (const auto &p : gpu_ms_)
            std::fprintf(stderr, "qwen35 GPU profile: %s=%.3f ms\n", p.first.c_str(), p.second);
    }
    void reset() override {
        placement_frozen_ = false;
        std::fill(prompt_routes_.begin(), prompt_routes_.end(), 0);
        checked(cudaStreamSynchronize(stream_));
        for (auto &s : state_)
            if (s.rec.p) {
                s.rec.zero(stream_);
                s.conv.zero(stream_);
            }
        pos_ = 0;
        verifying_ = false;
        checkpos_.clear();
        draft_pos_ = 0;
        prompt_cache_state_.forget();
        periodic_.clear();
        periodic_slots_.clear();
        periodic_writes_ = 0;
        next_periodic_ = 0;
    }
    void begin_decode() override {
        // Learn from the complete prompt, not its first distinct expert ids. The selected
        // placement stays fixed throughout decoding/verification, so rejected drafts cannot
        // change it. A later request may choose a different placement from its own prompt.
        if (cache_.valid() && profile_cache_) {
            std::vector<std::pair<int, int>> chosen;
            bool refill = false;
            for (int l = 0; l < g_.n_layers; ++l) {
                int64_t lo = 0, hi = 0;
                cache_.layer_slot_range(l, lo, hi);
                std::vector<int> rank(g_.n_expert);
                for (int e = 0; e < g_.n_expert; ++e)
                    rank[e] = e;
                auto count = [&](int e) { return prompt_routes_[size_t(l) * g_.n_expert + e]; };
                std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) { return count(a) > count(b); });
                for (int i = 0; i < std::min<int64_t>(hi - lo, g_.n_expert) && count(rank[i]); ++i) {
                    chosen.emplace_back(l, rank[i]);
                    refill = refill || cache_.slot_of(l, rank[i]) < 0;
                }
            }
            if (refill) {
                checked(cudaStreamSynchronize(stream_));
                const auto slots = cache_.slots();
                cache_.close();
                std::string err;
                if (!cache_.open(slots, g_.n_layers, g_.n_expert, banks_[0].layout.bytes, err))
                    throw std::runtime_error(err);
                cache_.set_per_layer_admission(true);
                for (auto [l, e] : chosen) {
                    const int slot = cache_.admit(l, e);
                    if (slot < 0 ||
                        !cache_.fill_slot(slot, banks_[l].host.data() + size_t(e) * banks_[l].layout.bytes,
                                          stream_, err, banks_[l].layout.bytes))
                        throw std::runtime_error("prompt cache placement: " + err);
                    cache_h2d_bytes_ += banks_[l].layout.bytes;
                }
                checked(cudaStreamSynchronize(stream_));
            }
        }
        placement_frozen_ = true;
    }
    int64_t position() const override { return pos_; }
    int64_t context() const override { return ctx_; }
    bool has_mtp() const override { return mtp_ != nullptr; }
    void target(int64_t token, float *logits, float *hidden) override {
        if (token < 0 || token >= g_.n_vocab || pos_ >= ctx_)
            throw std::out_of_range("GPU token/context");
        dequant_row(w_.token_embd, token, host_);
        checked(cudaMemcpyAsync(x_, host_, g_.n_embd * 4, cudaMemcpyHostToDevice, stream_));
        fillpos<<<1, 32, 0, stream_>>>(positions_, g_.n_head, pos_);
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.is_recurrent(l))
                graph(l, 0, 1, [&] { gdn(l); });
            else
                attention(l);
            add<<<(g_.n_embd + 255) / 256, 256, 0, stream_>>>(x_, a_, g_.n_embd);
            moe(l);
            if (g_stage_trace) {
                std::vector<float> v(g_.n_embd);
                checked(cudaMemcpyAsync(v.data(), x_, v.size() * 4, cudaMemcpyDeviceToHost, stream_));
                checked(cudaStreamSynchronize(stream_));
                g_stage_trace(l, "l_out", v.data(), v.size());
            }
        }
        mark(g_.n_layers, 7, "target output/head");
        norm(x_, w_.output_norm, xn_, g_.n_embd);
        if (hidden)
            checked(cudaMemcpyAsync(hidden, xn_, g_.n_embd * 4, cudaMemcpyDeviceToHost, stream_));
        if (logits) {
            mv(w_.output, xn_, logits_);
            checked(cudaMemcpyAsync(logits, logits_, g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_));
        }
        mark(g_.n_layers, 7, "target output/head", true);
        ++pos_;
        prompt_cache_state_.consumed.push_back(token);
        if (verifying_) {
            if (checkpos_.size() >= size_t(opt_.spec + 2))
                throw std::logic_error("verification exceeds window");
            checkpoint(checkpos_.size());
            checkpos_.push_back(pos_);
        } else {
            maybe_periodic_checkpoint();
        }
        checked(cudaStreamSynchronize(stream_));
        harvest();
    }
    int prefill_batch_size() const override { return std::min(Batch, opt_.prefill); }
    void target_batch(const int64_t *tokens, int nt, float *const *logits, float *const *hidden) override {
        if (nt < 1 || nt > Batch || pos_ + nt > ctx_)
            throw std::out_of_range("GPU batch/context");
        if (verifying_ && checkpos_.size() + nt > size_t(opt_.spec + 2))
            throw std::logic_error("verification exceeds window");
        for (int t = 0; t < nt; ++t)
            if (tokens[t] < 0 || tokens[t] >= g_.n_vocab)
                throw std::out_of_range("GPU token");
        if (nt == 1 || g_stage_trace) {
            for (int t = 0; t < nt; ++t)
                target(tokens[t], logits ? logits[t] : nullptr, hidden ? hidden[t] : nullptr);
            return;
        }
        const int H = g_.n_embd;
        for (int t = 0; t < nt; ++t)
            dequant_row(w_.token_embd, tokens[t], host_ + t * H);
        checked(cudaMemcpyAsync(x_, host_, nt * H * 4, cudaMemcpyHostToDevice, stream_));
        size_t off = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.is_recurrent(l)) {
                if (verifying_)
                    gdn_batch(l, nt, off);
                else
                    graph(l, 0, nt, [&] { gdn_batch(l, nt, off); });
                off += state_[l].conv.bytes + state_[l].rec.bytes;
            } else
                attention_batch(l, nt);
            add<<<(nt * H + 255) / 256, 256, 0, stream_>>>(x_, a_, nt * H);
            moe(l, nt);
        }
        mark(g_.n_layers, 7, "target output/head");
        for (int t = 0; t < nt; ++t) {
            norm(x_ + t * H, w_.output_norm, xn_ + t * H, H);
            if (hidden && hidden[t])
                checked(cudaMemcpyAsync(hidden[t], xn_ + t * H, H * 4, cudaMemcpyDeviceToHost, stream_));
        }
        bool all = logits != nullptr;
        for (int t = 0; t < nt && all; ++t)
            all = logits[t] != nullptr;
        if (all) {
            mv(w_.output, xn_, logits_, false, nt);
            for (int t = 0; t < nt; ++t)
                checked(cudaMemcpyAsync(logits[t], logits_ + size_t(t) * g_.n_vocab, g_.n_vocab * 4,
                                        cudaMemcpyDeviceToHost, stream_));
        } else if (logits)
            for (int t = 0; t < nt; ++t)
                if (logits[t]) {
                    mv(w_.output, xn_ + t * H, logits_);
                    checked(
                        cudaMemcpyAsync(logits[t], logits_, g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_));
                }
        mark(g_.n_layers, 7, "target output/head", true);
        if (verifying_)
            for (int t = 0; t < nt; ++t)
                checkpos_.push_back(pos_ + t + 1);
        pos_ += nt;
        prompt_cache_state_.consumed.insert(prompt_cache_state_.consumed.end(), tokens, tokens + nt);
        maybe_periodic_checkpoint();
        checked(cudaStreamSynchronize(stream_));
        harvest();
    }
    void draft(int64_t token, const float *h, float *logits, float *next) override {
        if (!mtp_ || !h || !next)
            throw std::logic_error("no draft");
        if (token < 0 || token >= g_.n_vocab || draft_pos_ >= ctx_)
            throw std::out_of_range("GPU draft token/context");
        const int H = g_.n_embd;
        dequant_row(mtp_->embedding, token, host_);
        checked(cudaMemcpyAsync(x_, host_, H * 4, cudaMemcpyHostToDevice, stream_));
        norm(x_, draft_enorm_, qkv_, H);
        checked(cudaMemcpyAsync(x_, h, H * 4, cudaMemcpyHostToDevice, stream_));
        norm(x_, draft_hnorm_, qkv_ + H, H);
        mv(draft_eh_, qkv_, x_);
        fillpos<<<1, g_.n_head, 0, stream_>>>(positions_, g_.n_head, draft_pos_);
        attention(g_.n_layers);
        add<<<(H + 255) / 256, 256, 0, stream_>>>(x_, a_, H);
        moe(g_.n_layers);
        mark(g_.n_layers, 7, "draft output/head");
        norm(x_, draft_head_norm_, xn_, H);
        checked(cudaMemcpyAsync(next, xn_, H * 4, cudaMemcpyDeviceToHost, stream_));
        if (logits) {
            mv(draft_output_, xn_, logits_);
            checked(cudaMemcpyAsync(logits, logits_, g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_));
        }
        ++draft_pos_;
        mark(g_.n_layers, 7, "draft output/head", true);
        checked(cudaStreamSynchronize(stream_));
        harvest();
    }
    void begin_verify() override {
        if (verifying_ || !mtp_)
            throw std::logic_error("invalid verification");
        checkpos_ = {pos_};
        prompt_cache_state_.verify_from = (int64_t) prompt_cache_state_.consumed.size();
        checkpoint(0);
        verifying_ = true;
    }
    void commit_verify(int64_t accepted) override {
        if (!verifying_ || accepted < 0 || size_t(accepted) >= checkpos_.size())
            throw std::logic_error("invalid accepted prefix");
        if (size_t(accepted) + 1 != checkpos_.size())
            checkpoint(accepted, true);
        pos_ = checkpos_[accepted];
        checkpos_.clear();
        verifying_ = false;
        // Rejected draft tokens were read but never accepted: the cache must hold only the accepted
        // prefix, or a later prompt would be matched against tokens the model did not keep.
        prompt_cache_state_.consumed.resize((size_t) (prompt_cache_state_.verify_from + accepted));
    }
    void rewind_draft(int64_t p) override {
        if (!mtp_ || p < 0 || p > draft_pos_)
            throw std::logic_error("draft rewind");
        draft_pos_ = p;
    }
    int64_t consumed_tokens() const override { return (int64_t) prompt_cache_state_.consumed.size(); }
    const float *cached_logits() const override {
        return prompt_cache_state_.logits.empty() ? nullptr : prompt_cache_state_.logits.data();
    }
    int64_t prepare_prompt(const int64_t *tokens, int64_t count) override {
        // Same rule as the CPU tier (PromptCacheState::plan): resume from the deepest checkpoint the
        // prompt still contains, and read nothing at all when the prompt *is* the end-of-prompt prefix,
        // because its logits were kept.  GDN state cannot be rewound token by token, so checkpoints are
        // all the granularity there is.
        const int64_t at = opt_.prompt_cache
                               ? prompt_cache_state_.plan(tokens, count, g_.n_vocab, periodic_)
                               : 0;
        if (std::getenv("STRATA_QWEN35_CACHE_DEBUG"))
            std::fprintf(stderr,
                         "qwen35 prompt cache: count=%lld checkpoint=%lld prefix=%lld periodic=%lld "
                         "resume=%lld\n",
                         (long long) count, (long long) prompt_cache_state_.checkpoint,
                         (long long) reused_prefix(prompt_cache_state_.consumed, tokens, count),
                         (long long) periodic_.size(), (long long) at);
        if (at > 0) {
            int64_t slot = prompt_end_slot_;
            if (at != prompt_cache_state_.checkpoint) {
                const auto it = std::find(periodic_.begin(), periodic_.end(), at);
                if (it == periodic_.end()) throw std::logic_error("prompt resume without a checkpoint");
                slot = periodic_slots_[(size_t) (it - periodic_.begin())];
            }
            checkpoint(slot, true);
            pos_ = at;
            draft_pos_ = at;
            prompt_cache_state_.consumed.resize((size_t) at);
            checkpos_.clear();
            verifying_ = false;
            // The routing profile is *accumulated* across a resumed conversation rather than rebuilt from
            // the few new tokens: decode placement is ranked from the experts the whole conversation has
            // exercised, so it stays stable turn to turn instead of being chosen from a short tail.
            // begin_decode() freezes it again from this request's reads.
            placement_frozen_ = false;
            // Resume points beyond the one used describe state this request is about to re-read; keeping
            // them would let a later prompt resume into a state the conversation no longer matches.  The
            // interval is for the span that is actually left to read, not for the whole prompt: a resumed
            // turn re-reads a tail, and its checkpoints belong inside that tail.
            periodic_.clear();
            periodic_slots_.clear();
            prompt_cache_state_.every =
                PromptCacheState::interval_for(count - at, opt_.prompt_cache_slots);
            next_periodic_ = at + prompt_cache_state_.every;
            return at;
        }
        reset();
        prompt_cache_state_.every = PromptCacheState::interval_for(count, opt_.prompt_cache_slots);
        next_periodic_ = prompt_cache_state_.every;
        return 0;
    }
    void checkpoint_prompt(const float *logits) override {
        if (!opt_.prompt_cache || !snapshot_.p) return;
        checkpoint(prompt_end_slot_);
        prompt_cache_state_.checkpoint = pos_;
        prompt_cache_state_.valid = true;
        prompt_cache_state_.remember(logits, g_.n_vocab);
    }
    void maybe_periodic_checkpoint() {
        if (!opt_.prompt_cache || verifying_ || prompt_cache_state_.every <= 0 ||
            opt_.prompt_cache_slots <= 0 || !snapshot_.p)
            return;
        if (pos_ < next_periodic_) return;
        if (!periodic_.empty() && periodic_.back() == pos_) return;
        if ((int) periodic_.size() >= opt_.prompt_cache_slots) {
            periodic_.erase(periodic_.begin());
            periodic_slots_.erase(periodic_slots_.begin());
        }
        const int64_t slot = prompt_end_slot_ + 1 + (periodic_writes_++ % opt_.prompt_cache_slots);
        checkpoint(slot);
        periodic_.push_back(pos_);
        periodic_slots_.push_back(slot);
        next_periodic_ = pos_ + prompt_cache_state_.every;
    }
};
} // namespace
std::unique_ptr<InferenceSession> make_gpu_session(const Qwen35Geometry &g, const TrunkWeights &w,
                                                   const MtpWeights *mtp, int64_t ctx, const GpuOptions &o) {
    return std::make_unique<GpuSession>(g, w, mtp, ctx, o);
}
// Everything GpuSession puts in VRAM, summed the same way it allocates it. `context` is the KV capacity,
// so the KV term is the one a launcher's --max-context/--kv choice moves.
GpuMemoryPlan qwen35_gpu_plan(const Qwen35Geometry &g, const TrunkWeights &w, const MtpWeights *mtp,
                              int64_t ctx, const std::string &kv, int spec, bool prompt_cache, int prompt_cache_slots) {
    if (mtp && (spec < 1 || spec > 4))
        throw std::invalid_argument("GPU MTP plan spec must be 1..4");
    GpuMemoryPlan p;
    const int64_t H = g.n_embd, Q = g.n_head, D = g.head_dim, KV = g.n_head_kv;
    const size_t kv_bytes = kv == "f16" ? 2 : 4;
    const auto f4 = [](int64_t n) { return uint64_t(n) * 4; };
    const auto mat = [](const Mat &m) { return uint64_t(m.row_bytes) * uint64_t(m.n_out); };
    const auto attn = [&](const AttnLayerWeights &a) {
        p.dense += f4(H) + mat(a.wq) + mat(a.wk) + mat(a.wv) + mat(a.wo) + f4(D * Q) + f4(D * KV);
        const uint64_t cell = uint64_t(ctx) * KV * D * kv_bytes;
        p.kv += 2 * cell;
    };
    const auto moe = [&](const MoeLayerWeights &m) {
        p.dense += f4(H * g.n_expert) + f4(H) + mat(m.gate_shexp) + mat(m.up_shexp) + mat(m.down_shexp);
    };
    for (int64_t l = 0; l < g.n_layers; ++l) {
        p.dense += f4(H); // the post-attention norm every layer uploads
        if (g.is_recurrent(l)) {
            const auto &d = w.gdn[size_t(l)];
            p.dense += f4(H) + mat(d.wqkv) + mat(d.wgate) + mat(d.ssm_beta) + mat(d.ssm_alpha) +
                       mat(d.ssm_out) + f4(g.conv_channels() * 4) + f4(g.ssm_dt_rank) + f4(g.ssm_dt_rank) +
                       f4(g.ssm_state);
            p.state +=
                uint64_t(g.conv_channels()) * 3 * 4 + uint64_t(g.ssm_state) * g.ssm_state * g.ssm_dt_rank * 4;
        } else {
            attn(w.attn[size_t(l)]);
        }
        moe(w.moe[size_t(l)]);
    }
    p.dense += mat(w.output) + f4(H);
    if (mtp) { // the draft is one full attention+MoE block, its own KV, and spec+2 state checkpoints
        p.dense += f4(H) + f4(H) * 3;
        attn(mtp->attn);
        moe(mtp->moe);
        p.dense += mat(mtp->eh_proj) + mat(mtp->output);
        p.scratch += (spec + 2) * p.state;
    }
    // The prompt cache's checkpoints: one at the end of each prompt (needed with or without MTP; spec 0 is
    // the shipping default) plus the periodic ones that make a resume usable with a chat template.  The
    // session allocates the same number of state copies with the same expression.
    if (prompt_cache)
        p.scratch += p.state * uint64_t(1 + std::max(0, prompt_cache_slots));
    p.scratch += scratch_floats(g, ctx) * 4 + Batch * size_t(g.n_expert_used) * 4 + size_t(Q) * 4 +
                 k::native_q8_1_bytes(g.conv_channels(), Batch) +
                 k::native_expert_scratch_bytes(Batch * g.n_expert_used, g.n_ff_exp) +
                 8 * Batch * size_t(g.n_expert_used) + 4 * (4 * Batch * size_t(g.n_expert_used) + 2);
    if (!w.moe.empty() && w.moe[0].experts != nullptr)
        p.expert_blob = k::native_expert_layout(w.moe[0].experts[0].gate.type, w.moe[0].experts[0].down.type,
                                                g.n_embd, g.n_ff_exp)
                            .bytes;
    p.n_expert_pairs = g.n_layers * g.n_expert;
    return p;
}

bool gpu_free_bytes(uint64_t &free_bytes, std::string &err) {
    size_t free = 0, total = 0;
    const cudaError_t e = cudaMemGetInfo(&free, &total);
    if (e != cudaSuccess) {
        err = cudaGetErrorString(e);
        return false;
    }
    free_bytes = free;
    return true;
}

int64_t gpu_expert_slot_capacity(const GpuMemoryPlan &plan, uint64_t free_bytes) {
    if (plan.expert_blob == 0)
        return 0;
    const uint64_t reservation = 256ull << 20;
    const uint64_t usable = free_bytes > reservation ? free_bytes - reservation : 0;
    return (int64_t)std::min<uint64_t>(usable / plan.expert_blob, uint64_t(plan.n_expert_pairs));
}
} // namespace strata::qwen35
