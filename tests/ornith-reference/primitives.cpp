// Tests Strata's native kernels against independently executed pinned GGML HIP graphs.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/qwen35/gpu.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata::kernels;
static void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Device {
    float *p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc((void **)&p, n * sizeof(float))); }
    explicit Device(const std::vector<float> &x) : Device(x.size()) {
        check(cudaMemcpy(p, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    }
    ~Device() {
        if (p)
            cudaFree(p);
    }
    std::vector<float> get(size_t n) {
        std::vector<float> x(n);
        check(cudaMemcpy(x.data(), p, n * 4, cudaMemcpyDeviceToHost));
        return x;
    }
};
struct Oracle {
    ggml_context *ctx;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer = nullptr;
    Oracle() {
        ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        backend = ggml_backend_cuda_init(0);
        if (!ctx || !backend)
            throw std::runtime_error("oracle init failed");
    }
    ~Oracle() {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    ggml_tensor *tensor(int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
        return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d);
    }
    void alloc() {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer)
            throw std::runtime_error("oracle allocation failed");
    }
    void put(ggml_tensor *t, const std::vector<float> &v) {
        if (v.size() * 4 != ggml_nbytes(t))
            throw std::runtime_error("oracle input shape");
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
    }
    std::vector<float> run(ggml_tensor *t) {
        auto *g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, t);
        if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("oracle compute failed");
        std::vector<float> x(ggml_nelements(t));
        ggml_backend_tensor_get(t, x.data(), 0, x.size() * 4);
        return x;
    }
};
static std::mt19937 rng(35);
static std::vector<float> randoms(size_t n, float scale = 1) {
    std::normal_distribution<float> d(0, scale);
    std::vector<float> x(n);
    for (auto &v : x)
        v = d(rng);
    return x;
}
static void compare(const char *name, const std::vector<float> &a, const std::vector<float> &b,
                    double tolerance = 1e-6) {
    if (a.size() != b.size())
        throw std::runtime_error("comparison shape");
    double max = 0, ss = 0, rr = 0;
    size_t at = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::runtime_error("nonfinite primitive");
        double e = double(a[i]) - b[i];
        if (std::abs(e) > max) {
            max = std::abs(e);
            at = i;
        }
        ss += e * e;
        rr += double(b[i]) * b[i];
    }
    double rel = std::sqrt(ss / std::max(rr, 1e-30));
    std::printf("%s n=%zu max_abs=%.9g rms=%.9g relative_rms=%.9g worst=%zu\n", name, a.size(), max,
                std::sqrt(ss / a.size()), rel, at);
    if (rel > tolerance || max > tolerance * std::max(1.0, 10 * std::sqrt(rr / a.size())))
        throw std::runtime_error(std::string(name) + " parity failed");
}
static void preprocessing(cudaStream_t stream) {
    {
        Oracle o;
        auto *x = o.tensor(128, 32);
        auto *y = ggml_scale(o.ctx, ggml_rms_norm(o.ctx, x, 1e-6f / 128), 1 / std::sqrt(128.f));
        o.alloc();
        auto v = randoms(128 * 32);
        o.put(x, v);
        Device d(v);
        native_gdn_l2_norm(d.p, 32, 128, 1e-6f, stream);
        check(cudaStreamSynchronize(stream));
        compare("L2", d.get(v.size()), o.run(y));
    }
    {
        Oracle o;
        auto *x = o.tensor(128, 32);
        auto *z = o.tensor(128, 32);
        auto *g = o.tensor(128);
        auto *y = ggml_mul(o.ctx, ggml_mul(o.ctx, ggml_rms_norm(o.ctx, x, 1e-6f), g), ggml_silu(o.ctx, z));
        o.alloc();
        auto xv = randoms(4096), zv = randoms(4096), gv = randoms(128);
        o.put(x, xv);
        o.put(z, zv);
        o.put(g, gv);
        Device dx(xv), dz(zv), dg(gv), dy(4096);
        native_gdn_out_norm_silu(dx.p, dz.p, dg.p, dy.p, 32, 128, 1e-6f, stream);
        check(cudaStreamSynchronize(stream));
        compare("GDN SiLU output", dy.get(4096), o.run(y));
    }
    {
        Oracle o;
        auto *a = o.tensor(32);
        auto *dt = o.tensor(32);
        auto *ssm = o.tensor(32);
        auto *y = ggml_mul(o.ctx, ggml_softplus(o.ctx, ggml_add(o.ctx, a, dt)), ssm);
        o.alloc();
        auto av = randoms(32, 20), dv = randoms(32), sv = randoms(32);
        av[0] = -100;
        av[1] = 1000;
        o.put(a, av);
        o.put(dt, dv);
        o.put(ssm, sv);
        Device da(av), dd(dv), ds(sv), dy(32);
        native_gdn_gate_qwen35(da.p, dd.p, ds.p, dy.p, 32, stream);
        check(cudaStreamSynchronize(stream));
        compare("softplus gate", dy.get(32), o.run(y));
    }
    {
        Oracle o;
        auto *x = o.tensor(32);
        auto *y = ggml_sigmoid(o.ctx, x);
        o.alloc();
        auto v = randoms(32, 20);
        o.put(x, v);
        Device d(v);
        native_gdn_beta_gate(d.p, 32, stream);
        check(cudaStreamSynchronize(stream));
        compare("beta sigmoid", d.get(32), o.run(y));
    }
    {
        Oracle o;
        auto *x = o.tensor(4, 8192);
        auto *w = o.tensor(4, 8192);
        auto *raw = ggml_ssm_conv(o.ctx, x, w);
        auto *y = ggml_silu(o.ctx, raw);
        o.alloc();
        auto v = randoms(8192 * 4), wv = randoms(8192 * 4);
        o.put(x, v);
        o.put(w, wv);
        std::vector<float> hist(8192 * 3), in(8192);
        for (int c = 0; c < 8192; ++c) {
            for (int i = 0; i < 3; ++i)
                hist[c * 3 + i] = v[c * 4 + i];
            in[c] = v[c * 4 + 3];
        }
        Device dh(hist), di(in), dw(wv), dr(8192), dy(8192);
        native_gdn_conv_silu(dh.p, di.p, dw.p, dr.p, dy.p, 8192, 4, stream);
        check(cudaStreamSynchronize(stream));
        compare("conv raw", dr.get(8192), o.run(raw));
        compare("conv SiLU", dy.get(8192), o.run(y));
        auto hn = dh.get(hist.size());
        for (int c = 0; c < 8192; ++c)
            for (int i = 0; i < 3; ++i)
                if (hn[c * 3 + i] != v[c * 4 + i + 1])
                    throw std::runtime_error("conv history");
    }
    for (int cols : {128, 256, 2048}) {
        Oracle o;
        auto *x = o.tensor(cols, 16);
        auto *g = o.tensor(cols, 16);
        auto *y = ggml_mul(o.ctx, ggml_rms_norm(o.ctx, x, 1e-6f), g);
        o.alloc();
        auto xv = randoms(cols * 16), gv = randoms(cols * 16);
        o.put(x, xv);
        o.put(g, gv);
        Device dx(xv), dg(gv), dy(xv.size());
        native_gr_rms_norm_weighted(dx.p, dg.p, dy.p, cols, 16, 1e-6f, stream);
        check(cudaStreamSynchronize(stream));
        compare("weighted RMS", dy.get(xv.size()), o.run(y));
    }
}
static void recurrence(cudaStream_t stream) {
    Oracle o;
    auto *q = o.tensor(128, 16), *k = o.tensor(128, 16), *v = o.tensor(128, 32), *g = o.tensor(1, 32),
         *b = o.tensor(1, 32), *s = o.tensor(128, 128, 32);
    auto *y = ggml_gated_delta_net(o.ctx, q, k, v, g, b, s, 1);
    o.alloc();
    auto state = randoms(128 * 128 * 32, 0.02f);
    std::vector<float> native(state.size());
    for (int h = 0; h < 32; ++h)
        for (int j = 0; j < 128; ++j)
            for (int i = 0; i < 128; ++i)
                native[(i * 32 + h) * 128 + j] = state[(h * 128 + j) * 128 + i];
    Device ds(native), dq(2048), dk(2048), dv(4096), dg(32), db(32), out(4096);
    for (int step = 0; step < 32; ++step) {
        auto qv = randoms(2048, .08f), kv = randoms(2048, .08f), vv = randoms(4096), gv = randoms(32),
             bv = randoms(32);
        for (auto &x : gv)
            x = -std::abs(x);
        for (auto &x : bv)
            x = 1 / (1 + std::exp(-x));
        o.put(q, qv);
        o.put(k, kv);
        o.put(v, vv);
        o.put(g, gv);
        o.put(b, bv);
        o.put(s, state);
        for (auto pair : {std::make_pair(dq.p, &qv), {dk.p, &kv}, {dv.p, &vv}, {dg.p, &gv}, {db.p, &bv}})
            check(cudaMemcpyAsync(pair.first, pair.second->data(), pair.second->size() * 4,
                                  cudaMemcpyHostToDevice, stream));
        native_gdn_step(ds.p, dq.p, dk.p, dv.p, dg.p, db.p, out.p, {128, 16, 32}, stream);
        check(cudaStreamSynchronize(stream));
        auto ref = o.run(y);
        compare("GDN readout", out.get(4096), std::vector<float>(ref.begin(), ref.begin() + 4096));
        state.assign(ref.begin() + 4096, ref.end());
        auto ns = ds.get(native.size());
        for (int h = 0; h < 32; ++h)
            for (int j = 0; j < 128; ++j)
                for (int i = 0; i < 128; ++i)
                    native[(h * 128 + j) * 128 + i] = ns[(i * 32 + h) * 128 + j];
        compare("GDN state", native, state);
    }
}
static void rope(cudaStream_t stream) {
    for (int pos : {0, 1, 511, 4096, 131071}) {
        Oracle o;
        auto *x = o.tensor(256, 16, 1);
        auto *p = ggml_new_tensor_1d(o.ctx, GGML_TYPE_I32, 4);
        int sections[4] = {11, 11, 10, 0};
        auto *y = ggml_rope_multi(o.ctx, x, p, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE, 262144, 1e7, 1,
                                  0, 1, 32, 1);
        o.alloc();
        auto xv = randoms(4096);
        o.put(x, xv);
        int ps[4] = {pos, pos, pos, pos};
        ggml_backend_tensor_set(p, ps, 0, sizeof(ps));
        std::vector<float> pv(16);
        Device dx(xv), dy(4096), dp(16);
        std::vector<int> heads(16, pos);
        check(cudaMemcpy(dp.p, heads.data(), 64, cudaMemcpyHostToDevice));
        native_rope_apply(dx.p, dy.p, 16, 256, 64, {}, (int *)dp.p, stream);
        check(cudaStreamSynchronize(stream));
        compare("IMRoPE", dy.get(4096), o.run(y));
    }
}

static void router(cudaStream_t stream) {
    for (int which = 0; which < 4; ++which) {
        Oracle o;
        auto *x = o.tensor(256);
        auto *p = ggml_soft_max(o.ctx, x);
        auto *ids = ggml_argsort_top_k(o.ctx, p, 8);
        auto *pr = ggml_get_rows(o.ctx, ggml_reshape_2d(o.ctx, p, 1, 256), ids);
        auto *w = ggml_div(o.ctx, pr,
                           ggml_clamp(o.ctx, ggml_sum_rows(o.ctx, ggml_reshape_2d(o.ctx, pr, 8, 1)),
                                      0.00006103515625f, INFINITY));
        o.alloc();
        auto v = randoms(256, which == 1 ? 20 : 1);
        if (which == 2)
            std::fill(v.begin(), v.end(), 0);
        if (which == 3)
            for (int i = 0; i < 256; ++i)
                v[i] = i % 7;
        o.put(x, v);
        Device dx(v), di(8), dw(8);
        strata::qwen35::gpu_router(dx.p, 256, 8, (int *)di.p, dw.p, stream);
        check(cudaStreamSynchronize(stream));
        auto reference = o.run(w);
        compare("router weights", dw.get(8), reference);
        auto *g = ggml_new_graph(o.ctx);
        ggml_build_forward_expand(g, ids);
        if (ggml_backend_graph_compute(o.backend, g) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("sort compute");
        std::vector<int> ri(8), ni(8);
        ggml_backend_tensor_get(ids, ri.data(), 0, 32);
        check(cudaMemcpy(ni.data(), di.p, 32, cudaMemcpyDeviceToHost));
        if (ri != ni)
            throw std::runtime_error("router IDs/tie order");
    }
}
static void mmvq(cudaStream_t stream) {
    for (int type : {8, 12, 23})
        for (int ni : {512, 2048}) {
            Oracle o;
            int no = type == 8 ? 32 : 512;
            auto *w = ggml_new_tensor_2d(o.ctx, (ggml_type)type, ni, no);
            auto *x = o.tensor(ni);
            auto *y = ggml_mul_mat(o.ctx, w, x);
            o.alloc();
            auto wf = randoms(size_t(ni) * no, .025f), xv = randoms(ni);
            std::vector<float> im(ni, 1);
            std::vector<uint8_t> quant(ggml_nbytes(w));
            ggml_quantize_init((ggml_type)type);
            ggml_quantize_chunk((ggml_type)type, wf.data(), quant.data(), 0, no, ni, im.data());
            ggml_backend_tensor_set(w, quant.data(), 0, quant.size());
            o.put(x, xv);
            Device dw((quant.size() + 3) / 4), dx(xv), out(no), q8((native_q8_1_bytes(ni) + 3) / 4);
            check(cudaMemcpy(dw.p, quant.data(), quant.size(), cudaMemcpyHostToDevice));
            native_quantize_q8_1(dx.p, q8.p, ni, 1, stream);
            native_mmvq_rdna3(type, dw.p, q8.p, out.p, ni, no, 1, stream);
            check(cudaStreamSynchronize(stream));
            compare("native MMVQ", out.get(no), o.run(y));
        }
}
// Separate multiply and add graphs mirror the canonical unfused model contract.
static void moe_combine(cudaStream_t stream) {
    for (int K : {1, 2, 8, 15}) {
        Oracle products;
        auto *experts = products.tensor(2048, K);
        auto *weights = products.tensor(1, K);
        auto *weighted = ggml_mul(products.ctx, experts, weights);
        products.alloc();
        auto ev = randoms(size_t(2048) * K), wv = randoms(K, .2f);
        products.put(experts, ev);
        products.put(weights, wv);
        auto pv = products.run(weighted);
        Oracle reduction;
        auto *input = reduction.tensor(2048, K);
        auto *shared = reduction.tensor(2048);
        auto *sum = ggml_view_1d(reduction.ctx, input, 2048, 0);
        for (int e = 1; e < K; ++e)
            sum =
                ggml_add(reduction.ctx, sum, ggml_view_1d(reduction.ctx, input, 2048, size_t(e) * 2048 * 4));
        auto *out = ggml_add(reduction.ctx, sum, shared);
        reduction.alloc();
        auto sv = randoms(2048);
        reduction.put(input, pv);
        reduction.put(shared, sv);
        Device de(ev), dw(wv), ds(sv), dout(2048);
        strata::qwen35::gpu_moe_combine(de.p, dw.p, ds.p, dout.p, 2048, K, stream);
        check(cudaStreamSynchronize(stream));
        compare("MoE weighted combine + shared", dout.get(2048), reduction.run(out), 0);
    }
}
static void grouped_experts(cudaStream_t stream) {
    constexpr int H = 2048, F = 512, E = 8, K = 8;
    const auto layout = native_expert_layout(23, 12, H, F);
    for (int nt : {1, 2, 4}) {
        Oracle o;
        auto *gate = ggml_new_tensor_3d(o.ctx, GGML_TYPE_IQ4_XS, H, F, E),
             *up = ggml_new_tensor_3d(o.ctx, GGML_TYPE_IQ4_XS, H, F, E),
             *down = ggml_new_tensor_3d(o.ctx, GGML_TYPE_Q4_K, F, H, E);
        auto *x = o.tensor(H, 1, nt);
        auto *ids = ggml_new_tensor_2d(o.ctx, GGML_TYPE_I32, K, nt);
        auto *act = ggml_swiglu_split(o.ctx, ggml_mul_mat_id(o.ctx, gate, x, ids),
                                      ggml_mul_mat_id(o.ctx, up, x, ids));
        auto *y = ggml_mul_mat_id(o.ctx, down, act, ids);
        o.alloc();
        auto make_weights = [&](ggml_tensor *t, int width, int rows) {
            auto v = randoms(size_t(width) * rows, .025f);
            std::vector<float> im(width, 1);
            std::vector<uint8_t> q(ggml_nbytes(t));
            ggml_quantize_init(t->type);
            ggml_quantize_chunk(t->type, v.data(), q.data(), 0, rows, width, im.data());
            ggml_backend_tensor_set(t, q.data(), 0, q.size());
            return q;
        };
        auto gq = make_weights(gate, H, F * E), uq = make_weights(up, H, F * E),
             dq = make_weights(down, F, H * E);
        std::vector<uint8_t> pack(E * layout.bytes);
        for (int e = 0; e < E; ++e) {
            auto *p = pack.data() + e * layout.bytes;
            std::copy_n(gq.data() + e * layout.up_off, layout.up_off, p);
            std::copy_n(uq.data() + e * layout.up_off, layout.up_off, p + layout.up_off);
            std::copy_n(dq.data() + e * (layout.bytes - layout.down_off), layout.bytes - layout.down_off,
                        p + layout.down_off);
        }
        auto xv = randoms(H * nt);
        o.put(x, xv);
        std::vector<int> selected(nt * K);
        for (int t = 0; t < nt; ++t)
            for (int j = 0; j < K; ++j)
                selected[t * K + j] = (j + 3 * t) % E;
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * 4);
        Device dp((pack.size() + 3) / 4), dx(xv), q8((native_q8_1_bytes(H) * nt + 3) / 4),
            scratch((native_expert_scratch_bytes(nt * K, F) + 3) / 4), out(H * nt * K),
            plan(2 * E + E + 1 + 2 * nt * K + 1);
        check(cudaMemcpy(dp.p, pack.data(), pack.size(), cudaMemcpyHostToDevice));
        std::vector<unsigned long long> ptr(E);
        std::vector<int> start{0}, dst, tok;
        for (int e = 0; e < E; ++e) {
            ptr[e] = (unsigned long long)((uint8_t *)dp.p + e * layout.bytes);
            for (int t = 0; t < nt; ++t)
                for (int j = 0; j < K; ++j)
                    if (selected[t * K + j] == e) {
                        dst.push_back(t * K + j);
                        tok.push_back(t);
                    }
            start.push_back(dst.size());
        }
        auto *gp = (unsigned long long *)plan.p;
        auto *st = (int *)(gp + E);
        auto *ed = st + E + 1;
        auto *et = ed + nt * K;
        auto *ng = et + nt * K;
        check(cudaMemcpy(gp, ptr.data(), E * 8, cudaMemcpyHostToDevice));
        check(cudaMemcpy(st, start.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(ed, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(et, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice));
        int count = E;
        check(cudaMemcpy(ng, &count, 4, cudaMemcpyHostToDevice));
        native_quantize_q8_1(dx.p, q8.p, H, nt, stream);
        native_expert_grouped(layout, gp, st, ng, ed, et, E, nt * K, q8.p, scratch.p, out.p, stream);
        check(cudaStreamSynchronize(stream));
        compare("IQ4_XS/Q4_K grouped experts", out.get(H * nt * K), o.run(y), 1e-3);
    }
}
static void attention(cudaStream_t stream) {
    for (int cells : {1, 8, 63, 256, 513, 4096}) {
        Oracle o;
        auto *q = o.tensor(256, 1, 16), *key = o.tensor(256, cells, 2), *value = o.tensor(cells, 256, 2),
             *gate = o.tensor(256, 1, 16);
        auto *s = ggml_mul_mat(o.ctx, key, q);
        auto *p = ggml_soft_max_ext(o.ctx, s, nullptr, 1 / std::sqrt(256.f), 0);
        auto *raw = ggml_mul_mat(o.ctx, value, p);
        auto *y = ggml_mul(o.ctx, raw, ggml_sigmoid(o.ctx, gate));
        o.alloc();
        auto qv = randoms(4096), kv = randoms(size_t(cells) * 512), vv = randoms(size_t(cells) * 512),
             gv = randoms(4096);
        std::vector<float> kc(kv.size()), vc(vv.size());
        for (int h = 0; h < 2; ++h)
            for (int c = 0; c < cells; ++c)
                for (int d = 0; d < 256; ++d) {
                    kc[(size_t(c) * 2 + h) * 256 + d] = kv[(size_t(h) * cells + c) * 256 + d];
                    vc[(size_t(c) * 2 + h) * 256 + d] = vv[(size_t(h) * 256 + d) * cells + c];
                }
        o.put(q, qv);
        o.put(key, kv);
        o.put(value, vv);
        o.put(gate, gv);
        Device dq(qv), dk(kc), dv(vc), dg(gv), out(4096), scores(size_t(cells) * 16);
        strata::qwen35::gpu_dense_attention(dq.p, dg.p, dk.p, dv.p, out.p, scores.p, cells, 16, 2, 256, false,
                                            stream);
        check(cudaStreamSynchronize(stream));
        compare("dense attention", out.get(4096), o.run(y));
    }
}
int main() {
    try {
        cudaStream_t s;
        check(cudaStreamCreate(&s));
        preprocessing(s);
        recurrence(s);
        rope(s);
        router(s);
        mmvq(s);
        moe_combine(s);
        grouped_experts(s);
        attention(s);
        check(cudaStreamDestroy(s));
        std::puts("PASS independent HIP primitives");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
