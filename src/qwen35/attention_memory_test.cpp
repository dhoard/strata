// The coalesced value reader must retain every bit of the independent per-row
// implementation, including half-pair accumulation and padding boundaries.
#include "ggml.h"
#include "strata/qwen35/gpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
namespace q = strata::qwen35;
static void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n)); }
    ~Buffer() { (void)cudaFree(p); }
    void put(const void *src, size_t n) { check(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
    float *f() { return (float *)p; }
};
static void old(bool yes) {
#ifdef _WIN32
    _putenv_s("STRATA_QWEN35_OLD_ATTENTION", yes ? "1" : "");
#else
    if (yes)
        setenv("STRATA_QWEN35_OLD_ATTENTION", "1", 1);
    else
        unsetenv("STRATA_QWEN35_OLD_ATTENTION");
#endif
}
int main() {
    try {
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        for (int cells : {1, 8, 63, 255, 256, 257, 511, 512, 513, 1023, 4096, 8193, 65537}) {
            std::vector<float> kv(size_t(cells) * 512), v(kv.size()), query(4096), gate(4096), ref(4096),
                got(4096);
            for (size_t i = 0; i < kv.size(); ++i) {
                kv[i] = std::sin(float(i) * .021f) * .5f;
                v[i] = std::cos(float(i) * .019f) * 4.f;
            }
            for (int i = 0; i < 4096; ++i) {
                query[i] = std::cos(float(i) * .031f);
                gate[i] = float(i % 13) - 6;
            }
            Buffer dq(query.size() * 4), dg(gate.size() * 4), dr(ref.size() * 4), dy(got.size() * 4),
                scores(size_t(cells) * 16 * 4);
            dq.put(query.data(), query.size() * 4);
            dg.put(gate.data(), gate.size() * 4);
            for (bool half : {false, true}) {
                Buffer dk(kv.size() * (half ? 2 : 4)), dv(v.size() * (half ? 2 : 4));
                if (half) {
                    std::vector<ggml_fp16_t> hk(kv.size()), hv(v.size());
                    for (size_t i = 0; i < kv.size(); ++i) {
                        hk[i] = ggml_fp32_to_fp16(kv[i]);
                        hv[i] = ggml_fp32_to_fp16(v[i]);
                    }
                    dk.put(hk.data(), hk.size() * 2);
                    dv.put(hv.data(), hv.size() * 2);
                } else {
                    dk.put(kv.data(), kv.size() * 4);
                    dv.put(v.data(), v.size() * 4);
                }
                old(true);
                q::gpu_dense_attention(dq.f(), dg.f(), dk.p, dv.p, dr.f(), scores.f(), cells, 16, 2, 256,
                                       half, stream);
                old(false);
                q::gpu_dense_attention(dq.f(), dg.f(), dk.p, dv.p, dy.f(), scores.f(), cells, 16, 2, 256,
                                       half, stream);
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(ref.data(), dr.p, ref.size() * 4, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost));
                if (std::memcmp(ref.data(), got.data(), ref.size() * 4))
                    throw std::runtime_error("tiled attention differs");
                std::printf("cells=%d kv=%s all 4096 outputs bitwise equal\n", cells, half ? "f16" : "f32");
            }
        }
        check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "attention memory: %s\n", e.what());
        return 1;
    }
}
