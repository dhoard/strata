// Architecture-specific router and direct conventional dense attention.
#include "strata/qwen35/gpu.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>
namespace strata::qwen35 {
namespace {
void checked(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
__device__ float sum32(float v) {
    for (int d = 16; d; d /= 2)
        v += __shfl_xor_sync(0xffffffffu, v, d, 32);
    return v;
}
__device__ float max32(float v) {
    for (int d = 16; d; d /= 2)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d, 32));
    return v;
}
template <int N> __device__ float sumblock(float v, float *tmp) {
    v = sum32(v);
    int lane = threadIdx.x % 32;
    if (lane == 0)
        tmp[threadIdx.x / 32] = v;
    __syncthreads();
    return sum32(lane < N / 32 ? tmp[lane] : 0);
}
// Match upstream HIP's full softmax, bitonic descending argsort and SUM_ROWS contract.
__global__ void route(const float *logits, int E, int K, int *ids, float *weights) {
    int t = threadIdx.x;
    __shared__ float p[1024], tmp[32], mx;
    __shared__ int order[1024];
    float v = t < E ? logits[t] : -INFINITY;
    float m = max32(v);
    if (t % 32 == 0)
        tmp[t / 32] = m;
    __syncthreads();
    m = max32(t % 32 < blockDim.x / 32 ? tmp[t % 32] : -INFINITY);
    if (t == 0)
        mx = m;
    __syncthreads();
    p[t] = t < E ? expf(v - mx) : 0;
    float s;
    // E is padded to a power of two; the loader admits 256/512 here.
    if (blockDim.x == 256)
        s = sumblock<256>(p[t], tmp);
    else
        s = sumblock<512>(p[t], tmp);
    p[t] = p[t] * (1.0f / s);
    order[t] = t;
    __syncthreads();
    for (int n = 2; n <= blockDim.x; n *= 2)
        for (int d = n / 2; d; d /= 2) {
            int other = t ^ d;
            if (other > t) {
                int a = order[t], b = order[other];
                bool swap = (t & n) ? p[a] > p[b] : p[a] < p[b];
                if (swap) {
                    order[t] = b;
                    order[other] = a;
                }
            }
            __syncthreads();
        }
    // GGML SUM_ROWS on the eight selected probabilities uses one warp with padding zeros.
    float picked = t < K ? p[order[t]] : 0;
    float selected = sum32(picked);
    if (t < K) {
        ids[t] = order[t];
        weights[t] = picked / fmaxf(selected, 0.00006103515625f);
    }
}
template <typename T>
__global__ void scores_kernel(const float *q, const T *key, float *s, int cells, int H, int KV, int D) {
    int cell = blockIdx.x, head = blockIdx.y, t = threadIdx.x, kh = head / (H / KV);
    float v = 0;
    if (cell >= cells)
        return;
    for (int pair = t; pair < D / 2; pair += 128) {
        int d = pair * 2;
        v = fmaf(q[head * D + d], float(key[(size_t(cell) * KV + kh) * D + d]), v);
        v = fmaf(q[head * D + d + 1], float(key[(size_t(cell) * KV + kh) * D + d + 1]), v);
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
            s[size_t(head) * cells + cell] = v * rsqrtf(float(D));
    }
}
template <int N> __global__ void softmax_kernel(float *s, int cells) {
    int h = blockIdx.x, t = threadIdx.x;
    s += size_t(h) * cells;
    __shared__ float tmp[32], mx, inv;
    float v = -INFINITY;
    for (int c = t; c < cells; c += N)
        v = fmaxf(v, s[c]);
    v = max32(v);
    if (t % 32 == 0)
        tmp[t / 32] = v;
    __syncthreads();
    v = max32(t % 32 < N / 32 ? tmp[t % 32] : -INFINITY);
    if (t == 0)
        mx = v;
    __syncthreads();
    v = 0;
    for (int c = t; c < cells; c += N) {
        float e = expf(s[c] - mx);
        s[c] = e;
        v += e;
    }
    v = sumblock<N>(v, tmp);
    if (t == 0)
        inv = 1 / v;
    __syncthreads();
    for (int c = t; c < cells; c += N)
        s[c] *= inv;
}
template <typename T>
__global__ void values_kernel(const float *s, const T *v, const float *gate, float *out, int cells, int H,
                              int KV, int D) {
    int d = blockIdx.x, head = blockIdx.y, t = threadIdx.x, kh = head / (H / KV);
    float acc = 0;
    const int padded = ((cells + 255) / 256) * 256;
    if constexpr (std::is_same_v<T, __half>) {
        // GGML's F16 value MMVF uses paired half products and half accumulators,
        // followed by F32 warp reductions. KQ explicitly requests F32 accumulation.
        __half2 sum = __floats2half2_rn(0, 0);
        for (int pair = t; pair < padded / 2; pair += blockDim.x) {
            int c = pair * 2;
            const float a = c < cells ? __half2float(v[(size_t(c) * KV + kh) * D + d]) : 0;
            const float b = c + 1 < cells ? __half2float(v[(size_t(c + 1) * KV + kh) * D + d]) : 0;
            const float p = c < cells ? s[size_t(head) * cells + c] : 0;
            const float q = c + 1 < cells ? s[size_t(head) * cells + c + 1] : 0;
            sum = __hadd2(sum, __hmul2(__floats2half2_rn(a, b), __floats2half2_rn(p, q)));
        }
        acc = __low2float(sum) + __high2float(sum);
    } else {
        for (int pair = t; pair < padded / 2; pair += blockDim.x) {
            int c = pair * 2;
            if (c < cells)
                acc = fmaf(s[size_t(head) * cells + c], float(v[(size_t(c) * KV + kh) * D + d]), acc);
            if (c + 1 < cells)
                acc = fmaf(s[size_t(head) * cells + c + 1], float(v[(size_t(c + 1) * KV + kh) * D + d]), acc);
        }
    }
    __shared__ float tmp[32];
    if (t < 32)
        tmp[t] = 0;
    __syncthreads();
    acc = sum32(acc);
    if (t % 32 == 0)
        tmp[t / 32] = acc;
    __syncthreads();
    if (t < 32) {
        acc = sum32(tmp[t]);
        if (t == 0)
            out[head * D + d] = acc * (1.0f / (1.0f + expf(-gate[head * D + d])));
    }
}
// Transpose eight value columns in shared memory, streaming at most 512 cells
// at a time. Global reads are contiguous across columns. Each output keeps the
// reference's paired products, accumulation type, thread count and XOR reduction.
// The tile size does not grow with context and no host synchronization is needed.
template <typename T>
__global__ void values_tiled(const float *scores, const T *value, const float *gate, float *out, int cells,
                             int H, int KV, int D) {
    constexpr int Columns = 8;
    const int first = blockIdx.x * Columns, head = blockIdx.y, t = threadIdx.x, kh = head / (H / KV);
    __shared__ T tile[512 * Columns];
    __shared__ float tmp[32];
    float acc[Columns] = {};
    __half2 hs[Columns];
    if constexpr (std::is_same_v<T, __half>)
        for (int d = 0; d < Columns; ++d)
            hs[d] = __floats2half2_rn(0, 0);
    const int padded = ((cells + 255) / 256) * 256, step = 2 * blockDim.x;
    for (int base = 0; base < padded; base += step) {
        for (int i = t; i < step * Columns; i += blockDim.x) {
            const int c = base + i / Columns, d = i % Columns;
            tile[i] = c < cells ? value[(size_t(c) * KV + kh) * D + first + d] : T(0);
        }
        __syncthreads();
        const int c = base + 2 * t;
        const float p = c < cells ? scores[size_t(head) * cells + c] : 0;
        const float q = c + 1 < cells ? scores[size_t(head) * cells + c + 1] : 0;
        if (c < padded)
            for (int d = 0; d < Columns; ++d) {
                const float a = float(tile[(2 * t) * Columns + d]),
                            b = float(tile[(2 * t + 1) * Columns + d]);
                if constexpr (std::is_same_v<T, __half>)
                    hs[d] = __hadd2(hs[d], __hmul2(__floats2half2_rn(a, b), __floats2half2_rn(p, q)));
                else {
                    if (c < cells)
                        acc[d] = fmaf(p, a, acc[d]);
                    if (c + 1 < cells)
                        acc[d] = fmaf(q, b, acc[d]);
                }
            }
        __syncthreads();
    }
    for (int d = 0; d < Columns; ++d) {
        float v = acc[d];
        if constexpr (std::is_same_v<T, __half>)
            v = __low2float(hs[d]) + __high2float(hs[d]);
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
                out[head * D + first + d] = v * (1.0f / (1.0f + expf(-gate[head * D + first + d])));
        }
        __syncthreads();
    }
}
__global__ void combine_rounded(const float *parts, const float *weights, const float *shared, float *out,
                                int H, int K) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= H)
        return;
    float sum = __fmul_rn(parts[i], weights[0]);
    for (int e = 1; e < K; ++e)
        sum = __fadd_rn(sum, __fmul_rn(parts[e * H + i], weights[e]));
    out[i] = shared ? __fadd_rn(sum, shared[i]) : sum;
}
} // namespace
void gpu_moe_combine(const float *p, const float *w, const float *s, float *out, int H, int K, void *stream) {
    if (!stream || H < 1 || K < 1 || K > 15)
        throw std::invalid_argument("Qwen35 combine geometry");
    combine_rounded<<<(H + 255) / 256, 256, 0, (cudaStream_t)stream>>>(p, w, s, out, H, K);
    checked(cudaGetLastError());
}
void gpu_router(const float *x, int E, int K, int *ids, float *w, void *stream) {
    if (!stream || (E != 256 && E != 512) || K < 1 || K > 32)
        throw std::invalid_argument("Qwen35 router geometry");
    route<<<1, E, 0, (cudaStream_t)stream>>>(x, E, K, ids, w);
    checked(cudaGetLastError());
}
void gpu_dense_attention(const float *q, const float *gate, const void *key, const void *value, float *out,
                         float *scores, int cells, int H, int KV, int D, bool half, void *stream) {
    if (!stream || cells <= 0 || H <= 0 || KV <= 0 || H % KV || D != 256)
        throw std::invalid_argument("dense attention geometry");
    auto s = (cudaStream_t)stream;
    dim3 blocks(cells, H);
    int threads = 128;
    int padded = ((cells + 255) / 256) * 256, best = 32, iters = (padded + 63) / 64;
    for (int n = 64; n <= 256; n += 32) {
        int candidate = (padded + 2 * n - 1) / (2 * n);
        if (candidate < iters) {
            best = n;
            iters = candidate;
        }
    }
    if (half)
        scores_kernel<<<blocks, threads, 0, s>>>(q, (const __half *)key, scores, cells, H, KV, D);
    else
        scores_kernel<<<blocks, threads, 0, s>>>(q, (const float *)key, scores, cells, H, KV, D);
    if (padded <= 256)
        softmax_kernel<256><<<H, 256, 0, s>>>(scores, cells);
    else if (padded <= 512)
        softmax_kernel<512><<<H, 512, 0, s>>>(scores, cells);
    else
        softmax_kernel<1024><<<H, 1024, 0, s>>>(scores, cells);
    const bool tiled = !std::getenv("STRATA_QWEN35_OLD_ATTENTION");
    if (tiled && half)
        values_tiled<<<dim3(D / 8, H), best, 0, s>>>(scores, (const __half *)value, gate, out, cells, H, KV,
                                                     D);
    else if (tiled)
        values_tiled<<<dim3(D / 8, H), best, 0, s>>>(scores, (const float *)value, gate, out, cells, H, KV,
                                                     D);
    else if (half)
        values_kernel<<<dim3(D, H), best, 0, s>>>(scores, (const __half *)value, gate, out, cells, H, KV, D);
    else
        values_kernel<<<dim3(D, H), best, 0, s>>>(scores, (const float *)value, gate, out, cells, H, KV, D);
    checked(cudaGetLastError());
}
} // namespace strata::qwen35
