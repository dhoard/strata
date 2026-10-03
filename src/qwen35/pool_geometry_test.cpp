// A native pool must use the artifact's dimensions, including arbitrary worker row partitions.
#include "strata/qwen35/qwen35.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include <cstring>
#include <random>
#include <cstdio>
namespace q = strata::qwen35;
namespace cpu = strata::kernels::cpu;
int main() {
    q::qwen35_enable_ggml();
    ggml_quantize_init(GGML_TYPE_IQ4_XS);
    const int H = 2048, F = 512, E = 8, T = 4;
    std::mt19937 rng(35);
    std::normal_distribution<float> d(0, .025f);
    cpu::NativeFmt fmt;
    std::string err;
    if (!cpu::native_fmt(23, 12, H, F, fmt, err))
        return 1;
    fmt.swiglu = q::g_swiglu_vec;
    std::vector<std::vector<uint8_t>> blobs(E, std::vector<uint8_t>(fmt.bytes));
    std::vector<float> w(H * F), importance(H, 1);
    for (auto &b : blobs)
        for (int part = 0; part < 3; ++part) {
            for (auto &v : w)
                v = d(rng);
            int rows = part == 2 ? H : F, cols = part == 2 ? F : H;
            ggml_quantize_chunk(part == 2 ? GGML_TYPE_Q4_K : GGML_TYPE_IQ4_XS, w.data(),
                                b.data() + (part == 0   ? 0
                                            : part == 1 ? fmt.up_off
                                                        : fmt.down_off),
                                0, rows, cols, importance.data());
        }
    std::vector<std::vector<float>> x(T, std::vector<float>(H));
    std::vector<std::vector<uint8_t>> act(T, std::vector<uint8_t>(fmt.act_bytes));
    for (int t = 0; t < T; ++t) {
        for (auto &v : x[t])
            v = d(rng) * 40;
        cpu::native_quant_act(fmt, x[t].data(), act[t].data());
    }
    std::vector<std::vector<float>> ref(E * T, std::vector<float>(H)),
        got(E * T, std::vector<float>(H + 64, -1e30f));
    std::vector<cpu::ExpertJobMulti> jobs(E);
    for (int e = 0; e < E; ++e) {
        jobs[e].blob = blobs[e].data();
        jobs[e].nt = e % T + 1;
        for (int t = 0; t < jobs[e].nt; ++t) {
            auto *b = blobs[e].data();
            q::Mat gm{b, nullptr, 23, H, F, fmt.gu_row}, um{b + fmt.up_off, nullptr, 23, H, F, fmt.gu_row},
                dm{b + fmt.down_off, nullptr, 12, F, H, fmt.d_row};
            std::vector<float> gate(F), up(F), ff(F);
            q::matvec(gm, x[t].data(), gate.data());
            q::matvec(um, x[t].data(), up.data());
            q::g_swiglu_vec(F, ff.data(), gate.data(), up.data());
            q::matvec(dm, ff.data(), ref[e * T + t].data());
            jobs[e].nact[t] = act[t].data();
            jobs[e].out[t] = got[e * T + t].data();
        }
    }
    for (int workers : {1, 3, 8, 11, 12}) {
        cpu::ExpertPool pool(workers, false, true);
        pool.run_split_multi_native(fmt, jobs.data(), jobs.size());
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                if (std::memcmp(ref[e * T + t].data(), got[e * T + t].data(), H * 4)) {
                    std::fprintf(stderr, "FAIL workers=%d expert=%d token=%d\n", workers, e, t);
                    return 1;
                }
                for (int i = H; i < H + 64; ++i)
                    if (got[e * T + t][i] != -1e30f)
                        return 1;
            }
        std::printf("native pool 2048x512 workers=%d: bitwise PASS\n", workers);
    }
    return 0;
}
