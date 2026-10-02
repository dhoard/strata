// Regression: allocator reuse must not make a GPU batch consume the preceding batch's activation.
#include "strata/qwen35/qwen35.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace q = strata::qwen35;
int main() {
    q::qwen35_enable_ggml();
    constexpr int H = 2048, O = 32;
    std::vector<float> weights(H*O), x(H), got(O), expected(O), single(O);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin((float) i*0.017f)*0.02f;
    std::vector<uint8_t> blocks(ggml_row_size(GGML_TYPE_Q8_0, H)*O);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, weights.data(), blocks.data(), 0, O, H, nullptr);
    q::TrunkWeights w;
    w.output = {blocks.data(), nullptr, GGML_TYPE_Q8_0, H, O, ggml_row_size(GGML_TYPE_Q8_0,H)};
    std::string err;
    if (!q::qwen35_gpu_init(err) || !q::qwen35_gpu_upload(w,err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 1;
    }
    const q::Mat* mats[] = {&w.output};
    const float* inputs[] = {x.data()};
    float* outputs[] = {got.data()};
    for (int step = 0; step < 8; ++step) {
        for (int i = 0; i < H; ++i) x[i] = std::cos(i*0.011f+step)* (step+1);
        if (q::g_gpu_batch(mats,inputs,outputs,1) != 1) return 1;
        q::g_gpu_matvec(w.output,x.data(),single.data());
        if (got != single) { std::fprintf(stderr,"stale batch at step %d\n",step); return 1; }
        auto host = w.output; host.dev = nullptr;
        q::matvec(host,x.data(),expected.data());
        double ss=0, rr=0;
        for (int i=0;i<O;++i) { const double d=got[i]-expected[i]; ss+=d*d; rr+=(double)expected[i]*expected[i]; }
        const double rel=std::sqrt(ss/std::max(rr,1e-30));
        std::printf("batch step=%d relative_rms=%.8g\n",step,rel);
        if (!std::isfinite(rel) || rel>0.015) return 1;
        auto row = q::mat_row(w.output,13);
        float r=0;
        q::matvec(row,x.data(),&r);
        if (std::fabs(r-single[13])>1e-5f) { std::fprintf(stderr,"device row offset failed\n"); return 1; }
    }
    return 0;
}
