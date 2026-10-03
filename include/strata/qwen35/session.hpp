#pragma once
#include "strata/qwen35/qwen35.hpp"
#include <functional>

namespace strata::qwen35 {

// One inference thread owns a session. Verification checkpoints contain recurrent state and
// logical KV cursors; accepted-prefix commit never copies the context's entire attention cache.
class InferenceSession {
public:
    virtual ~InferenceSession() = default;
    virtual void reset() = 0;
    virtual int64_t position() const = 0;
    virtual int64_t context() const = 0;
    virtual bool has_mtp() const = 0;
    virtual void target(int64_t token, float* logits, float* hidden) = 0;
    // Contiguous causal tokens; output pointer arrays may contain null entries (prefill).
    // Every token has the same arithmetic as target(), including verification checkpoints.
    virtual int prefill_batch_size() const { return 1; }
    virtual void target_batch(const int64_t* tokens, int count, float* const* logits, float* const* hidden) {
        for (int i = 0; i < count; ++i) target(tokens[i], logits ? logits[i] : nullptr, hidden ? hidden[i] : nullptr);
    }
    virtual void begin_decode() {}
    virtual void draft(int64_t token, const float* hidden, float* logits, float* next_hidden) = 0;
    virtual void begin_verify() = 0;
    virtual void commit_verify(int64_t accepted) = 0;
    virtual void rewind_draft(int64_t position) = 0;
};

class CpuSession final : public InferenceSession {
public:
    CpuSession(const Qwen35Geometry& g, const TrunkWeights& w, const MtpWeights* mtp, int64_t context);
    void reset() override;
    int64_t position() const override { return state_.position; }
    int64_t context() const override { return context_; }
    bool has_mtp() const override { return mtp_ != nullptr; }
    void target(int64_t token, float* logits, float* hidden) override;
    void draft(int64_t token, const float* hidden, float* logits, float* next_hidden) override;
    void begin_verify() override;
    void commit_verify(int64_t accepted) override;
    void rewind_draft(int64_t position) override;
private:
    const Qwen35Geometry& g_;
    const TrunkWeights& w_;
    const MtpWeights* mtp_;
    int64_t context_;
    TrunkState state_;
    AttnState draft_;
    bool verifying_ = false;
    std::vector<TrunkSnapshot> checkpoints_;
};

struct GenerationStats {
    int64_t generated = 0, prompt_tokens = 0, proposed = 0, accepted = 0;
    double prompt_ms = 0, decode_ms = 0, draft_ms = 0, verify_ms = 0;
    std::string finish = "length";
};

// Greedy MTP verification is exact: the target chooses every emitted token. Other sampling
// modes use target-only generation. force_reject is a diagnostic accepted-draft prefix length.
GenerationStats generate(InferenceSession& session, const Qwen35Geometry& g, int64_t eos,
                         const std::vector<int64_t>& prompt, int64_t max_new, int spec,
                         const std::function<int64_t(const std::vector<float>&)>& select,
                         const std::function<void(int64_t)>& emit,
                         const std::function<bool()>& cancelled, int force_reject = -1);
} // namespace strata::qwen35
