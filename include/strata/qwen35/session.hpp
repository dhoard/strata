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

    // ---- prompt cache: the serve loop's conversation cache -------------------------------------
    // A session remembers the tokens it has consumed and one state checkpoint taken at the end of the
    // last prompt.  A request whose prompt still starts with those tokens restores that checkpoint and
    // reads only the new tail instead of the whole conversation, which is what makes an agentic client
    // (a fresh turn every few seconds, a 2k-100k token prompt every time) usable.  A prompt that
    // diverges before the checkpoint is a full read, exactly as without a cache.
    //
    // GDN state is a recurrence, so it cannot be rewound token by token: the checkpoint is what makes a
    // resume possible at all.  sessions without a cache keep the reset-and-read-everything behaviour,
    // which is what the parity tools ask for.
    //
    // `prepare_prompt` leaves the session ready to read the whole prompt and returns how many leading
    // tokens it may skip (0 = read everything, count = nothing to read).  `checkpoint_prompt` is called
    // once the prompt has been read, with the logits of its last token: it records the next request's
    // resume point.  Neither may be called mid-verification.
    virtual int64_t prepare_prompt(const int64_t* tokens, int64_t count) {
        (void) tokens; (void) count; reset(); return 0;
    }
    virtual void checkpoint_prompt(const float* logits) { (void) logits; }
    /// The kept logits, valid when `prepare_prompt` returned `count`.
    virtual const float* cached_logits() const { return nullptr; }
    virtual int64_t consumed_tokens() const { return 0; }
};

/// Leading tokens of `tokens` a session that has consumed `consumed` may skip, ignoring the checkpoint:
/// the longest shared prefix.  Declared here so both session kinds - and the tests - use one rule.
int64_t reused_prefix(const std::vector<int64_t>& consumed, const int64_t* tokens, int64_t count);

/// Prompt-cache bookkeeping shared by both tiers.  A session keeps the tokens it has consumed, a
/// checkpoint at the end of each prompt (with the logits it produced), and a few periodic checkpoints
/// inside the prompt.
///
/// The periodic ones are what make the cache usable with a chat template.  Qwen's template re-renders a
/// past assistant turn from its content, without the `<think>` scaffold its own generation prompt ended
/// with, so the next turn's prompt diverges *at the very end* of the previous one: an end-of-prompt
/// checkpoint sits inside that divergence and is never reusable.  GDN state is a recurrence and cannot
/// be rewound token by token, so the only way to resume at the last shared token is to have saved state
/// near it.  `interval` is a fixed share of the prompt (count / slots), which bounds what a resumed turn
/// re-reads to 1/slots of the prompt plus the new turn, for any prompt length.
struct PromptCacheState {
    std::vector<int64_t> consumed;
    std::vector<float> logits;   // the last prompt token's logits, so an identical repeat reads nothing
    int64_t checkpoint = 0;      // position of the end-of-last-prompt resume point
    bool valid = false;
    int64_t verify_from = 0;     // consumed.size() when the verification window opened
    int slots = 4;               // periodic resume points to keep (each costs one state copy)
    int64_t every = 0;           // tokens between periodic checkpoints for the current prompt

    void forget() {
        consumed.clear();
        logits.clear();
        checkpoint = 0;
        valid = false;
        every = 0;
    }
    /// The interval for a prompt of `count` tokens.  At least 8 tokens (a checkpoint per token would be
    /// all state and no throughput) and at most the whole prompt.
    static int64_t interval_for(int64_t count, int slots) {
        if (slots <= 0) return 0;
        return std::max<int64_t>(8, count / slots);
    }
    /// How many leading tokens of `tokens` may be skipped; 0 means read the whole prompt, and `count`
    /// means read nothing at all (the kept logits are used).  Prefers the end-of-prompt checkpoint - it
    /// is the deepest - and falls back to the deepest periodic checkpoint inside the shared prefix.
    int64_t plan(const int64_t* tokens, int64_t count, int64_t vocab,
                 const std::vector<int64_t>& periodic) const {
        const int64_t prefix = reused_prefix(consumed, tokens, count);
        if (valid && checkpoint > 0 && checkpoint <= count && prefix >= checkpoint &&
            (checkpoint != count || (int64_t) logits.size() == vocab))
            return checkpoint;
        int64_t deepest = 0;
        for (int64_t p : periodic)
            if (p > deepest && p <= prefix && p < count) deepest = p;
        return deepest;
    }
    void remember(const float* l, int64_t vocab) {
        if (l) logits.assign(l, l + vocab);
        else logits.clear();
    }
};

class CpuSession final : public InferenceSession {
public:
    CpuSession(const Qwen35Geometry& g, const TrunkWeights& w, const MtpWeights* mtp, int64_t context,
               bool prompt_cache = true, int prompt_cache_slots = 4);
    void reset() override;
    int64_t position() const override { return state_.position; }
    int64_t context() const override { return context_; }
    bool has_mtp() const override { return mtp_ != nullptr; }
    void target(int64_t token, float* logits, float* hidden) override;
    void draft(int64_t token, const float* hidden, float* logits, float* next_hidden) override;
    void begin_verify() override;
    void commit_verify(int64_t accepted) override;
    void rewind_draft(int64_t position) override;
    int64_t prepare_prompt(const int64_t* tokens, int64_t count) override;
    void checkpoint_prompt(const float* logits) override;
    const float* cached_logits() const override { return cache_.logits.empty() ? nullptr : cache_.logits.data(); }
    int64_t consumed_tokens() const override { return (int64_t) cache_.consumed.size(); }
private:
    /// A periodic resume point inside the prompt: a chat template re-renders a past assistant turn
    /// without its `<think>` scaffold, so the next prompt diverges at the previous prompt's very end and
    /// an end-of-prompt checkpoint can never be reused.  These are what make the cache pay off.
    void maybe_periodic_checkpoint();
    const Qwen35Geometry& g_;
    const TrunkWeights& w_;
    const MtpWeights* mtp_;
    int64_t context_;
    TrunkState state_;
    AttnState draft_;
    bool verifying_ = false;
    std::vector<TrunkSnapshot> checkpoints_;
    // Prompt cache: every token this session has consumed, and the state at the end of the last prompt
    // (the resume point).  `prompt_cache_` off keeps the pre-cache behaviour, a full read per request -
    // what the parity tools and the tests that compare against a fresh session ask for.
    bool prompt_cache_ = true;
    PromptCacheState cache_;
    TrunkSnapshot prompt_checkpoint_;
    // Periodic resume points, oldest first, parallel to `periodic_snapshots_`.
    std::vector<int64_t> periodic_;
    std::vector<TrunkSnapshot> periodic_snapshots_;
    int64_t next_periodic_ = 0;   // position at which the next periodic checkpoint is due
};

struct GenerationStats {
    int64_t generated = 0, prompt_tokens = 0, proposed = 0, accepted = 0;
    int64_t reused = 0;   // prompt tokens the session already held (the conversation cache's hit)
    double prompt_ms = 0, decode_ms = 0, draft_ms = 0, verify_ms = 0;
    std::string finish = "length";
};

// Greedy MTP verification is exact: the target chooses every emitted token. Other sampling
// modes use target-only generation. force_reject is a diagnostic accepted-draft prefix length.
// When the session has a prompt cache, a prompt that extends the conversation it already holds is
// read from the checkpoint instead of from token zero; `GenerationStats::reused` says how much was
// skipped, and the tokens read are the same ones a full read would have read.  A prompt equal to the
// cached prefix reads nothing at all - the logits the checkpoint produced are reused.
// `progress(read, total)` is called after each prefill tile (the serve loop's PP line); the CLI
// leaves it empty so its stdout stays the token stream.
GenerationStats generate(InferenceSession& session, const Qwen35Geometry& g, int64_t eos,
                         const std::vector<int64_t>& prompt, int64_t max_new, int spec,
                         const std::function<int64_t(const std::vector<float>&)>& select,
                         const std::function<void(int64_t)>& emit,
                         const std::function<bool()>& cancelled, int force_reject = -1,
                         const std::function<void(int64_t, int64_t)>& progress = {});
} // namespace strata::qwen35
