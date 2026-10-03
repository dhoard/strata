#include "strata/qwen35/session.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace strata::qwen35 {
CpuSession::CpuSession(const Qwen35Geometry& g, const TrunkWeights& w, const MtpWeights* mtp, int64_t context,
                       bool prompt_cache, int prompt_cache_slots)
    : g_(g), w_(w), mtp_(mtp), context_(context), prompt_cache_(prompt_cache) {
    if (context <= 0 || context > g.context_length) throw std::invalid_argument("context exceeds model maximum");
    cache_.slots = prompt_cache_slots;
    state_.reset(g,context);
    if (mtp) draft_.resize(context,g);
}
void CpuSession::reset() {
    state_.zero(g_);
    if (mtp_) draft_.zero();
    checkpoints_.clear();
    verifying_ = false;
    cache_.forget();
    periodic_.clear();
    periodic_snapshots_.clear();
    next_periodic_ = 0;
}

int64_t reused_prefix(const std::vector<int64_t>& consumed, const int64_t* tokens, int64_t count) {
    const int64_t n = std::min<int64_t>((int64_t) consumed.size(), count);
    int64_t i = 0;
    while (i < n && consumed[(size_t) i] == tokens[i]) ++i;
    return i;
}

int64_t CpuSession::prepare_prompt(const int64_t* tokens, int64_t count) {
    // Resume from the deepest checkpoint the prompt still contains.  A prompt that *is* the cached prefix
    // reads nothing at all, because the logits that checkpoint produced were kept; anything else re-reads
    // from the checkpoint, and a prompt that diverges before every checkpoint is a full read.
    const int64_t at = prompt_cache_ ? cache_.plan(tokens, count, g_.n_vocab, periodic_) : 0;
    if (at > 0) {
        const bool end = at == cache_.checkpoint;
        if (end) {
            std::string err;
            if (!state_.restore(prompt_checkpoint_, err))
                throw std::runtime_error(err);
        } else {
            const auto it = std::find(periodic_.begin(), periodic_.end(), at);
            std::string err;
            if (it == periodic_.end() ||
                !state_.restore(periodic_snapshots_[(size_t) (it - periodic_.begin())], err))
                throw std::runtime_error(err);
        }
        cache_.consumed.resize((size_t) at);
        checkpoints_.clear();
        verifying_ = false;
        if (mtp_) draft_.n = at;
        // Resume points beyond the one used describe state this request is about to re-read; keeping them
        // would let a later prompt resume into a state the conversation no longer matches.  The interval is
        // for the span that is actually left to read, not for the whole prompt: a resumed turn re-reads a
        // tail, and its checkpoints belong inside that tail.
        periodic_.clear();
        periodic_snapshots_.clear();
        cache_.every = PromptCacheState::interval_for(count - at, cache_.slots);
        next_periodic_ = at + cache_.every;
        return at;
    }
    reset();
    cache_.every = PromptCacheState::interval_for(count, cache_.slots);
    next_periodic_ = cache_.every;
    return 0;
}

void CpuSession::checkpoint_prompt(const float* logits) {
    if (!prompt_cache_) return;
    prompt_checkpoint_ = state_.snapshot();
    cache_.checkpoint = state_.position;
    cache_.valid = true;
    cache_.remember(logits, g_.n_vocab);
}

void CpuSession::maybe_periodic_checkpoint() {
    // A periodic resume point, taken to a fixed share of the prompt so a later turn that diverges near
    // the prompt's end still has state within `every` tokens of it.
    if (!prompt_cache_ || verifying_ || cache_.every <= 0 || cache_.slots <= 0) return;
    if ((int64_t) cache_.consumed.size() < next_periodic_) return;
    if (!periodic_.empty() && periodic_.back() == state_.position) return;
    if ((int) periodic_.size() >= cache_.slots) {
        periodic_.erase(periodic_.begin());
        periodic_snapshots_.erase(periodic_snapshots_.begin());
    }
    periodic_.push_back(state_.position);
    periodic_snapshots_.push_back(state_.snapshot());
    next_periodic_ = state_.position + cache_.every;
}

void CpuSession::target(int64_t token, float* logits, float* hidden) {
    trunk_forward(g_,w_,state_,token,logits,nullptr,nullptr,hidden);
    cache_.consumed.push_back(token);
    if (verifying_) checkpoints_.push_back(state_.snapshot());
    else maybe_periodic_checkpoint();
}
void CpuSession::draft(int64_t token, const float* hidden, float* logits, float* next_hidden) {
    if (!mtp_) throw std::logic_error("no MTP artifact loaded");
    mtp_forward(g_,*mtp_,draft_,token,hidden,logits,next_hidden);
}
void CpuSession::begin_verify() {
    if (verifying_) throw std::logic_error("nested verification");
    checkpoints_.clear(); checkpoints_.push_back(state_.snapshot()); verifying_ = true;
    cache_.verify_from = (int64_t) cache_.consumed.size();
}
void CpuSession::commit_verify(int64_t accepted) {
    if (!verifying_ || accepted < 0 || (size_t) accepted >= checkpoints_.size())
        throw std::logic_error("invalid accepted prefix");
    std::string err;
    if (!state_.restore(checkpoints_[(size_t) accepted],err)) throw std::runtime_error(err);
    checkpoints_.clear(); verifying_ = false;
    // The rejected draft tokens were read but not accepted; the cache must remember only the accepted
    // prefix, or the next prompt would be matched against tokens the model never really consumed.
    cache_.consumed.resize((size_t) (cache_.verify_from + accepted));
}
void CpuSession::rewind_draft(int64_t pos) {
    if (!mtp_ || pos < 0 || pos > draft_.n) throw std::logic_error("invalid draft rewind");
    draft_.n = pos;
}

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
int64_t greedy(const std::vector<float>& logits) {
    return std::max_element(logits.begin(),logits.end())-logits.begin();
}
}

GenerationStats generate(InferenceSession& s, const Qwen35Geometry& g, int64_t eos,
                         const std::vector<int64_t>& prompt, int64_t max_new, int spec,
                         const std::function<int64_t(const std::vector<float>&)>& select,
                         const std::function<void(int64_t)>& emit,
                         const std::function<bool()>& cancelled, int force_reject,
                         const std::function<void(int64_t, int64_t)>& progress) {
    if (prompt.empty() || (int64_t) prompt.size() > s.context() || max_new < 0 || spec < 0 || spec > 4)
        throw std::invalid_argument("invalid prompt/context/output/spec length");
    for (auto t : prompt) if (t < 0 || t >= g.n_vocab) throw std::invalid_argument("prompt token outside vocabulary");
    GenerationStats result; result.prompt_tokens = (int64_t) prompt.size();
    const int64_t skip = s.prepare_prompt(prompt.data(), (int64_t) prompt.size());
    if (skip < 0 || skip > (int64_t) prompt.size()) throw std::logic_error("invalid prompt resume");
    result.reused = skip;
    std::vector<float> logits((size_t) g.n_vocab), h((size_t) g.n_embd), prev((size_t) g.n_embd,0.0f);
    std::vector<float> dh((size_t) g.n_embd), dl((size_t) g.n_vocab);
    const auto p0 = Clock::now();
    bool have_logits = false;
    if (skip == (int64_t) prompt.size()) {
        // The whole prompt is the cached prefix and its logits were kept: nothing to read at all.
        const float* kept = s.cached_logits();
        if (!kept) throw std::logic_error("prompt resume without cached logits");
        std::copy_n(kept, (size_t) g.n_vocab, logits.begin());
        have_logits = true;
    }
    for (size_t i = (size_t) skip; i < prompt.size();) {
        if (cancelled()) { result.finish = "stop"; break; }
        const int count = int(std::min<size_t>(s.prefill_batch_size(), prompt.size()-i));
        std::vector<float> hs(size_t(count)*g.n_embd);
        std::vector<float*> lp(count, nullptr), hp(count);
        for (int j=0;j<count;++j) hp[j] = hs.data()+size_t(j)*g.n_embd;
        if (i+count == prompt.size()) { lp.back() = logits.data(); have_logits = true; }
        s.target_batch(prompt.data()+i,count,lp.data(),hp.data());
        for (int j=0;j<count;++j) {
            if (s.has_mtp() && spec) s.draft(prompt[i+j],prev.data(),nullptr,dh.data());
            std::copy_n(hp[j],g.n_embd,prev.data());
        }
        h = prev; i += count;
        if (progress) progress((int64_t) i, (int64_t) prompt.size());
    }
    // Only record a resume point when the prompt was read to its end: a cancelled read leaves logits
    // that were never produced for this prompt, and the next request must not be handed those.
    if (have_logits) s.checkpoint_prompt(logits.data());
    s.begin_decode();
    const auto d0 = Clock::now(); result.prompt_ms = ms(p0,d0);
    while (!cancelled() && result.generated < max_new && s.position() < s.context()) {
        const int64_t first = select(logits);
        if (first == eos) { result.finish = "stop"; break; }
        const int k = s.has_mtp() ? (int) std::min<int64_t>(spec,std::min(max_new-result.generated-1,s.context()-s.position()-1)) : 0;
        if (k == 0) {
            s.target(first,logits.data(),h.data());
            emit(first); ++result.generated;
            if (s.has_mtp() && spec) s.draft(first,prev.data(),nullptr,dh.data());
            prev = h; continue;
        }
        const int64_t start = s.position();
        std::vector<int64_t> proposals{first};
        dh = h;
        const auto draft_start = Clock::now();
        for (int i=0;i<k;++i) {
            s.draft(proposals.back(),dh.data(),dl.data(),dh.data());
            proposals.push_back(greedy(dl));
        }
        result.draft_ms += ms(draft_start,Clock::now()); result.proposed += k;
        std::vector<std::vector<float>> vl((size_t) k+1,std::vector<float>((size_t) g.n_vocab));
        std::vector<std::vector<float>> vh((size_t) k+1,std::vector<float>((size_t) g.n_embd));
        const auto verify_start = Clock::now(); s.begin_verify();
        std::vector<float*> lp(k+1), hp(k+1);
        for (int i=0;i<=k;++i) { lp[i]=vl[i].data(); hp[i]=vh[i].data(); }
        s.target_batch(proposals.data(),k+1,lp.data(),hp.data());
        int commit = 1;
        while (commit <= k && proposals[(size_t) commit] != eos &&
               proposals[(size_t) commit] == greedy(vl[(size_t) commit-1]) &&
               (force_reject < 0 || commit <= force_reject)) ++commit;
        s.commit_verify(commit);
        result.verify_ms += ms(verify_start,Clock::now()); result.accepted += commit-1;
        s.rewind_draft(start);
        for (int i=0;i<commit;++i) {
            // Rebuild the committed draft cache with TARGET hidden states, replacing speculative ones.
            s.draft(proposals[(size_t) i],i ? vh[(size_t) i-1].data() : h.data(),nullptr,dh.data());
            emit(proposals[(size_t) i]); ++result.generated;
        }
        logits = std::move(vl[(size_t) commit-1]); h = std::move(vh[(size_t) commit-1]); prev = h;
    }
    if (cancelled()) result.finish = "stop";
    result.decode_ms = ms(d0,Clock::now());
    return result;
}
} // namespace strata::qwen35
