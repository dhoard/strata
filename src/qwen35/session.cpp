#include "strata/qwen35/session.hpp"
#include <chrono>
#include <stdexcept>

namespace strata::qwen35 {
CpuSession::CpuSession(const Qwen35Geometry& g, const TrunkWeights& w, const MtpWeights* mtp, int64_t context)
    : g_(g), w_(w), mtp_(mtp), context_(context) {
    if (context <= 0 || context > g.context_length) throw std::invalid_argument("context exceeds model maximum");
    state_.reset(g,context);
    if (mtp) draft_.resize(context,g);
}
void CpuSession::reset() { state_.zero(g_); if (mtp_) draft_.zero(); checkpoints_.clear(); verifying_ = false; }
void CpuSession::target(int64_t token, float* logits, float* hidden) {
    trunk_forward(g_,w_,state_,token,logits,nullptr,nullptr,hidden);
    if (verifying_) checkpoints_.push_back(state_.snapshot());
}
void CpuSession::draft(int64_t token, const float* hidden, float* logits, float* next_hidden) {
    if (!mtp_) throw std::logic_error("no MTP artifact loaded");
    mtp_forward(g_,*mtp_,draft_,token,hidden,logits,next_hidden);
}
void CpuSession::begin_verify() {
    if (verifying_) throw std::logic_error("nested verification");
    checkpoints_.clear(); checkpoints_.push_back(state_.snapshot()); verifying_ = true;
}
void CpuSession::commit_verify(int64_t accepted) {
    if (!verifying_ || accepted < 0 || (size_t) accepted >= checkpoints_.size())
        throw std::logic_error("invalid accepted prefix");
    std::string err;
    if (!state_.restore(checkpoints_[(size_t) accepted],err)) throw std::runtime_error(err);
    checkpoints_.clear(); verifying_ = false;
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
                         const std::function<bool()>& cancelled, int force_reject) {
    if (prompt.empty() || (int64_t) prompt.size() > s.context() || max_new < 0 || spec < 0 || spec > 4)
        throw std::invalid_argument("invalid prompt/context/output/spec length");
    for (auto t : prompt) if (t < 0 || t >= g.n_vocab) throw std::invalid_argument("prompt token outside vocabulary");
    s.reset();
    GenerationStats result; result.prompt_tokens = (int64_t) prompt.size();
    std::vector<float> logits((size_t) g.n_vocab), h((size_t) g.n_embd), prev((size_t) g.n_embd,0.0f);
    std::vector<float> dh((size_t) g.n_embd), dl((size_t) g.n_vocab);
    const auto p0 = Clock::now();
    for (size_t i = 0; i < prompt.size();) {
        if (cancelled()) { result.finish = "stop"; break; }
        const int count = int(std::min<size_t>(s.prefill_batch_size(), prompt.size()-i));
        std::vector<float> hs(size_t(count)*g.n_embd);
        std::vector<float*> lp(count, nullptr), hp(count);
        for (int j=0;j<count;++j) hp[j] = hs.data()+size_t(j)*g.n_embd;
        if (i+count == prompt.size()) lp.back() = logits.data();
        s.target_batch(prompt.data()+i,count,lp.data(),hp.data());
        for (int j=0;j<count;++j) {
            if (s.has_mtp() && spec) s.draft(prompt[i+j],prev.data(),nullptr,dh.data());
            std::copy_n(hp[j],g.n_embd,prev.data());
        }
        h = prev; i += count;
    }
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
