// include/strata/core/progress.hpp - issue #29: whether a request is still moving, and where it is.
//
// The engine's host loop and the GPU wait on each other through flags; a protocol bug there does not crash, it
// spins forever (the GPU "100%", one CPU core busy, no tokens).  `--serve` runs a watchdog thread over this: a
// request whose heartbeat stops for `STRATA_WATCHDOG_S` seconds (default 60; 0 = off) ends the engine with the
// stage it was stuck in, and the server starts it again instead of hanging.  Tokens, prompt chunks and verify
// windows beat; the stage is two relaxed stores per layer, which the token path does not notice.
//
// #481: the SERVER has a second watchdog over the same request - an engine that prints nothing on stdout for
// `engine_silence_s` (default 300 s) has lost step with it, so it ends the engine and the next request starts it
// again.  A prompt chunk on a slow disk prints nothing until it ends (PP comes once per chunk) and can take many
// minutes, which that watchdog cannot tell from a lost engine; so while a request is in flight, progress_beat()
// also prints one `HB <chunk> <detail> <where>` line to stdout, at most every `STRATA_HB_S` seconds (default 10,
// 0 = off).  It is only printed while a request runs (`busy`): an engine that lost step sits in its command loop
// with no request in flight, says nothing, and is still ended exactly as before.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

struct Progress {
    std::atomic<uint64_t> beats{0};
    std::atomic<bool> busy{false};
    std::atomic<const char*> where{"idle"};
    std::atomic<int64_t> detail{-1};
    std::atomic<int64_t> chunk{-1};      ///< #251: the prompt chunk (its first position) a batched-read stage is in
    std::atomic<int64_t> since_ms{0};    ///< when `where` was set (steady clock): how long a stage has lasted
    std::atomic<uint64_t> ticks{0};      ///< layers served: a window that still moves, slowly, against one that stopped
    std::atomic<int64_t> hb_ms{0};       ///< #481: when progress_beat() last printed an HB line (0: never in this run)
};

inline int64_t progress_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// What the watchdog prints about a part of the engine when it fires (the expert pool, the verify window).
using DiagFn = void (*)(std::FILE*);
inline std::atomic<DiagFn>& diag_pool_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }
inline std::atomic<DiagFn>& diag_verify_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }
/// #267: what a path that ends the engine runs first - it releases the GPU's spin waits on host flags (the verify
/// window's), so no kernel stays resident while the process goes away (on Windows that left the GPU "lost").
inline std::atomic<DiagFn>& release_gpu_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }
inline void release_gpu_waits(std::FILE* f) {
    if (auto fn = release_gpu_fn().load()) fn(f);
}

inline Progress& progress() {
    static Progress p;
    return p;
}

inline void progress_at(const char* where, int64_t detail = -1, int64_t chunk = -1) {
    Progress& p = progress();
    p.where.store(where, std::memory_order_relaxed);
    p.detail.store(detail, std::memory_order_relaxed);
    p.chunk.store(chunk, std::memory_order_relaxed);
    p.since_ms.store(progress_now_ms(), std::memory_order_relaxed);
}

/// #481: how often a request may print a heartbeat to the server: `STRATA_HB_S` seconds (default 10), 0 turns it
/// off.  Read once, on the first beat.
inline int64_t progress_hb_interval_ms() {
    static const int64_t ms = [] {
        const char* e = std::getenv("STRATA_HB_S");
        const double s = e != nullptr ? std::atof(e) : 10.0;
        return (int64_t) (s > 0.0 ? s * 1000.0 : 0.0);
    }();
    return ms;
}

inline void progress_beat() {
    Progress& p = progress();
    p.beats.fetch_add(1, std::memory_order_relaxed);
    // #481: a line the server hears, so a request that is only SLOW is not taken for one that lost step.  Never
    // while no request is in flight: that is exactly the state the server's watchdog exists to end.
    const int64_t every = progress_hb_interval_ms();
    if (every <= 0 || !p.busy.load(std::memory_order_relaxed)) return;
    const int64_t now = progress_now_ms();
    int64_t last = p.hb_ms.load(std::memory_order_relaxed);
    if (now - last < every) return;
    if (!p.hb_ms.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;   // one printer per interval
    std::printf("HB %lld %lld %s\n", (long long) p.chunk.load(std::memory_order_relaxed),
                (long long) p.detail.load(std::memory_order_relaxed), p.where.load(std::memory_order_relaxed));
    std::fflush(stdout);
}
inline void progress_tick() { progress().ticks.fetch_add(1, std::memory_order_relaxed); }

}  // namespace strata::core
