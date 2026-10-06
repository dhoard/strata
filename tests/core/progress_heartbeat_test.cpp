// tests/core/progress_heartbeat_test.cpp - #481: the heartbeat progress_beat() prints while a request runs.
//
// The server ends an engine that prints nothing for `engine_silence_s` during a request (it has lost step with the
// server).  A prompt chunk on a slow disk prints nothing until it ends, so the engine must say it is still moving -
// and it must NOT say anything when no request is in flight, or the server would never end an engine that lost
// step.  This test pins both halves on the host, with no GPU and no engine: it needs nothing but the header.
//
// Run: `./build-cpu/progress_heartbeat_test` (ctest `progress_heartbeat_test`), or:
//   g++ -std=c++17 -Iinclude tests/core/progress_heartbeat_test.cpp -o /tmp/hb && /tmp/hb
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <io.h>                       // _dup/_dup2/_close: the same trick without unistd
#define dup _dup
#define dup2 _dup2
#define close _close
#else
#include <unistd.h>
#endif

#include "strata/core/progress.hpp"

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

/// Run `fn` with stdout on a temporary file and return what it wrote (the heartbeat goes to stdout, as PP and T do).
static std::string captured(void (*fn)()) {
    std::fflush(stdout);
    std::FILE* f = std::tmpfile();
    if (f == nullptr) { std::perror("tmpfile"); std::exit(2); }
    const int saved = ::dup(::fileno(stdout));
    ::dup2(::fileno(f), ::fileno(stdout));
    fn();
    std::fflush(stdout);
    ::dup2(saved, ::fileno(stdout));
    ::close(saved);
    std::rewind(f);
    std::string out;
    char buf[256];
    while (std::fgets(buf, sizeof buf, f) != nullptr) out += buf;
    std::fclose(f);
    return out;
}

static int lines(const std::string& s) {
    int n = 0;
    for (char c : s) n += c == '\n';
    return n;
}

static void idle_beats() {                       // no request in flight: every beat must stay silent
    for (int i = 0; i < 50; ++i) strata::core::progress_beat();
}

static void short_step() {                       // one step inside one interval: exactly one line
    strata::core::progress_at("reading the prompt (batched): layer", 7, 2048);
    strata::core::progress_beat();
    for (int i = 0; i < 10; ++i) {               // ~60 ms of beats, under the 100 ms interval
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
        strata::core::progress_beat();
        strata::core::progress_beat();
    }
}

static void long_step() {                        // a step over two intervals: one line each, and no more
    strata::core::progress_at("reading the prompt (verify windows), from token", 1150);
    strata::core::progress_beat();
    for (int i = 0; i < 40; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(6));   // ~240 ms
        strata::core::progress_beat();
    }
}

int main() {
    setenv("STRATA_HB_S", "0.1", 1);             // 100 ms: read once, by the first beat's static
    strata::core::Progress& p = strata::core::progress();

    // 1. Idle: nothing at all, however often the (unrelated) beats come in.
    p.busy.store(false);
    check(captured(idle_beats).empty(), "no heartbeat while no request is in flight (what ends a lost-step engine)");

    // 2. A request that is moving: one line, and it says where the request is.
    p.busy.store(true);
    const std::string one = captured(short_step);
    check(lines(one) == 1, "one heartbeat for a step inside one interval, not one per beat");
    check(one.rfind("HB 2048 7 reading the prompt (batched): layer\n", 0) == 0,
          "the heartbeat carries the chunk, the detail and the stage (HB <chunk> <detail> <where>)");

    // 3. A step that runs longer keeps beating, so the server's silence clock keeps being reset.
    const std::string many = captured(long_step);
    check(lines(many) >= 2, "a step that runs longer than the interval prints more heartbeats");
    check(many.rfind("HB -1 1150 reading the prompt (verify windows), from token\n", 0) == 0,
          "a non-batched read reports the token it is at (chunk -1)");

    // 4. Silent again once the request is over (a stale line would keep a lost engine alive).
    p.busy.store(false);
    check(captured(idle_beats).empty(), "silent again once the request has ended");

    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
