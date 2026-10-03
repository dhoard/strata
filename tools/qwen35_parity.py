"""Run independent upstream CPU full-model parity on reproducible Ornith token prompts.

Build the oracle with ./build.sh, then run this inside the HIP builder container with the model and
tokenizer mounted. Numerical results are not throughput benchmarks: both engines run and trace layers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from strata_tokenizer import Tokenizer

CODE = """// Review this implementation and explain how to fix the race.
class WorkQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> pending;
    bool stopping = false;
public:
    void submit(std::function<void()> job) {
        std::unique_lock lock(mutex);
        pending.push_back(std::move(job));
        ready.notify_one();
    }
    void run() {
        while (!stopping) {
            std::unique_lock lock(mutex);
            ready.wait(lock, [&] { return stopping || !pending.empty(); });
            if (stopping && pending.empty()) return;
            auto job = std::move(pending.front());
            pending.pop_front();
            lock.unlock();
            job();
        }
    }
};
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--exe", default="build-hip/qwen35_model_parity")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", default="1,8,64,512,4096")
    ap.add_argument("--greedy", type=int, default=32)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    vocab = json.loads((a.tokenizer / "vocab.json").read_text())
    tokens = [None]*len(vocab)
    for token, i in vocab.items():
        tokens[i] = token
    merges = (a.tokenizer / "merges.txt").read_text().splitlines()
    merges = [m for m in merges if m and not m.startswith("#")]
    tokenizer = Tokenizer(tokens, merges, json.loads((a.tokenizer / "token_type.json").read_text()))
    base = tokenizer.encode(CODE)
    lengths = [int(n) for n in a.lengths.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    results_path = a.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for length in lengths:
        if length <= 0:
            ap.error("lengths must be positive")
        prompt = (base*((length+len(base)-1)//len(base)))[:length]
        case = f"code-{length}"
        (a.out / f"{case}-tokens.json").write_text(json.dumps(prompt)+"\n")
        command = [a.exe, a.model, ",".join(map(str,prompt)), str(a.greedy), str(a.threads)]
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
        started = time.monotonic()
        with (a.out / f"{case}.log").open("w") as log:
            status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        rows = []
        for line in (a.out / f"{case}.log").read_text().splitlines():
            if line.startswith("logits "):
                rows.append(dict(re.findall(r"([\w_]+)=([^ ]+)",line)))
        results = [r for r in results if r["case"] != case]
        results.append({"case":case,"prompt_tokens":length,"greedy_steps":a.greedy,
                        "threads":a.threads,"returncode":status,"elapsed_seconds":time.monotonic()-started,
                        "logits":rows})
        (a.out / "results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(f"{case}: {'PASS' if status == 0 else 'FAIL'} ({len(rows)} logits comparisons)",flush=True)
        if status != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
