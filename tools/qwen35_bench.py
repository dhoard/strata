"""Warmed, sequential Ornith engine benchmarks; run inside the build/runtime container.

Use JSON token prompts from qwen35_parity.py. Each configuration gets its own process,
one excluded warmup, and three timed requests. No numerical tracing is enabled.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import re
import subprocess
import threading
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default="/src/build-hip/strata-qwen35")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp")
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--prefill", type=int, default=0,
                    help="prompt tokens per engine pass (0 = the engine's own default of 8; 1..8 are "
                         "the implemented native tile sizes, larger requests are split into tiles)")
    ap.add_argument("--configs", default="gpu:12:0:f32:0,cpu:12:0:f32:0")
    ap.add_argument("--repeats", type=int, default=4,
                    help="requests per configuration: the first is a warmup, the rest are timed "
                         "(default 4 = one warmup + three timed, the shipping methodology)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = json.loads(a.tokens.read_text())
    results_path = a.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for config in a.configs.split(","):
        backend, workers, slots, kv, spec = config.split(":")
        cmd = [a.exe, "--serve", "--model", a.model, "--"+backend,
               "--max-context", str(a.context), "--pool-workers", workers,
               "--expert-cache", slots, "--kv", kv, "--spec", spec]
        if int(spec):
            if not a.mtp:
                ap.error("speculation requires --mtp")
            cmd += ["--mtp", a.mtp]
        if a.prefill:
            cmd += ["--prefill", str(a.prefill)]
        start = time.monotonic()
        rows = []
        memory = {"host_rss_peak_bytes":0,"host_vmhwm_peak_bytes":0,"device_vram_peak_bytes":0}
        stop_monitor = threading.Event()
        with (a.out / (config.replace(":", "-")+".stderr")).open("w") as errors:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=errors, text=True, bufsize=1)
            def monitor():
                vram = list(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
                while not stop_monitor.is_set():
                    try:
                        fields = dict(re.split(r":\s*",line,1) for line in
                                      Path(f"/proc/{p.pid}/status").read_text().splitlines() if ":" in line)
                        for key,field in (("host_rss_peak_bytes","VmRSS"),("host_vmhwm_peak_bytes","VmHWM")):
                            memory[key]=max(memory[key],int(fields.get(field,"0 kB").split()[0])*1024)
                        memory["device_vram_peak_bytes"]=max(memory["device_vram_peak_bytes"],
                            max((int(f.read_text()) for f in vram),default=0))
                    except (OSError,ValueError):
                        pass
                    stop_monitor.wait(.05)
            watcher = threading.Thread(target=monitor,daemon=True);watcher.start()
            try:
                line = p.stdout.readline().strip()
                if not line.startswith("READY "):
                    raise RuntimeError(f"engine failed to start: {line}")
                load_seconds = time.monotonic()-start
                with (a.out / (config.replace(":", "-")+".stdout")).open("w") as output:
                    output.write(line+"\n")
                    for repeat in range(a.repeats):
                        p.stdin.write(f"GEN {a.new} temperature=0 penalty_repeat=1 "
                                      + ",".join(map(str, ids))+"\n")
                        p.stdin.flush()
                        tokens = []
                        while True:
                            line = p.stdout.readline().strip()
                            output.write(line+"\n")
                            if line.startswith("T "):
                                tokens.append(int(line.split()[1]))
                            elif line.startswith("DONE "):
                                f = line.split()
                                n, prompt, pp, decode = int(f[1]), int(f[2]), float(f[3]), float(f[4])
                                rows.append(dict(warmup=repeat == 0, generated=n, prompt_tokens=prompt,
                                                 prefill_ms=pp, decode_ms=decode,
                                                 tok_s=1000*n/decode if decode else 0,
                                                 accepted=int(f[6]), proposed=int(f[7]), tokens=tokens))
                                break
                            elif not line or line.startswith("ERR "):
                                raise RuntimeError(f"generation failed: {line}")
                    if any(row["tokens"] != rows[0]["tokens"] for row in rows[1:]):
                        raise RuntimeError("greedy tokens changed across identical sequential requests")
                    p.stdin.write("QUIT\n"); p.stdin.flush()
                    p.wait(timeout=60)
                    if p.returncode:
                        raise RuntimeError(f"engine exit {p.returncode}")
            finally:
                if p.poll() is None:
                    p.kill(); p.wait()
                stop_monitor.set();watcher.join()
        speeds = [r["tok_s"] for r in rows if not r["warmup"]]
        result = dict(config=config, command=cmd, context=a.context, prefill=a.prefill,
                      repeats=a.repeats, load_seconds=load_seconds,
                      median_tok_s=statistics.median(speeds), min_tok_s=min(speeds), max_tok_s=max(speeds),
                      runs=rows)
        result.update(memory)
        result["environment"]={k:os.environ[k] for k in (
            "OMP_NUM_THREADS","STRATA_QWEN35_MMVQ_ROWS","HIP_VISIBLE_DEVICES",
            "STRATA_VRAM_BUDGET_MIB","STRATA_VRAM_RUNTIME_RESERVE_MIB",
            "STRATA_QWEN35_GRAPHS","STRATA_QWEN35_CACHE_PROFILE",
            "STRATA_QWEN35_NO_HOST_PIN","STRATA_QWEN35_OLD_ATTENTION") if k in os.environ}
        results = [r for r in results if r["config"] != config or r["context"] != a.context]
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2)+"\n")
        print(f"{config} ctx={a.context}: {result['median_tok_s']:.2f} tok/s "
              f"({min(speeds):.2f}–{max(speeds):.2f}), generated={rows[-1]['generated']}", flush=True)


if __name__ == "__main__":
    main()
