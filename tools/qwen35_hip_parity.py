"""Independent HIP model parity ladder. Run inside Docker after ./build.sh --ornith-reference.

The oracle executes pinned upstream graphs with fusion disabled, F32 KV, and CPU
routed experts. Dumps live under --work, outside the repository; compact metrics
and full logs live under --out. These traced runs are correctness tests.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--lengths", default="1,8,64,512,4096")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results_path = a.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for length in map(int, a.lengths.split(",")):
        case = f"code-{length}"
        tokens = json.loads((a.prompts / (case+"-tokens.json")).read_text())
        work = a.work / case
        env = dict(os.environ, GGML_CUDA_DISABLE_FUSION="1", OMP_NUM_THREADS="12")
        commands = [["/src/build-ornith-reference/ornith_reference", a.model,
                     ",".join(map(str, tokens)), str(work), "99", str(a.steps), "12"]]
        with (a.out / (case+"-oracle.log")).open("w") as f:
            subprocess.run(commands[-1], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        meta = json.loads((work / "metadata.json").read_text())
        commands.append(["/src/build-hip/qwen35_gpu_parity", a.model,
                         ",".join(map(str, meta["tokens"])), str(work), "0"])
        with (a.out / (case+".log")).open("w") as f:
            status = subprocess.run(commands[-1], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        log = (a.out / (case+".log")).read_text()
        logits = re.findall(r"^logits .*relative_rms=([\deE.+-]+)", log, re.M)
        result = dict(case=case, prompt_tokens=length, greedy_steps=a.steps, returncode=status,
                      comparisons=len(logits), max_logits_relative_rms=max(map(float, logits), default=None),
                      kl_max=max(map(float, re.findall(r"distribution kl=([\deE.+-]+)", log)), default=None),
                      top10_min=min(map(int, re.findall(r"top10_overlap=(\d+)", log)), default=None),
                      commands=commands, oracle_environment={k:env[k] for k in ("GGML_CUDA_DISABLE_FUSION","OMP_NUM_THREADS")})
        results = [r for r in results if r["case"] != case]+[result]
        results_path.write_text(json.dumps(results, indent=2)+"\n")
        print(f"{case}: {'PASS' if status == 0 else 'FAIL'} comparisons={len(logits)} "
              f"max_relative_rms={result['max_logits_relative_rms']}", flush=True)
        if status:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
