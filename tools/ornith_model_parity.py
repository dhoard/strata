#!/usr/bin/env python3
"""Full-model logits parity for the Ornith quality suite (ORNITH_QWEN35MOE.md phases 18E, 18F, 23).

The greedy comparison in tools/ornith_quality.py says whether two engines pick the same token; this
tool says how far apart they are numerically, which is the measurement phase 18E asks for
(max absolute error, RMS, relative RMS, KL divergence, top-1 and top-10 agreement).

For each suite prompt it:

  1. dumps the upstream llama.cpp oracle (tests/ornith-reference/reference.cpp) for the exact token
     sequence - per-layer residuals, hidden states and full-vocabulary logits, KV layout matched
  2. runs qwen35_gpu_parity, which replays the same tokens through Strata's shipped GPU session and
     compares every one of those tensors

    python tools/ornith_model_parity.py --model Ornith.gguf --tokenizer /work/tokenizer/ornith \
        --reference build-ornith-reference/ornith_reference \
        --gpu-parity build-hip/qwen35_gpu_parity --steps 16 --kv f16 --out /work/parity

The summary records, per token, the logits relative RMS, KL and top-10 overlap, plus the worst
residual relative RMS, so a first divergence is visible as a layer, not just a token.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ornith_quality import eos_token, load_tokenizer, render_chat  # noqa: E402
from ornith_quality_cases import CASES  # noqa: E402

FLOAT = r"[-+0-9.eE]+"
LOGITS = re.compile(r"^logits max_abs=(%s) rms=(%s) relative_rms=(%s)$" % (FLOAT, FLOAT, FLOAT))
HIDDEN = re.compile(r"^hidden max_abs=(%s) rms=(%s) relative_rms=(%s)$" % (FLOAT, FLOAT, FLOAT))
DIST = re.compile(r"^distribution kl=(%s) top10_overlap=(\d+)/10$" % FLOAT)
TOP1 = re.compile(r"^top1=(-?\d+)/(-?\d+)$")
RESIDUAL = re.compile(r"^layer (\d+) residual max_abs=(%s) rms=(%s) relative_rms=(%s)$" % (FLOAT, FLOAT, FLOAT))


def run(command: list[str], env: dict | None = None) -> str:
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                       env=env)
    if p.returncode:
        raise RuntimeError("%s failed (%d): %s" % (command[0], p.returncode, p.stderr.strip()[-400:]))
    return p.stdout


def parse(stdout: str) -> dict:
    tokens, worst = [], None
    pending: dict = {}
    for line in stdout.splitlines():
        m = RESIDUAL.match(line)
        if m:
            rel = float(m.group(4))
            if worst is None or rel > worst["relative_rms"]:
                worst = dict(layer=int(m.group(1)), relative_rms=rel)
            continue
        m = LOGITS.match(line)
        if m:
            pending["logits_max_abs"], pending["logits_rms"] = float(m.group(1)), float(m.group(2))
            pending["logits_relative_rms"] = float(m.group(3))
            continue
        m = HIDDEN.match(line)
        if m:
            pending["hidden_relative_rms"] = float(m.group(3))
            continue
        m = DIST.match(line)
        if m:
            pending["kl"], pending["top10_overlap"] = float(m.group(1)), int(m.group(2))
            continue
        m = TOP1.match(line)
        if m:
            pending["top1_strata"], pending["top1_reference"] = int(m.group(1)), int(m.group(2))
            pending["top1_match"] = pending["top1_strata"] == pending["top1_reference"]
            tokens.append(pending)
            pending = {}
    return dict(tokens=tokens, worst_residual=worst,
                passed=bool(re.search(r"^GPU model parity: PASS$", stdout, re.M)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--gpu-parity", default="build-hip/qwen35_gpu_parity")
    ap.add_argument("--mtp", default="", help="also compare the MTP draft graphs (optional)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--kv", default="f16", choices=("f16", "f32"))
    ap.add_argument("--slots", default="0")
    ap.add_argument("--gpu-layers", type=int, default=999)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    wanted = [c for c in a.cases.split(",") if c]
    cases = [c for c in CASES if not wanted or c["name"] in wanted]
    if not cases:
        ap.error("no cases selected")
    tokenizer = load_tokenizer(a.tokenizer)
    eos = eos_token(a.model)
    a.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if a.kv == "f16":
        env["Q35_REFERENCE_KV"] = "f16"
    results = []
    failures = 0
    for case in cases:
        ids = tokenizer.encode(render_chat(a.tokenizer, case["prompt"]))
        case_dir = a.out / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        oracle = case_dir / "oracle"
        oracle.mkdir(exist_ok=True)
        record = dict(case=case["name"], category=case["category"], prompt_tokens=len(ids),
                      steps=a.steps, kv=a.kv, eos=eos)
        try:
            reference = [a.reference, a.model, ",".join(map(str, ids)), str(oracle),
                         str(a.gpu_layers), str(a.steps), str(a.threads)]
            if a.mtp:
                reference.append(a.mtp)   # dumps the draft logits/hidden the GPU arm compares
            run(reference, env=env)
            command = [a.gpu_parity, a.model, ",".join(map(str, ids)), str(oracle), str(a.slots)]
            if a.mtp:
                command.append(a.mtp)
            stdout = run(command, env=env)
            (case_dir / "parity.log").write_text(stdout)
            parsed = parse(stdout)
        except RuntimeError as error:
            record["error"] = str(error)
            results.append(record)
            (a.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print("%-22s FAIL %s" % (case["name"], error), flush=True)
            failures += 1
            continue
        tokens = parsed["tokens"]
        if parsed["worst_residual"]:
            record["worst_residual_relative_rms"] = parsed["worst_residual"]["relative_rms"]
            record["worst_residual_layer"] = parsed["worst_residual"]["layer"]
        if tokens:
            worst = max(tokens, key=lambda t: t.get("logits_relative_rms", 0.0))
            record.update(
                tokens_compared=len(tokens),
                max_logits_relative_rms=worst.get("logits_relative_rms"),
                max_logits_max_abs=max(t.get("logits_max_abs", 0.0) for t in tokens),
                max_kl=max(t.get("kl", 0.0) for t in tokens),
                min_top10_overlap=min(t.get("top10_overlap", 0) for t in tokens),
                top1_matches=sum(1 for t in tokens if t.get("top1_match")),
                top1_total=len(tokens),
                per_token=tokens)
        record["gpu_parity_passed"] = parsed["passed"]
        if not parsed["passed"]:
            failures += 1
        results = [r for r in results if r["case"] != case["name"]] + [record]
        (a.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print("%-22s %-22s %s  logits rel_rms max %.3g, KL max %.3g, top10 min %d/10, top1 %d/%d" % (
            case["name"], case["category"], "PASS" if parsed["passed"] else "FAIL",
            record.get("max_logits_relative_rms", 0.0), record.get("max_kl", 0.0),
            record.get("min_top10_overlap", 0), record.get("top1_matches", 0),
            record.get("top1_total", 0)), flush=True)
    print("model parity: %d cases, %d failures" % (len(cases), failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
