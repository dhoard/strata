#!/usr/bin/env python3
"""Deterministic Ornith quality/greedy regression across three arms (ORNITH_QWEN35MOE.md, phases 18F/23).

For each prompt the *same token sequence* is pushed through:

  1. upstream llama.cpp (the pinned revision, tests/ornith-reference/reference.cpp) on the same GGUF
  2. Strata Qwen35MoE with speculation off (`--spec 0`)
  3. Strata Qwen35MoE with the trained external Qwen3.6 MTP (`--spec 1`)

Greedy sampling only, one fresh context per case, and the comparison is exact token equality between
the continuations - not generated-text similarity.  The upstream arm is the independent reference:
Strata's numeric parity against it is established separately, so an exact token match here is the
end-to-end statement that the shipped path generates what llama.cpp generates.

Three details decide whether the comparison means anything, and all three are handled here:

* **Chat template.**  These prompts are chat turns, so they are rendered through the model's own
  `chat_template.jinja` (the same `serve.frontend.ChatTemplate` the server uses).  A raw
  completion prompt makes this model emit `<|im_end|>` at once, which is not a useful test.
* **KV layout.**  The reference defaults to f32 KV; Strata's shipping layout is f16.  The reference
  arm is given `Q35_REFERENCE_KV` to match `--kv`, because a different KV format is different
  arithmetic and can flip an argmax.
* **Expert placement.**  The reference keeps routed experts on the CPU, so the controlled Strata
  arms use `--expert-cache 0` (all CPU) as well.  With `auto` the spec-off and spec-on arms get
  different slot counts - hence different rounding - so the shipping arms are measured separately
  and only reported, never used to claim MTP equivalence.

Streams stop at the model's first end-of-text token (read from the GGUF), since the reference tool
emits a fixed number of tokens and would otherwise continue past the end of the answer.

    python tools/ornith_quality.py --model Ornith.gguf --mtp mtpdraft.gguf \
        --tokenizer /work/tokenizer/ornith --reference build-ornith-reference/ornith_reference \
        --exe build-hip/strata-qwen35 --steps 32 --out /work/quality

The suite covers the workloads phase 23 names (code generation, code repair, repository reasoning,
structured JSON, prose reasoning, instruction following) plus the long repeated structure of 18F.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ornith_quality_cases import CASES  # noqa: E402
from strata_tokenizer import Tokenizer  # noqa: E402


def load_tokenizer(directory: Path) -> Tokenizer:
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, i in vocab.items():
        tokens[i] = token
    merges = [m for m in (directory / "merges.txt").read_text().splitlines()
              if m and not m.startswith("#")]
    return Tokenizer(tokens, merges, json.loads((directory / "token_type.json").read_text()))


def render_chat(directory: Path, prompt: str) -> str:
    """Render one user turn with the model's template, thinking off, exactly as the server does."""
    from serve.frontend import ChatTemplate
    template = ChatTemplate(directory / "chat_template.jinja")
    return template.render([{"role": "user", "content": prompt}], add_generation_prompt=True,
                           enable_thinking=False)


def eos_token(model: str) -> int:
    """The artifact's end-of-text id, read from the GGUF rather than assumed."""
    from gguf_reader import GGUFFile
    metadata = GGUFFile(model).metadata
    return int(metadata["tokenizer.ggml.eos_token_id"])


def truncate_at_eos(ids: list[int], eos: int) -> list[int]:
    if eos in ids:
        return ids[:ids.index(eos)]
    return ids


def parse_stream(stdout: str) -> list[int]:
    """The engine's non-interactive greedy output: one comma-separated line of token ids."""
    lines = stdout.splitlines()
    first = lines[0] if lines else ""
    return [int(t) for t in first.split(",") if t != ""]


def first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_strata(exe: str, model: str, mtp: str, tokens: list[int], steps: int, *,
               backend: str, kv: str, slots: str, workers: int, context: int,
               spec: int) -> dict:
    cmd = [exe, "--" + backend, "--model", model, "--spec", str(spec), "--kv", kv,
           "--expert-cache", slots, "--pool-workers", str(workers),
           "--max-context", str(context), "--max-new", str(steps),
           "--tokens", ",".join(map(str, tokens))]
    if spec:
        cmd += ["--mtp", mtp]
    started = time.monotonic()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError("strata-qwen35 failed (%d): %s" % (p.returncode, p.stderr.strip()[-400:]))
    return dict(tokens=parse_stream(p.stdout), seconds=round(time.monotonic() - started, 3),
                command=cmd)


def run_reference(reference: str, model: str, tokens: list[int], out: Path, steps: int, *,
                  gpu_layers: int, threads: int, kv: str) -> dict:
    env = dict(os.environ)
    if kv == "f16":
        env["Q35_REFERENCE_KV"] = "f16"
    cmd = [reference, model, ",".join(map(str, tokens)), str(out), str(gpu_layers), str(steps),
           str(threads)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if p.returncode:
        raise RuntimeError("ornith_reference failed (%d): %s" % (p.returncode, p.stderr.strip()[-400:]))
    metadata = json.loads((out / "metadata.json").read_text())
    # metadata["tokens"] is the prompt followed by the greedy continuation.
    return dict(tokens=[int(t) for t in metadata["tokens"]][len(tokens):], command=cmd, kv=kv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", default="")
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--exe", default="build-hip/strata-qwen35")
    ap.add_argument("--reference", default="", help="upstream ornith_reference; omit to skip arm 1")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--gpu-layers", type=int, default=999)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--kv", default="f16", choices=("f16", "f32"))
    ap.add_argument("--slots", default="0", help="expert cache for the controlled arms (0 = all CPU)")
    ap.add_argument("--shipping-slots", default="auto", help="expert cache for the reported shipping arms")
    ap.add_argument("--shipping", action="store_true", help="also run spec 0/1 with the shipping cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--raw", action="store_true", help="skip the chat template (raw completion prompts)")
    ap.add_argument("--cases", default="", help="comma-separated case names; default all")
    ap.add_argument("--require-reference-exact", action="store_true",
                    help="fail when any case's greedy stream disagrees with upstream anywhere in the "
                         "window; off by default because late-token drift under f16 KV is expected "
                         "and is measured, not hidden, as reference agreement length")
    a = ap.parse_args()
    if not a.mtp:
        ap.error("--mtp is required: the suite compares spec 0 against spec 1")

    wanted = [c for c in a.cases.split(",") if c]
    cases = [c for c in CASES if not wanted or c["name"] in wanted]
    if not cases:
        ap.error("no cases selected")
    tokenizer = load_tokenizer(a.tokenizer)
    eos = eos_token(a.model)
    a.out.mkdir(parents=True, exist_ok=True)
    results_path = a.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    failures = 0
    for case in cases:
        text = case["prompt"] if a.raw else render_chat(a.tokenizer, case["prompt"])
        ids = tokenizer.encode(text)
        case_dir = a.out / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "prompt-tokens.json").write_text(json.dumps(ids) + "\n")
        (case_dir / "prompt-text.txt").write_text(text + "\n")
        record = dict(case=case["name"], category=case["category"], prompt_tokens=len(ids),
                      steps=a.steps, kv=a.kv, context=a.context, workers=a.workers,
                      controlled_slots=a.slots, eos=eos)
        arms: dict[str, list[int]] = {}
        try:
            if a.reference:
                ref = run_reference(a.reference, a.model, ids, case_dir / "reference", a.steps,
                                    gpu_layers=a.gpu_layers, threads=a.threads, kv=a.kv)
                arms["reference"] = truncate_at_eos(ref["tokens"], eos)
                record["reference_command"] = ref["command"]
            arms["spec0"] = run_strata(a.exe, a.model, a.mtp, ids, a.steps, backend=a.backend,
                                       kv=a.kv, slots=a.slots, workers=a.workers,
                                       context=a.context, spec=0)["tokens"]
            arms["spec1"] = run_strata(a.exe, a.model, a.mtp, ids, a.steps, backend=a.backend,
                                       kv=a.kv, slots=a.slots, workers=a.workers,
                                       context=a.context, spec=1)["tokens"]
            if a.shipping:
                arms["spec0-shipping"] = run_strata(
                    a.exe, a.model, a.mtp, ids, a.steps, backend=a.backend, kv=a.kv,
                    slots=a.shipping_slots, workers=a.workers, context=a.context, spec=0)["tokens"]
                arms["spec1-shipping"] = run_strata(
                    a.exe, a.model, a.mtp, ids, a.steps, backend=a.backend, kv=a.kv,
                    slots=a.shipping_slots, workers=a.workers, context=a.context, spec=1)["tokens"]
        except RuntimeError as error:
            record["error"] = str(error)
            results = [r for r in results if r["case"] != case["name"]] + [record]
            results_path.write_text(json.dumps(results, indent=2) + "\n")
            print("%-22s FAIL %s" % (case["name"], error), flush=True)
            failures += 1
            continue
        for name, stream in arms.items():
            record[name + "_tokens"] = stream
            record[name + "_text"] = tokenizer.decode(stream)
        record["spec1_equals_spec0"] = arms["spec1"] == arms["spec0"]
        record["spec1_vs_spec0_first_diff"] = first_difference(arms["spec1"], arms["spec0"])
        if "reference" in arms:
            record["reference_equals_spec0"] = arms["reference"] == arms["spec0"]
            record["reference_vs_spec0_first_diff"] = first_difference(arms["reference"], arms["spec0"])
        if a.shipping:
            record["shipping_spec1_equals_spec0"] = arms["spec1-shipping"] == arms["spec0-shipping"]
        if not record["spec1_equals_spec0"]:
            failures += 1
        if a.require_reference_exact and record.get("reference_equals_spec0") is False:
            failures += 1
        agreement = (record.get("reference_vs_spec0_first_diff")
                     if "reference" in arms else None)
        record["reference_agreement_tokens"] = (len(arms["spec0"]) if agreement is None
                                                else agreement) if "reference" in arms else None
        results = [r for r in results if r["case"] != case["name"]] + [record]
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        verdict = "PASS" if record["spec1_equals_spec0"] else "FAIL"
        print("%-22s %-22s %s (prompt %d, spec1==spec0 %s, ref agreement %s/%d, shipping spec1==spec0 %s)" % (
            case["name"], case["category"], verdict, len(ids), record["spec1_equals_spec0"],
            record["reference_agreement_tokens"], len(arms["spec0"]),
            record.get("shipping_spec1_equals_spec0")), flush=True)
    summary = dict(cases=len(cases), failures=failures, steps=a.steps, kv=a.kv, context=a.context,
                   controlled_slots=a.slots, shipping=a.shipping, raw=a.raw,
                   model=a.model, mtp=a.mtp, upstream=a.reference or "skipped")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("quality: %d cases, %d failures" % (len(cases), failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
