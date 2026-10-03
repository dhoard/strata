#!/usr/bin/env python3
"""Build deterministic coding-context token prompts for the Ornith context sweep (phase 19).

The throughput sweep must use coding content, not repeated filler, so the prompt is built by
concatenating this repository's own sources (C++ kernels, the Python tools and the docs) in a fixed
file order until the requested token count is reached.  The result is a JSON list of token ids that
`tools/qwen35_bench.py --tokens` consumes; the same file therefore reproduces the same prompt.

    python tools/qwen35_context.py --tokenizer /work/tokenizer/ornith --out /work/prompts \
        --lengths 1024,8192,32768,130000

The lengths are prompt tokens.  Keep them under the KV capacity the benchmark passes to the engine
(131072 by default) with room for the generated tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ornith_quality import load_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Coding content, in a fixed order: kernels and core first, then the server, then the docs.
SOURCES = ["src/core/*.cpp", "src/kernels/cuda/*.cu", "src/qwen35/*.cu", "src/qwen35/*.cpp",
           "include/strata/**/*.hpp", "serve/*.py", "tools/*.py", "docs/*.md"]


def corpus() -> list[str]:
    parts: list[str] = []
    for pattern in SOURCES:
        for path in sorted(ROOT.glob(pattern)):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append("\n\n=== %s ===\n%s" % (path.relative_to(ROOT).as_posix(), text))
    if not parts:
        raise SystemExit("no source files found to build the prompt from")
    return parts


def prompt_tokens(tokenizer, length: int) -> list[int]:
    """Encode the corpus and cut it to exactly `length` tokens, looping if the corpus is shorter."""
    parts = corpus()
    ids: list[int] = []
    while len(ids) < length:
        for part in parts:
            ids.extend(tokenizer.encode(part))
            if len(ids) >= length:
                break
    return ids[:length]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lengths", default="1024,8192,32768,130000")
    a = ap.parse_args()
    tokenizer = load_tokenizer(a.tokenizer)
    a.out.mkdir(parents=True, exist_ok=True)
    index = []
    for raw in a.lengths.split(","):
        length = int(raw)
        ids = prompt_tokens(tokenizer, length)
        if len(ids) != length:
            ap.error("could not build a %d-token prompt" % length)
        path = a.out / ("context-%d-tokens.json" % length)
        path.write_text(json.dumps(ids) + "\n")
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        head = tokenizer.decode(ids[:64])
        index.append(dict(length=length, file=path.name, sha256=digest, first_tokens=head[:80]))
        print("%7d tokens -> %s (%s)" % (length, path.name, digest), flush=True)
    (a.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
