#!/usr/bin/env python3
"""Write the server's tokenizer files from an Ornith/Qwen35MoE GGUF.

`serve/server.py` loads a tokenizer directory with `vocab.json`, `merges.txt` and `token_type.json`.  Ornith
ships its tokenizer only inside the GGUF (`tokenizer.ggml.tokens` / `.merges` / `.token_type`), so this
extracts them.  The GGUF header is parsed with tools/gguf_reader.py; no tensor data is read.

    python tools/ornith_tokenizer.py --model Ornith.gguf --out /work/tokenizer/ornith
"""
from __future__ import annotations

import argparse
import json
import hashlib
import pathlib
import os
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402


def prepare(model: pathlib.Path, out: pathlib.Path) -> dict:
    g = GGUFFile(model)
    with model.open("rb") as f:
        identity = {"header_sha256":hashlib.sha256(f.read(g.data_start)).hexdigest(),"bytes":model.stat().st_size}
    marker = out / ".complete.json"
    required = ["vocab.json","merges.txt","token_type.json"]
    if marker.is_file() and all((out/p).is_file() for p in required):
        try:
            if json.loads(marker.read_text()) == identity:
                print(f"ornith_tokenizer: reusing verified tokenizer {out}")
                return identity
        except (ValueError,OSError): pass
    md = g.metadata
    tokens = md.get("tokenizer.ggml.tokens")
    merges = md.get("tokenizer.ggml.merges")
    types = md.get("tokenizer.ggml.token_type")
    if tokens is None or merges is None or types is None:
        raise ValueError("the GGUF has no gpt2 tokenizer metadata (tokens/merges/token_type)")
    if not (len(tokens) == len(types)):
        raise ValueError(f"{len(tokens)} tokens but {len(types)} token types")

    out.parent.mkdir(parents=True,exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-",dir=out.parent))
    # The server indexes by id, so the map is token -> id (duplicates keep the first, as the model does).
    vocab: dict[str, int] = {}
    for i, t in enumerate(tokens):
        vocab.setdefault(t, i)
    (stage / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (stage / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (stage / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (stage / "chat_template.jinja").write_text(tpl, encoding="utf-8")
    # Completion is written only after every file. A failed extraction leaves no reusable tokenizer.
    (stage / ".complete.json").write_text(json.dumps(identity,sort_keys=True)+"\n")
    backup = out.with_name(out.name+".previous")
    try:
        if out.exists():
            if backup.exists(): shutil.rmtree(backup)
            os.replace(out,backup)
        os.replace(stage,out)
    except BaseException:
        if not out.exists() and backup.exists(): os.replace(backup,out)
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f"ornith_tokenizer: wrote {out} ({len(tokens)} tokens, {len(merges)} merges, "
          f"template={'yes' if tpl else 'no'})")
    return identity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,type=pathlib.Path)
    ap.add_argument("--out", required=True,type=pathlib.Path)
    a = ap.parse_args()
    try: prepare(a.model,a.out); return 0
    except (ValueError,OSError) as e:
        print(f"ornith_tokenizer: {e}",file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
