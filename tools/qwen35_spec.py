"""Compare greedy streams with speculation disabled, lengths 1..4, and forced rejection prefixes 0..4."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default="/src/build-hip/strata-qwen35")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--kv", default="f32")
    ap.add_argument("--slots", default="0")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    tokens = json.loads(a.tokens.read_text());a.out.mkdir(parents=True,exist_ok=True)
    results=[];reference=None
    for spec,reject in [(s,-1) for s in range(5)]+[(4,p) for p in range(5)]:
        cmd=[a.exe,"--"+a.backend,"--model",a.model,"--mtp",a.mtp,"--spec",str(spec),
             "--force-reject",str(reject),"--kv",a.kv,"--expert-cache",a.slots,
             "--pool-workers","12","--max-context",str(max(512,len(tokens)+a.new+8)),
             "--max-new",str(a.new),"--tokens",",".join(map(str,tokens))]
        stem=a.out/f"spec-{spec}-reject-{reject}"
        with stem.with_suffix(".stderr").open("w") as err:
            p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=err,text=True)
        stem.with_suffix(".stdout").write_text(p.stdout)
        if p.returncode:raise RuntimeError(f"engine failed: {stem}")
        stream=[int(t) for t in p.stdout.splitlines()[0].split(",") if t]
        if reference is None:reference=stream
        result=dict(spec=spec,forced_reject=reject,tokens=stream,equal=stream==reference,command=cmd)
        results.append(result);(a.out/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(f"spec={spec} forced_reject={reject}: {'PASS' if result['equal'] else 'FAIL'} tokens={len(stream)}",flush=True)
        if not result['equal']:return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
