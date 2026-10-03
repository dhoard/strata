"""Gate row tiling with independent primitive/model parity, then benchmark each passing variant."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",required=True)
    ap.add_argument("--tokens",required=True,type=Path)
    ap.add_argument("--oracle",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--rows",default="2,4,8")
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sequence=json.loads((a.oracle/"metadata.json").read_text())["tokens"]
    for rows in a.rows.split(","):
        env=dict(os.environ,STRATA_QWEN35_MMVQ_ROWS=rows)
        variant=a.out/("rows-"+rows);variant.mkdir(parents=True,exist_ok=True)
        for name,cmd in (
            ("primitives",["/src/build-ornith-reference/ornith_primitives"]),
            ("model-parity",["/src/build-hip/qwen35_gpu_parity",a.model,
                             ",".join(map(str,sequence)),str(a.oracle),"0"])):
            with (variant/(name+".log")).open("w") as log:
                subprocess.run(cmd,env=dict(env,GGML_CUDA_DISABLE_FUSION="1"),stdout=log,
                               stderr=subprocess.STDOUT,check=True)
            print(f"rows={rows}: {name} PASS",flush=True)
        subprocess.run(["python3","/src/tools/qwen35_bench.py","--model",a.model,"--tokens",str(a.tokens),
                        "--configs","gpu:12:0:f32:0","--context","131072","--new","128",
                        "--out",str(variant)],env=env,check=True)


if __name__=="__main__":
    main()
