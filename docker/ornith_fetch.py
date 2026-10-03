#!/usr/bin/env python3
"""Resolve/download exact Ornith main and external MTP files into the shared HF cache.

Hugging Face owns resumable downloads and atomic blob completion. Only complete files count
as cached. No vision files or alternative quantizations are downloaded.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import shutil
import sys
import hfmodel


def fetch(cache: Path, *, native: str = "", mtp: str = "", spec: int = 0,
          offline: bool = False, revision: str = "", download=None, free_bytes=None) -> dict:
    if spec not in range(5):
        raise ValueError("spec must be 0..4")
    family = hfmodel.FAMILIES["ornith"]
    source = hfmodel.mtp_source("ornith")
    items = [("NATIVE",native,hfmodel.shard_path(cache,"ornith",1,revision),
              family["repo"],family["file"],revision,20.1e9)]
    if spec:
        mr = os.environ.get("STRATA_MTP_REV", "")
        items.append(("MTP_GGUF",mtp,hfmodel.mtp_path(cache,"ornith",mr),
                      source["repo"],source["file"],mr,2.0e9))
    result = {"MTP_GGUF":""}; missing = []
    for key, explicit, cached, repo, filename, rev, size in items:
        path = Path(explicit) if explicit else cached
        if explicit and not path.is_file():
            raise ValueError(f"explicit artifact does not exist: {path}")
        if path is not None and path.is_file():
            result[key] = str(path)
        else:
            missing.append((key,repo,filename,rev,size))
    if missing and offline:
        commands = [f"HF_HUB_CACHE={shlex.quote(str(cache))} hf download {repo} --include {shlex.quote(filename)}"
                    for _,repo,filename,_,_ in missing]
        raise ValueError("offline: missing artifacts; fetch them with:\n  "+"\n  ".join(commands))
    if missing:
        ancestor = cache
        while not ancestor.exists(): ancestor = ancestor.parent
        available = shutil.disk_usage(ancestor).free if free_bytes is None else free_bytes
        required = int(sum(x[4] for x in missing))+(20<<30)+(1<<30)
        if available < required:
            raise ValueError(f"download needs {required/(1<<30):.1f} GiB including a 20 GiB floor; cache has {available/(1<<30):.1f} GiB free")
        if download is None:
            from huggingface_hub import hf_hub_download
            download = hf_hub_download
        for key,repo,filename,rev,_ in missing:
            print(f"ornith_fetch: downloading {repo}/{filename} (resumable)",file=sys.stderr,flush=True)
            path = Path(download(repo_id=repo,filename=filename,revision=rev or None,cache_dir=str(cache)))
            if not path.is_file(): raise ValueError(f"download did not produce a complete file: {path}")
            result[key] = str(path)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--native",default="")
    ap.add_argument("--mtp",default="")
    ap.add_argument("--spec",type=int,default=0)
    ap.add_argument("--revision",default="")
    ap.add_argument("--offline",action="store_true")
    a = ap.parse_args()
    try:
        for key,value in fetch(a.cache,native=a.native,mtp=a.mtp,spec=a.spec,revision=a.revision,offline=a.offline).items():
            print(f"{key}={shlex.quote(value)}")
        return 0
    except (ValueError,OSError) as e:
        print(f"ornith_fetch: {e}",file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
