#!/usr/bin/env bash
# Ornith phase measurements (docs/ORNITH_QWEN35MOE.md phases 11, 19, 20, 22) against the exact
# shipping artifact at the shipping context capacity of 131072.
#
# Run from the repository root, on the gfx1101 machine, with the Strata runtime image built:
#
#     ./build.sh --ornith-reference          # once: engine + independent llama.cpp reference
#     bash bench/results/2026-10-03-ornith-phases/run.sh
#
# Everything runs in the runtime image (the project's build/run contract), with the repository at
# /src and the shared Hugging Face cache at /hf.  Results land next to this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/bench/results/2026-10-03-ornith-phases"
IMAGE="${STRATA_IMAGE:-strata-hip:gfx1101-latest}"
PHASE="${1:-all}"

MAIN=/hf/models--AtomicChat--Ornith-1.5-35B-A3B-GGUF/snapshots/7aa8fc1d9b861d797880f4a341166d4bb3439f74/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf
MTP=/hf/models--EryriLabs--Ornith-1.5-35B-A3B-BigBang-MTP-GGUF/snapshots/2e9f6f487b82200f2a045e67d01a3f1e2982b00a/mtpdraft-Q8_0.gguf

run() {   # run <container-name> <shell command>
  docker run --rm --name "$1" \
    --device /dev/kfd --device /dev/dri/renderD128 --group-add video --group-add render \
    -e HIP_VISIBLE_DEVICES=0 --ulimit memlock=-1:-1 \
    -v "$ROOT":/src -v "$HOME/.cache/huggingface/hub":/hf \
    --entrypoint bash "$IMAGE" -lc "$2"
}

# ------------------------------------------------- phase 18E/23: full-model logits parity
if [ "$PHASE" = all ] || [ "$PHASE" = parity ]; then
  mkdir -p "$OUT/model-parity"
  run "strata-phase-parity" "
    python3 /src/tools/ornith_tokenizer.py --model $MAIN \
      --out /src/bench/results/2026-10-03-ornith-phases/tokenizer &&
    python3 /src/tools/ornith_model_parity.py --model $MAIN --mtp $MTP \
      --tokenizer /src/bench/results/2026-10-03-ornith-phases/tokenizer \
      --reference /src/build-ornith-reference/ornith_reference \
      --gpu-parity /src/build-hip/qwen35_gpu_parity --steps 16 --kv f16 --slots 0 \
      --out /src/bench/results/2026-10-03-ornith-phases/model-parity" \
    2>&1 | tee "$OUT/model-parity/run.log"
fi

# ---------------------------------------------------------------- phase 19: filled-context sweep
if [ "$PHASE" = all ] || [ "$PHASE" = 19 ]; then
  mkdir -p "$OUT/context-sweep"
  for N in 1024 8192 32768; do
    mkdir -p "$OUT/context-sweep/ctx-$N"
    run "strata-phase-ctx-$N" "
      python3 /src/tools/qwen35_bench.py --model $MAIN \
        --tokens /src/bench/results/2026-10-03-ornith-phases/prompts/context-$N-tokens.json \
        --context 131072 --new 128 --configs gpu:8:auto:f16:0 \
        --out /src/bench/results/2026-10-03-ornith-phases/context-sweep/ctx-$N" \
      2>&1 | tee "$OUT/context-sweep/ctx-$N/run.log"
  done
fi

# ------------------------------------------------- phase 19b: filled 128K prompt (one diagnostic)
# A 130000-token prompt costs ~44 minutes of prompt processing per request, so this row is one
# excluded warmup plus one timed request rather than three.  It is an execution/capacity diagnostic
# like the 8K row in docs/ORNITH_QWEN35MOE.md, not a three-run median.
if [ "$PHASE" = all ] || [ "$PHASE" = 128k ]; then
  mkdir -p "$OUT/context-sweep/ctx-130000"
  run "strata-phase-ctx-130000" "
    python3 /src/tools/qwen35_bench.py --model $MAIN \
      --tokens /src/bench/results/2026-10-03-ornith-phases/prompts/context-130000-tokens.json \
      --context 131072 --new 32 --repeats 2 --configs gpu:8:auto:f16:0 \
      --out /src/bench/results/2026-10-03-ornith-phases/context-sweep/ctx-130000" \
    2>&1 | tee "$OUT/context-sweep/ctx-130000/run.log"
fi

# ---------------------------------------------------------------- phase 11: prefill tile sizes
if [ "$PHASE" = all ] || [ "$PHASE" = 11 ]; then
  # One output directory per tile size: tools/qwen35_bench.py keys its results by the engine config,
  # and every tile size uses the same config string, so a shared directory would overwrite them.
  for P in 1 2 4 8; do
    mkdir -p "$OUT/prefill-tiles/tile-$P"
    run "strata-phase-prefill-$P" "
      python3 /src/tools/qwen35_bench.py --model $MAIN \
        --tokens /src/bench/results/2026-10-03-ornith-phases/prompts/context-4096-tokens.json \
        --context 131072 --new 16 --prefill $P --configs gpu:8:auto:f16:0 \
        --out /src/bench/results/2026-10-03-ornith-phases/prefill-tiles/tile-$P" \
      2>&1 | tee "$OUT/prefill-tiles/tile-$P/run.log"
  done
fi

# ---------------------------------------------------------------- phase 20: per-part profile
if [ "$PHASE" = all ] || [ "$PHASE" = 20 ]; then
  mkdir -p "$OUT/profile"
  run "strata-phase-profile" "
    T=\$(python3 -c \"import json;print(','.join(map(str,json.load(open('/src/bench/results/2026-10-03-ornith-phases/prompts/context-1024-tokens.json')))))\")
    /usr/local/bin/strata-qwen35 --gpu --model $MAIN --spec 0 --kv f16 --expert-cache auto \
      --pool-workers 8 --max-context 131072 --max-new 128 --profile --tokens \"\$T\"" \
    2>&1 | tee "$OUT/profile/profile.log"
fi

# ---------------------------------------------------------------- phase 22: MTP policy at 128K
if [ "$PHASE" = all ] || [ "$PHASE" = 22 ]; then
  mkdir -p "$OUT/mtp-policy"
  run "strata-phase-mtp" "
    python3 /src/tools/qwen35_bench.py --model $MAIN --mtp $MTP \
      --tokens /src/bench/results/2026-10-03-ornith-phases/prompts/context-1024-tokens.json \
      --context 131072 --new 128 \
      --configs gpu:8:auto:f16:0,gpu:8:auto:f16:1 \
      --out /src/bench/results/2026-10-03-ornith-phases/mtp-policy" \
    2>&1 | tee -a "$OUT/mtp-policy/run.log"
fi

echo "phase measurements written under $OUT"
