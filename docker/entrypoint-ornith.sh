#!/usr/bin/env bash
# Container entrypoint for the Ornith-1.5 / Qwen35MoE path (../run3.sh).
#
# It exists as a SEPARATE entrypoint on purpose: the Qwen3.8 flow (entrypoint-hip.sh) is built around a
# two-shard GGUF, an experts.bin pack and a Qwen4Exp MTP runtime, and pretending an Ornith checkpoint is
# one of those is exactly what the engine's architecture guard refuses.  This entrypoint:
#
#   guard -> device -> VRAM budget -> resolve the single GGUF + external MTP -> validate the artifact
#         -> build the engine config -> serve
#
# The artifact validation is `strata-qwen35-check`, the compiled-in Qwen35MoE geometry/tensor guard (the
# same code `src/core/qwen35.cpp` uses).  It reads the header only, so it is fast even on a 20 GB file.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # /opt/strata/docker
REPO="$(dirname "$DIR")"
PY="${PYTHON:-python3}"

log() { printf 'strata-ornith: %s\n' "$*"; }
die() { printf 'strata-ornith: ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. environment guards
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ "${STRATA_ALLOW_HSA_OVERRIDE:-0}" != "1" ]; then
  die "HSA_OVERRIDE_GFX_VERSION='$HSA_OVERRIDE_GFX_VERSION' is set; on gfx1101 it makes the code object
       fail to load. Remove it, or set STRATA_ALLOW_HSA_OVERRIDE=1 if you know why."
fi
unset HSA_OVERRIDE_GFX_VERSION
[ -n "${AMD_SERIALIZE_KERNEL:-}" ] && { log "unsetting AMD_SERIALIZE_KERNEL=$AMD_SERIALIZE_KERNEL"; unset AMD_SERIALIZE_KERNEL; }
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
# CPU fallback uses physical cores; count cores rather than /proc/cpuinfo's per-thread entries.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS="$("$PY" -c 'import os, psutil; print(psutil.cpu_count(logical=False) or os.cpu_count() or 1)')"
fi

# ---------------------------------------------------------------- 2. the device
ARCH="$("$PY" "$DIR/hipinfo.py" --arch)" \
  || die "no AMD GPU visible: run with --device /dev/kfd --device /dev/dri/renderD<N>"
case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "the GPU is $ARCH; this launcher targets gfx1100/gfx1101 (docs/ORNITH_QWEN35MOE.md)" ;;
esac
log "GPU (HIP index $HIP_VISIBLE_DEVICES): $ARCH"
[ -x /usr/local/bin/strata-device ] && { /usr/local/bin/strata-device >/tmp/strata-device.txt 2>&1 \
  || die "the engine cannot open this GPU"; head -3 /tmp/strata-device.txt | sed 's/^/  /'; }

# ---------------------------------------------------------------- 3. VRAM budget (same contract as run.sh/run2.sh)
BUDGET_MIB="${STRATA_VRAM_BUDGET_MIB:-10240}"
[[ "$BUDGET_MIB" =~ ^[0-9]+$ ]] && [ "$BUDGET_MIB" -gt 1280 ] && [ "$BUDGET_MIB" -le 10240 ] \
  || die "VRAM budget must be 1281..10240 MiB; 10 GiB is the hard ceiling"
export STRATA_VRAM_BUDGET_MIB="$BUDGET_MIB"
export STRATA_VRAM_LATER_MIB="${STRATA_VRAM_LATER_MIB:-768}"
export STRATA_VRAM_SLACK_MIB="${STRATA_VRAM_SLACK_MIB:-256}"
export STRATA_VRAM_RUNTIME_RESERVE_MIB="${STRATA_VRAM_RUNTIME_RESERVE_MIB:-1024}"
log "VRAM: ${BUDGET_MIB} MiB ceiling; runtime reserve ${STRATA_VRAM_RUNTIME_RESERVE_MIB} MiB, slack ${STRATA_VRAM_SLACK_MIB} MiB"

# ---------------------------------------------------------------- 4. the model, from the HF cache
MODEL="${STRATA_MODEL:-ornith}"
export STRATA_HF_CACHE="${STRATA_HF_CACHE:-${HF_HUB_CACHE:-/hf-cache}}"
export STRATA_WORK="${STRATA_WORK:-/work}"
LOG="${STRATA_LOG:-$STRATA_WORK/logs/ornith/engine.log}"
RUN_DIR="${STRATA_RUNTIME_DIR:-/run}"
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-131072}"
MAX_TOKENS="${STRATA_MAX_TOKENS:-32768}"
REASONING="${STRATA_REASONING_EFFORT:-high}"
PREFILL="${STRATA_PREFILL:-8}"
EXPERT_CACHE="${STRATA_EXPERT_CACHE:-auto}"
POOL_WORKERS="${STRATA_POOL_WORKERS:-0}"
# Which tier runs what.  `gpu` is the optimized split the launcher uses by default: dense projections, GDN,
# attention, KV and the expert cache live on the card, the expert pool that does not fit stays in system RAM
# and is streamed in.  `cpu` runs every layer on CPU through ggml-cpu.  There is no silent fallback between
# them: a failed GPU tier dies with its own error rather than quietly serving from the slower one.
TIER="${STRATA_TIER:-gpu}"
# The two KV layouts the backend implements and has reference parity for.  f16 halves the KV's VRAM, and on
# this card that freed space buys expert-cache slots, which is where the measured decode win comes from.
KV="${STRATA_KV:-f16}"
SPEC="${STRATA_SPEC:-0}"
MODEL_NAME="${STRATA_MODEL_NAME:-ornith-1.5-35b-a3b-ad-q4-iq4}"
MTP_MODE="${STRATA_MTP_MODE:-auto}"      # auto | 0 | a path inside the container

case "$TIER" in gpu|cpu) ;; *) die "STRATA_TIER must be gpu or cpu (got '$TIER')" ;; esac
case "$KV" in f32|f16) ;; *) die "STRATA_KV must be f32 or f16, the KV layouts Qwen35MoE implements (got '$KV')" ;; esac
[[ "$EXPERT_CACHE" =~ ^(auto|[0-9]+)$ ]] || die "STRATA_EXPERT_CACHE must be auto or a slot count (got '$EXPERT_CACHE')"
[[ "$PREFILL" =~ ^[0-9]+$ ]] && [ "$PREFILL" -ge 1 ] && [ "$PREFILL" -le 4096 ] \
  || die "STRATA_PREFILL must be 1..4096 (native execution uses tiles of at most 8 tokens)"

[ "$MODEL" = ornith ] || die "this entrypoint requires STRATA_MODEL=ornith"
if [ "$MTP_MODE" = 0 ] || [ "$MTP_MODE" = off ]; then SPEC=0; fi
FETCH=("$PY" "$DIR/ornith_fetch.py" --cache "$STRATA_HF_CACHE" --spec "$SPEC")
[ -z "${STRATA_NATIVE:-}" ] || FETCH+=(--native "$STRATA_NATIVE")
DRAFT_OVERRIDE="${STRATA_MTP_PATH:-}"
[ "$MTP_MODE" = auto ] || [ "$MTP_MODE" = 0 ] || [ "$MTP_MODE" = off ] || DRAFT_OVERRIDE="$MTP_MODE"
[ -z "$DRAFT_OVERRIDE" ] || FETCH+=(--mtp "$DRAFT_OVERRIDE")
[ -z "${HF_REVISION:-${STRATA_HF_REV:-}}" ] || FETCH+=(--revision "${HF_REVISION:-$STRATA_HF_REV}")
[ "${STRATA_DOWNLOAD_MODEL:-1}" != 0 ] && [ "${HF_HUB_OFFLINE:-0}" != 1 ] || FETCH+=(--offline)
# A check is a question about what is already on the machine, so it never downloads: without this the
# resolver would fetch ~22 GB in answer to `--check-only`.
[ "${STRATA_CHECK_ONLY:-0}" = "1" ] && FETCH+=(--offline)
log "resolving exact main and trained MTP artifacts ..."
RESOLVED="$("${FETCH[@]}")" || die "artifact preparation failed"
eval "$RESOLVED"
log "model $MODEL"
log "  gguf=$NATIVE"
log "  mtp=${MTP_GGUF:-<none>}"

# ---------------------------------------------------------------- 5. validate the artifact (the guard)
log "validating the qwen35moe geometry and tensor set ..."
VALIDATE=(strata-qwen35 --check --model "$NATIVE" --max-context "$MAX_CONTEXT")
[ -z "$MTP_GGUF" ] || VALIDATE+=(--mtp "$MTP_GGUF" --spec "$SPEC")
"${VALIDATE[@]}" >/dev/null || die "the artifact failed the Qwen35MoE guard above"

# The server needs a tokenizer directory; Ornith ships its tokenizer inside the GGUF.
TOKENIZER_DIR="${STRATA_TOKENIZER:-$STRATA_WORK/tokenizer/ornith}"
"$PY" "$REPO/tools/ornith_tokenizer.py" --model "$NATIVE" --out "$TOKENIZER_DIR" \
  || die "could not prepare a complete tokenizer from $NATIVE"

# ---------------------------------------------------------------- 6. the engine config
mkdir -p "$RUN_DIR" "$(dirname "$LOG")" 2>/dev/null || true
CONFIG="${STRATA_CONFIG:-$RUN_DIR/strata-ornith.json}"
export CONFIG NATIVE MTP_GGUF LOG MODEL_NAME MAX_CONTEXT MAX_TOKENS REASONING PREFILL KV SPEC POOL_WORKERS EXPERT_CACHE TOKENIZER_DIR TIER
"$PY" - <<'PY'
import json, os
e = os.environ
args = ["--serve", "--model", e["NATIVE"], "--max-context", e["MAX_CONTEXT"], "--pool-workers", e["POOL_WORKERS"],
        "--kv", e["KV"], "--expert-cache", e["EXPERT_CACHE"], "--prefill", e["PREFILL"], "--" + e["TIER"]]
# The prompt cache is on by default; these pass a launcher's override through (see docs/ORNITH_QWEN35MOE.md).
if e.get("PROMPT_CACHE"):
    args += ["--prompt-cache", e["PROMPT_CACHE"]]
if e.get("PROMPT_CACHE_SLOTS"):
    args += ["--prompt-cache-slots", e["PROMPT_CACHE_SLOTS"]]
if e["MTP_GGUF"]:
    args += ["--mtp", e["MTP_GGUF"], "--spec", e["SPEC"]]
cfg = {"exe": os.environ.get("STRATA_EXE", "/usr/local/bin/strata-qwen35"), "args": args,
       "cwd": "/opt/strata", "tokenizer": e["TOKENIZER_DIR"], "model_name": e["MODEL_NAME"],
       "log": e["LOG"], "host": os.environ.get("STRATA_HOST", "0.0.0.0"),
       "max_tokens": int(e["MAX_TOKENS"]), "reasoning_effort": e["REASONING"]}
open(e["CONFIG"], "w").write(json.dumps(cfg, indent=1) + "\n")
print("strata-ornith: wrote %s\n               %s" % (e["CONFIG"], " ".join(args)))
PY

# ---------------------------------------------------------------- 6b. will it actually fit the card?
# Predicts the resident tier's VRAM from the artifact's own geometry before anything is allocated, so an
# impossible context/KV/slot combination is refused here, in seconds, instead of failing after ~20 GB of
# loading.  Nothing is lowered silently: the report names what to change.
if [ "$TIER" = gpu ]; then
  FIT=(strata-qwen35 --fit-report --gpu --model "$NATIVE" --max-context "$MAX_CONTEXT" --kv "$KV"
       --expert-cache "$EXPERT_CACHE")
  [ -z "$MTP_GGUF" ] || FIT+=(--mtp "$MTP_GGUF" --spec "$SPEC")
  log "checking the plan against the ${BUDGET_MIB} MiB budget ..."
  if ! "${FIT[@]}"; then
    die "this configuration does not fit the card.  Lower --max-context, keep --kv f16, or ask for fewer
       expert slots (--expert-cache); run3.sh never reduces a requested context silently."
  fi
fi

# ---------------------------------------------------------------- 7. check-only stops here
if [ "${STRATA_CHECK_ONLY:-0}" = "1" ]; then
  log "check-only: the artifact and its geometry are valid; not starting the server"
  exit 0
fi

# ---------------------------------------------------------------- 8. serve
# The engine is the source of truth for what it can serve.  run3.sh checks this before the download too;
# this is the in-container backstop (e.g. an image swapped in later).
CAPS="$(/usr/local/bin/strata-qwen35 --capabilities 2>/dev/null | tr '\n' ' ')"
case " $CAPS " in
  *" qwen35moe "*) ;;
  *) log "this engine build has no qwen35moe execution backend (it serves: ${CAPS:-unknown})."
     log "Rebuild the image with ./build.sh; the artifact and geometry themselves validated."
     exit 78 ;;
esac
case " $CAPS " in
  *" gpu "*) ;;
  *) [ "$TIER" != gpu ] || die "this engine build has no GPU tier (it reported: ${CAPS}); run ./run3.sh --cpu
      for the CPU tier"
esac
cd "$REPO"
log "starting the server on ${STRATA_HOST:-0.0.0.0}:${STRATA_PORT:-8080} (tier=$TIER kv=$KV expert-cache=$EXPERT_CACHE workers=$POOL_WORKERS spec=$SPEC)"
exec "$PY" -m serve.server --engine strata --config "$CONFIG" --port "${STRATA_PORT:-8080}"
