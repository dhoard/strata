#!/usr/bin/env bash
# Run the Ornith-1.5-35B-A3B server container on this machine's AMD GPU (Qwen35MoE path).
#
#   ./run3.sh                     Ornith AD-Q4_K-IQ4_XS, http://127.0.0.1:9931
#   ./run3.sh --no-mtp            target-only (spec off) for a direct speed comparison
#   ./run3.sh --spec 1            enable trained external Qwen3.6 MTP
#   ./run3.sh --cpu               every layer on CPU instead of the GPU/CPU split (slower, no card needed
#                                 for the compute itself, but the image and its GPU device access are)
#   ./run3.sh --kv f32            the other KV layout the backend implements (f16 is the default)
#   ./run3.sh --vram-report        print what the tier needs against the VRAM budget, then exit
#   ./run3.sh --detach            background;  ./run3.sh --check  asks it afterwards whether it is up
#   ./run3.sh --offline           never download: fail with the command to run instead
#   ./run3.sh --env STRATA_PROMPT_CACHE=0        read every prompt in full (no conversation cache)
#   ./run3.sh --env STRATA_PROMPT_CACHE_SLOTS=8  more resume points, more VRAM (see docs/ORNITH_QWEN35MOE.md)
#   ./run3.sh --dry-run           print the docker command and the reasoning, change nothing
#
# Ornith-1.5 is a DIFFERENT architecture from Qwen3.8-Flash-Next (Qwen35MoE: 40 layers, 30 gated-delta-net
# recurrent + 10 full-attention, 256 experts top-8, 2048-wide).  It therefore has its own launcher, its own
# container name and its own work directory; run.sh and run2.sh are untouched.  See
# docs/ORNITH_QWEN35MOE.md for the architecture, the artifacts and the validation behind them.
#
# Main model:  AtomicChat/Ornith-1.5-35B-A3B-GGUF / Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf  (~20.1 GB)
# MTP draft:   EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF / mtpdraft-Q8_0.gguf           (~2.0 GB)
#
# Both live in the shared Hugging Face cache; prepared artifacts go under the writable /work mount, kept
# separate from the Qwen3.8 pack/MTP trees.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCH=""                                        # empty: whatever this machine actually has
MODEL="${STRATA_MODEL:-ornith}"                # the hfmodel.py key for the single-file family
IMAGE="${STRATA_IMAGE:-}"
NAME="${STRATA_CONTAINER_NAME:-}"
export STRATA_VRAM_LATER_MIB="${STRATA_VRAM_LATER_MIB:-768}"
BUDGET="${STRATA_VRAM_BUDGET_MIB:-10240}"      # this card has 12 272 MiB; the contract is 10 GiB
DEFAULT_MODEL_DIR="${DEFAULT_MODEL_DIR:-$HOME/Development/models}"
HF_CACHE="${STRATA_HF_CACHE_HOST:-}"
WORK="${STRATA_WORK_HOST:-}"
PORT="${STRATA_PORT_HOST:-9931}"
BIND="${STRATA_BIND:-127.0.0.1}"
MEMORY="${STRATA_CONTAINER_MEMORY:-96g}"
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-131072}"    # 128K default; 262144 fits the 10 GiB budget only with --kv f16
TIER="${STRATA_TIER:-gpu}"
# f16 halves KV storage so more routed experts fit on the card. Shipping gfx1101 measurements:
# 2472 cache slots, 52.0 tok/s over 128 outputs and 50.6 over 256 (64-token code prompt, 8 workers).
# Both f32/f16 have reference parity; Qwen3.8's int8/int4/int2 layouts do not apply to Qwen35MoE.
KV="${STRATA_KV:-f16}"
SPEC="${STRATA_SPEC:-0}"                       # MTP measured slower than plain decode today; off by default
VRAM_REPORT=0
DETACH=0 CHECK=0 FRESH=0 OFFLINE=0 DRY=0 CHECK_ONLY=0 EXPLICIT_CACHE=""
MODEL_FILE="" MTP_PATH=""
API_KEY="${STRATA_API_KEY:-}"
EXTRA_ENV=()
EXTRA_ARGS=()

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
xrun()   { if [ "$DRY" = 1 ]; then printf '  +'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  cat <<'USAGE'

Options:
      --model KEY         hfmodel.py family key (default: ornith)
      --model-file PATH   the main GGUF directly (skips the cache lookup)
      --hf-cache PATH     model cache in the Hugging Face layout (default: ~/Development/models)
      --work PATH         writable Strata state: packs, MTP, logs (default: ~/Development/strata-work)
      --budget MiB        VRAM Strata may use for itself (default 10240 on a 12 GiB card)
  -p, --port PORT         host port (default 9931)
      --bind ADDR         host interface to publish on (default 127.0.0.1)
      --api-key KEY       required when publishing beyond localhost
  -d, --detach            run in the background instead of in the foreground
      --check             query /v1/models of the running container and exit
      --check-only        validate the GPU, the artifacts and the geometry, then exit (no server)
      --offline           never download: fail with instructions instead
      --fresh             remove an existing container of the same name first
      --max-context N     KV context (default 131072; 262144 needs --kv f16 to fit the 10 GiB budget)
      --max-tokens N      output cap a request gets when it names none (default 32768)
      --reasoning-effort L  off, minimal, low, medium, high (default high)
      --prefill N         causal prefill tile (default 8; 1 disables batching, larger requests use 8-token tiles)
      --expert-cache N    VRAM expert slots held on the card, or 'auto' (default auto: as many as fit)
      --pool-workers N    CPU expert workers (default 8, measured on this machine; 0 selects auto)
      --gpu               GPU/CPU split tier: dense, GDN, attention, KV and expert cache on the card,
                          expert pool in system RAM (default)
      --cpu               every layer on CPU through ggml-cpu; measured 9.1 tok/s on this card's machine
      --kv F              KV layout: f16 (default) or f32
      --vram-report       ask the engine what the chosen tier/context/KV/cache needs and exit (no server)
      --spec N            trained external MTP draft length, 0..4 (default 0; use measured fastest setting)
      --no-mtp            disable the external MTP draft (--spec 0)
      --mtp PATH          a specific MTP GGUF (default: the cached mtpdraft-Q8_0.gguf)
  -e, --env KEY=VALUE     pass extra environment through (repeatable)
      --image NAME:TAG    override the image (default strata-hip:<arch>-latest)
      --allow-hsa-override  let HSA_OVERRIDE_GFX_VERSION through (breaks gfx1101; you have been told)
  -n, --dry-run           print the command instead of running it
  -h, --help              this text
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)           MODEL="${2:?--model needs a value}"; shift 2 ;;
    --model-file)      MODEL_FILE="${2:?}"; shift 2 ;;
    --hf-cache)        HF_CACHE="${2:?}"; shift 2 ;;
    --work)            WORK="${2:?}"; shift 2 ;;
    --budget)          BUDGET="${2:?}"; shift 2 ;;
    -p|--port)         PORT="${2:?}"; shift 2 ;;
    --bind)            BIND="${2:?}"; shift 2 ;;
    --api-key)         API_KEY="${2:?}"; shift 2 ;;
    -d|--detach)       DETACH=1; shift ;;
    --check)           CHECK=1; shift ;;
    --check-only)      CHECK_ONLY=1; DETACH=0; shift ;;
    --offline)         OFFLINE=1; shift ;;
    --fresh)           FRESH=1; shift ;;
    --max-context)     MAX_CONTEXT="${2:?--max-context needs a value}"; shift 2 ;;
    --gpu)             TIER=gpu; shift ;;
    --cpu)             TIER=cpu; shift ;;
    --kv)              KV="${2:?--kv needs a value}"; shift 2 ;;
    --vram-report)     VRAM_REPORT=1; CHECK_ONLY=1; TIER=gpu; shift ;;
    --max-tokens)      EXTRA_ENV+=(-e "STRATA_MAX_TOKENS=${2:?}"); shift 2 ;;
    --reasoning-effort) EXTRA_ENV+=(-e "STRATA_REASONING_EFFORT=${2:?}"); shift 2 ;;
    --prefill)         EXTRA_ENV+=(-e "STRATA_PREFILL=${2:?}"); shift 2 ;;
    --expert-cache)    EXPLICIT_CACHE="${2:?--expert-cache needs a value}"; shift 2 ;;
    --pool-workers)    EXTRA_ENV+=(-e "STRATA_POOL_WORKERS=${2:?}"); shift 2 ;;
    --spec)            SPEC="${2:?--spec needs a value}"; shift 2 ;;
    --no-mtp)          SPEC=0; shift ;;
    --mtp)             MTP_PATH="${2:?--mtp needs a value}"; shift 2 ;;
      --allow-hsa-override) EXTRA_ENV+=(-e "STRATA_ALLOW_HSA_OVERRIDE=1" -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-}"); shift ;;
    -e|--env)          EXTRA_ENV+=(-e "${2:?--env needs KEY=VALUE}"); shift 2 ;;
    --image)           IMAGE="${2:?}"; shift 2 ;;
    -n|--dry-run)      DRY=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 usage >&2; die "unknown option '$1'" ;;
  esac
done

[[ "$BUDGET" =~ ^[0-9]+$ ]] && [ "$BUDGET" -gt 1280 ] && [ "$BUDGET" -le 10240 ] \
  || die "budget must be 1281..10240 MiB; 10 GiB is the hard ceiling"
[[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] && [ "$MAX_CONTEXT" -gt 0 ] && [ "$MAX_CONTEXT" -le 262144 ] \
  || die "--max-context must be 1..262144 (Ornith's native maximum)"
[[ "$SPEC" =~ ^[0-9]+$ ]] && [ "$SPEC" -le 4 ] || die "--spec must be 0..4"
case "$TIER" in gpu|cpu) ;; *) die "--tier/--cpu/--gpu: gpu or cpu (got '$TIER')" ;; esac
case "$KV" in f32|f16) ;; *) die "--kv must be f32 or f16 (Qwen35MoE implements these two KV layouts; int8/int4
                      are the Qwen3.8 layouts and do not exist here)" ;; esac
[ "$MODEL" = ornith ] || die "run3.sh requires the ornith family; use --model-file for a compatible Qwen35MoE GGUF"
case "$BIND" in
  127.0.0.1|localhost|::1) ;;
  *) [ -n "${API_KEY//[[:space:]]/}" ] || die "--bind $BIND requires --api-key (AGENTS.md: localhost-only without authentication)" ;;
esac

PY="${PYTHON:-python3}"
HIPINFO="$ROOT/docker/hipinfo.py"
command -v docker >/dev/null || die "docker not found.  Build first: ./build.sh"
[ -f "$HIPINFO" ] || die "$HIPINFO is missing (is this the Strata repository root?)"

# ---------------------------------------------------------------- the card
ARCH_GUESSED="$("$PY" "$HIPINFO" --arch 2>/dev/null)" || die "no AMD GPU visible through the KFD topology." \
  note "Strata's HIP backend needs the amdgpu driver and /dev/kfd."
ARCH="${ARCH:-$ARCH_GUESSED}"
case "$ARCH" in
  gfx1100|gfx1101) ;;
  *) die "this machine's GPU is $ARCH; the Ornith launcher targets gfx1100/gfx1101 (docs/ORNITH_QWEN35MOE.md)" ;;
esac
RENDER="$("$PY" "$HIPINFO" --render-node)"
read -r VRAM_TOTAL VRAM_USED _VRAM_FREE <<<"$("$PY" "$HIPINFO" --vram)"
RESERVE="$("$PY" "$HIPINFO" --reserve-mib --allocation-guard --budget-mib "$BUDGET" --slack-mib "${STRATA_VRAM_SLACK_MIB:-256}")"
log "GPU: $ARCH ($RENDER), ${VRAM_TOTAL} MiB VRAM; ${BUDGET} MiB Strata ceiling"
note "allocation guard preserves runtime overhead and slack; cache reserves ${RESERVE} MiB for later buffers"

if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ "${STRATA_ALLOW_HSA_OVERRIDE:-0}" != "1" ]; then
  die "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set in your environment. On gfx1101 it makes
       the code object fail to load. Unset it, or pass --allow-hsa-override if you really mean it."
fi

# ---------------------------------------------------------------- the image
if [ -z "$IMAGE" ]; then
  for cand in "strata-hip:${ARCH}-latest" "strata-hip:${ARCH}"; do
    if docker image inspect "$cand" >/dev/null 2>&1; then IMAGE="$cand"; break; fi
  done
fi
[ -n "$IMAGE" ] || die "no strata-hip image for $ARCH. Build it:  ./build.sh   (takes 10-25 minutes the first time)"
NAME="${NAME:-strata-ornith-${ARCH}}"
log "image: $IMAGE   container: $NAME"

# The engine is the single source of truth for what it can serve.  If this build has no qwen35moe backend,
# say so BEFORE spending ~22 GB of download on an artifact the engine will refuse at load.
[ "$FRESH" = 1 ] && [ "$DRY" != 1 ] && docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$DRY" != 1 ]; then
  CAPS="$(docker run --rm --entrypoint /usr/local/bin/strata-qwen35 "$IMAGE" --capabilities 2>/dev/null | tr '\n' ' ')"
  case " $CAPS " in
    *" qwen35moe "*) ;;
    *) die "this engine image has no qwen35moe backend (strata-qwen35 reported: ${CAPS:-nothing}).
       Rebuild it with ./build.sh; see docs/ORNITH_QWEN35MOE.md." ;;
  esac
fi

# ---------------------------------------------------------------- the cache and the space it needs
if [ -z "$HF_CACHE" ]; then
  HF_CACHE="$DEFAULT_MODEL_DIR"
  [ -e "$HOME/.cache/huggingface/hub" ] \
    && note "hint: export HF_HUB_CACHE=$HF_CACHE  makes 'hf' and this script share one cache"
fi
WORK="${WORK:-$(dirname "$DEFAULT_MODEL_DIR")/strata-work}"
WORK="$(readlink -m "$WORK" 2>/dev/null || printf '%s' "$WORK")"
HF_CACHE="$(readlink -m "$HF_CACHE" 2>/dev/null || printf '%s' "$HF_CACHE")"
OR_NATIVE="${MODEL_FILE:-${STRATA_NATIVE:-}}"
OR_MTP="${MTP_PATH:-${STRATA_MTP_PATH:-}}"
if [ -n "$OR_NATIVE" ]; then
  eval "$("$PY" "$ROOT/docker/hfmodel.py" --model "$MODEL" --cache "$HF_CACHE" --print shell --allow-missing 2>/dev/null || true)"
  note "using --model-file $OR_NATIVE (cache lookup skipped for the main GGUF)"
fi
log "HF cache:  $HF_CACHE  (mounted read-write: a first run may download into it)"
log "work dir:  $WORK  packs=$WORK/packs/ornith-ad-q4-iq4-xs  mtp=$WORK/mtp/ornith-qwen36  logs=$WORK/logs/ornith"

# What is already there?  (asks the same resolver the container uses, so no drift between them)
eval "$("$PY" "$ROOT/docker/hfmodel.py" --model "$MODEL" --cache "$HF_CACHE" --print shell --allow-missing)"
existing_ancestor() { local p="$1"; while [ ! -e "$p" ] && [ "$p" != "/" ]; do p="$(dirname "$p")"; done; printf '%s' "$p"; }
free_gib() { df -BG --output=avail "$(existing_ancestor "$1")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0; }
dev_of()   { stat -c %d "$(existing_ancestor "$1")" 2>/dev/null || echo none; }
same_fs()  { [ "$(dev_of "$1")" = "$(dev_of "$2")" ]; }
if [ "${STRATA_CACHED:-0}" != 1 ] && [ -z "$OR_NATIVE" ]; then
  # ~20.1 GB main GGUF + ~2 GB MTP + a 20 GB floor; prepared artifacts are small (the native path reads
  # the GGUF itself), so unlike Qwen3.8 there is no 50 GB experts.bin term.
  need="$(awk -v d="${STRATA_DOWNLOAD_GB:-20.1}" -v m="${STRATA_MTP_DOWNLOAD_GB:-2.0}" \
            'BEGIN{printf "%d", d + m + 8 + 20}')"
  have="$(free_gib "$HF_CACHE")"
  log "Ornith is not in the cache yet: ~${need} GB free required (main + MTP + a 20 GB floor)"
  if [ "$OFFLINE" = 1 ]; then
    die "--offline, and $MODEL is not cached. Fetch it on the host:
         HF_HUB_CACHE=$HF_CACHE hf download $STRATA_REPO --include '$STRATA_HF_INCLUDE'
         HF_HUB_CACHE=$HF_CACHE hf download $STRATA_MTP_REPO --include '$STRATA_MTP_FILE'"
  fi
  if [ "${have:-0}" -lt "$need" ]; then
    die "$MODEL needs ${need} GB free - the main GGUF, the MTP draft and 20 GB that must stay free - but
       the filesystem holding $HF_CACHE has ${have} GB free.  Pick a bigger place for the cache:
         ./run3.sh --hf-cache /mnt/storage/models --work /mnt/storage/strata-work"
  fi
else
  note "$MODEL is cached"
fi

# ---------------------------------------------------------------- port
if [ "$DRY" != 1 ] && [ "$CHECK" != 1 ] && [ "$CHECK_ONLY" != 1 ]; then
  if "$PY" - "$BIND" "$PORT" <<'PY'
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((sys.argv[1], int(sys.argv[2]))); sys.exit(1)
except OSError:
    sys.exit(0)
finally:
    s.close()
PY
  then
    die "port $PORT on $BIND is already in use - Strata is probably already running:
       docker logs -f $NAME    (or ./run3.sh --check)"
  fi
fi

# ---------------------------------------------------------------- --check: just ask the running server
if [ "$CHECK" = 1 ]; then
  log "asking http://$BIND:$PORT/v1/models ..."
  STRATA_CHECK_API_KEY="$API_KEY" "$PY" - "$BIND" "$PORT" <<'PY'
import json, os, sys, urllib.request
try:
    request = urllib.request.Request(f"http://{sys.argv[1]}:{sys.argv[2]}/v1/models")
    if os.environ.get("STRATA_CHECK_API_KEY"):
        request.add_header("Authorization", "Bearer " + os.environ["STRATA_CHECK_API_KEY"])
    with urllib.request.urlopen(request, timeout=10) as r:
        models = json.load(r).get("data", [])
except Exception as e:
    print(f"    not answering yet: {e}", file=sys.stderr)
    raise SystemExit(1)
if not models:
    print("    answering, but no model is listed", file=sys.stderr)
    raise SystemExit(1)
for m in models:
    status = (m.get("status") or {}).get("value", "?")
    ctx = (m.get("meta") or {}).get("n_ctx", "?")
    print(f"    {m.get('id')}  {status}  ctx={ctx}")
    if status != "loaded":
        raise SystemExit(1)
PY
  exit $?
fi

# Refuse inference with an older engine image that cannot admit allocations.
if [ "$DRY" = 0 ] && [ "$CHECK_ONLY" = 0 ]; then
  budget_guard="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "io.strata.vram-budget"}}' 2>/dev/null || true)"
  [ "$budget_guard" = 1 ] || die "image $IMAGE predates the HIP allocation guard; rebuild with ./build.sh before inference"
fi

# ---------------------------------------------------------------- run it
CHECK_ONLY_SUFFIX=""
[ "$CHECK_ONLY" = 1 ] && CHECK_ONLY_SUFFIX="-check"
ARGS=(docker run --rm --name "$NAME$CHECK_ONLY_SUFFIX"
      --entrypoint /opt/strata/docker/entrypoint-ornith.sh
      --device /dev/kfd --device "$RENDER"
      --group-add video --group-add render
      --ulimit memlock=-1:-1
      --shm-size=16g --memory "$MEMORY"
      -e "STRATA_MODEL=$MODEL"
      -e "STRATA_VRAM_BUDGET_MIB=$BUDGET"
      -e "STRATA_MAX_CONTEXT=$MAX_CONTEXT"
      -e "STRATA_TIER=$TIER"
      -e "STRATA_KV=$KV"
      -e "STRATA_POOL_WORKERS=${STRATA_POOL_WORKERS:-8}"
      -e "STRATA_PREFILL=${STRATA_PREFILL:-8}"
      -e "STRATA_EXPERT_CACHE=${EXPLICIT_CACHE:-${STRATA_EXPERT_CACHE:-auto}}"
      -e "STRATA_SPEC=$SPEC"
      -e "STRATA_VRAM_LATER_MIB=$STRATA_VRAM_LATER_MIB"
      -e "STRATA_VRAM_RUNTIME_RESERVE_MIB=${STRATA_VRAM_RUNTIME_RESERVE_MIB:-1024}"
      -e "STRATA_VRAM_SLACK_MIB=${STRATA_VRAM_SLACK_MIB:-256}"
      -e "STRATA_PORT=$PORT"
      -e "STRATA_WORK=/work"
      -e "HIP_VISIBLE_DEVICES=0" -e "HSA_ENABLE_SDMA=1"
      -v "$HF_CACHE:/hf-cache" -v "$WORK:/work")
[ "$CHECK_ONLY" = 1 ] || ARGS+=(-p "$BIND:$PORT:$PORT")
[ -n "$OR_NATIVE" ] && ARGS+=(-v "$OR_NATIVE:/models/ornith.gguf:ro" -e "STRATA_NATIVE=/models/ornith.gguf")
[ -n "$OR_MTP" ]    && ARGS+=(-v "$OR_MTP:/models/mtp.gguf:ro" -e "STRATA_MTP_PATH=/models/mtp.gguf")
[ -n "${HF_TOKEN:-}" ]    && ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
[ -n "${HF_REVISION:-}" ] && ARGS+=(-e "HF_REVISION=$HF_REVISION")
[ -n "$API_KEY" ] && ARGS+=(-e "STRATA_API_KEY=$API_KEY")
[ "$OFFLINE" = 1 ] && ARGS+=(-e "STRATA_DOWNLOAD_MODEL=0" -e "HF_HUB_OFFLINE=1")
for tuning_key in STRATA_MAX_TOKENS STRATA_REASONING_EFFORT STRATA_PREFILL_RING STRATA_STAGER_RING \
                  STRATA_STAGER_THREADS STRATA_IO_THREADS STRATA_HIPBLASLT_TUNING \
                  STRATA_MTP_REPO STRATA_MTP_FILE; do
  [ -z "${!tuning_key:-}" ] || ARGS+=(-e "$tuning_key=${!tuning_key}")
done
[ "$CHECK_ONLY" = 1 ] && ARGS+=(-e "STRATA_CHECK_ONLY=1")
[ "$VRAM_REPORT" = 1 ] && ARGS+=(-e "STRATA_VRAM_REPORT=1")
[ ${#EXTRA_ENV[@]} -gt 0 ] && ARGS+=("${EXTRA_ENV[@]}")
if [ "$DETACH" = 1 ]; then ARGS+=(-d "$IMAGE")
elif [ "$CHECK_ONLY" = 0 ] && [ -t 0 ] && [ -t 1 ]; then ARGS+=(-it "$IMAGE")
else ARGS+=("$IMAGE"); fi

log "starting Ornith on $ARCH (server on http://$BIND:$PORT, logs: $WORK/logs/ornith/)"
note "tier=$TIER kv=$KV context=$MAX_CONTEXT expert-cache=${EXPLICIT_CACHE:-auto} spec=$SPEC ($([ "$SPEC" = 0 ] && echo 'no MTP' || echo 'external Qwen3.6 MTP'))"
note "the container first checks the plan against the ${BUDGET} MiB budget and refuses what cannot fit"
note "first start prepares the ~20 GB model and, with --spec, the ~2 GB trained MTP draft"
if [ "$DETACH" = 1 ]; then
  xrun "${ARGS[@]}"
  [ "$DRY" = 1 ] || { note "container $NAME;  logs: docker logs -f $NAME"; note "ready check: ./run3.sh --check"; }
else
  xrun "${ARGS[@]}"
fi
