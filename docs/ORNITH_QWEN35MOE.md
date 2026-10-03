# Ornith-1.5 (Qwen35MoE) on Strata

Ornith-1.5-35B-A3B is a Qwen3.5-family mixture-of-experts model. Strata added a separate Qwen35MoE
architecture path for it rather than teaching the existing Qwen4Exp path a second meaning. This page
records the model, the architecture, the artifacts, what is implemented and verified on the RX 7700 XT
(gfx1101), and what remains.

## Status

| piece | state |
|---|---|
| GGUF inspection and the checked-in layout report | done, measured |
| Architecture identity (`ModelKind`) and the Qwen35MoE geometry/tensor guard | done, 16-case test + the real header |
| Native routed experts at 2048/512 with IQ4_XS gate/up + Q4_K down (grouped HIP) | done, GPU parity |
| Single-file (Ornith) cache resolution and `run3.sh` host-side launcher | done, contract-tested (9 new `run3.sh` cases in `docker/test_runtime_contract.py`) |
| Qwen35 forward pass: GDN, full attention, MoE, ordinary-residual trunk | done, host tests + runs on the real model |
| GGUF loader + quantized matvec (ggml-cpu vec_dot for the projections) | done, loads the 20 GB artifact |
| CPU tier (`./run3.sh --cpu`): every layer on CPU through ggml-cpu, served by the same server surface | done, 9.1 tok/s measured |
| GPU tier (the default): dense projections, GDN, attention, KV, norms, shared expert and the expert cache on the card; the expert pool that does not fit streamed from system RAM | done, **52.0 tok/s** with 131072 KV capacity (64-token prompt, 128 outputs; measurements below) |
| VRAM expert cache with per-layer admission and `--expert-cache auto` sized from measured free bytes | done; automatic capacity, prompt frequency placement frozen during decode |
| External Qwen3.6 MTP backend and greedy speculative rollback | done; batched verification and exact rollback; spec 1 measures **50.0 vs 52.0 tok/s** target-only, so spec defaults to 0 |
| VRAM fit report (`strata-qwen35 --fit-report`), run by the container before it serves | done; predicts the session's own allocation exactly (planned 8.482 GiB = allocated 8.482 GiB) |
| Chunked prefill and multi-token verification (one pass over the weights for several tokens) | done, 8-token causal tiles; full logits/hidden/state and all rollback prefixes bitwise equal to single-token execution |
| Prompt cache: recurrent-state checkpoints so a turn reads only its new tail, not the whole conversation | done, measured **39.7 s -> 2.9 s** for a 2K-token agentic turn; bitwise equal to a full read |

`./run3.sh` runs the GPU/CPU split by default, the same contract `run.sh` and `run2.sh` use: whatever fits
stays on the card, and the routed expert set that does not fit is streamed from system RAM. The latest warmed code measurement is 52.0 tok/s, within the lower end of the 50-70 target. This is
**decode throughput with 131072 KV capacity**, not a filled 128K prompt benchmark. Prefill, cache placement,
and server overhead are reported separately. Trained MTP is available through `--spec`; it is slower on
this workload and is not the default.

## Prompt cache

Ornith's GDN layers are a recurrence: state cannot be rewound token by token, so before this work every
request started with `reset()` and **re-read its whole prompt**, at roughly 50 tok/s. That is what made an
agentic client feel slow - amanda's turns carry ~2,000 prompt tokens (system prompt, `AGENTS.md`, tool
schemas, transcript) and paid ~40 s of prompt processing on *every* turn, while a short web chat paid ~1 s.
The cache keeps recurrent-state checkpoints so a turn reads only what is new:

- one checkpoint at the end of each prompt, with the logits it produced - an identical repeat (a retry)
  then reads **nothing at all**;
- `--prompt-cache-slots N` (default 4) periodic checkpoints inside the prompt, spaced `prompt/slots`
  apart, which bound what a resumed turn re-reads to about `1/slots` of the prompt.

The periodic points are not an optimisation detail, they are what makes the cache work at all. Qwen's own
chat template re-renders a past assistant turn from its content, without the `<think>` scaffold that its
generation prompt ended with, so the next turn's prompt diverges **at the very end of the previous one**:
an end-of-prompt checkpoint sits inside that divergence and can never be reused. (This is why the same
problem exists for Mamba/RWKV models in other engines, which also keep periodic state checkpoints.)

Measured on gfx1101 through `./run3.sh`, `reasoning_effort=none`, f16 KV, 131072 capacity, auto expert
cache, one conversation whose first prompt is 2,020 tokens (the same three-file prompt a coding agent
sends):

| turn | prompt tokens | prompt tokens reused | wall | engine prompt processing |
|---|---|---|---|---|
| 1 (cold) | 2020 | 0 | 35.6 s | 34.8 s |
| 2 | 2035 | 1536 | 9.1 s | 8.7 s |
| 3 | 2050 | 1920 | 2.9 s | 2.8 s |
| 4 | 2069 | 1944 | 2.9 s | 2.8 s |
| 5 | 2085 | 2040 | 1.3 s | 1.2 s |
| 6 | 2101 | 2072 | 0.9 s | 0.9 s |
| 6 again | 2117 | 2096 | 0.8 s | 0.7 s |

The first turn still reads everything (there is nothing cached yet); from the third turn on a turn costs
about a second instead of forty. The server reports the reuse as OpenAI's `usage.prompt_tokens_details.cached_tokens`,
and `strata-qwen35 --serve` prints it as the 9th `DONE` field.

What it costs and what it does not do:

- **VRAM**: one state copy per checkpoint, 61 MiB each on this model (30 GDN layers), so `1 + slots` =
  **305 MiB** at the default. It comes out of the expert cache: 2278 slots instead of 2472 in the
  measured session. `--prompt-cache-slots 0` keeps only the end-of-prompt checkpoint; `--prompt-cache 0`
  disables the cache entirely (a full read per request, the pre-cache behaviour), which is what the parity
  tools use.
- **Correctness**: a resumed turn's logits and greedy tokens are **bitwise** a full read's. The periodic
  resume is exercised on the real artifact at f32 and f16 KV (`qwen35_batch_session_test`) and on a
  synthetic trunk (`qwen35_layers_test`), both against a session with the cache off.
- **Not a KV cache in RAM**: the checkpoints hold recurrent state and KV cursors, not the conversation's
  K/V storage; the attention KV stays in VRAM as before.
- A prompt that diverges *before* every checkpoint (a client that rewrites its history, or a different
  conversation arriving at the server) is still a full read. The engine's
  `STRATA_QWEN35_CACHE_DEBUG=1` prints the decision per request (`count`, `checkpoint`, shared `prefix`,
  `resume`).


## The model

- **Main:** `AtomicChat/Ornith-1.5-35B-A3B-GGUF`, file `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf` (20.1 GB).
  This is the architecture-aware ~4-bit quant: routed expert gate/up `IQ4_XS`, routed expert down `Q4_K`.
- **Speculative draft:** `EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF`, file `mtpdraft-Q8_0.gguf` (2.0 GB).
  A trained Qwen3.6-derived MTP head, intended to run as a separate draft with stock Ornith GGUFs. Ornith's
  own embedded MTP tensors are NOT the default.
- **Text only.** Images are not handled on this path.

`docs/ornith/gguf-layout.txt` is the full checked-in tensor layout of the main artifact;
`bench/results/2026-10-02-ornith-rdna3/gguf-metadata.json` is the derived geometry in one file.

## Architecture

Ornith-1.5-35B-A3B is a 40-layer hybrid. Every fourth layer is conventional full attention
(`full_attention_interval = 4`); the other 30 are gated delta-net (GDN) recurrent layers. Both families
run the same 256-expert top-8 MoE FFN with a sigmoid-gated shared expert.

Verified from the artifact:

| | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| layers | 40 (30 GDN recurrent + 10 full attention) |
| residual width | 2048 |
| routed experts / selected | 256 / 8 |
| routed expert FF | 512 |
| shared expert FF | 512 |
| attention heads / KV heads | 16 / 2 |
| head width | 256 |
| partial RoPE | 64, sections `[11, 11, 10, 0]`, base 1e7 |
| GDN state | 128, key heads 16, value heads 32, conv 4 |
| GDN inner (value dim) | 4096 |
| qkv projection width | 8192 (= 2·16·128 + 32·128) |
| RMS epsilon | 1e-6 |
| context | 262144 |
| vocab | 248320 |

Routed expert types, every layer: `ffn_gate_exps` and `ffn_up_exps` `IQ4_XS`, `ffn_down_exps` `Q4_K`.
Shared expert `Q8_0`. Router `ffn_gate_inp` and the shared-expert scalar gate `F32`. Embedding and head
`Q8_0`; final norm `F32`.

The reference is `src/models/qwen35moe.cpp` and the Qwen3.5 conversion code in the llama.cpp revision this
repository pins (`third_party/ggml/VERSION.txt`). The guard's shapes are exactly the `create_tensor` shapes
that file builds.

## Why a separate backend

Qwen4Exp (Qwen3.8-Flash-Next) and Qwen35MoE disagree on almost everything structural: four-stream
hyper-connection residual versus an ordinary residual; QSA sparse selection with an indexer versus dense
attention; 512 experts top-10 and 2560 hidden versus 256 experts top-8 and 2048 hidden; a 2:1 attention
split versus 3:1 with GDN; PLE/ngram state versus none. Folding both into one `ModelGeometry` would make
every kernel ambiguous and every field a possible wrong answer.

So the Qwen4Exp code is left as it is, and Qwen35MoE is introduced beside it:

- `include/strata/core/model_kind.hpp` - `ModelKind` (`Qwen4Exp`, `Qwen35Moe`, `Unknown`).
- `include/strata/core/qwen35.hpp` + `src/core/qwen35.cpp` - `Qwen35Geometry`, `detect_model_kind`,
  `qwen35_geometry`, `check_qwen35_tensors`, `check_qwen35_all`.

`check_qwen35_all` validates, at load time, the architecture plus every dimension the kernels depend on,
and the required tensor set per layer family. It refuses, naming the tensor, its shape/type and the
required shape/type:

- a dense attention projection on a recurrent layer, or a GDN tensor on a full-attention layer;
- a Qwen4Exp-only hyper-connection (`hc_*`), QSA indexer (`indexer.*`) or PLE (`ple_*`) tensor under a
  Qwen35 block;
- a norm or router in a type the kernels do not read as `F32`;
- a weight whose contiguous dimension is not a whole number of its format's blocks;
- inconsistent metadata (`value_length != key_length`, a non-dividing interval, non-positive geometry).

`qwen35_layout_test` exercises all of these on a synthetic header and accepts a real artifact header when
given a path. `strata-qwen35-check` is the same binary installed in the runtime image for `run3.sh
--check-only`.

## Routed expert execution

The grouped native expert HIP path gained Q4_K as a DOWN format (`src/kernels/cuda/iq_kernels.cu`,
`STRATA_D_FMTS`). Q4_K was already a gate/up format (Unsloth UD-Q4_K_XL); Ornith is the first checkpoint
whose routed experts use Q4_K down. The down kernel goes through the same `Fmt<12>::dot` the gate/up path
uses, so this is an admission change, not a new kernel - and it is admitted only because the parity test
says so, at the real 2048/512 geometry, against ggml's float reference and ggml-cpu:

```
synthetic iq4_xs /q4_K  cpu rel 2.04e-02  gpu rel 1.24e-02  cpu-gpu 2.28e-02  ok
q4_K down rows, AVX2 multi-token vs ggml vec_dot: 0 of 36 token-sets differ in any bit
gpu decode-once vs per-entry kernels: bitwise equal
```

`native_expert_parity` gained `STRATA_PARITY_H` / `STRATA_PARITY_FF` so the same harness can check another
model's expert geometry, and the CMake tests are now registered for HIP as well as CUDA.

## `run3.sh`

```
./run3.sh                     # Ornith AD-Q4_K-IQ4_XS, http://127.0.0.1:9931
./run3.sh --no-mtp            # target-only, spec 0
./run3.sh --mtp /path/draft.gguf
./run3.sh --model-file /path/Ornith-....gguf
./run3.sh --check-only        # GPU + artifact + geometry validation, then exit
./run3.sh --dry-run
./run3.sh --max-context 131072      # default; 262144 also accepted
./run3.sh --spec 4 --expert-cache auto --pool-workers 10
./run3.sh --env STRATA_PROMPT_CACHE=0        # no conversation cache (a full read per request)
```

It uses the same 10 GiB VRAM contract as `run.sh`/`run2.sh`, a distinct container name
(`strata-ornith-gfx1101`), and a distinct work tree (`/work/packs/ornith-ad-q4-iq4-xs`,
`/work/mtp/ornith-qwen36`, `/work/logs/ornith`). The main GGUF stays in the shared HF cache. It never
touches the Qwen3.8 pack or MTP directories, and it does not modify `run.sh` or `run2.sh`.

The container entrypoint is `docker/entrypoint-ornith.sh`. It resolves the single GGUF and the external MTP
through `docker/hfmodel.py` (generalized so a family describes one or more files), validates the artifact
with `strata-qwen35-check`, builds the engine config and starts the server.

The engine advertises `qwen35moe mtp greedy-spec cancellation cpu gpu` through
`strata-qwen35 --capabilities`. The launcher checks this before preparation. The container validates the
artifact, writes the server config, checks the memory plan, then starts the same API and web app.

`--prefill` now reaches the engine. The default is 8; 1 uses single-token prompt processing, and larger
requests are split into native tiles of at most 8 tokens. The MMVQ projections share weights across the
tile, routed expert entries are grouped by expert on CPU and GPU, and GDN updates and attention masks
stay causal. Verification snapshots are assembled per layer after each token, so every accepted prefix
can be restored without replaying rejected tokens.

The host waits for the router/activation copy event while the shared expert continues on the GPU. Expert
results and routing plans use pinned host storage. Attention streams eight contiguous value columns
through a bounded shared-memory tile, retaining the original dot-product order and f16 accumulation.
The entire prompt's routing counts choose the expert cache placement before decode; it stays fixed
through speculative verification. Graph capture is available with `STRATA_QWEN35_GRAPHS=1`, but disabled
by default because its measured run did not improve throughput.

## Measured on gfx1101

### Latest implementation measurement

Evidence: `bench/results/2026-10-02-ornith-optimization/`. RX 7700 XT/gfx1101, Ryzen 9 7900X,
128 GB RAM, ROCm 7.2.1; main HF revision `7aa8fc1d9b861d797880f4a341166d4bb3439f74`, trained
Q8_0 draft revision `2e9f6f487b82200f2a045e67d01a3f1e2982b00a`. One warmup excluded, then three
64-token code prompts, context capacity 131072, f16 KV, automatic expert cache, desktop GPU
shared with the OS. Each row has one excluded warmup and three timed requests; the benchmark rejects
any change in greedy tokens between identical sequential requests.

| setting | outputs | workers | median decode tok/s | range | prompt processing |
|---|---|---|---|---|---|
| target-only, shipping defaults | 128 | 8 | **52.01** | **51.87-52.02** | 65.2 tok/s |
| target-only, longer decode | 256 | 8 | **50.64** | **50.26-50.88** | 62.7 tok/s |
| target-only, 1024-token prompt | 128 | 8 | 50.56 | 49.88-51.23 | 57.5 tok/s |
| trained MTP, same 8 workers, spec 1 | 128 | 8 | 49.03 | 47.35-49.24 | 61.3 tok/s |
| target-only, auto workers | 128 | 11 | 51.24 | 50.99-52.01 | 64.1 tok/s |
| trained MTP, spec 1 | 128 | 11 | 49.97 | 48.84-50.31 | 61.4 tok/s |

An 8192-token repeated-code diagnostic with 64 outputs measures 40.77 tok/s decode and 50.31 tok/s
prefill (one run, no excluded warmup). The 50-52 tok/s result applies to the shorter measured prompts,
not all context lengths.

Prompt timing includes cache placement/refill before decode; warming does not omit that work.
Target-only uses 2472 expert slots and peaks at 34.74 GiB process RSS. The HIP allocation guard reports
8.50 GiB explicit allocations; sampled total card VRAM, including desktop/runtime allocations, peaks at
9.40 GiB in the shipping runs. MTP spec 1 uses 1855 slots and 36.90 GiB RSS. Its warm draft acceptance is
59/69 (85.5%) on each request. Checkpoint storage scales with the configured draft length.

Earlier tuning folders in the evidence directory are provisional: a cache placement inherited from
previous requests changed CPU/GPU arithmetic and greedy tokens between repeats. The shipping runs use
fixed CPU placement during prefill, collect routing counts across the whole prompt, then select and
freeze GPU cache placement for decode. Identical requests now reproduce identical output. Earlier
throughput numbers are excluded from the shipping claim.

The causal batch test compares the full vocabulary, hidden vectors and subsequent state bitwise for
2/3/5/8 tokens, both KV formats, with cache 0/256. All accepted verification prefixes 0..5 match the
single-token path. The attention memory test compares every output bit against the original value
reader at 13 lengths from 1 through 65537, both KV formats. Independent upstream HIP primitives and
full-model reference comparisons pass. Longer-context capacity and filled-context quality remain
distinct checks; the latest throughput row does not establish a 128K needle test.

The table below records the earlier implementation, before causal batching and prompt cache placement.

### Throughput

All rows: RX 7700 XT (gfx1101, 12 272 MiB), 10 240 MiB Strata budget, Ryzen 9 7900X (24 threads),
`tools/qwen35_bench.py` against the built `strata-qwen35`, the 64-token code prompt from
`tools/qwen35_parity.py` (regenerated from the tokenizer, so the prompts are the same ones the parity runs
used), 128 tokens generated, median of 3 runs with the range in brackets, `--spec 0` unless stated, warm
page cache.  `auto` is the cache size the engine itself derived from the bytes free at that moment.

| tier | KV | expert cache | workers | context | decode tok/s |
|---|---|---|---|---|---|
| cpu | - | - | 12 | 131072 | 9.08 (8.98-9.14) |
| gpu | f32 | 0 | 12 | 131072 | 25.97 (25.48-26.17) |
| gpu | f16 | 0 | 12 | 131072 | 26.46 (26.36-26.99) |
| gpu | f32 | auto | 12 | 131072 | 29.28 (28.90-29.38), measured 2026-10-02 |
| gpu | f16 | auto = 2477 | 12 | 131072 | 37.80 (36.65-38.54) |
| gpu | f16 | auto = 2477 | 8 | 131072 | 40.42 (40.42-40.57) |
| gpu | f16 | auto = 2477 | 10 | 131072 | 40.68 (40.60-40.68) |
| gpu | f16 | auto = 2477 | 11 | 131072 | **41.10** (41.04-41.15) |
| gpu | f16 | 2048 fixed | 12 | 131072 | 30.46 (29.19-36.02) |
| gpu | f16 | auto = 2627 | 11 | 8192 | **46.48** (46.35-46.73) |
| gpu | f16 | auto = 2477 | 11 | 32768 | 38.63 (38.63-40.55) |
| gpu | f16 | auto = 1745 + MTP | 12 | 131072 | spec 2: 24.09 (23.51-24.65), spec 3: 20.42, spec 4: 17.12 |

What the numbers say:

- **The KV format is the biggest single lever**, because on a 12 GB card the KV's bytes are cache slots:
  f16 KV at context 131072 leaves room for 2488 expert slots where f32 leaves 913, and the same run goes
  from 26.0 to 41.1 tok/s.  f16 and f32 both have reference parity (`logits relative_rms=0`, top-10 overlap
  10/10, `top1 2/2` on the code prompt at context 8192), and the launcher offers only these two.
- **The cache pays, and `auto` beats a fixed number.** 2477 slots measured 43% hits (87159 of 202880 expert
  positions in a server run).  An explicit 2048 in the same session measured *fewer* hits (42%) and 30.5
  tok/s; that ordering is not explained by the per-layer slot split and is recorded here as unexplained -
  `auto` is the default, and no fixed default is offered until it is understood.
- **Worker count is flat between 8 and 11 and worse at 12** (41.1 vs 37.8): the pool's workers spin, so the
  engine's own thread needs a core.  The engine's auto setting picks 11 here, which is why `--pool-workers`
  is left at 0 by the launcher.
- **Context cost less than the run-to-run spread in this session** (41.1 at 131072 vs 38.6 at 32768 vs 46.5
  at 8192): the desktop's own VRAM use moves the cache size between runs, so short-context superiority is
  real in direction but not clean in size.  The f32-to-f16 gain above is far outside that noise.
- **Speculation is a loss today even though the draft is good** (72% acceptance: 75 of 104 proposed tokens
  accepted on the code prompt, in the range public Qwen3.6-MTP testing reports).  Verifying a draft token
  costs a whole trunk pass, and the CPU expert work does not amortise over the tokens being verified, so
  `--spec 2` costs ~3 single-token passes for 1.72 tokens out.  The fix is the multi-token pass below, not
  a different draft.
- Run-to-run spread across containers in this session was up to ±10% (the card is shared with a desktop);
  the medians in brackets are within-run.

### Earlier end-to-end server measurement

`./run3.sh --detach` loads the 20 GB `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf` and `/v1/models` reports
`ornith-1.5-35b-a3b-ad-q4-iq4 loaded ctx=131072`.  A 128-token chat completion on a 23-token prompt took
3.86 s and 3.80 s - **33.1 and 33.6 tok/s wall**, prompt processing included; the same model with the same
tier measured 41.1 tok/s decode-only in the bench, so ~20% of the wall time is prompt processing and server
overhead.  A 204-token prompt took 8.6 s wall, the same 128 tokens: prompt processing is one token at a
time at 45-55 tok/s, which is what makes long prompts feel slow even when decode is fast.

### Earlier allocation report

`strata-qwen35 --fit-report --gpu` at context 131072 with the 10 240 MiB budget (8.750 GiB admissible after
the 1024 MiB runtime reserve and 256 MiB slack), expert blob 1.62 MiB, 10 240 expert pairs:

| configuration | dense | GDN state | KV | fixed | slots that fit |
|---|---|---|---|---|---|
| f32 KV, no draft | 1.980 GiB | 0.061 GiB | 5.000 GiB | 7.051 GiB | 913 |
| f16 KV, no draft | 1.980 GiB | 0.061 GiB | 2.500 GiB | 4.551 GiB | 2488 |
| f16 KV, context 262144, no draft | 1.980 GiB | 0.061 GiB | 5.000 GiB | 7.059 GiB | 908 |
| f16 KV + trained MTP draft, context 131072 | - | - | - | 5.357 GiB | 1980 |

These rows precede causal batching and the checkpoint allocation correction. The current fit report
includes the full recurrent-state snapshots and sizes their count as spec + 2; the shipping spec-1
allocation uses three snapshots. Use `./run3.sh --check-only --spec 1` for the current report.

In the older report the trained draft was counted as 0.806 GiB over the f16 row above.  The session's own report agrees with the plan bit for bit (`planned 8.482 GiB,
allocated 8.482 GiB`), which is what lets the launcher refuse an impossible combination before the load
instead of after it: `--expert-cache 4096` at f16/131072 is refused with "4096 expert slots need 6.500 GiB
of the 4.199 GiB free after the fixed 4.551 GiB", where the session previously died 40 s into the load.  The
predicted slot count is conservative against what the session actually took (2488 predicted, 2477 taken).
With f16 the 262144 context fits the same 10 GiB budget with room for 908 expert slots; f32 at 262144 is
refused - "the dense weights, state and KV alone need 12.059 GiB and 8.750 GiB is free" - so run3.sh takes
262144 as a request and lets the plan say yes or no rather than promising it unconditionally.

### Correctness

- Full HIP CTest: 70 pass, 3 skip, 3 failures across 76 registered tests. The three failures
  (`ple_parity`, `expert_parity`, `pool_test`) require absent local PLE/expert-pack fixtures and also
  failed before this work. The real-model batch test skips without a model path; its separate artifact
  run passes, including both KV formats, cache 0/256, every rollback prefix, repeated requests and cancellation.
- Independent pinned llama.cpp HIP primitive comparison passes. Native IQ4_XS gate/up and Q4_K down
  have zero difference; the independent attention comparison has at most 4.2e-7 absolute difference.
- Independent full-model f32/f16 comparisons over 40 code tokens have zero logits/residual difference.
  Earlier CPU full-model fixtures through 4096 prompt tokens also have zero logits difference.
- The attention reader matches the old kernel bitwise at 13 lengths through 65537 for both KV formats.
- Trained MTP spec 0..4 and forced rejection prefixes 0..4 produce the same 64-token greedy output.
- Docker artifact/launcher tests: 33 pass. Setup choices: 15 pass. AMD setup: 18 pass.
- The rebuilt Qwen3.8 image starts through unchanged `run.sh` and returns a chat completion.

## Remaining work

- Larger MMQ prefill tiles beyond the native eight-token path, and long-context prefill optimization. The
  prompt cache removes the *repeat* cost of a long prompt, but a cold turn still reads its whole prompt at
  ~50 tok/s.
- Filled 128K/262K needle and quality runs. Capacity 131072 fits and runs in the shipping configuration;
  262144 is an explicit request subject to the memory fit gate, not a completed long-context quality claim.
- Broader code, repository, JSON and prose reference sweeps at large filled contexts, and KV-quality sweeps.
- More efficient MTP verification/draft execution: speculation is correct but still loses to target-only
  on the measured coding prompt. Default speculation remains off.
- DFlash is out of scope. The CPU tier still uses the HIP runtime image and requires a visible card.

## Building and testing

```
./build.sh                    # compile the engine in the HIP builder container and package the image
./build.sh --tests            # ... and run the HIP ctest set on the GPU
python -m unittest discover -s docker -p 'test_*.py'      # launcher + fetch contracts, no GPU needed
python tools/ornith_inspect.py --repo AtomicChat/Ornith-1.5-35B-A3B-GGUF \
       --file Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf \
       --out docs/ornith/gguf-layout.txt --header-out /tmp/ornith-header.gguf
strata-qwen35 --check --model <the GGUF>          # inside the runtime image; --fit-report --gpu adds the plan
```

The five `qwen35_*` ctest cases (geometry guard, GDN math, layer math, pool geometry, GPU batch) pass on
gfx1101 with this build; `run3.sh`'s own contract lives with the other runtime contracts in
`docker/test_runtime_contract.py`.
