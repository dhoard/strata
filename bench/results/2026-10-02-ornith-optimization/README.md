# Ornith on RX 7700 XT: shipping implementation evidence

Baseline HEAD: `8c7030b`. Final implementation commit: `94b93e8` (full SHAs in
`baseline-sha.txt` and `final-sha.txt`). This checkout already contained uncommitted Ornith work;
that work was preserved and completed. The documentation/evidence commit follows the implementation
commit. `changed-files.txt` records the implementation's 39 changed/added files.

RX 7700 XT (gfx1101, 12272 MiB), Ryzen 9 7900X (12 physical cores), 128 GiB RAM,
ROCm 7.2.1. See `hardware.txt`, `toolchain.txt`, and `runtime-image.txt`.

## Artifacts and design

Main: `AtomicChat/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf`,
HF revision `7aa8fc1d9b861d797880f4a341166d4bb3439f74`.
Trained draft: `EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF/mtpdraft-Q8_0.gguf`,
HF revision `2e9f6f487b82200f2a045e67d01a3f1e2982b00a`.
Independent upstream reference: llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`,
as pinned in `third_party/ggml/VERSION.txt`.

Qwen35MoE has its own ModelKind, geometry guard, session and executable. Its 40 layers use
30 recurrent GDN blocks and 10 dense attention blocks, width 2048, 256 routed experts top-8,
512-wide routed/shared experts. Dense weights, recurrent state, KV, shared experts and selected
native IQ4_XS/Q4_K expert pairs live on the GPU; remaining experts run in the native CPU pool.
The mmap source GGUF stays in system RAM. The Qwen3.8 launchers are unchanged.

Causal eight-token tiles share projection/expert weight reads. Router/activation copies use pinned
memory and a completion event while shared-expert work continues. The attention value reader
coalesces eight contiguous columns with bounded shared storage and preserves arithmetic order.
Prompt routing counts determine GPU expert placement before decoding; placement freezes for decode
and verification. Prefill uses fixed CPU placement so earlier requests cannot change rounding.
Trained MTP uses batched target verification and full recurrent/convolution/KV-prefix rollback;
snapshot count is spec + 2. Graph capture is optional and disabled after measuring no gain.

## Shipping measurements

Engine-only timing, greedy sampling, repeat penalty 1, context **capacity** 131072, f16 KV,
10 GiB budget. One excluded warmup, three timed requests, one engine process per configuration.
Every shipping configuration rejects any token change across identical sequential requests.
The desktop shares the GPU; figures are measurements of this machine, not guarantees.

| prompt | output | workers | spec | median decode tok/s | range | median prefill tok/s |
|---|---|---|---|---|---|---|
| 64 | 128 | 8 | 0 | 52.01 | 51.87–52.02 | 65.2 |
| 64 | 256 | 8 | 0 | 50.64 | 50.26–50.88 | 62.7 |
| 64 | 128 | 11 | 0 | 51.24 | 50.99–52.01 | 64.1 |
| 64 | 128 | 8 | 1 | 49.03 | 47.35–49.24 | 61.3 |
| 64 | 128 | 11 | 1 | 49.97 | 48.84–50.31 | 61.4 |
| 1024 | 128 | 8 | 0 | 50.56 | 49.88–51.23 | 57.5 |

Defaults: **8 workers, auto cache (2472 slots here), spec 0, f16 KV, prefill tile 8**.
Spec 1 acceptance is 59/69 = 85.5% on each warm measured request, but its overhead and smaller
expert cache make it slower than target-only. Its cache uses 1855 slots; it remains opt-in.
Target-only RSS peaks at 34.74 GiB, MTP at 36.90 GiB. Explicit target allocations are 8.482 GiB;
sampled total card use reaches 9.40 GiB in shipping measurements, including desktop/runtime.
The allocator enforces the 10240 MiB ceiling minus 1024 MiB runtime reserve and 256 MiB slack.

`shipping-128`, `shipping-256`, `shipping-mtp-8`, `shipping-1k` contain full commands, tokens,
timings, environment, sampled RAM/VRAM and engine stdout/stderr. `summary.json` provides compact
machine-readable results. Cache sizes are computed from actual available device bytes.

The 8K diagnostic is an execution/performance check, not a needle or model-quality test.

Earlier tuning folders (`baseline`, `batched`, `graphs-pinned`, `final`, `old-attention`,
`final-workers`, `final-spec-sweep`) are **provisional**. They include worker and spec-length sweeps
but predate a fix for inherited cache placement: greedy output could change between requests.
Their speed figures are excluded from the shipping claim. The final cache 0/256 tests establish
correctness; auto versus fixed-cache historical speed sweeps are not a shipping comparison.

## Validation and reproduction

Build and package:

```sh
./build.sh --ornith-reference
./build.sh --runtime-only
```

In the HIP builder with `/src` and `/models` mounted, model and draft variables below identify
the exact files above:

```sh
cmake --build /src/build-hip -j 12
ctest --test-dir /src/build-hip --output-on-failure
/src/build-hip/qwen35_batch_session_test "$ORNITH_MAIN" "$ORNITH_MTP"
/src/build-hip/qwen35_attention_memory_test
/src/build-ornith-reference/ornith_primitives
python3 /src/tools/qwen35_spec.py --model "$ORNITH_MAIN" --mtp "$ORNITH_MTP" \
  --tokens /src/bench/results/2026-10-02-ornith-optimization/prompts/code-64-tokens.json \
  --backend gpu --kv f16 --slots 256 --new 64 --out /tmp/ornith-spec
python3 /src/tools/qwen35_bench.py --model "$ORNITH_MAIN" --mtp "$ORNITH_MTP"   --tokens /src/bench/results/2026-10-02-ornith-optimization/prompts/code-64-tokens.json   --context 131072 --new 128 --configs gpu:8:auto:f16:0,gpu:8:auto:f16:1 --out /tmp/ornith-bench
```

Host checks:

```sh
python -m unittest discover -s docker -p 'test_*.py'
python tools/test_setup_choices.py
python tools/test_setup_amd.py
./run3.sh --dry-run
./run3.sh --check-only
./run3.sh --check-only --spec 1
./run3.sh --detach
./run3.sh --check
```

Full CTest: 70 pass, 3 skip, 3 fixture-dependent failures across 76 registered tests.
`ple_parity`, `expert_parity`, `pool_test` cannot find local PLE/expert-pack fixtures; these same
three failures are present in `baseline-ctest.log`. Two HIP features skip as unsupported; the
real-model batch test skips in CTest without artifact environment variables and passes separately.
Docker tests: 33 pass. Setup choices: 15 pass. AMD setup: 18 pass.

`shipping-batch-parity.log`: every full vocabulary and hidden vector is bitwise equal for
batch sizes 2/3/5/8, f32/f16 KV, cache 0/256. All accepted verification prefixes 0..5, next-token
state, repeated greedy requests, spec/no-spec with the same placement, and cancellation recovery pass.
`stable-spec-parity`: spec 0..4 plus forced rejection positions 0..4 produce identical 64-token output.
`attention-memory-parity.log`: all output bits match the original reader at 13 lengths through 65537,
both KV formats. `final-primitives.log`: independent upstream HIP primitive comparisons pass;
native IQ4_XS/Q4_K experts have zero difference, attention max absolute difference <=4.2e-7.
`upstream-parity.log` and `tiled-f16-upstream-parity.log`: independent full-model comparisons have
zero logits/residual difference over 40 code tokens. `earlier-cpu-full-model-summary.json` records
the preserved prior CPU comparisons through 4096 prompt tokens, zero logits difference.

`qwen38-api-smoke.json` and logs: rebuilt runtime launches unchanged `run.sh`, loads IQ3_XXS at
131072 capacity, and answers a chat completion. This is a correctness/startup smoke, not a Qwen3.8
performance measurement; the first request reads cold expert pages.

`ornith-api-smoke.json`: the final `run3.sh --detach` server emits a correct Python add function,
identical greedy repeated messages, completed SSE streams and identical output after stream cancellation.
Images are rejected with HTTP 400; the Anthropic endpoint returns a completion. The published port is
127.0.0.1:9931. The server was restarted and readiness rechecked, reusing prepared artifacts.

## Context limits and remaining measurements

131072 capacity fits and is exercised by shipping benchmarks and the launcher. An actual 1024-token
prompt is measured above. A single 8192-token repeated-code diagnostic (64 generated tokens, no
excluded warmup, not a three-run median) measures **40.77 tok/s decode**, **50.31 tok/s prefill**
(162.84 seconds). See `filled-8k.json` and its stdout/stderr. The 50–52 tok/s result is limited to
the shorter measured prompts; larger prompt workloads can be slower. Full-model CPU oracle parity extends through 4096 prompt tokens in earlier
fixtures; the attention operator's bitwise stress test extends through 65537 cached positions.
These are different evidence levels. No filled-32K/128K/256K throughput or needle/quality claim is made.
262144 is accepted only if the memory fit gate passes (f16 required on this card).
Larger MMQ tiles, long-context prefill optimization and broad agentic coding/JSON quality suites remain.
NVIDIA/CUDA behavior was not tested on this AMD machine. The CPU tier uses a HIP runtime image.

Final command:

```sh
./run3.sh
./run3.sh --no-mtp
```

These two commands select the same fastest setting. To compare the trained draft:

```sh
./run3.sh --spec 1
```
