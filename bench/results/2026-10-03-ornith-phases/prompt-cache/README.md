# Ornith prompt cache (conversation cache) — measurement

`docs/ORNITH_QWEN35MOE.md` records the design; this directory keeps the raw evidence.

## What was wrong

The Qwen35 server session called `reset()` for every request and re-read the whole prompt. Nothing was
reused, at ~50 prompt tokens/s. Measured on the exact shipping artifact before the change, one 2,020-token
prompt sent twice:

| request | prompt tokens | `usage.prompt_tokens_details.cached_tokens` | wall |
|---|---|---|---|
| 1 | 2020 | 0 | 34.8 s |
| 2 (identical) | 2020 | 0 | 34.7 s |

Two identical requests costing the same is the whole diagnosis.

## What the cache does

`src/qwen35/session.{hpp,cpp}`, `src/qwen35/gpu_session.cu`, wired through `generate()`:

- every request records the tokens it consumed;
- a checkpoint at the end of each prompt (with the logits it produced) — an identical repeat reads nothing;
- `--prompt-cache-slots N` periodic checkpoints spaced `prompt/slots` apart inside the prompt, so a prompt
  that diverges at the previous prompt's end still resumes near it. That divergence is not hypothetical:
  Qwen's chat template re-renders a past assistant turn without the `<think>` scaffold its generation
  prompt ended with, so the next turn's prompt always diverges at the very end of the previous one.

## After, through `./run3.sh` (gfx1101)

`reasoning_effort=none`, f16 KV, 131072 capacity, auto expert cache, one conversation, first prompt 2,020
tokens. Server `usage` and the engine's own `qwen35 timing:` lines agree:

| turn | prompt tokens | reused (cached_tokens) | wall | engine prefill |
|---|---|---|---|---|
| 1 (cold) | 2020 | 0 | 35.6 s | 34.8 s |
| 2 | 2035 | 1536 | 9.1 s | 8.7 s |
| 3 | 2050 | 1920 | 2.9 s | 2.8 s |
| 4 | 2069 | 1944 | 2.9 s | 2.8 s |
| 5 | 2085 | 2040 | 1.3 s | 1.2 s |
| 6 | 2101 | 2072 | 0.9 s | 0.9 s |
| 6 again | 2117 | 2096 | 0.8 s | 0.7 s |

`engine.log` holds the engine's `qwen35 prompt cache: count=... checkpoint=... prefix=... resume=...` lines
(`STRATA_QWEN35_CACHE_DEBUG=1`) and the matching timing lines.

Cost: one 61 MiB state copy per checkpoint, so 5 copies (1 end + 4 periodic) = 305 MiB of VRAM, which comes
out of the expert cache — 2278 slots in this session against 2472 without the cache.

## Correctness

Both are against a session built with `prompt_cache = false`, so the comparison is a resumed turn against a
full read of the same prompt:

- `qwen35_layers_test` (synthetic trunk, no GPU): first request reads in full; a resumed turn, an identical
  repeat, a longer prompt, an edited prefix, and a prompt cut at the previous prompt's end each behave as
  documented, and every resumed arm's logits and tokens are **bitwise** the full read's.
- `qwen35_batch_session_test` (the real 20 GB artifact, f32 and f16 KV, slots 0 and 256):
  `12 of 21 prompt tokens reused, logits bitwise equal` and
  `periodic prompt cache: 8 of 12 reused, logits bitwise equal`.
