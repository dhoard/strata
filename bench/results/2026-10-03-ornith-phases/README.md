# Ornith phase measurements

This directory keeps the measurement scripts, prompts, metadata, numerical results and text reports.
The model and MTP revisions used by the phase runner are pinned in `run.sh`.

Reference programs also write raw float32 hidden states, residuals and logits (`*.f32`). Those dumps
are generated artifacts: together with the extracted `tokenizer/` they occupied about 2.82 GiB in the
original measurement commit, and nine individual dumps exceeded 100 MiB. They are ignored by Git.
Existing local copies can still be used by the parity tools. The recorded JSON results remain tracked.

To regenerate the tokenizer and full-model parity dumps from the pinned models in the local Hugging
Face cache, run from the repository root on the gfx1101 measurement machine:

```sh
./build.sh --ornith-reference
bash bench/results/2026-10-03-ornith-phases/run.sh parity
```

The runner extracts the tokenizer from the main GGUF before running parity. The Python measurement
tools under `tools/` write their own reference dumps when rerun; `tools/qwen35_tune.py --oracle` expects
an already generated oracle directory. Raw dumps are local inputs, not files supplied by a fresh clone.

Other phase commands and their hardware-specific settings are documented in `run.sh` and
`docs/ORNITH_QWEN35MOE.md`. Regenerating measurements overwrites the corresponding local results;
copy results you want to preserve before running them again.
