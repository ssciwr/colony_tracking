Status: ready-for-agent
Blocked by: 02

# 07: `finetune` track — prepare / train / evaluate

Goal: the finetuning track as public functions with the data conversion kept
in a **private module**, reusing io helpers (no duplicated io logic).
`train`/`evaluate` are ported faithfully from the prototype; only the
conversion is covered by hermetic tests (no ground-truth labels, GPU, or
network exist on this machine — spec "Out of Scope").

Spec: `.scratch/colony-tracking/spec.md` — "Options and CLI contract →
train prepare | train | evaluate" (hyperparameter defaults), "Finetuning
track" (all bullets), user stories 32–35. Reference implementations
(READ-ONLY): `/home/kriedmiller/projects/colony_size/prototype/
prepare_finetune_data.py`, `train_sam.py`, `evaluate_sam.py`, plus
`README_finetune.md` for context.

## Build — `src/colony_tracking/finetune/` (package)

- `finetune/__init__.py` — the public surface:
  - `PrepareOptions` — raw frame dir + ground-truth label dir, out dir,
    `val_fraction: float = 0.2`, `max_val_wells: int = 3`,
    `min_instances: int = 1`, `n_workers: int = 8`, optional explicit
    val-well list.
  - `TrainOptions` — model `vit_t`, `Minimal` configuration, samples per
    image 3 / val 2, objects per batch 2, patch 512, 100 epochs, early
    stopping 15, lr 1e-5, seed 42; out dir (checkpoints + TensorBoard logs
    under it).
  - `EvaluateOptions` — 512 px tiles, checkpoint, split (held-out val),
    optional JSON output path.
  - `run_prepare / run_train / run_evaluate` each returning a summary
    dataclass (counts: pairs written / wells split / smoke-test status /
    mean+median IoU, IoU@0.5/0.75/0.9, clipped count, ...).
- `finetune/_convert.py` — **private** data conversion (not part of the
  public API; not re-exported from the package root): pairs raw frames with
  ground-truth labels by file stem; drops labels with < `min_instances` (1)
  instances (an all-zero label would abort the micro-sam sampler); stretches
  raw to uint8 via the **shared io 1/99 helper**; **splits by well, never by
  file** (neighbouring timepoints correlate r > 0.99 — a file split leaks):
  greedy instance-richest wells until `val_fraction`, capped by
  `max_val_wells`, or an explicit val-well list; per-pair h5 (`raw` uint8,
  `label` int32, gzip 4) under `<out>/{train,val}`.
- **train**: micro-sam's `train_sam` on the h5 split; a **smoke test** (one
  batch, shapes printed via the logger) runs before training; the prototype's
  default model is `vit_t` with the `Minimal` configuration (the laptop GPU
  OOMs on `vit_b`). Checkpoints + TensorBoard logs under the out directory.
  micro-sam / torch_em imported lazily inside the function only — importing
  `colony_tracking.finetune` must not import torch.
- **evaluate**: held-out split; each ground-truth instance prompts the model
  with its own bounding box (scores the mask decoder, not the prompt
  generator); per-instance IoU in the best containing 512 px tile (instances
  larger than a tile flagged `clipped`); handle the micro-sam `[y0,x0,y1,x1]`
  vs SAM `[x0,y0,x1,y1]` box-order swap; greyscale replicated to 3 channels;
  report mean/median IoU, IoU@0.5/0.75/0.9, optional JSON file.
- Log via the package logger; never print.

## Tests — `tests/test_finetune.py` (hermetic: conversion only)

Build a tiny synthetic labeled mini-plate in tmp_path (a few wells, a few
frames, planted disk labels; reuse/extend the conftest builders):

- well-based split: no well crosses the split; `val_fraction` respected
  (greedy instance-richest); the `max_val_wells` cap; an explicit val-well
  list is honoured.
- low-instance labels dropped (< `min_instances`); a well whose labels are all
  dropped does not crash the run (logged).
- h5 contents: dtypes (raw uint8, label int32), shapes match the source
  frames, values match the io stretch output, gzip compression present.
- import check: `import colony_tracking.finetune` does not import torch /
  micro_sam / torch_em (assert via `sys.modules`).
- `run_prepare` summary counters correct.

`train` / `evaluate`: no automated tests (no GPU/network). They must at least
parse options, validate inputs, and fail with a clear logged error when
prerequisites (h5 split / checkpoint) are missing.

## Acceptance

- `uv run pytest` passes.
- Public API: `PrepareOptions`, `TrainOptions`, `EvaluateOptions`,
  `run_prepare`, `run_train`, `run_evaluate` + summaries re-exported from the
  package root; `_convert` NOT exported.
- No duplicated io logic; no hardcoded data paths.
