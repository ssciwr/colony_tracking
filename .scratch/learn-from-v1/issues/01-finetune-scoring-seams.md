Status: ready-for-agent

# Adopt v1's testable finetune seams: module-level `predict_instances` and `build_loader`, ported hermetic tests

Spec: `.scratch/learn-from-v1/spec.md` (item 1). v1 reference repo:
`../colony_tracking` (absolute: `/home/kriedmiller/projects/colony_size/colony_tracking`).

## What

v2's scoring logic and dataset plumbing are untestable: `_predict_instances`
lives in `src/colony_tracking/finetune/__init__.py` (263-320) with a hard
`from micro_sam.util import get_centers_and_bounding_boxes` inside (line 271),
and `build_loader` is a closure inside `run_train` (147-168) closing over
`options`. v1 keeps both module-level with injectable seams —
`predict_instances(predictor, raw, label, patch_shape, prompt,
n_max_instances=None, boxes_fn=None)` (v1 `src/colony_tracking/finetune/evaluate.py:49-116`)
and `build_loader(paths, options, is_train)` with the torch/micro-sam imports
lazy inside (v1 `src/colony_tracking/finetune/train.py:62-85`) — and tests them
hermetically with a `FakePredictor` (v1 `tests/test_finetune.py:125-208`).
v2's `tests/test_finetune.py` covers only the conversion and the
logged-prerequisite errors; the scoring behaviour (tile choice, box-order
swap, IoU, clipped flag) and the loader have no tests.

## Fix sketch

In `src/colony_tracking/finetune/__init__.py` (keep the single module — the
spec explicitly rejects the train/evaluate split as not worth the churn):

1. Extract `_predict_instances` → module-level
   `predict_instances(predictor, raw, label, patch_shape, prompt="box",
   n_max_instances=None, boxes_fn=None)`, mirroring v1's evaluate.py:49-116:
   - `boxes_fn(label) -> (points, boxes)` injectable; only when None, lazily
     import `micro_sam.util.get_centers_and_bounding_boxes` inside the
     function (keeps `test_importing_finetune_does_not_import_torch` green);
   - `n_max_instances` slices the sorted positive instance ids (v1 semantics,
     evaluate.py:73-74);
   - `prompt` exists for seam parity with v1; anything other than `"box"`
     (i.e. `"point"`, `"box_point"`) raises `ValueError` — v2's spec scores
     the decoder with box prompts only, so rejection is the contract;
   - otherwise keep today's body (per-instance scoring in the best
     containing tile, `[y0,x0,y1,x1]` → `[x0,y0,x1,y1]` swap, IoU, `clipped`)
     behaviour-identical.
   `run_evaluate` calls `predict_instances(predictor, raw, label,
   patch_shape)` where it called `_predict_instances` (line 404).
   Make both new names importable from `colony_tracking.finetune` and add
   them to that module's `__all__`; do NOT touch the package root `__all__`
   (its exact list is pinned by `tests/test_skeleton.py:19`).
2. Extract the `build_loader` closure → module-level
   `build_loader(paths: list[Path], options: TrainOptions, is_train: bool,
   *, num_workers: int = 4)`; body = today's closure with the lazy
   `torch_em` / `micro_sam.training` imports inside it. `run_train` calls
   `build_loader(train_paths, options, is_train=True)`. The keyword-only
   `num_workers` keeps production behaviour (the closure hardcodes 4 and
   v2's `TrainOptions` has no such field) while the ported smoke test can
   pass 0 like v1's does.
3. Port v1's three scoring tests (v1 `tests/test_finetune.py:140-208`) into
   v2's `tests/test_finetune.py`: the `gt_boxes` helper, `FakePredictor`,
   `test_predict_instances_perfect` (including the tile-offset `(4, 4)` box
   assertion), and `test_predict_instances_clipped_and_capped`
   (`n_max_instances=1`, `clipped is True`). Port
   `test_predict_instances_point_prompt` as a rejection contract instead:
   `prompt="point"` and `"box_point"` must raise `ValueError`. Small local
   helpers (64x64 arrays, patch 32) — no files, no micro_sam import; the
   helpers stay local to the test file.
4. Port the loader smoke test (v1 `tests/test_finetune.py:125-135`):
   `_h5_dataset`-style tiny fixture (64x64 uint8 raw + int32 label h5 files
   with `raw`/`label` datasets — v2's existing `_write_split_h5` writes 8x8,
   too small for a 32-px patch; extend or add a variant), `TrainOptions` with
   `patch_shape=(32, 32)`, `samples_per_image=1`, `val_samples_per_image=1`,
   `build_loader(paths, options, is_train=True, num_workers=0)`, assert
   `tuple(x.shape[-2:]) == (32, 32)` and `x.is_floating_point()`. This test
   imports torch in-process — that is fine; only importing the package must
   stay model-free.
5. Update the `tests/test_finetune.py` module docstring: the scoring and
   loader seams are now hermetically tested (only the real model runs stay
   outside the suite).

## Risk

- The lazy-import subprocess test (`test_importing_finetune_does_not_import_torch`)
  must keep passing: no torch/micro_sam/torch_em import at module level.
- `prompt` must default to `"box"` so `run_evaluate`'s behaviour is unchanged.

## Tests

The new tests above plus the full suite (`uv run pytest`); the finetune
option-contract tests are untouched by this ticket.

## Acceptance criteria

- Module-level `predict_instances` and `build_loader` with the signatures
  above; `run_train`/`run_evaluate` behaviour unchanged.
- The three ported scoring tests and the loader smoke test pass hermetically.
- `uv run pytest` passes.
