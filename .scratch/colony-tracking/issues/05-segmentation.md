Status: resolved
Blocked by: 02, 04

# 05: `segmentation` stage — growth-mask gating, drift gate, re-ID

Goal: the segment stage as a public API: model-based instance segmentation
gated by the growth mask (micro-sam, lazy import), the drift gate, output
writing, resume, and the model-free re-ID mode that re-runs the shared
tracking logic.

Spec: `.scratch/colony-tracking/spec.md` — "Options and CLI contract →
SegmentationOptions", "Segment stage (growth-mask gating — ADR 0002)" (all
bullets), "I/O contract" (label images, colonies.csv, done marker, scale.txt),
user stories 10–18, 35, 45. Reference implementation:
`/home/kriedmiller/projects/colony_size/prototype/segment_growth_pipeline.py`
(port faithfully; the merged `exclude_before` semantics of ticket 04 override
the prototype's two flags). Real prototype outputs to compare shapes against
(READ-ONLY): `/home/kriedmiller/projects/colony_size/prototype/data/
seg_growth_diff_LO005/` (per-well label TIFFs, `colonies.csv`, `_done.txt`,
`ref__labels.tif`).

## Build — `src/colony_tracking/segmentation.py`

- `SegmentationOptions` — fields EXACTLY as the spec (defaults from the
  spec): `input: "raw"|"diff" = "raw"`, `raw_dir: Path | None` (required
  when input="raw"), `diff_dir: Path | None` (required when input="diff"),
  `out_dir: Path`, `wells: list[str] | None = None`, `model_type: str =
  "vit_b_lm"`, `checkpoint: Path | None = None`, `fg_ref: float = 0.5`,
  `fg_early: float = 0.3`, `min_size: int = 200`, `min_solidity: float =
  0.7`, `mask_margin: int = 15`, `min_overlap: float = 0.5`,
  `max_fg_ratio: float = 1.5` (0 disables the drift gate),
  `exclude_before: int | None = None`, `merge_touching: bool = True`,
  `redo: bool = False`, `reid_only: bool = False`. Validation error
  (usage error): missing `raw_dir`/`diff_dir` for the chosen `input`.
- `run_segmentation(options) -> SegmentationSummary` (per-well
  processed/skipped/failed + skipped-frame/drift counts).
- **micro-sam is imported lazily inside the model path only** (module import
  of `segmentation` must not import torch/micro_sam — `difference`/
  `analyze`/`train` must not pay that cost). `checkpoint` (finetuned decoder)
  replaces the pretrained model when given.
- **Input**: raw → each frame 1/99 percentile-stretched to uint8 (shared io
  helper); diff → frames fed as written (128 = no change); the reference
  frame and display limit come from the well's `scale.txt`; **a difference
  reference equal to the final frame is rejected up front** (clear logged
  error — the growth mask would be empty).
- **Growth mask from the final frame**: instance segmentation at decoder
  foreground ≥ `fg_ref` (0.5); drop instances with area < `min_size` (200) or
  solidity < `min_solidity` (0.7) (well-ring junk); survivors' union →
  connected components = **groups**; groups dilated by `mask_margin` (15 px)
  = the **growth mask**. `ref__labels.tif` holds the group labels of the
  reference frame.
- **Earlier frames**: foreground ≥ `fg_early` (0.3); keep an instance only if
  ≥ `min_overlap` (0.5) of its pixels lie inside the growth mask **and**
  inside a single group.
- **Drift gate**: skip a frame when its elevated-foreground fraction exceeds
  `max(max_fg_ratio × reference-frame fraction, 0.05)` (port the prototype's
  exact formulation of "elevated foreground"); `max_fg_ratio = 0` disables.
  Skipped frames still get output files (blank labels) to stay 1:1.
- **Tracking** via the `tracking` module (pre-merge touching, continue /
  absorb-merge / fork / new ID, `exclude_before` blanking + pre-existing
  flagging, ancestors).
- **Outputs per well** (per-well subdirectories under `out_dir`):
  - `<frame-stem>__labels.tif` per input frame, 1:1 alignment, uint32
    (0 = background, colony IDs); frames skipped by the gate or blanked by
    `exclude_before` are all-zero;
  - `ref__labels.tif`: group labels of the reference frame;
  - `colonies.csv` — canonical columns (`frame`, NOT `lo`);
  - `_done.txt` — the seven canonical keys.
  - resume: well with a complete `_done.txt` is skipped unless `redo`.
- **Re-ID** (`reid_only = True`): no model, no GPU — segments re-derived from
  the saved label images (each label value re-split into connected
  components), tracking re-run (from the first non-blank frame), labels /
  CSV / marker rewritten.
- Log per well, per frame via the package logger; never print.

## Structure for testability

Keep the model call behind one small internal function (the only place
micro-sam is touched) and expose the pure parts as testable helpers:
drift-gate decision (given foreground fractions + reference fraction),
growth-mask construction (given a list of instance masks → groups + mask),
mask-overlap filter (given instances + mask + groups). Tests never call the
model path.

## Tests — `tests/test_segmentation.py`

No model, no GPU, no network:

- drift-gate frame selection on synthetic foreground fractions (gate on/off,
  ratio 0 disables, floor 0.05 behaviour).
- growth-mask construction: a synthetic final-frame segmentation (planted
  instance masks, including one tiny / non-solid junk instance) → correct
  group count, junk dropped, dilated margin.
- mask-overlap filter: an instance straddling the mask edge keeps/discards
  per the 0.5 rule; an instance touching two groups is dropped.
- rejection of a difference reference equal to the final frame (clear
  logged error, no outputs).
- **re-ID end-to-end on a synthetic label tree** (build per-well label TIFFs
  + `colonies.csv` in tmp_path with the conftest label-series helper; use
  small images here): re-tracking continues IDs, records merges in
  `merged_with`, applies `exclude_before` blanking + pre-existing flagging
  (including a well whose series lacks frame N), rewrites labels/CSV/
  `_done.txt` with correct counters; outputs stay 1:1 with input frames.
- import check: `import colony_tracking.segmentation` does not import
  micro_sam/torch (assert via `sys.modules`).

## Acceptance

- `uv run pytest` passes.
- Public API: `SegmentationOptions`, `run_segmentation`,
  `SegmentationSummary` re-exported from the package root.
- Module docstring carries the algorithm detail (growth-mask gating, drift
  gate, re-ID) per the spec.

## Comments

- 2026-10-12: Resolved by branch t05-segmentation (merge commit 64a79ded312fa5af8a3bee3cd8abfaf1f9231ce5); 103 passed; API: run_segmentation(SegmentationOptions) -> SegmentationSummary(wells, processed, skipped, failed, failed_wells, gate_skipped_frames); up-front checks before model load; micro-sam imported lazily in the model path only. Deviations: final frame segmented once (reused), rejected wells leave no directory, missing requested wells raise ValueError, colonies.csv writes 1-based frame number.
