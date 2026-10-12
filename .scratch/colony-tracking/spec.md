# Spec: colony_tracking — stable package for the colony-size pipeline

> **Context note (2026-10-12).** This spec was drafted when the new repo was to
> be created at `colony_size/colony_tracking`. That directory is now occupied by
> a competing implementation. **The implementation target is the sibling repo
> `colony_size/colony_tracking_2`** (this spec and its tickets live in its
> `.scratch/colony-tracking/`). Everywhere this spec refers to the new repo or
> the "`colony_tracking` directory", read `colony_tracking_2`. The prototype in
> `colony_size/prototype/` remains the frozen reference: read it, never modify
> it, and never point the package at it by default.


## Problem Statement

The colony-size measurement pipeline exists only as a pile of ad-hoc prototype
scripts in a flat, unversioned directory. It works on the 96-well plate, but it
cannot be maintained or reused: there is no git history, no package, no
dependency manifest, no tests, no entry point, no public API. Data paths are
relative to the scripts' directory and already broke once when the directory
contents moved; core functionality is entangled with experimental comparison
code; and the prototype TUI drives the scripts by shelling out, so nothing is
available as a reusable implementation. The researcher wants the same
functionality as stable software: a single Python package in a git repo called
`colony_tracking`, with the experimental parts cleaned out, a clean Python API
so that a TUI can later be built as a module *on top of the implementation*,
and no hardcoded data locations. The TUI itself is out of scope for now.

## Solution

A single Python package `colony_tracking` in a fresh git repo (a new
`colony_tracking` directory inside the existing `colony_size` directory; the
prototype — scripts, old docs, and all data — stays frozen in the sibling
`prototype/` directory, is never modified, and its location is never baked
into the package).

- The three pipeline stages become a **public Python API**: *difference*
  (per-well difference frames), *segment* (growth-masked instance segmentation
  and colony tracking), *analyze* (well circle, rim and bad-segment filtering,
  growth rates, pooled aggregates, plots). Each stage is an options dataclass
  plus a `run_*` function; all runtime output goes through the standard
  library logger. The future TUI module imports this API rather than shelling
  out.
- One CLI entry point, `colony-tracking`, with subcommands
  `difference | segment | analyze | pipeline | train`. The `pipeline`
  subcommand runs the standard chain (difference → segment on the difference
  tree → analyze) from a TOML config file, with run-level command-line
  overrides. The `train` subcommand exposes the finetuning track
  (`prepare | train | evaluate`); its data conversion is a private module.
- The package is **uv-managed** (Python ≥ 3.12, committed lock file, MIT
  license); micro-sam comes from PyPI as a hard dependency — no conda.
- Domain terminology (per the repo `GLOSSARY.md`) is used in identifiers,
  help text, and docs. On-disk outputs keep the prototype's *values*; the
  agreed renames are written to disk (`frame` for the frame number,
  `ancestors` for the transitive self-inclusive merge count), and the analyze
  stage also reads prototype-written trees (legacy column names).
- Verification: a hermetic pytest suite (no real data, GPU, or network) plus a
  **one-time manual full-plate run** in a fresh uv venv, judged by visual QA
  and documented numbers (protocol in Further Notes).

## User Stories

1. As a researcher, I want a single uv-syncable project with a committed lock file, so that I can reproduce the exact environment — including micro-sam from PyPI — on a new machine.
2. As a researcher, I want to run the package via the `colony-tracking` command, so that I stop remembering script names and flags from memory.
3. As a researcher, I want the difference stage to produce difference frames from a raw plate directory, so that every well gets uint8 previews with 128 = no change.
4. As a researcher, I want to select the reference frame as `first` or by frame number, so that the required input type is unambiguous on the command line.
5. As a researcher, I want a per-well scale file recording the reference frame and the display limit, so that previews can be converted back to counts.
6. As a researcher, I want the difference run to skip complete wells on re-run, so that an interrupted run resumes instead of restarting.
7. As a researcher, I want to restrict any stage to a subset of wells, so that I can iterate on a few wells.
8. As a researcher, I want a dry-run mode for the difference stage, so that I can check plate discovery before committing to a long run.
9. As a researcher, I want parallel workers in the difference stage, so that a full plate is processed in reasonable time.
10. As a researcher, I want the segment stage to segment raw frames or difference frames, so that I can compare the two input modes.
11. As a researcher, I want the segment stage to build the growth mask from the final frame (size/solidity filtering, groups), so that detection is confined to where colonies actually grow.
12. As a researcher, I want the drift gate to skip frames with elevated foreground, so that drift bursts around the reference frame do not create phantom colonies.
13. As a researcher, I want colony IDs tracked forward across frames with continue / merge / fork rules, so that each colony has a stable identity over time.
14. As a researcher, I want the surviving colony to keep its ID in a merge and the absorbed colonies to be recorded, so that lineage is preserved in the outputs.
15. As a researcher, I want `--exclude-before N` (blank frames before N, flag pre-existing colonies), so that everything present before a chosen frame is removed from growth counts with one flag.
16. As a researcher, I want a re-ID mode that re-applies the identity logic and the pre-existing filter without the model, so that I can change the filter without a GPU.
17. As a researcher, I want per-well label images, a colonies CSV, and a done marker, so that results are inspectable per well and resumable.
18. As a researcher, I want the segment stage to reject a difference reference equal to the final frame, so that an empty growth mask is caught up front with a clear message.
19. As a researcher, I want the analyze stage to fit the well circle from the earliest raw frames and cache it, so that rim handling is stable and re-runs are fast.
20. As a researcher, I want rim exclusion to apply monotonically from the frame a colony enters the rim, so that wall-corrupted measurements never enter the aggregates.
21. As a researcher, I want pre-existing colonies of a difference root probed against the raw root at the reference frame, so that colonies invisible to the difference reference are still flagged.
22. As a researcher, I want bad-segment flags (`from_start`, `no_growth`) and bad-well exclusion, so that debris and broken wells do not poison the pooled results.
23. As a researcher, I want per-colony growth rates (exponential and absolute) over valid frames, so that I can quantify how fast each colony grows.
24. As a researcher, I want pooled aggregates and plots grouped by ancestor count, so that I can compare growth by lineage.
25. As a researcher, I want per-well QA overlays showing the well circle, rim, and kept/excluded colonies, so that I can visually verify every exclusion.
26. As a researcher, I want a plots-only mode that re-renders from an existing trajectories CSV, so that I can iterate on plots without re-analyzing.
27. As a researcher, I want the analyze stage to read segmentation trees written by the prototype (legacy column names), so that existing runs can be re-analyzed without re-segmenting.
28. As a researcher, I want a `pipeline` command that runs difference → segment → analyze from a TOML config file, so that a standard run is one command against one file.
29. As a researcher, I want the pipeline to derive intermediate tree names from the reference frame, so that successive runs never overwrite each other and naming stays consistent.
30. As a researcher, I want to override run-level settings (data paths, reference, wells) from the command line, so that I can adjust a run without editing the config.
31. As a researcher, I want an interrupted pipeline to resume via the per-stage done markers, so that a crash does not waste a full plate of segmentation.
32. As a researcher, I want `train prepare` to build a well-split h5 training set from raw frames and ground-truth labels, so that I can finetune the decoder without leaking between correlated timepoints.
33. As a researcher, I want `train train` to run micro-sam decoder training, so that I can produce a checkpoint.
34. As a researcher, I want `train evaluate` to report per-instance IoU on the held-out split, so that I can benchmark a checkpoint.
35. As a researcher, I want the segment stage to accept a finetuned checkpoint, so that I can use a better decoder when one exists.
36. As a future TUI module, I want to import the option dataclasses with defaults and help text, so that I can render option forms and command previews.
37. As a future TUI module, I want to call the run functions in-process, so that I can drive the pipeline without shelling out to scripts.
38. As a future TUI module, I want all progress to flow through the package logger, so that I can stream run output into the UI.
39. As a future TUI module, I want config round-tripping (dict ↔ TOML), so that I can persist and restore run configurations.
40. As a future TUI module, I want each run function to return a summary, so that I can update status without parsing log text.
41. As a maintainer, I want hermetic unit tests that need no real data, GPU, or network, so that the suite runs anywhere.
42. As a maintainer, I want a marked integration test that runs the standard chain on the real plate, so that I can verify against real data when I choose.
43. As a maintainer, I want tests verifying the renamed columns carry the prototype's values (frame = old lo, ancestors = old n_merged), so that the rename cannot silently change any number.
44. As a maintainer, I want clear logged errors and non-zero exit codes for missing data or misconfiguration, so that failures are obvious.
45. As a researcher, I want both flat (all wells in one directory) and per-well input trees accepted, so that the package points at raw plates and at intermediate trees alike.
46. As a researcher, I want no hardcoded data paths anywhere in the package, so that the data can live anywhere.
47. As a researcher, I want consistent domain terminology in help text, outputs, and docs, so that the tool's language matches how I talk about the data.
48. As a maintainer, I want a glossary and short ADRs in the repo, so that terminology and hard decisions are recorded where future readers look.

## Implementation Decisions

### Repository and tooling

- Fresh git repo named `colony_tracking` inside the existing `colony_size`
  directory. No history to carry (the prototype was never versioned);
  incremental commits during implementation.
- **uv** is the environment and package manager (replaces the prototype's
  conda workflow — ADR 0003). pyproject with the hatchling backend; project
  name `colony-tracking`, version `0.1.0`, MIT license (LICENSE file),
  `requires-python >= 3.12`, pinned interpreter 3.13, `uv.lock` committed.
- Core dependencies: `numpy`, `scipy`, `scikit-image`, `tifffile`, `Pillow`,
  `matplotlib`, `micro-sam` (available on PyPI as `micro-sam`), `torch_em`,
  `h5py`, `tomli-w`.
- PEP 735 `[dependency-groups]` dev group: `pytest`.
- Console script `colony-tracking`.
- `micro-sam` is imported lazily inside the segment step only (it is a hard
  dependency, but `difference`/`analyze`/`train` must not pay the torch
  import cost).

### Package layout and public API

- Modules: **io** (input-tree discovery for both layouts, well/frame/time
  parsing from file names, the shared 1/99 percentile stretch, label/CSV/
  marker/scale I/O, legacy column reading); **difference**; **segmentation**
  (model-based); **tracking** (pure identity logic, shared by the full
  segment run and the re-ID mode); **analysis**; **pipeline** (config +
  orchestration); **finetune** (public prepare/train/evaluate, with the data
  conversion as a *private* module); **cli**.
- The package root re-exports the public API: the option dataclasses, the
  `run_*` functions, `PipelineConfig`, and the summary types.
- Each stage = an options dataclass (fields are exactly the CLI flags,
  defaults below) + `run_*(options) -> Summary`. Dataclasses expose per-field
  help text and dict/TOML round-tripping (consumed by the config file and,
  later, the TUI).
- All runtime output via stdlib `logging` under the single package logger —
  no prints — with the prototype's progress granularity (per well, per
  frame).
- No option may default to a data path; every path is a required option.
- Exit codes: 0 success, 1 runtime error (logged), 2 usage error.

### Options and CLI contract

Type shapes below come from the prototype's CLI defaults, adjusted as stated
in this spec. (Kept here because they encode the decisions more precisely
than prose can.)

`DifferenceOptions`
```
raw_dir: Path                      # required
out_dir: Path                      # required
reference: "first" | int = "first" # frame number, 1-based
wells: list[str] | None = None
display_percentile: float = 99.5
subsample: int = 8
n_workers: int = 6
dry_run: bool = False
```
(Prototype's `--flatfield`/`--sigma`, `--int16`/`--skip_uint8`, and the
`--diff_scaling` variants are **cut**.)

`SegmentationOptions`
```
input: "raw" | "diff" = "raw"
raw_dir: Path | None               # required when input = "raw"
diff_dir: Path | None              # required when input = "diff"
out_dir: Path                      # required
wells: list[str] | None = None
model_type: str = "vit_b_lm"
checkpoint: Path | None = None     # finetuned decoder
fg_ref: float = 0.5
fg_early: float = 0.3
min_size: int = 200
min_solidity: float = 0.7
mask_margin: int = 15
min_overlap: float = 0.5
max_fg_ratio: float = 1.5          # 0 disables the drift gate
exclude_before: int | None = None  # replaces prototype --exclude_lo and
                                   # --drop_labels_before (merged; see below)
merge_touching: bool = True        # flag: --no-merge-touching
redo: bool = False
reid_only: bool = False
```

`AnalysisOptions`
```
seg_dirs: list[Path]               # required, repeatable
raw_dir: Path                      # required
out_root: Path                     # required
wells: list[str] | None = None
rim_width_px: int = 60
rim_overlap_max: float = 0.1
max_bad_segments: int = 2
keep_bad_wells: bool = False
keep_pre_existing: bool = False
raw_seg_dir: Path | None = None    # raw root probed for difference roots
start_frames: int = 1
min_growth_2step: float = 0.3
refresh_geometry: bool = False
plots_only: bool = False
no_plots: bool = False
no_qa: bool = False
no_group_alpha: bool = False
```

`train prepare | train | evaluate` sub-subcommands keep the prototype's
hyperparameter defaults: prepare — `val_fraction 0.2`, `max_val_wells 3`,
`min_instances 1`, `n_workers 8`, optional explicit val-well list; train —
model `vit_t`, `Minimal` configuration, samples per image 3 / val 2, objects
per batch 2, patch 512, 100 epochs, early stopping 15, lr 1e-5, seed 42;
evaluate — 512 px tiles, optional JSON output.

CLI: kebab-case flags map 1:1 to option fields; `--wells` is repeatable and
accepts comma-separated values (well names start with a dash, so they must
stay inside the flag value).

### I/O contract

- **Input tree** — two accepted layouts: *flat* (all wells' frames in one
  directory; the well is parsed from the leading `-<A-H>\d{3}` of the file
  name) and *per-well* (one subdirectory per well; the well is the directory
  name). Frame metadata is parsed from the file name in both layouts: frame
  number from `LO<nnn>`, acquisition time from the `T` field. **All stage
  inputs accept both layouts; all stage outputs are per-well nested
  subdirectories.**
- **colonies.csv** (prototype schema, renamed — `lo` → `frame`):
  `well, frame, t_seconds, image, colony_id, group_id, merged_with,
  pre_existing, area_px, centroid_y, centroid_x`. `frame` = 1-based frame
  number; `t_seconds` = the raw `T` field (milliseconds in this dataset) or
  0; `image` = the input frame's file name; `merged_with` = ';'-separated
  colony IDs absorbed in this frame's merge (empty when none).
- **Label images**: one per input frame, 1:1 alignment,
  `<frame-stem>__labels.tif` (uint32; 0 = background; colony IDs);
  `ref__labels.tif` holds the group labels of the reference frame.
- **Done marker**: per-well `_done.txt`, tab-separated key/value lines:
  `frames, groups, colonies, pre_existing_colonies, merge_events,
  forked_segments, discarded_instances`. Presence = complete (skipped on
  re-run unless `redo`).
- **scale.txt** (difference output, per well): key/value lines; the new
  writer writes `reference` (the reference frame's file name) and
  `display_limit_int16` (float limit). Readers must tolerate missing or extra
  lines (the prototype also wrote `flatfield`/`sigma`, which are no longer
  written).
- **well_geometry.csv** (shared cache in the analysis out root):
  `well, cy, cx, R` (one decimal).
- **Analysis outputs** per segmentation root (prototype schemas, renamed):
  - `trajectories.csv`: `well, frame, t_hours, colony_id, group_id,
    merged_with, ancestors, pre_existing, area_px, age_h, rim_frac,
    rim_excluded, bad, bad_reason`
  - `growth_rates.csv`: `well, colony_id, group_id, first_frame, last_frame,
    n_frames, n_frames_valid, area_first, area_last, max_ancestors,
    rim_excluded_frame, growth_rate_per_h, growth_doublings_per_day,
    abs_growth_px_per_h, bad, bad_reason, pre_existing`
  - `bad_segments.csv`: `well, colony_id, bad_reason, first_frame, last_frame,
    n_frames_valid, area_first, area_last, n_bad_in_well, well_excluded`
  - pooled: `areas_by_ancestors.csv` (`frame, t_hours, ancestors, n_colonies,
    n_wells, mean_area_px, median_area_px`) and `rates_by_ancestors.csv`
    (`max_ancestors, n_colonies, mean_doublings_per_day,
    median_doublings_per_day, mean_abs_growth_px_per_h,
    median_abs_growth_px_per_h`)
  - plots: `growth_curves_by_ancestors.png`, `mean_area_by_ancestors.png`,
    `growth_rates_by_ancestors.png`; QA overlays: per-well
    `qa/<well>_rim.png`.
- **Renames written to disk**: the transitive self-inclusive ancestor count
  (prototype `n_merged`; 1 = unmerged) is written as **`ancestors`**
  everywhere (columns, pooled files, plot names) with values unchanged —
  verified in the one-time check. The frame number is written as
  **`frame`** (prototype `lo`).
- **Legacy reading**: the analyze stage accepts prototype-written trees —
  when reading colonies/trajectories CSVs, legacy column names
  (`lo`, `n_merged`, `max_n_merged`, `first_lo`, `last_lo`,
  `rim_excluded_lo`) are mapped to the canonical names. New writes always use
  canonical names.
- **Tree naming** keeps the `LO<nnn>` suffix convention
  (`difference_LO005`, `seg_growth_diff_LO005`), consistent with the input
  file vocabulary.

### Difference stage

- Per well: `cur − ref` in float32 counts; the per-well display limit is the
  `display_percentile` (99.5) of |diff|, estimated on the subsampled grid
  (stride `subsample`); uint8 = `128 + diff × 127 / limit`, clamped —
  **128 = no change** (ADR 0001: the 8-bit format is the internal contract
  between stages).
- Frames acquired before the reference frame come out polarity-flipped —
  documented, intended, unchanged.
- Thread pool over wells (`n_workers`); resume via the done marker;
  `dry_run` reports wells/frames/reference and writes nothing.

### Segment stage (growth-mask gating — ADR 0002)

- Raw input: each frame percentile-stretched (1/99) to uint8 via the shared
  io helper. Difference input: frames fed as written; the reference frame and
  display limit come from the well's `scale.txt`; a reference equal to the
  final frame is rejected (the growth mask would be empty).
- Final frame: instance segmentation at decoder foreground ≥ `fg_ref` (0.5);
  instances with area < `min_size` (200 px) or solidity < `min_solidity`
  (0.7) are dropped (well-ring junk); survivors' union is split into
  connected components = **groups**; groups dilated by `mask_margin` (15 px)
  = the **growth mask**.
- Earlier frames: foreground ≥ `fg_early` (0.3 — the decoder under-fires on
  faint colonies); an instance is kept only if ≥ `min_overlap` (0.5) of its
  pixels lie inside the growth mask **and** inside a single group.
- **Drift gate**: a frame is skipped when its elevated-foreground fraction
  exceeds `max(max_fg_ratio × reference-frame fraction, 0.05)` — drift bursts
  around the reference frame make the decoder hallucinate giant instances;
  `max_fg_ratio = 0` disables the gate.
- Touching segments of one group are pre-merged (3 px dilation) unless
  `merge_touching` is off — colonies that touch have grown together.
- **Tracking** (pure module, shared with re-ID): forward in time; IoU ≥ 0.3
  against a colony's last-known mask continues it; a segment covering ≥ 0.5
  of a further uncontinued colony **absorbs** it (a *merge*: the survivor
  keeps its ID, the absorbed colony is listed in `merged_with`, and an
  absorbed colony resumes its own ID if it separates again); an unmatched
  segment splitting one colony is a **fork** (the same ID continues, logged);
  otherwise it starts a new colony ID.
- **The pre-existing filter** (`--exclude-before N`) merges the prototype's
  `--exclude_lo` and `--drop_labels_before`, which the user decided must
  always be applied together with the same number:
  1. every frame with frame number **< N** gets an empty label image and no
     CSV rows (outputs stay 1:1 with the input frames);
  2. colonies whose first detection is **at or before frame N** (frame number
     ≤ N) are flagged `pre_existing=1`; rows are kept.
  Because frames before N are blanked, (2) effectively means "first detected
  at N". The rule compares frame numbers directly, so N need not exist in a
  particular well (generalizing the prototype's exact-frame lookup, which
  warned and disabled flagging when N was missing). Applies to full runs and
  to re-ID; re-ID re-tracks from the first non-blank frame.
  *This intentionally changes the prototype's raw-root behavior, where
  flagging ran without blanking — see Further Notes.*
- **Re-ID** (`reid_only`): no model, no GPU; segments are re-derived from the
  saved label images (each label value re-split into connected components),
  tracking is re-run, labels/CSV/marker rewritten.
- `checkpoint`: optional finetuned decoder in place of the pretrained model.

### Analyze stage

- **Well circle**: fitted per well from up to the three earliest raw frames
  (LO001–003, no colonies yet). Per frame: Gaussian blur (σ 6); stage 1
  samples the darkest point per angle (240 angles, radii 650–1120 px from
  the image centre) and keeps only real local dips (at least 0.08 × profile
  range below the ±60 px radial neighbourhood — illumination gradients create
  flat-dark regions whose minimum is not the ring); < 30 points ⇒ no fit; a
  coarse Kasa fit with iterative trimming (tol 30, ≥ 20 points) must land in
  the plausible radius band (700–1100 px); stage 2 re-locks every angle in a
  narrow window (R ± 70 px, dip 0.02, ≥ 40 points), trims (tol 25), and
  accepts only if the radius std < 20 px, the centre is within 200–1850 px,
  and ≥ 8 of the 12 angular sectors (30°) are covered. Frame fits whose R
  deviates > 40 px from the median are dropped; the rest are averaged.
  Cached in `well_geometry.csv` (`refresh_geometry` forces refits). A well
  with no plausible circle has rim exclusion disabled (logged).
- **Rim**: a colony is rim-excluded from the first frame in which more than
  `rim_overlap_max` (0.1) of its pixels lie within `rim_width_px` (60) of the
  well circle; exclusion is monotone (colonies only grow into the rim).
- **Time**: `t_hours` from the colonies CSV's `t_seconds` (the raw `T` field)
  via the prototype's magnitude heuristic (values > 1e6 are read as
  milliseconds).
- **Ancestors**: per colony-frame row, the number of distinct original
  colonies combined in the segment, resolved transitively through the
  `merged_with` histories, the colony itself included (an A+B where B had
  merged with C ends at 3; unmerged = 1). The prototype computed the same
  quantity as `n_merged`; values are unchanged, the name is now
  `ancestors`.
- **Bad segments**: `from_start` — first valid frame within `start_frames`
  (1) of the series start (anchored at the earliest label image); `no_growth`
  — every exact 2-frame window among the valid frames grows by less than
  `1 + min_growth_2step` (1.3).
- **Bad wells**: a well with ≥ `max_bad_segments` (2) distinct bad colonies
  is excluded from aggregates/plots (`keep_bad_wells` overrides); always
  reported in the log and `bad_segments.csv`.
- **Pre-existing colonies** (flag from the segment stage, or probed for
  difference roots) are excluded from aggregates/plots over their whole
  lifetime (`keep_pre_existing` overrides); rows stay in the CSVs.
- **Difference-root probe**: for roots named `*diff_LO<nnn>`, the reference
  frame is invisible to the root (it is differenced against itself), so each
  colony's first-detection centroid is probed against the raw root's
  (`raw_seg_dir`) label image at the reference frame: a hit at the centroid ±
  12 px, or inside an area-equivalent disk of a raw colony (+ 15 px slack),
  flags the colony pre-existing.
- **Growth rates**: over a colony's valid (not rim-excluded) frames with ≥ 2
  points and positive area: specific rate = least-squares slope of ln(area)
  vs hours (h⁻¹) and doublings/day = slope × 24 / ln 2; absolute growth =
  (last − first) valid area over elapsed time (px/h). Colonies with a single
  valid frame get NaN rates (areas stay in trajectories).
- **Aggregates and plots** use only non-bad, non-pre-existing rows from
  non-excluded wells: pooled per (frame, ancestors) area statistics; pooled
  per max-ancestors rate statistics; growth curves (individual trajectories,
  absolute and relative time, coloured by ancestor count); mean area per
  ancestor group (buckets with < 3 members not drawn; group opacity fades
  with member count unless `no_group_alpha`); rate distributions per group
  (box + strip, symlog absolute-growth panel).
- `plots_only`: re-render everything derived from the existing
  trajectories CSV with no image IO (legacy column names accepted).
- **QA overlays**: per-well 512×512 raw-frame composite — fitted well circle,
  rim boundary, colony edges coloured valid / pre-existing / rim-excluded /
  bad, with translucent fill.

### Finetuning track

- Subcommands `train prepare | train | evaluate`; the **data conversion is a
  private module** inside the finetune package (not public API) and reuses
  the io helpers — no duplication of io logic.
- **prepare**: pairs raw frames with ground-truth labels by file stem; drops
  labels with < `min_instances` (1) instances (all-zero labels abort the
  micro-sam sampler); stretches to uint8 (1/99 percentile) via the shared io
  helper; **splits by well, never by file** (neighbouring timepoints correlate
  r > 0.99 — a file split leaks): greedy instance-richest wells until
  `val_fraction` (0.2), capped by `max_val_wells` (3), or an explicit val-well
  list; per-pair h5 (raw uint8, label int32, gzip 4) under
  `<out>/{train,val}`.
- **train**: micro-sam's `train_sam` on the h5 split; a smoke test (one
  batch, shapes printed) runs before training; checkpoints and TensorBoard
  logs under the out directory. The prototype's default model is `vit_t` with
  the `Minimal` configuration (the laptop GPU OOMs on `vit_b`); the smoke
  test and all hyperparameter defaults are retained.
- **evaluate**: held-out split; each ground-truth instance prompts the model
  with its own bounding box (scores the mask decoder, not the prompt
  generator); per-instance IoU in the best containing 512 px tile (instances
  larger than a tile flagged `clipped`); the micro-sam `[y0,x0,y1,x1]` vs SAM
  `[x0,y0,x1,y1]` box-order swap is handled; greyscale replicated to 3
  channels; report: mean/median IoU, IoU@0.5/0.75/0.9, optional JSON.

### Pipeline and config

- `PipelineConfig` = run-level settings (`raw_dir`, `out_root`,
  `reference`, `wells`) + the three stage option dataclasses (each fully
  overridable per stage).
- **Derived names** (all overridable in the config): difference tree
  `<out_root>/difference_LO<nnn>` (or `<out_root>/difference` for
  `reference = "first"`); segment root `<out_root>/seg_growth_diff_LO<nnn>`
  (or `<out_root>/seg_growth_diff`); the segment stage's `exclude_before`
  defaults to the reference frame number (when the reference is numeric);
  analysis out root `<out_root>/analysis`.
- The pipeline runs the stages in order; segment consumes the difference tree
  it just produced (`input = diff`); analyze consumes the segment root. Each
  stage's own resume logic applies, so an interrupted run re-enters at the
  first incomplete stage's incomplete wells.
- **Config file**: TOML — a `[pipeline]` section plus one section per stage
  with any of its options; omitted keys take defaults. Read with stdlib
  `tomllib`; a write helper (via `tomli-w`) exists for the future TUI. The
  `pipeline` subcommand takes `--config` plus **run-level overrides only**
  (`--raw-dir`, `--out-root`, `--reference`, `--wells`); per-stage tuning
  belongs in the config file.
- The pipeline always runs the standard chain (difference → segment on
  difference input → analyze); raw-input segmentation stays available via the
  `segment` subcommand directly.

### Documentation and records

- A new, minimal README: install via uv, the five subcommands with option
  tables, the two input-tree layouts, output conventions.
- `GLOSSARY.md` at the repo root — the final agreed version (Data / Colonies
  / Exclusions-and-flags / Analysis sections; the terms *valid*,
  *trajectory*, *aggregate* and the whole model section were dropped).
- `docs/adr/` with three short ADRs: 0001 — the 8-bit difference format with
  128 = no change (int16 counts and the `abs`/`abs_percentile` presentations
  were considered and rejected); 0002 — growth-mask gating from the final
  frame, including the drift gate; 0003 — uv + PyPI `micro-sam` instead of
  the prototype's conda-forge workflow.
- Module docstrings carry the algorithm detail; nothing else is carried over
  from the prototype's docs (they may be outdated).

## Testing Decisions

### What makes a good test

Tests assert **external behaviour only**: files written (names, pixel values,
CSV contents), returned summary counters, log records, CLI exit codes. Pure
logic (parsing, ancestor counting, circle fitting, split logic) is tested as
input → output on synthetic data. No test may require real plate data, a GPU,
or network access; the model-dependent path is deliberately outside the
automated suite.

### Seams

- **One automated seam, the highest available**: the public API (option
  dataclasses, `run_*` functions, `PipelineConfig`) and the CLI, driven by
  synthetic input trees in temporary directories. This seam covers the
  difference stage end-to-end, the tracking logic through the re-ID path (no
  model), the analysis stage end-to-end, the finetune conversion, config
  round-trips, and CLI behaviour.
- The model-dependent segment call (micro-sam inference) is **not** covered
  by the automated suite (weights, GPU, network). It is validated by the
  one-time manual full-plate run (protocol in Further Notes). The pure
  tracking logic that inference feeds is fully covered via re-ID on synthetic
  label images.

### What is tested (module → behaviour)

- **io**: discovery in both input layouts; well/frame/`T`-field parsing; the
  128-neutral mapping and the percentile limit against known values;
  scale/marker/CSV writing; legacy column mapping (`lo` → `frame`,
  `n_merged` → `ancestors`, `first/last/rim_excluded` `lo` → `frame`).
- **difference**: uint8 values (128 at zero difference; a known count maps
  through the known limit); polarity flip before the reference frame;
  `scale.txt` contents; resume (second run skips complete wells); `dry_run`
  writes nothing; worker parallelism does not change results.
- **tracking** (via re-ID on synthetic label series): IoU continuation;
  absorb-merge (survivor keeps the ID, `merged_with` recorded, transitive
  ancestor count 3 in an A+B←C chain); fork (same ID continues, counted in
  the marker); new IDs; pre-merge of touching segments; `exclude_before`
  blanking + pre-existing flagging (including a well whose series lacks frame
  N); `_done.txt` counters.
- **segmentation** (non-model parts): drift-gate frame selection on synthetic
  foreground fractions; growth-mask construction (group count from a
  synthetic final-frame segmentation); rejection of reference = final frame.
- **analysis**: well-circle fit on synthetic frames with a planted meniscus
  ring (recovers the planted centre/radius within tolerance; rejects a broken
  ring); rim exclusion (a planted colony entering the rim is excluded
  monotonically from the right frame); bad flags (planted `from_start` and
  `no_growth` colonies); bad-well exclusion; the difference-root probe
  (planted pre-existing colony); transitive ancestors; growth rates (planted
  exponential colony → known slope/doublings); pooled files and plots
  (existence, non-trivial content, correct grouping); `plots_only`
  re-render from a written trajectories CSV; reading a legacy (prototype)
  tree.
- **finetune conversion**: well-based split (no well crosses the split;
  `val_fraction` respected; well cap; explicit list); low-instance labels
  dropped; h5 contents (dtypes, shapes, values match the stretch).
- **pipeline**: config parse/round-trip (TOML ↔ `PipelineConfig`); derived
  root names (numeric and `first` reference); `exclude_before` derivation;
  CLI run-level override precedence.
- **cli**: flag parsing (kebab-case mapping; repeatable `--wells` with
  dash-leading values); exit codes (missing data → 1 with a logged message;
  usage error → 2); `dry_run` output.

### Fixtures

- A synthetic mini-plate builder: 2–3 wells × 5–8 frames of 2048×2048 uint16
  (the circle-fit constants assume the real plate's size), with a planted
  dark meniscus ring (R ≈ 920 px) in the earliest frames, planted growing
  disks (a pair that touches → merge; a static disk → `no_growth`; one present
  from frame 1 → `from_start`; one approaching the rim → rim exclusion), and
  frame names carrying millisecond `T` fields; emitted in both flat and
  per-well layouts.
- Synthetic label series (integer label images, no model) for tracking tests.
- Matplotlib forced to the Agg backend in tests.

### Integration test and prior art

- A pytest-marked **integration** test (excluded from the default run)
  requires a real data root (via environment variable) and a GPU: runs the
  standard pipeline into a scratch out root and asserts the one-time-check
  criteria — colony-count sanity against the documented numbers and column
  equality (new `frame` == old `lo`, new `ancestors` == old `n_merged`) for
  rows present in both.
- **Prior art**: none — the prototype has no tests; this suite is new.

## Out of Scope

- The TUI itself (planned as a later module on top of this API).
- Conda environment management / environment doctoring (a prototype TUI
  behaviour, superseded by uv).
- The cut comparison knobs: flat-field correction, int16 difference output,
  the `abs`/`abs_percentile` difference presentations.
- End-to-end validation of the finetuning track on real labels — no
  ground-truth labels exist on this machine; the conversion is tested
  hermetically, and train/evaluate are ported faithfully.
- Migration of existing data trees (the package *reads* them via legacy
  column mapping; it never rewrites a prototype tree).
- Generalization beyond this acquisition format (96-well, 2048×2048 uint16,
  `LO`/`T` file naming; the circle-fit constants are calibrated to it).
- Multi-plate pooling; other species or microscopy modalities.
- PyPI publishing, CI, Windows support.

## Further Notes

- **`exclude_before` is an intentional behaviour change.** The prototype's two
  flags are now one, so a raw-root run also blanks frames before the cutoff
  (the prototype's raw root was flagged without blanking, keeping its early
  rows). Consequences for the one-time check: raw-root numbers from the
  prototype (260 colonies; 229 after flagging) are not directly comparable;
  the difference-root chains are the primary reference.
- **One-time verification protocol** (fresh uv venv, real plate, outputs into
  a scratch root — the prototype's trees are never touched):
  1. `pipeline` completes all 96 wells with 0 failures on the standard chain
     (reference 5).
  2. Visual QA against the prototype's published outputs: per-well rim
     overlays and the three plots, compared with the prototype's analysis
     outputs and its example images.
  3. Number sanity against the prototype's documented results: difference
     root with reference 5 → 369 colonies / 354 merge events; reference 1 →
     326 colonies / 9 forks.
  4. Column equality for rows present in both trees: new `frame` == old `lo`;
     new `ancestors` == old `n_merged` (and `max_ancestors` == old
     `max_n_merged`).
  Note: the fresh uv venv takes torch from PyPI, possibly a different version
  than the prototype's conda torch, so small pixel-level segmentation
  differences are expected and covered by visual + sanity acceptance rather
  than byte identity (explicit user decision).
- Glossary terms are binding for code identifiers, help text, and
  documentation.
- `t_seconds` keeps the raw `T` field (milliseconds in this dataset); the
  magnitude heuristic for hours is retained.
- The three ADRs listed under Documentation and records are written into the
  repo as part of the implementation.
- Commit strategy: incremental — skeleton + tooling, then io, difference,
  tracking, segmentation, analysis, finetune, cli + pipeline, tests, docs.
