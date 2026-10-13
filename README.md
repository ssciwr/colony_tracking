# colony_tracking

Measure colony growth on 96-well agar plates. The package takes a plate of
raw microscope frames (2048×2048 uint16 TIFFs, hourly acquisition), finds
and tracks the colonies in every well, and writes per-colony growth
measurements, pooled aggregates, and plots.

Three stages make up the standard chain:

1. **difference** subtracts a reference frame from every other frame of a
   well and stores the result as uint8 preview frames (128 = no change,
   ADR 0001).
2. **segment** runs micro-sam instance segmentation over the difference
   frames, gates it with a growth mask built from the final frame, and
   tracks colonies across frames by IoU (ADR 0002).
3. **analyze** fits the well circles, excludes rim-touching and bad
   colonies, and writes trajectories, growth rates, pooled aggregates, and
   plots.

The `pipeline` subcommand runs the three stages from one TOML config file.
The `train` subcommands (`prepare`, `train`, `evaluate`) are the finetuning
track: they convert raw frames plus ground-truth label images into a
well-split h5 training set, run micro-sam decoder training, and benchmark
the checkpoint against the held-out wells.

## Install

[uv](https://docs.astral.sh/uv/) manages the environment (ADR 0003).
Python ≥ 3.12 is required; development uses 3.13.

```
uv sync
```

This creates `.venv`, installs the package editable, and honours the
committed `uv.lock`. `micro-sam` and `torch` come from PyPI. Segmentation
needs a CUDA GPU; the difference, analyze, and finetune-conversion stages
do not.

The test suite is hermetic (no real data, no GPU, no network):

```
uv run pytest
```

## Usage

The `colony-tracking` console script has five subcommands. Runtime output
goes to stderr through the package logger; nothing prints. Exit codes:
0 success, 1 runtime error (a stage failure, logged with context), 2 usage
error (bad flags, invalid option combination, invalid config).

Resume behaviour is shared by the difference and segment stages: a well
with a complete done marker is skipped on a re-run unless `redo` is set.

### difference

Per-well uint8 difference frames from a raw plate.

| Option | Default | Description |
| --- | --- | --- |
| `--raw-dir` | `required` | Directory of raw plate frames (flat or per-well layout). |
| `--out-dir` | `required` | Directory to write the per-well difference trees into. |
| `--reference` | `first` | Reference frame: "first" or a 1-based frame number. |
| `--wells` | `none` | Process only these wells (repeatable, comma-separated). Default: all wells. |
| `--display-percentile` | `99.5` | Percentile of \|diff\| that sets the per-well display limit. |
| `--subsample` | `8` | Stride of the subsampled grid used to estimate the display limit. |
| `--n-workers` | `6` | Thread-pool size over wells. |
| `--dry-run` | `off` | Report wells, frames, and reference; write nothing. |

### segment

Colony segmentation and tracking of raw or difference frames. This is the
only stage that touches the GPU (micro-sam is imported lazily inside it).
`--reid-only` re-derives segments from saved label images and re-tracks
without the model.

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `raw` | Input mode: "raw" frames or "diff" difference frames. |
| `--raw-dir` | `none` | Raw plate directory (required when input = 'raw'). |
| `--diff-dir` | `none` | Difference tree directory (required when input = 'diff'). |
| `--out-dir` | `required` | Directory to write the per-well label/CSV/marker trees into. |
| `--wells` | `none` | Segment only these wells (repeatable, comma-separated). Default: all wells. |
| `--model-type` | `vit_b_lm` | micro-sam model type. |
| `--checkpoint` | `none` | Optional finetuned decoder checkpoint, used in place of the pretrained model. |
| `--fg-ref` | `0.5` | Decoder foreground threshold for the final frame. |
| `--fg-early` | `0.3` | Decoder foreground threshold for earlier frames. |
| `--min-size` | `200` | Drop final-frame instances smaller than this (px). |
| `--min-solidity` | `0.7` | Drop final-frame instances with lower solidity. |
| `--mask-margin` | `15` | Dilation (px) of the final-frame groups forming the growth mask. |
| `--min-overlap` | `0.5` | Minimum fraction of an instance's pixels that must lie inside the growth mask and inside a single group. |
| `--max-fg-ratio` | `1.5` | Drift gate: skip frames whose elevated-foreground fraction exceeds this multiple of the reference frame's (0 disables the gate). |
| `--exclude-before` | `none` | Blank frames before this frame number and flag colonies first detected at or before it as pre-existing. |
| `--no-merge-touching` | `on` | Pre-merge touching segments of one group (disable with --no-merge-touching). |
| `--redo` | `off` | Reprocess wells even when their done marker is present. |
| `--reid-only` | `off` | Re-derive segments from the saved label images and re-track without the model (no GPU). |

`--exclude-before` replaces the prototype's `--exclude_lo` and
`--drop_labels_before`, which were always applied together with the same
number.

### analyze

Growth measurements, aggregates, and plots for one or more segmentation
trees.

| Option | Default | Description |
| --- | --- | --- |
| `--seg-dir` | `required` | Segmentation trees to analyze (repeatable). |
| `--raw-dir` | `required` | Raw plate directory (well-circle fit and the difference-root probe). |
| `--out-root` | `required` | Directory to write the analysis outputs into. |
| `--wells` | `none` | Analyze only these wells (repeatable, comma-separated). Default: all wells. |
| `--rim-width-px` | `60` | Width of the rim band inside the well circle (px). |
| `--rim-overlap-max` | `0.1` | A colony is rim-excluded from the first frame in which more than this fraction of its pixels lies in the rim. |
| `--max-bad-segments` | `2` | A well with at least this many distinct bad colonies is a bad well. |
| `--keep-bad-wells` | `off` | Include bad wells in aggregates and plots. |
| `--keep-pre-existing` | `off` | Include pre-existing colonies in aggregates and plots. |
| `--raw-seg-dir` | `none` | Raw-root segmentation tree probed to flag pre-existing colonies of difference roots. |
| `--start-frames` | `1` | A colony whose first valid frame lies within this many frames of the series start is bad (from_start). |
| `--min-growth-2step` | `0.3` | A colony whose every exact 2-frame window grows by less than 1 + this factor is bad (no_growth). |
| `--refresh-geometry` | `off` | Re-fit the well circles instead of using the cached well_geometry.csv. |
| `--plots-only` | `off` | Re-render the plots from the existing trajectories CSV without image IO. |
| `--no-plots` | `off` | Skip the plots. |
| `--no-qa` | `off` | Skip the per-well QA overlays. |
| `--no-group-alpha` | `off` | Do not fade group opacity with member count. |

### pipeline

The standard chain (difference → segment on the difference tree → analyze)
from a TOML config file. The command line carries the config path and the
run-level overrides only; per-stage tuning belongs in the config file,
where every option of the tables above is valid in its stage's section
(`[difference]`, `[segmentation]`, `[analysis]`, snake_case keys).

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `required` | TOML config file: a [pipeline] section plus one optional section per stage. |
| `--raw-dir` | `none` | Directory of raw plate frames (flat or per-well layout). Overrides the config file. |
| `--out-root` | `none` | Root directory every tree of the run is written into. Overrides the config file. |
| `--reference` | `none` | Reference frame: "first" or a 1-based frame number; drives the derived tree names. Overrides the config file. |
| `--wells` | `none` | Run only these wells (repeatable, comma-separated). Default: all wells. Overrides the config file. |

A minimal config:

```toml
[pipeline]
raw_dir = "data/20260825_160101_Plate1"
out_root = "out"
reference = 5
```

Tree names derive from the out root and the reference: `difference_LO005`,
`seg_growth_diff_LO005`, `analysis/`. A stage section can override its
tree's location (`[difference].out_dir`, `[segmentation].out_dir`,
`[analysis].out_root` and `[analysis].seg_dirs`). With a numeric
reference, the segment stage's `exclude_before` defaults to the reference
frame number.

### train

The finetuning track. Three steps, each with its own flags: prepare the
data, train the decoder, evaluate the result.

#### prepare

Build the well-split h5 training set from raw frames and label images.

| Option | Default | Description |
| --- | --- | --- |
| `--raw-dir` | `required` | Directory of raw plate frames (flat or per-well layout). |
| `--label-dir` | `required` | Directory of ground-truth label images (same layout, file names matching the raw frames); the label set defines the image set. |
| `--out-dir` | `required` | Directory to write the well-split h5 set (train/ and val/) into. |
| `--val-fraction` | `0.2` | Approximate fraction of instances to hold out for validation (rounded to whole wells). |
| `--max-val-wells` | `3` | Upper bound on the number of wells moved into the validation split. |
| `--min-instances` | `1` | Drop labels with fewer than this many instances (an all-zero label would abort the micro-sam sampler). |
| `--n-workers` | `8` | Threads used for scanning/inspection. |
| `--val-wells` | `none` | Explicit wells (e.g. -C006) for validation; overrides val_fraction. |

Wells never cross the split; validation wells are held out whole.

#### train

Run micro-sam decoder training on the prepared split.

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `required` | Directory with the train/ and val/ h5 subfolders (the train prepare output). |
| `--out-dir` | `required` | Directory for the checkpoints and TensorBoard logs. |
| `--model-type` | `vit_t` | SAM variant ('vit_t' is MobileSAM and the lightest option; the laptop GPU OOMs on vit_b). |
| `--configuration` | `Minimal` | micro-sam hardware preset (Minimal, gtx1080, gtx3080, ...). |
| `--samples-per-image` | `3` | Crops drawn per training image per epoch (micro-sam defaults to 16, which makes one epoch take ~90 min on a laptop GPU). |
| `--val-samples-per-image` | `2` | Crops drawn per validation image per epoch. |
| `--n-objects-per-batch` | `2` | Objects sampled per training step (the Minimal preset asks for 4, which leaves little VRAM headroom). |
| `--patch-shape` | `512 512` | Crop (tile) size in px, (height, width). |
| `--n-epochs` | `100` | Number of training epochs. |
| `--early-stopping` | `15` | Stop after this many epochs without validation improvement (0 disables it). |
| `--lr` | `1e-05` | Learning rate. |
| `--seed` | `42` | Random seed. |

#### evaluate

Benchmark a checkpoint: per-instance IoU on the held-out split.

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `required` | Directory with the train/ and val/ h5 subfolders (the train prepare output). |
| `--split` | `val` | Split to evaluate (val / train). |
| `--checkpoint` | `none` | Finetuned checkpoint; omit to evaluate the pretrained base model. |
| `--model-type` | `vit_t` | SAM variant the checkpoint was trained with. |
| `--patch-shape` | `512 512` | Tile size in px, (height, width); instances larger than a tile are flagged clipped. |
| `--n-max-images` | `none` | Evaluate only the first N h5 images of the split (a quick smoke run). |
| `--n-max-instances` | `none` | Score only the first N instances per image. |
| `--output` | `none` | Optional path for a JSON report. |

## Python API

```python
import logging
from colony_tracking import DifferenceOptions, run_difference

logging.basicConfig(level=logging.INFO)   # all output goes through logging

opts = DifferenceOptions(raw_dir="plate", out_dir="difference_LO005", reference=5)
summary = run_difference(opts)
```

Every stage follows the same shape — an options dataclass whose defaults
mirror the CLI flags, a `run_*` function, and a small summary dataclass.
The pipeline step takes a `PipelineConfig` instead;
`read_config(path)` and `write_config(config, path)` — module-level
functions in `colony_tracking.pipeline` — round-trip it as TOML. The
finetune track exposes `PrepareOptions`/`TrainOptions`/`EvaluateOptions`
with their `run_*` functions the same way; the conversion module is
private.

## Input trees

Every stage accepts two input layouts:

- **flat** — all wells' frames in one directory; the well is parsed from
  the leading `-<A-H>\d{3}` of the file name.
- **per-well** — one subdirectory per well; the well is the directory
  name.

File-name example (the real acquisition's vocabulary):

```
-A001--PO01--LO001--CO1--SL001--PX32500--PW0050--IN0020--TM366--X014222--Y011210--Z220046--T0000000000--WE00001.tif
```

The frame number is parsed from the `LO` field (`LO001` = frame 1), the
acquisition time from the `T` field (milliseconds in this dataset). Frame
metadata comes from the file name in both layouts; stage outputs are
always per-well nested subdirectories.

## Output conventions

Tree names keep the `LO<nnn>` suffix convention. A pipeline run with
reference frame 5 writes `difference_LO005/`, `seg_growth_diff_LO005/`,
and `analysis/` under its out root.

**A difference tree** holds, per well, the uint8 `_diff_uint8.tif` frames
plus `scale.txt` — key/value lines with `reference` (the reference frame's
file name) and `display_limit_int16` (the float display limit). Readers
tolerate missing or extra lines. Frames acquired before the reference
frame come out polarity-flipped; that is intended (128 = no change, ADR
0001).

**A segmentation tree** holds, per well:

- `<frame-stem>__labels.tif` — one label image per input frame (uint32;
  0 = background; colony IDs), 1:1 with the input frames.
  `ref__labels.tif` holds the group labels of the reference frame.
- `colonies.csv` — one row per colony per frame, columns
  `well, frame, t_seconds, image, colony_id, group_id, merged_with,
  pre_existing, area_px, centroid_y, centroid_x`. `frame` is the 1-based
  frame number, `t_seconds` the raw `T` field (milliseconds here),
  `merged_with` the ';'-separated colony IDs absorbed in that frame's
  merge (empty when none).
- `_done.txt` — tab-separated counters: `frames`, `groups`, `colonies`,
  `pre_existing_colonies`, `merge_events`, `forked_segments`,
  `discarded_instances`. Presence means the well is complete.

**The analyze stage** writes, under `<out-root>/<seg-root-name>/`:

- `trajectories.csv` — one row per colony per frame: `well, frame,
  t_hours, colony_id, group_id, merged_with, ancestors, pre_existing,
  area_px, age_h, rim_frac, rim_excluded, bad, bad_reason`
- `growth_rates.csv` — one row per colony: `well, colony_id, group_id,
  first_frame, last_frame, n_frames, n_frames_valid, area_first,
  area_last, max_ancestors, rim_excluded_frame, growth_rate_per_h,
  growth_doublings_per_day, abs_growth_px_per_h, bad, bad_reason,
  pre_existing`
- `bad_segments.csv` — one row per bad colony (reason, frames, areas)
- pooled aggregates: `areas_by_ancestors.csv`, `rates_by_ancestors.csv`
- plots: `growth_curves_by_ancestors.png`, `mean_area_by_ancestors.png`,
  `growth_rates_by_ancestors.png`
- per-well QA overlays: `qa/<well>_rim.png`

`well_geometry.csv` (the fitted well circles, `well, cy, cx, R`) is a
shared cache in the analysis out root; `--refresh-geometry` re-fits it.

**Renamed columns.** The transitive self-inclusive ancestor count
(prototype `n_merged`; 1 = unmerged) is written as `ancestors` everywhere
(columns, pooled files, plot names); the frame number (prototype `lo`) is
written as `frame`. Values are unchanged. The analyze stage reads
prototype-written trees and maps the legacy names (`lo`, `n_merged`,
`max_n_merged`, `first_lo`, `last_lo`, `rim_excluded_lo`) to the canonical
ones; new writes always use the canonical names.

## Integration test

`tests/test_integration.py` runs the standard chain on the real plate and
checks it against the prototype's published outputs. It is excluded from
the default run and skips cleanly unless both prerequisites are present:

- the data root via the `COLONY_TRACKING_DATA_ROOT` environment variable —
  the directory holding the raw plate (a `*_Plate*` subdirectory) and the
  prototype's outputs (`seg_growth_diff_LO005/`, `analysis/`, ...). The
  prototype trees are read-only;
- a CUDA GPU.

Run it with:

```
uv run pytest -m integration
```

The test runs `pipeline` (reference 5 by default; set
`COLONY_TRACKING_REFERENCE` to change it) into a scratch out root under
the system temporary directory and never writes inside the data root. It
asserts the one-time-check criteria:

1. the pipeline completes (all wells done, no failures);
2. colony-count sanity against the documented numbers, within ±5% (a fresh
   PyPI torch may segment slightly differently): reference 5 → 369
   colonies / 354 merge events; reference 1 → 326 colonies / 9 forks;
3. column equality for rows present in both the new and the prototype
   trees: new `frame` == old `lo`, new `ancestors` == old `n_merged`, and
   new `max_ancestors` == old `max_n_merged`.

Expect roughly half an hour on the laptop GPU (96 wells × 19 frames of
micro-sam inference). Without the env var or GPU the test skips with a
clear message, so CI and dataless checkouts are unaffected.

## Documentation

- `GLOSSARY.md` — the binding domain vocabulary (data, colonies,
  exclusions and flags, analysis).
- `docs/adr/` — decision records: 0001 (8-bit difference frames with
  128 = no change), 0002 (growth-mask gating from the final frame), 0003
  (uv + PyPI micro-sam).

## License

MIT — see `LICENSE`.