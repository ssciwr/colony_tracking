# colony_tracking

Measure colony growth on 96-well agar plates. The package takes a plate of
raw microscope frames (2048×2048 uint16 TIFFs, hourly acquisition), finds
and tracks the colonies in every well, and writes per-colony growth
measurements, pooled aggregates, and plots.

Three stages make up the standard chain:

1. **difference** subtracts a reference frame from all other frames of a
   well and stores the result as uint8 preview frames (128 = no change,
   ADR 0001).
2. **segment** runs micro-sam instance segmentation over the difference
   frames, gates it with a growth mask built from the final frame, and
   tracks colonies across frames by IoU and mask containment (ADR 0002).
3. **analyze** fits the well circles, excludes rim-touching and bad
   colonies, and writes trajectories, growth rates, pooled aggregates, and
   plots.

The `pipeline` subcommand runs the three stages, configured by one TOML
config file or by command-line flags alone.
The `train` subcommands (`prepare`, `train`, `evaluate`) are the finetuning
track: they convert raw frames plus ground-truth label images into a
well-split h5 training set, run micro-sam decoder training, and benchmark
the checkpoint against the held-out wells.
The `tui` subcommand opens a terminal interface over the same stages
(see below).
The `view` subcommand opens a run's wells in
[napari](https://napari.org): the raw time series, the difference stack, the
segmentation labels, and the QA overlay of each well side by side, so the QA
protocol becomes "look at the run" instead of "open the PNG".

## Install

The project installs from source only. Clone or download the repository,
then install with uv:

[uv](https://docs.astral.sh/uv/) manages the environment (ADR 0003).
Python ≥ 3.12 is required; development uses 3.13.

```
uv sync
```

This creates `.venv`, installs the package editable, and honours the
committed `uv.lock`. Segmentation needs a CUDA GPU;
the difference, analyze, and finetune-conversion stages do not.

The test suite is hermetic: no real data, no GPU, no network.

```
uv run pytest
```

## Usage

To run `colony-tracking`, you can use `uv run colony-tracking` inside the repository.  
To run from a different directory, first activate the environment, followed by
starting colony-tracking:  

```bash
source [path/to/repo/].venv/bin/activate
colony-tracking
```

The `colony-tracking` console script has seven subcommands, which are explained below.

Runs can be resumed after stopping, a well with a complete done marker is
skipped on a re-run unless `redo` is set.

### difference

Per-well uint8 difference frames from a raw plate.

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `none` | TOML config file: the stage's own flat config (what the TUI writes) or a pipeline config (this stage's section plus the run-level wiring). Command-line flags override its values. |
| `--raw-dir` | `required` | Directory of raw plate frames (flat or per-well layout). |
| `--out-dir` | `required` | Directory to write the per-well difference trees into. |
| `--reference` | `first` | Reference frame: "first" or a 1-based frame number. |
| `--wells` | `none` | Process only these wells (repeatable, comma-separated). Default: all wells. |
| `--display-percentile` | `99.5` | Percentile of \|diff\| that sets the per-well display limit. |
| `--subsample` | `8` | Stride of the subsampled grid used to estimate the display limit. |
| `--n-workers` | `6` | Thread-pool size over wells. |
| `--dry-run` | `unset` | Report wells, frames, and reference; write nothing. |

`--config` may be this stage's own flat TOML (what the TUI's Difference
screen writes) or a `pipeline.toml` (the `[difference]` section plus the
run-level wiring). Command-line flags override the file's values, and
required options may come from the file instead of the flags.

### segment

Colony segmentation and tracking of raw or difference frames. This is the
only stage that touches the GPU (micro-sam is imported lazily inside it).
`--reid-only` re-derives segments from saved label images and re-tracks
without the model.

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `none` | TOML config file: the stage's own flat config (what the TUI writes) or a pipeline config (this stage's section plus the run-level wiring). Command-line flags override its values. |
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
| `--exclude-before` | `none` | Flag colonies first detected at or before this frame number as pre-existing (excluded from the analysis); frames below it are still segmented and tracked. |
| `--no-merge-touching` | `unset` | Pre-merge touching segments of one group (disable with --no-merge-touching). |
| `--redo` | `unset` | Reprocess wells even when their done marker is present. |
| `--reid-only` | `unset` | Re-derive segments from the saved label images and re-track without the model (no GPU). |

`--exclude-before` replaces the prototype's `--exclude_lo` and
`--drop_labels_before`, which were always applied together with the same
number.

`--config` may be this stage's own flat TOML (what the TUI's Segment
screen writes) or a `pipeline.toml` (the `[segmentation]` section plus the
run-level wiring — for a pipeline config the input is always the chain's
difference tree). Command-line flags override the file's values, and
required options may come from the file instead of the flags.

### analyze

Growth measurements, aggregates, and plots for one or more segmentation
trees.

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `none` | TOML config file: the stage's own flat config (what the TUI writes) or a pipeline config (this stage's section plus the run-level wiring). Command-line flags override its values. |
| `--seg-dir` | `required` | Segmentation trees to analyze (repeatable). |
| `--raw-dir` | `required` | Raw plate directory (well-circle fit). |
| `--out-root` | `required` | Directory to write the analysis outputs into. |
| `--wells` | `none` | Analyze only these wells (repeatable, comma-separated). Default: all wells. |
| `--rim-width-px` | `60` | Width of the rim band inside the well circle (px). |
| `--rim-overlap-max` | `0.1` | A colony is rim-excluded from the first frame in which more than this fraction of its pixels lies in the rim. |
| `--max-bad-segments` | `2` | A well with at least this many distinct bad colonies is a bad well. |
| `--keep-bad-wells` | `unset` | Include bad wells in aggregates and plots. |
| `--keep-pre-existing` | `unset` | Include pre-existing colonies in aggregates and plots. |
| `--min-growth-per-frame` | `0.1` | A colony for which any valid frame, compared to the earliest valid frame >= 2 images ahead, grows by this factor or less per image is bad (no_growth); pre-existing colonies are exempt. |
| `--refresh-geometry` | `unset` | Re-fit the well circles instead of using the cached well_geometry.csv. |
| `--plots-only` | `unset` | Re-render the plots from the existing trajectories CSV without image IO. |
| `--no-plots` | `unset` | Skip the plots. |
| `--no-qa` | `unset` | Skip the per-well QA overlays. |
| `--no-group-alpha` | `unset` | Do not fade group opacity with member count. |

`--config` may be this stage's own flat TOML (what the TUI's Analyze
screen writes) or a `pipeline.toml` (the `[analysis]` section plus the
run-level wiring — the segment directories and analysis out root derive
from the chain's difference/segment trees). Command-line flags override
the file's values, and required options may come from the file instead of
the flags.

### pipeline

The standard chain (difference → segment on the difference tree → analyze)
from a TOML config file, or from the command line alone. The command line
carries the config path and the run-level overrides only; per-stage tuning
belongs in the config file, where every stage option (the stage's dataclass
fields — its flags above, `--config` excepted) is valid in its stage's
section (`[difference]`, `[segmentation]`, `[analysis]`, snake_case keys).

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `none` | TOML config file: a [pipeline] section plus one optional section per stage. Omit to drive the run from the command line alone (--raw-dir and --out-root required then). |
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

The config file is optional. A flag-only run is equivalent to the minimal
config above:

```
colony-tracking pipeline --raw-dir data/20260825_160101_Plate1 --out-root out --reference 5
```

`--raw-dir` and `--out-root` are required when `--config` is omitted.
Per-stage tuning still needs a config file.

Tree names derive from the out root and the reference: `difference_LO005`,
`seg_growth_diff_LO005`, `analysis/` (with `reference = "first"` the first
two are plain `difference` and `seg_growth_diff`). A stage section can
override its tree's location (`[difference].out_dir`, `[segmentation].out_dir`,
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
| `--n-max-images` | `none` | Evaluate only the first N h5 images of the split (a quick smoke run); 0 = uncapped. |
| `--n-max-instances` | `none` | Score only the first N instances per image. |
| `--output` | `none` | Optional path for a JSON report. |

### tui

The terminal user interface: `colony-tracking tui` (no flags). Four run
types — the whole pipeline chain, and Difference / Segment / Analyze
each on its own — each edit their own in-progress config and write and
read their own config TOML; nothing on a screen is derived from another
run type's configuration or outputs (the chain's own wiring — segment
consuming the difference tree, the shared out root — applies only
within one pipeline run). Every runtime line the stages log streams
live into the run screen. The home menu offers:

- **Quick start (full-auto)** — the raw directory and the out root (the
  reference frame and wells optional), *Run now*: the whole chain runs
  in-process with every stage option at its default. Edits and writes
  the chain's `pipeline.toml`.
- **Difference / Segment / Analyze** — one screen per stage on its own:
  every option of the stage's dataclass as a form field (the same
  option dataclasses the CLI flags come from; labels are the kebab-case
  flag names) — including the ones the chain would wire for you,
  e.g. segmentation's input mode and its difference-tree directory —
  plus *Run this stage*, *Write TOML…*, and *Load TOML…* for the
  stage's own flat config file (`difference.toml` / `segmentation.toml`
  / `analysis.toml`). Re-running a stage re-enters it — per-well done
  markers make that a resume, not a wipe.
- **Pipeline config** — the chain's config in one place: the run-level
  settings, the per-stage sections (the fields the chain wiring derives
  within one run shown read-only), and the effective TOML preview;
  *Write TOML…* / *Load TOML…* for `pipeline.toml`.
- **View wells in napari** — the wells the chain's resolved sources
  contain: the pipeline draft's raw directory and its three derived
  output trees (difference, segmentation, analysis), discovered exactly
  as the `view` command discovers them, so a well none of them holds is
  never offered. Select one or more and *Open in napari*: the TUI starts
  `colony-tracking view` as a separate process with the wells and the
  draft's resolved paths, so the viewer shows the same layers the command
  produces by hand for those wells. The well box is a multi-selection
  list (arrow keys move, space toggles), not a text field; selecting from
  the discovered wells is the input, because `view` only opens wells that
  its sources actually hold. *Load pipeline TOML…* opens the same file
  dialog as Pipeline config, so the viewer can pick which `pipeline.toml`
  to read even when no draft has been configured yet; the wells refresh
  from the loaded file. When discovery finds nothing, the hint above the
  list names every source directory it checked. The TUI never imports
  napari and stays usable while the viewer is open; the child's lines and
  any launch failure (napari missing, no display) appear in the screen's
  log pane, never as a crash.
- **Train** — shown disabled (no training screen).

Path fields (the raw directory, the out root, output directories, and
similar) carry a *browse* button that opens a directory picker; the
choice lands in the field like typed text. The *Load TOML…* and *Write
TOML…* dialogs carry the same button for their config-file path: it opens
a file picker that prefers `*.toml` (with an all-files fallback), and the
choice is loaded or written at once. A loaded config's relative paths are
resolved against the directory it was written from (the config's own
directory or nearest ancestor where its raw tree exists), so a
`pipeline.toml` saved while the TUI ran in a project directory still finds
its data when the TUI is started elsewhere.

Each run type is a decision point: after its options are filled, either
*Write TOML…* or a direct start — and a direct start writes that run
type's own TOML first, so every run is reproducible from its config
file. A stage screen suggests its config beside the stage's output
tree (`<out-dir's parent>/difference.toml`; analysis, which writes into
an out root, suggests `<out-root>/analysis.toml`); the chain suggests
`<out-root>/pipeline.toml`. *Load TOML…* reads a run type's config back
into its screens (a load resets, it never merges); a stage screen accepts
its own flat TOML as well as a sectioned `pipeline.toml` (the stage's
wired options are derived from it). Config-only mode is
the same flow without a run: fill what you need, *Write TOML…*, quit,
and run the CLI later — the chain via
`colony-tracking pipeline --config <file>`, or a stage via
`colony-tracking difference|segment|analyze --config <file>`. A stage CLI
accepts either that stage's own flat TOML or a `pipeline.toml` (whose
section and run-level wiring resolve to the stage's options).

A run opens on its own screen: the live log, the stage status line, and
the final per-stage summaries (from the summary objects, never from the
log text); a stage failure — an exception or failed wells — becomes an
error panel and the app stays alive. There is no cancellation: quitting
(Ctrl-Q) mid-run is safe, the stages are crash-safe via their done
markers, and the next start resumes where the run left off.

### view

Open a run's wells in napari. A well gets up to four layers, one shared
frame slider, every layer visible:

- `<well> raw` — the acquisition frames as one greyscale image stack;
- `<well> diff` — the stored difference images as an `uint8` image stack,
  exactly as written on disk (`128` = no change, ADR 0001; no remapping
  toward a diverging palette);
- `<well> labels` — the segmentation label images as a napari `Labels`
  layer, IDs untouched (`uint32`);
- `<well> qa` — `qa/<well>_rim.png` from the analysis output of the
  configured segmentation root, loaded as-is. When a raw stack is loaded
  it is placed on the raw frame's footprint by a coordinate scale
  (`H/512`, `W/512` for the 512 px overlay); without a raw stack it is
  loaded unscaled.

Every source is optional and independent: a run with only segmentations,
only difference frames, or only a QA overlay is a normal session. A missing
artifact (no such tree, or a well without its files) drops that one layer
for that well with an INFO line and the session still opens. napari ships
with micro-sam, so `view` needs nothing extra; it is imported only when the
command runs.

| Option | Default | Description |
| --- | --- | --- |
| `--wells` | `none` | Wells to display (repeatable, comma-separated), e.g. -A001. Always explicit: every well must exist in the configured sources. |
| `--config` | `none` | TOML config file to read the paths from (default: pipeline.toml in the current directory, when that file exists). |
| `--raw-dir` | `none` | Directory of raw plate frames (flat or per-well layout). Overrides the config file's raw directory. |
| `--diff-dir` | `none` | Difference tree to show (one directory per well of uint8 previews). Overrides the config file's difference.out_dir. |
| `--seg-dir` | `none` | Segmentation tree to show (one directory per well of label images), and the root whose QA overlay is read. Overrides the config file's segmentation.out_dir. |
| `--analysis-dir` | `none` | Analysis output root holding the per-root QA overlays. Overrides the config file's analysis.out_root. |

From a run directory that holds a `pipeline.toml`, one flag is enough:

```
colony-tracking view --wells=-A001
```

Each layer is fed by one path: the raw stack by `[pipeline].raw_dir`
(`--raw-dir`), the difference stack by `[difference].out_dir`
(`--diff-dir`), the labels by `[segmentation].out_dir` (`--seg-dir`), and
the QA overlay by `[analysis].out_root` (`--analysis-dir`) — the overlay is
read from `<analysis-dir>/<segmentation-root-name>/qa/<well>_rim.png`,
never by searching across analysis roots for the file name, so an analysis
root holding several segmentation trees cannot show the wrong one. A
QA-only session without `--seg-dir` reads a well's overlay from an analysis
root only when exactly one root's `qa/` directory holds it; otherwise the
layer is skipped with an INFO line.

Paths resolve in this order: `--config PATH` when given, else the current
directory's `pipeline.toml` when that file exists, else the explicit path
flags. The config's paths are its resolved stages, so the derived tree
names follow its reference frame (`difference_LO005`, ...). An explicit
path flag overrides only its own config value, so `--raw-dir` alone points
at a different raw plate while the rest of the config stays in force; with
neither a config file nor any path flag, the command exits 2 naming the
flags.

Wells are always named on the command line — `--wells` is never inherited
from the config's `[pipeline].wells`. Well names start with a dash, so the
value stays inside the flag: `--wells=-A001,-B004` (or `--wells -A001`
once, repeatable). A well qualifies when any configured source has it. A
missing `--wells` and a well that none of the configured sources holds are
usage errors (exit 2) naming the wells that are available; a session with
nothing to load is reported as an error naming the wells that are
available.
`view` writes nothing and performs no pixel conversion: the raw stack keeps
the raw uint16 counts, the difference stack the stored uint8 values, and
the label images their `uint32` IDs.

The TUI offers the same session: **View wells in napari** on the home menu
lists the wells the chain draft's resolved sources hold and starts this
command as a separate process (see [tui](#tui)); *Load pipeline TOML…*
there picks the pipeline file when the draft is still empty. The TUI
itself never imports napari.

## Python API

```python
import logging
from colony_tracking import DifferenceOptions, run_difference

logging.basicConfig(level=logging.INFO)   # all output goes through logging

opts = DifferenceOptions(raw_dir="plate", out_dir="difference_LO005", reference=5)
summary = run_difference(opts)
```

Every stage follows the same shape: an options dataclass whose defaults
mirror the CLI flags, a `run_*` function, and a small summary dataclass.
The pipeline step takes a `PipelineConfig` instead; `read_config(path)`
and `write_config(config, path)`, module-level functions in
`colony_tracking.pipeline`, round-trip it as TOML. The finetune track
exposes `PrepareOptions`/`TrainOptions`/`EvaluateOptions` with their
`run_*` functions the same way; the conversion module is private.

The viewer follows the same shape: `view_wells(ViewOptions(...), viewer=None)`
builds the layers and returns the viewer without starting napari's event
loop (the CLI starts it). `view_wells` and `ViewOptions` are exported from
the package root; `viewer` accepts a napari session of the caller's own
(the tests pass a recording stub, so they never touch napari).

## Input trees

Every stage accepts two input layouts:

- flat: all wells' frames in one directory; the well is parsed from the
  leading `-<A-H>\d{3}` of the file name.
- per-well: one subdirectory per well; the well is the directory name.

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
plus `scale.txt`, key/value lines with `reference` (the reference frame's
file name) and `display_limit_int16` (the float display limit). Readers
tolerate missing or extra lines. Frames acquired before the reference
frame come out polarity-flipped; that is intended (128 = no change, ADR
0001).

**A segmentation tree** holds, per well:

- `<frame-stem>__labels.tif`: one label image per input frame (uint32;
  0 = background; colony IDs), 1:1 with the input frames.
  `ref__labels.tif` holds the group labels of the reference frame.
- `colonies.csv`: one row per colony per frame, columns
  `well, frame, t_seconds, image, colony_id, group_id, merged_with,
  pre_existing, area_px, centroid_y, centroid_x`. `frame` is the 1-based
  frame number, `t_seconds` the raw `T` field (milliseconds here),
  `merged_with` the ';'-separated colony IDs absorbed in that frame's
  merge (empty when none).
- `_done.txt`: tab-separated counters for `frames`, `groups`, `colonies`,
  `pre_existing_colonies`, `merge_events`, `forked_segments`,
  `discarded_instances`. Presence means the well is complete.

**The analyze stage** writes, under `<out-root>/<seg-root-name>/`:

- `trajectories.csv`: one row per colony per frame, columns `well, frame,
  t_hours, colony_id, group_id, merged_with, ancestors, pre_existing,
  area_px, age_h, rim_frac, rim_excluded, no_growth`
- `trajectories_good.csv`: the good trajectory rows — not `no_growth`, not
  from an excluded bad well, not `pre_existing` (unless `keep_pre_existing`),
  and not `rim_excluded` (the rows the pooled aggregates and plots use)
- `trajectories_ancestors_<n>.csv`: the same good rows, one file per
  row-level `ancestors` count present, so a colony that later merges appears
  in several buckets
- `growth_rates.csv`: one row per colony, columns `well, colony_id,
  group_id, first_frame, last_frame, n_frames, n_frames_valid,
  area_first, area_last, max_ancestors, rim_excluded_frame,
  growth_rate_per_h, growth_doublings_per_day, abs_growth_px_per_h,
  no_growth, pre_existing`
- `rates_good.csv`: every good colony that has a computed rate (colonies
  whose rate window is too narrow have blank rate cells and are skipped)
- `rates_ancestors_<n>.csv`: the same good rate rows, one file per
  colony-level `max_ancestors` value present (each colony lands entirely in
  one bucket)
- `bad_segments.csv`: one row per bad colony (frames, areas)
- pooled aggregates: `areas_by_ancestors.csv`, `rates_by_ancestors.csv`
- plots: `growth_curves_by_ancestors.png`, `mean_area_by_ancestors.png`,
  `growth_rates_by_ancestors.png`, and one `growth_curves_<n>_ancestor(s).png`
  growth-curve plot per ancestor count
- per-well QA overlays: `qa/<well>_rim.png`

`abs_growth_by_well.png` puts one absolute-growth box plot per well on the
96-well plate grid (rows A-H, columns 1-12) with a shared y-axis, so a well
whose colonies grow differently stands out by its position on the plate.
Red titles mark wells excluded as bad wells, orange titles wells with bad
segments below that threshold; this overview deliberately includes every
computed rate (bad and pre-existing colonies too), unlike the pooled plots.

`well_geometry.csv` (the fitted well circles, `well, cy, cx, R`) is a
shared cache in the analysis out root; `--refresh-geometry` re-fits it.

**Renamed columns.** The transitive self-inclusive ancestor count
(prototype `n_merged`; 1 = unmerged) is written as `ancestors` everywhere
(columns, pooled files, plot names); the frame number (prototype `lo`) is
written as `frame`. Values are unchanged. The analyze stage reads
prototype-written trees and maps the legacy names (`lo`, `n_merged`,
`max_n_merged`, `first_lo`, `last_lo`, `rim_excluded_lo`) to the canonical
ones; new writes always use the canonical names.

**QA overlay colors.** Each `qa/<well>_rim.png` composites the colony
statuses of the frame with the highest frame number that has rows onto the
raw frame. Colony tints (translucent fill plus solid edge, at the shown
frame, in this order): magenta — bad segment (from the series start or not
growing); orange — pre-existing (excluded over its whole lifetime); red —
rim-excluded; green edge without fill — valid. Beneath the colonies, the
fitted well circle is blue and the rim-zone boundary orange (the same hue
as the pre-existing tint; the boundary is a thin ring). A well with no
colony rows gets a rim-only overlay on its last raw frame.

## Integration test

`tests/test_integration.py` runs the standard chain on the real plate and
checks it against the prototype's published outputs. It is excluded from
the default run and skips cleanly unless both prerequisites are present:

- the data root via the `COLONY_TRACKING_DATA_ROOT` environment variable:
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

- `GLOSSARY.md`: the binding domain vocabulary (data, colonies,
  exclusions and flags, analysis).
- `docs/adr/`: decision records on 0001 (8-bit difference frames with
  128 = no change), 0002 (growth-mask gating from the final frame), 0003
  (uv + PyPI micro-sam), and 0004 (`exclude_before` flags pre-existing
  colonies instead of blanking their frames).

## License

MIT; see `LICENSE`.
