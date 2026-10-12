Status: ready-for-agent
Blocked by: 02

# 06: `analysis` stage — circle, rim, flags, rates, aggregates, plots, QA

Goal: the analyze stage as a public API. This is the largest stage; keep the
module organized (clear section functions mirroring the spec bullets) and
port the prototype faithfully where it encodes decisions.

Spec: `.scratch/colony-tracking/spec.md` — "Options and CLI contract →
AnalysisOptions", "Analyze stage" (all bullets), "I/O contract" (analysis
outputs, renames, legacy reading), user stories 19–27, 47. Reference
implementation: `/home/kriedmiller/projects/colony_size/prototype/
growth_analysis.py` (port the algorithms; rename columns per the spec).
Real prototype outputs for shape reference (READ-ONLY):
`/home/kriedmiller/projects/colony_size/prototype/data/analysis/`.

## Build — `src/colony_tracking/analysis.py`

- `AnalysisOptions` — fields EXACTLY as the spec (defaults from the spec):
  `seg_dirs: list[Path]` (required, repeatable), `raw_dir: Path` (required),
  `out_root: Path` (required), `wells: list[str] | None = None`,
  `rim_width_px: int = 60`, `rim_overlap_max: float = 0.1`,
  `max_bad_segments: int = 2`, `keep_bad_wells: bool = False`,
  `keep_pre_existing: bool = False`, `raw_seg_dir: Path | None = None`,
  `start_frames: int = 1`, `min_growth_2step: float = 0.3`,
  `refresh_geometry: bool = False`, `plots_only: bool = False`,
  `no_plots: bool = False`, `no_qa: bool = False`, `no_group_alpha: bool = False`.
- `run_analysis(options) -> AnalysisSummary` (per-well / per-root processed,
  bad segments, excluded wells, pre-existing colonies, failures).
- One analysis output directory per segmentation root:
  `<out_root>/<seg-root-name>/` (e.g. `analysis/seg_growth_diff_LO005/`),
  plus the shared `well_geometry.csv` cache in `out_root`.

Port each of these (exact constants from the spec — they are calibrated to
the 2048×2048 acquisition format):

- **Well circle** (fitted per well from up to the three earliest RAW frames,
  no colonies yet): per frame Gaussian blur σ 6; stage 1 samples the darkest
  point per angle (240 angles, radii 650–1120 px from image centre) keeping
  only real local dips (≥ 0.08 × profile range below the ±60 px radial
  neighbourhood); < 30 points ⇒ no fit; coarse Kasa fit with iterative
  trimming (tol 30, ≥ 20 points) must land in 700–1100 px; stage 2 re-locks
  every angle in R ± 70 px (dip 0.02, ≥ 40 points), trims (tol 25), accepts
  only if radius std < 20 px, centre within 200–1850 px, and ≥ 8 of 12
  angular sectors (30°) covered. Frame fits with R deviating > 40 px from the
  median are dropped; the rest are averaged. Cached in `well_geometry.csv`
  (`refresh_geometry` forces refits). No plausible circle ⇒ rim exclusion
  disabled for that well (logged).
- **Rim**: a colony is rim-excluded from the first frame in which more than
  `rim_overlap_max` (0.1) of its pixels lie within `rim_width_px` (60) of the
  well circle; exclusion is **monotone** over the colony's lifetime.
- **Time**: `t_hours` from the colonies CSV `t_seconds` (raw `T` field) via
  the prototype's magnitude heuristic (values > 1e6 read as milliseconds).
- **Ancestors**: per colony-frame row, distinct original colonies combined in
  the segment, resolved transitively through `merged_with` histories, self
  included (unmerged = 1) — the prototype's `n_merged`, now named
  `ancestors`, values unchanged.
- **Bad segments**: `from_start` — first valid frame within `start_frames`
  (1) of the series start (anchored at the earliest label image);
  `no_growth` — every exact 2-frame window among the valid frames grows by
  less than `1 + min_growth_2step` (1.3).
- **Bad wells**: a well with ≥ `max_bad_segments` (2) distinct bad colonies is
  excluded from aggregates/plots (`keep_bad_wells` overrides); always
  reported in the log and `bad_segments.csv`.
- **Pre-existing**: flag from the segment stage CSV; for difference roots
  (roots named `*diff_LO<nnn>`) additionally **probe** each colony's
  first-detection centroid against the raw root's (`raw_seg_dir`) label image
  at the reference frame: hit at centroid ± 12 px, or inside an
  area-equivalent disk of a raw colony (+ 15 px slack) ⇒ pre-existing.
  Pre-existing colonies are excluded from aggregates/plots over their whole
  lifetime (`keep_pre_existing` overrides); rows stay in the CSVs.
- **Growth rates**: over a colony's valid (not rim-excluded) frames with ≥ 2
  points and positive area: specific rate = least-squares slope of ln(area)
  vs hours (h⁻¹); doublings/day = slope × 24 / ln 2; absolute growth =
  (last − first) valid area / elapsed time (px/h). Single valid frame → NaN
  rates (areas stay in trajectories).
- **Aggregates** (only non-bad, non-pre-existing rows from non-excluded
  wells): pooled per (frame, ancestors) area statistics; pooled per
  max-ancestors rate statistics.
- **Plots**: `growth_curves_by_ancestors.png` (individual trajectories,
  absolute and relative time, coloured by ancestor count),
  `mean_area_by_ancestors.png` (buckets with < 3 members not drawn; group
  opacity fades with member count unless `no_group_alpha`),
  `growth_rates_by_ancestors.png` (box + strip, symlog absolute-growth
  panel). Port the prototype's plot code (including its group-color/
  opacity helpers) — rename `merged` → `ancestors` in titles/legends.
- **QA overlays**: per-well 512×512 raw-frame composite — fitted well circle,
  rim boundary, colony edges coloured valid / pre-existing / rim-excluded /
  bad, translucent fill — `qa/<well>_rim.png`.
- **`plots_only`**: re-render everything derived from the existing
  trajectories CSV with no image IO (legacy column names accepted);
  `no_plots` / `no_qa` suppress the respective outputs.
- **Legacy reading**: when reading colonies/trajectories CSVs, map prototype
  column names (`lo`, `n_merged`, `max_n_merged`, `first_lo`, `last_lo`,
  `rim_excluded_lo`) to canonical names (use the io legacy mapping). New
  writes always use canonical names:
  - `trajectories.csv`: `well, frame, t_hours, colony_id, group_id,
    merged_with, ancestors, pre_existing, area_px, age_h, rim_frac,
    rim_excluded, bad, bad_reason`
  - `growth_rates.csv`: `well, colony_id, group_id, first_frame, last_frame,
    n_frames, n_frames_valid, area_first, area_last, max_ancestors,
    rim_excluded_frame, growth_rate_per_h, growth_doublings_per_day,
    abs_growth_px_per_h, bad, bad_reason, pre_existing`
  - `bad_segments.csv`: `well, colony_id, bad_reason, first_frame,
    last_frame, n_frames_valid, area_first, area_last, n_bad_in_well,
    well_excluded`
  - pooled: `areas_by_ancestors.csv` (`frame, t_hours, ancestors,
    n_colonies, n_wells, mean_area_px, median_area_px`) and
    `rates_by_ancestors.csv` (`max_ancestors, n_colonies,
    mean_doublings_per_day, median_doublings_per_day,
    mean_abs_growth_px_per_h, median_abs_growth_px_per_h`)
- Log per well / per root via the package logger; never print.

## Tests — `tests/test_analysis.py`

Use the mini-plate fixture (2048² with planted ring R ≈ 920 and planted
disks) + synthetic segmentation trees (label TIFFs + `colonies.csv` built in
tmp_path — no model). Matplotlib is Agg in tests.

- well-circle fit: recovers the planted centre/radius within tolerance; a
  broken ring (e.g. ring with a gap / no ring) is rejected (no fit, rim
  disabled, logged).
- rim exclusion: a planted colony entering the rim is excluded **monotonically
  from the right frame** (before: not excluded, at/after: excluded).
- bad flags: a planted `from_start` colony and a static `no_growth` colony get
  the right `bad_reason`.
- bad-well exclusion: a well with ≥ 2 bad colonies drops out of aggregates
  (and `keep_bad_wells` overrides).
- difference-root probe: a planted pre-existing colony (present in the raw
  root's reference-frame labels at the colony's first-detection centroid) is
  flagged.
- transitive ancestors: an A+B←C chain yields 3.
- growth rates: a planted exponential colony (area = A0·e^(λt)) recovers λ as
  `growth_rate_per_h` and λ·24/ln 2 as doublings/day within tolerance; a
  single-valid-frame colony gets NaN.
- pooled files + plots: exist, non-trivial content (rows > 0; PNGs non-empty),
  correct grouping (one row per (frame, ancestors) / (max_ancestors)).
- `plots_only`: re-renders from a written trajectories CSV with no raw-image
  IO (point raw_dir at an empty dir to prove it is not read).
- legacy tree: reading a prototype-style tree (write `lo`/`n_merged`-style
  CSVs in the test) yields canonical columns with unchanged values.

## Acceptance

- `uv run pytest` passes.
- Public API: `AnalysisOptions`, `run_analysis`, `AnalysisSummary`
  re-exported from the package root.
- Module docstring carries the algorithm detail per the spec.
- No hardcoded data paths.
