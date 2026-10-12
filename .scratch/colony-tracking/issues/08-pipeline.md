Status: ready-for-agent
Blocked by: 03, 05, 06

# 08: `pipeline` module — config, derived names, orchestration

Goal: the standard chain (difference → segment on the difference tree →
analyze) driven by a TOML config, with run-level settings, derived tree
names, and per-stage resume. The CLI wiring for it is ticket 09.

Spec: `.scratch/colony-tracking/spec.md` — "Pipeline and config" (all
bullets), "Options and CLI contract" (the three stage option dataclasses),
"I/O contract → Tree naming", user stories 28–31, 39, 40.

## Build — `src/colony_tracking/pipeline.py`

- `PipelineConfig` = run-level settings (`raw_dir: Path`, `out_root: Path`,
  `reference: "first" | int = "first"`, `wells: list[str] | None = None`)
  + the three stage option dataclasses (`difference`, `segmentation`,
  `analysis` — each fully overridable per stage). Built on the shared
  `_options.py` machinery: dict/TOML round-tripping with per-field help text.
- **Config file** (TOML): a `[pipeline]` section (run-level) plus one section
  per stage (`[difference]`, `[segmentation]`, `[analysis]`) with any of its
  options; omitted keys take defaults. Read with stdlib `tomllib`; a write
  helper via `tomli-w` for the future TUI. Unknown keys in a known section =
  usage error (logged, clear message); unknown sections rejected.
- **Derived names** (all overridable in the config):
  - difference tree: `<out_root>/difference_LO<nnn>` for a numeric reference,
    `<out_root>/difference` for `reference = "first"`;
  - segment root: `<out_root>/seg_growth_diff_LO<nnn>` /
    `<out_root>/seg_growth_diff`;
  - the segment stage's `exclude_before` **defaults to the reference frame
    number** when the reference is numeric (explicit config values win);
  - analysis out root: `<out_root>/analysis`; the segment root is passed as
    the analysis `seg_dirs` entry and the difference tree's reference is the
    probe reference for difference-root analysis.
  Keep the `LO<nnn>` suffix convention (zero-padded 3 digits).
- `run_pipeline(config) -> PipelineSummary`:
  - run the stages in order: difference → segment (`input = "diff"` on the
    difference tree just produced) → analyze (on the segment root);
  - each stage's own resume logic applies, so an interrupted run re-enters at
    the first incomplete stage's incomplete wells (no extra bookkeeping);
  - the pipeline always runs the standard chain (raw-input segmentation stays
    available via the `segment` subcommand directly — ticket 09);
  - a stage failure aborts the pipeline with the logged error and a non-OK
    summary (CLI exit code handling is ticket 09).
  - summary: per-stage embedded summaries + overall status.
- Log per stage via the package logger; never print.

## Tests — `tests/test_pipeline.py`

Use the mini-plate fixture + synthetic inputs in tmp_path. The model path
must NOT run in the suite: structure the test so orchestration is exercised
without micro-sam — e.g. monkeypatch the three `run_*` functions to record
their options and return canned summaries (the real difference/analyze
stages can additionally be exercised for real where no model is needed).

- config parse: TOML → `PipelineConfig` (defaults for omitted keys;
  run-level + per-stage values); round-trip `PipelineConfig` → TOML →
  `PipelineConfig` equality.
- derived names: numeric reference → `difference_LO005` /
  `seg_growth_diff_LO005`; `reference = "first"` → `difference` /
  `seg_growth_diff`; explicit overrides win over derived names.
- `exclude_before` derivation: numeric reference → segment
  `exclude_before` = reference number; explicit segment value wins;
  `reference = "first"` → stays `None`.
- orchestration: stages run in order; the segment stage receives
  `input = "diff"` with the difference tree as `diff_dir`; analyze receives
  the segment root; a failing stage aborts the chain (later stages not
  called) with a logged error.
- interrupted-run re-entry: with fake stage functions, a second
  `run_pipeline` call skips stages whose wells are already complete (assert
  via the recorded calls / summaries).
- invalid config (unknown key / unknown section / missing required) → clear
  logged error, raised as a usage error (ticket 09 maps it to exit code 2).

## Acceptance

- `uv run pytest` passes.
- Public API: `PipelineConfig`, `run_pipeline`, `PipelineSummary` re-exported
  from the package root.
- No hardcoded data paths; no model imports in this module.
