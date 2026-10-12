Status: ready-for-agent
Blocked by: 07, 08

# 09: `cli` — the `colony-tracking` entry point

Goal: one CLI entry point with subcommands
`difference | segment | analyze | pipeline | train` (train has sub-subcommands
`prepare | train | evaluate`), kebab-case flags mapping 1:1 to option fields,
and the exit-code contract.

Spec: `.scratch/colony-tracking/spec.md` — "Options and CLI contract" (all
five subcommands' flags, incl. the kebab-case/`--wells` rules), "Pipeline and
config → Config file" (pipeline takes `--config` + run-level overrides only),
"Package layout and public API → Exit codes", user stories 2, 28–30, 44.

## Build — `src/colony_tracking/cli.py`

- `main(argv=None) -> int` (the console-script target; returns the exit code,
  does not call `sys.exit` so it is testable via the public API too — the
  installed script wraps it).
- Subcommands:
  - `difference`, `segment`, `analyze`: every option field is a kebab-case
    flag mapping 1:1 to the dataclass field (`raw_dir` → `--raw-dir`,
    `fg_ref` → `--fg-ref`, `rim_width_px` → `--rim-width-px`, ...). Boolean
    options become flags (`--redo`, `--dry-run`, `--reid-only`,
    `--no-merge-touching`, `--keep-bad-wells`, `--plots-only`, `--no-plots`,
    `--no-qa`, `--no-group-alpha`, `--refresh-geometry` ...).
  - `--wells` is **repeatable AND accepts comma-separated values** (well
    names start with a dash, so they must stay inside the flag value:
    `--wells -A001,-A002 --wells -B005`).
  - `reference` accepts `first` or a frame number (`5` or `LO005`).
  - `segmentation.raw_dir` / `diff_dir` are required exactly when
    `--input raw` / `--input diff` (usage error otherwise).
  - `analyze.seg_dirs` is repeatable (`--seg-dir`, comma-separated allowed).
  - `pipeline`: `--config` (TOML) plus **run-level overrides only**:
    `--raw-dir`, `--out-root`, `--reference`, `--wells`. Per-stage tuning
    belongs in the config file. Overrides take precedence over the config.
  - `train prepare | train | evaluate`: the finetune options as kebab-case
    flags (defaults from the spec's hyperparameter list).
- Help text: every flag's help string comes from the dataclass per-field help
  (TUI reuse, spec user story 36); subcommand help lists the option table.
- **Exit codes**: 0 success; 1 runtime error (logged to the package logger
  with `error` level, with context — missing data, failed wells, model
  errors); 2 usage error (argparse errors, invalid option combinations,
  config errors). Set up logging once in `main` (package logger → stderr,
  INFO default) — this is the only logging configuration in the package.
- Missing data (e.g. `raw_dir` does not exist / no frames found) → logged
  error + exit 1, not a traceback.

## Tests — `tests/test_cli.py`

Drive `colony_tracking.cli.main([...])` in-process with tmp_path inputs from
the conftest builders (difference/analyze can run for real on small plates;
segment/train paths validated via option construction / error cases, never
the model):

- flag parsing: kebab-case mapping for representative fields of every
  subcommand (assert on the constructed options dataclass — expose a
  testable seam, e.g. `main` builds options via a per-subcommand function);
  repeatable `--wells` with comma-separated dash-leading values;
  `reference` `first` / `5` / `LO005`.
- exit codes: missing `raw_dir` → 1 with a logged message (caplog);
  usage error (e.g. `segment` without `--raw-dir`/`--diff-dir`, unknown flag)
  → 2; success path → 0.
- `difference --dry-run` via the CLI: exit 0, nothing written, discovery
  logged.
- `pipeline`: `--config` + run-level overrides — precedence over config
  values (assert on the resolved `PipelineConfig` via a monkeypatched
  `run_pipeline`); missing config file → 1; unknown config key → 2.
- `train prepare` option parsing (kebab-case, defaults) without running
  torch.

## Acceptance

- `uv run pytest` passes.
- `uv run colony-tracking --help` and each subcommand's `--help` render clean
  option tables (smoke-tested).
- No prints anywhere in the package (the CLI routes everything through
  logging).
