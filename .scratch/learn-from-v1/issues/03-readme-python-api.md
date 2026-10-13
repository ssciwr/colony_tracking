Status: ready-for-agent

# README: short in-process Python API section

Spec: `.scratch/learn-from-v1/spec.md` (item 3, README part). v1 reference
repo: `../colony_tracking` (absolute:
`/home/kriedmiller/projects/colony_size/colony_tracking`).

## What

v1's README documents the in-process Python API with an example
(v1 `README.md:137-156`). v2's README (`README.md`, repo root) covers the
CLI only — yet the importable API is a v2 deliverable (the v2 spec's TUI
stories 36-37 build on it), and the strict public-API test
(`tests/test_skeleton.py:19`) pins `colony_tracking.__all__` to exactly the
options/run_*-functions/summatures/PipelineConfig surface.

## Fix sketch

Add a `## Python API` section to `README.md` right after the Usage section
(i.e. before `## Input trees`), adapted to v2's actual API — v1's section is
~19 lines; keep this one that short:

- Example block in v1's shape, with v2's real names:

  ```python
  import logging
  from colony_tracking import DifferenceOptions, run_difference

  logging.basicConfig(level=logging.INFO)   # all output goes through logging

  opts = DifferenceOptions(raw_dir="plate", out_dir="diff_LO005", reference=5)
  summary = run_difference(opts)
  ```

- One short paragraph: every stage follows the same shape — an options
  dataclass whose defaults mirror the CLI flags, a `run_*` function, and a
  small summary dataclass. The pipeline step takes a `PipelineConfig`
  (`read_config(path)` / `write_config(config, path)` round-trip TOML —
  v2's module-level functions, NOT v1's `config.load`/`config.save`). The
  finetune track exposes `PrepareOptions`/`TrainOptions`/`EvaluateOptions`
  with their `run_*` functions the same way; the conversion module is
  private.

Everything the section names must exist exactly as written in
`colony_tracking.__all__` / the pipeline module — verify each name against
`src/colony_tracking/__init__.py` and `src/colony_tracking/pipeline.py:454-473`
before writing.

## Risk

The README contract tests (`tests/test_readme.py`) pin existing headings and
needles; adding a section is safe as long as nothing existing is renamed or
removed.

## Tests

`uv run pytest` (test_readme.py is the guard; no new tests needed).

## Acceptance criteria

- The section exists, the example is accurate for v2, the suite passes.
