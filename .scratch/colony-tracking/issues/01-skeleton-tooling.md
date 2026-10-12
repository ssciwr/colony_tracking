Status: resolved
Blocked by:

# 01: Project skeleton, tooling, and domain docs

Goal: a uv-managed project skeleton that matches the spec exactly, plus the
domain vocabulary (GLOSSARY.md) and the three ADRs, so every later ticket can
use binding terminology without re-deciding anything.

Spec: read `.scratch/colony-tracking/spec.md` — sections "Repository and
tooling", "Package layout and public API", "Documentation and records".

## Build

- `pyproject.toml` (rewrite the `uv init` stub):
  - hatchling build backend (replace the `uv_build` backend);
  - name `colony-tracking`, version `0.1.0`, `readme = "README.md"`,
    `requires-python = ">=3.12"`;
  - MIT license: add a `LICENSE` file, `license = { text = "MIT" }` (or the
    PEP 639 `license = "MIT"` form if the uv version supports it — check
    `uv --help`/docs, keep `uv lock` happy);
  - dependencies (exact list from the spec): `numpy`, `scipy`,
    `scikit-image`, `tifffile`, `Pillow`, `matplotlib`, `micro-sam`,
    `torch_em`, `h5py`, `tomli-w`;
  - PEP 735 `[dependency-groups]` dev group: `pytest`;
  - console script `colony-tracking` → `colony_tracking.cli:main`.
- `.python-version` stays `3.13`. Generate `uv.lock` and commit it
  (`uv lock`); `uv sync` must succeed (the uv cache on this machine is warm).
- Package layout under `src/colony_tracking/` (stubs with docstrings only in
  this ticket): `io.py`, `difference.py`, `tracking.py`, `segmentation.py`,
  `analysis.py`, `pipeline.py`, `cli.py`, and a `finetune/` package
  (`__init__.py` + `_convert.py` placeholder). A private `_options.py` holds
  the shared options machinery (see below).
- Package logger: `logging.getLogger("colony_tracking")` exposed as
  `colony_tracking.log`; all later modules log through it, never print.
- Options machinery in `_options.py` (shared by every stage): a base class /
  helpers giving each options dataclass (a) per-field help text (e.g. via
  `Field(metadata={"help": ...})` or a parallel mapping — pick one clean
  design), (b) `to_dict()` / `from_dict()` round-tripping that later TOML
  and TUI work consumes. Dataclasses must be plain, importable, with defaults
  exactly as the spec's "Options and CLI contract" section states.
- Root `src/colony_tracking/__init__.py`: re-exports for the public API
  (empty set for now, grows per ticket — keep a clear `__all__`).
- `README.md`: short placeholder (title + one line; the full README is
  ticket 10).
- `GLOSSARY.md` at repo root — the final agreed version: sections
  **Data / Colonies / Exclusions-and-flags / Analysis** only. Terms are drawn
  from the spec's vocabulary (well, plate, frame, frame number, reference
  frame, time field, difference frame, display limit, input tree layouts,
  segment, instance, group, growth mask, colony, colony ID, merge, absorbed,
  fork, pre-existing, rim, rim-excluded, bad segment, bad well, ancestors,
  drift gate, exclude_before, well circle, growth rate, pooled). Do NOT
  include entries for *valid*, *trajectory*, *aggregate*, and no Model
  section — those were deliberately dropped from the glossary.
- `docs/adr/`: three short ADRs per the spec:
  - `0001-8-bit-difference-format.md` — 8-bit difference frames with
    128 = no change; int16 counts and the `abs`/`abs_percentile`
    presentations were considered and rejected;
  - `0002-growth-mask-gating.md` — growth-mask gating from the final frame,
    including the drift gate;
  - `0003-uv-pypi-micro-sam.md` — uv + PyPI `micro-sam` instead of the
    prototype's conda-forge workflow.
- Test scaffolding: `tests/` with `conftest.py` that forces matplotlib to the
  Agg backend (module-import side effect is fine) and registers the
  `integration` marker (pyproject `[tool.pytest.ini_options]` with
  `markers` + `addopts = "-m 'not integration'"` so the marked test is
  excluded from the default run). One smoke test: importing the package, the
  logger exists, and `colony_tracking.__all__` is a list.
- `main()` stub in `cli.py` so the console script resolves (real CLI is
  ticket 09).

## Acceptance

- `uv sync && uv run pytest` passes.
- `uv run colony-tracking` runs the stub without error.
- GLOSSARY.md, LICENSE, uv.lock, and the three ADRs are committed.
- No hardcoded data paths anywhere.

## Comments

- 2026-10-12 — Resolved by branch t01-skeleton (merge commit bdcc6ee0a220307e468187e0bdae8dbed16bc3a0); 19 passed; deviations noted in implementer report: options dataclasses live in _options.py, kw_only dataclasses, integration marker placeholder added, PEP 639 license form.
