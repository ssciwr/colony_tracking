Status: ready-for-agent
Blocked by: 09

# 10: Marked integration test, full README, final suite green

Goal: the last spec deliverables — the marked (excluded-from-default)
integration test against the real plate, the full README, and a final check
that the whole hermetic suite passes and the lock file is current.

Spec: `.scratch/colony-tracking/spec.md` — "Testing Decisions → Integration
test and prior art", "Documentation and records → README", "Further Notes →
One-time verification protocol", "I/O contract" (output conventions).

## Build

- **`tests/integration/` (or `tests/test_integration_real_plate.py`)** — one
  pytest test marked `@pytest.mark.integration` (already excluded from the
  default run by ticket 01's addopts):
  - requires a real data root via **environment variable** (e.g.
    `COLONY_TRACKING_DATA_ROOT`) and a GPU; skip with a clear message when
    the env var is unset (so CI/developers without data are unaffected);
  - runs the **standard pipeline** (reference 5, per the one-time protocol)
    into a **scratch out root** under tmp (never touching the prototype's
    trees in `/home/kriedmiller/projects/colony_size/prototype/data/`);
  - asserts the one-time-check criteria: colony-count sanity against the
    documented numbers (difference root reference 5 → 369 colonies / 354
    merge events; reference 1 → 326 colonies / 9 forks — assert within a
    small documented tolerance, e.g. ±5%, since a fresh PyPI torch may
    segment slightly differently) and **column equality** for rows present in
    both the new tree and the prototype tree: new `frame` == old `lo`, new
    `ancestors` == old `n_merged` (and `max_ancestors` == old
    `max_n_merged`), read from the prototype outputs under
    `COLONY_TRACKING_DATA_ROOT` (e.g. `.../data/seg_growth_diff_LO005` and
    `.../data/difference_LO005`).
- **`README.md`** — the full minimal README:
  - what the package is (three stages + pipeline + finetuning track);
  - install via uv (`uv sync`, Python ≥ 3.12, interpreter 3.13, micro-sam
    from PyPI);
  - the five subcommands with **option tables** (every option, default, and
    help text — generate from the dataclasses to avoid drift);
  - the two input-tree layouts (flat / per-well) with a filename example;
  - output conventions (per-well trees, `colonies.csv`, label images,
    `_done.txt`, `scale.txt`, `well_geometry.csv`, analysis outputs, the
    `frame`/`ancestors` renames);
  - pointer to `GLOSSARY.md` and `docs/adr/`.
- **Final checks**:
  - `uv lock --check` (lock current with pyproject);
  - full hermetic suite green: `uv run pytest`;
  - no hardcoded data paths anywhere in `src/` (grep-check);
  - every module docstring carries its algorithm detail (spot-check the
    spec's "Module docstrings carry the algorithm detail" line);
  - package root `__all__` exports exactly the public API (option
    dataclasses, `run_*` functions, `PipelineConfig`, summary types).

## Acceptance

- `uv run pytest` (default markers) passes with zero skips that are not
  env-gated integration skips.
- `uv run pytest -m integration --collect-only` collects the integration
  test without importing torch/micro-sam at collect time.
- README option tables match the dataclasses (visually + via a quick script
  in the test if you want to be strict).
