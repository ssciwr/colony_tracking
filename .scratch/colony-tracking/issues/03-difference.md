Status: ready-for-agent
Blocked by: 02

# 03: `difference` stage

Goal: the difference stage as a public API — options dataclass +
`run_difference` + per-well outputs + resume + dry-run.

Spec: `.scratch/colony-tracking/spec.md` — "Options and CLI contract →
DifferenceOptions", "Difference stage", user stories 3–9, "I/O contract
→ scale.txt". Reference implementation:
`/home/kriedmiller/projects/colony_size/prototype/make_difference_images.py`
(port the plain-subtraction path only).

## Build — `src/colony_tracking/difference.py`

- `DifferenceOptions` (fields EXACTLY as the spec; defaults from the spec;
  per-field help text via the shared `_options.py` machinery):
  `raw_dir: Path`, `out_dir: Path`, `reference: "first" | int = "first"`
  (frame number, 1-based), `wells: list[str] | None = None`,
  `display_percentile: float = 99.5`, `subsample: int = 8`,
  `n_workers: int = 6`, `dry_run: bool = False`.
  The prototype's `--flatfield`/`--sigma`, `--int16`/`--skip_uint8`, and
  `--diff_scaling` variants are CUT — do not port them.
- `run_difference(options: DifferenceOptions) -> DifferenceSummary` (summary
  dataclass with per-well processed/skipped/failed counters and totals;
  return it — spec user story 40).
- Per well (thread pool over wells, `n_workers`):
  - load frames via `io` discovery (both input layouts), in frame order;
  - reference = first frame of the series, or frame number N (1-based);
  - `cur − ref` in float32 counts;
  - per-well display limit = `display_percentile` percentile of |diff|,
    estimated on the subsampled grid (stride `subsample`) — use the shared
    io helpers;
  - uint8 = `128 + diff × 127 / limit`, clamped (io helper). **128 = no
    change**. The reference frame itself is written as an all-128 frame so
    each well folder stays a complete, evenly numbered series.
  - frames acquired before the reference come out polarity-flipped —
    intended, document in the module docstring.
  - outputs per well: `<out_dir>/<well>/<frame-stem>_diff_uint8.tif`
    (1:1 with input frames, uint8) + `scale.txt` (`reference` = reference
    frame file name, `display_limit_int16` = float limit).
  - resume: a well with a complete marker is skipped on re-run (write a
    per-well done marker, e.g. `_done.txt` with at least `frames <n>`;
    presence = complete).
- `dry_run`: log (package logger) the discovered wells, frame counts, and
  chosen reference per well; write nothing, touch nothing.
- Log at prototype granularity: per well, per frame, via `colony_tracking.log`.
  Never print.

## Tests — `tests/test_difference.py`

Use the mini-plate fixture from `tests/conftest.py` (build only the wells/
frames you need).

- uint8 values: zero difference → exactly 128; a known planted count through
  a known limit maps to the expected byte (compute the limit yourself from
  the same rule or pin it with `display_percentile` on a synthetic array).
- polarity flip: a frame before the reference shows the expected sign
  (bright > 128 where colonies later grow).
- `scale.txt` contents: `reference` = reference frame file name;
  `display_limit_int16` = float limit.
- resume: run once, delete nothing, run again → complete wells skipped
  (counter reports skipped; files untouched — check mtime/bytes).
- `dry_run` writes nothing (out dir empty / not created).
- `n_workers=1` vs `n_workers=4` → identical output bytes.
- `wells=` subset runs only the named wells.

## Acceptance

- `uv run pytest` passes.
- Public API: `DifferenceOptions`, `run_difference`, `DifferenceSummary`
  re-exported from the package root (`__init__.py`, update `__all__`).
- Module docstring carries the algorithm detail (polarity flip, 128 contract,
  limit estimation).
