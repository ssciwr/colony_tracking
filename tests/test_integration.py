"""The one-time real-plate integration check (spec: Testing Decisions →
Integration test and prior art; Further Notes → One-time verification
protocol).

Excluded from the default run by ``addopts = "-m 'not integration'"``; run
explicitly with ``uv run pytest -m integration``. It replaces ticket 01's
placeholder. Two prerequisites, each skipped with a clear message when
absent so CI and dataless checkouts are unaffected:

- the real data root via ``COLONY_TRACKING_DATA_ROOT`` — the directory
  holding the raw plate (a ``*_Plate*`` subdirectory, or the frames
  directly for a flat tree) and the prototype's published outputs
  (``seg_growth_diff_LO005/``, ``analysis/``, ...). The prototype trees are
  only read; the test never writes inside the data root;
- a CUDA GPU (the segment stage's micro-sam inference path).

The reference frame is 5 (the one-time protocol's standard chain); set
``COLONY_TRACKING_REFERENCE`` to run another documented reference.

The test runs ``run_pipeline`` — the standard chain, in-process — into a
scratch out root under tmp, then asserts the one-time-check criteria:

1. the pipeline completes and every well's done marker is present (96
   wells on the real plate);
2. colony-count sanity against the documented numbers, within ±5%
   (``TOLERANCE`` — a fresh PyPI torch may segment slightly differently, an
   explicit user decision): difference root reference 5 → 369 colonies /
   354 merge events; reference 1 → 326 colonies / 9 forks. The counters
   are the per-well ``_done.txt`` sums of the new segmentation root (the
   documented numbers come from the prototype's full-plate runs of the
   same counters);
3. column equality for rows present in both the new and the prototype
   trees, joined on well + colony identity (+ frame): new ``frame`` ==
   old ``lo`` (per-well ``colonies.csv``), new ``ancestors`` == old
   ``n_merged`` (``trajectories.csv``), new ``max_ancestors`` == old
   ``max_n_merged`` (``growth_rates.csv``). Legacy column names are mapped
   by the shared io readers, so both trees are read canonically.

The whole run takes roughly half an hour on the laptop GPU (96 wells × 19
frames of inference).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from colony_tracking import PipelineConfig, io, run_pipeline

#: The data root (raw plate + the prototype's published outputs).
DATA_ENV = "COLONY_TRACKING_DATA_ROOT"
#: Optional reference-frame override (default 5, the protocol's standard).
REFERENCE_ENV = "COLONY_TRACKING_REFERENCE"

#: Documented one-time-check counters per reference frame (spec: Further
#: Notes → One-time verification protocol), keyed by done-marker name.
EXPECTED = {
    5: {"colonies": 369, "merge_events": 354},
    1: {"colonies": 326, "forked_segments": 9},
}
#: Relative tolerance of the sanity check (fresh PyPI torch may segment
#: slightly differently than the prototype's conda torch).
TOLERANCE = 0.05
#: Wells on the real plate (the protocol's "completes all 96 wells").
EXPECTED_WELLS = 96


def _cuda_available() -> tuple[bool, str]:
    """Whether a CUDA GPU is usable, with the reason when it is not."""
    try:
        import torch
    except Exception as exc:  # a broken torch install must not error the run
        return False, f"torch is not importable: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False — no usable CUDA GPU"
    return True, ""


def _raw_plate(data_root: Path) -> Path:
    """The raw plate under the data root: the root itself when it holds the
    frames directly (flat layout), else its single ``*_Plate*``
    subdirectory."""
    if io.discover_wells(data_root):
        return data_root
    plates = [p for p in sorted(data_root.glob("*Plate*")) if p.is_dir()]
    if len(plates) == 1:
        return plates[0]
    raise FileNotFoundError(
        f"{data_root}: no raw plate found (expected the frames directly, "
        "or exactly one *_Plate* subdirectory)"
    )


def _marker_sums(seg_root: Path) -> dict[str, int]:
    """The per-well done-marker counters of a segmentation tree, summed."""
    sums: dict[str, int] = {}
    for path in sorted(seg_root.iterdir()):
        if not (path.is_dir() and io.parse_well(path.name)):
            continue
        for key, value in io.read_done(path).items():
            try:
                sums[key] = sums.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue
    return sums


def _index(rows: list[dict[str, str]],
           int_fields: tuple[str, ...]) -> dict[tuple, dict[str, str]]:
    """``{key: row}`` for CSV rows; the well stays text, the other key
    fields are read as ints (the prototype's ``lo`` column is zero-padded)."""
    index: dict[tuple, dict[str, str]] = {}
    for row in rows:
        index[(row["well"], *(int(row[f]) for f in int_fields))] = row
    return index


def _value(row: dict[str, str], field: str) -> int | str:
    """A CSV value as an int when it parses (else the stripped text)."""
    text = (row.get(field) or "").strip()
    try:
        return int(text)
    except ValueError:
        return text


def _assert_join(new_rows: list[dict[str, str]],
                 old_rows: list[dict[str, str]], int_fields: tuple[str, ...],
                 field: str, label: str) -> int:
    """Assert the renamed column carries the prototype's value for every row
    present in both trees; returns the number of joined rows."""
    new_index = _index(new_rows, int_fields)
    old_index = _index(old_rows, int_fields)
    common = sorted(set(new_index) & set(old_index))
    assert common, f"{label}: no rows present in both the new and the prototype tree"
    for key in common:
        got, want = _value(new_index[key], field), _value(old_index[key], field)
        assert got == want, (
            f"{label}: new {field}={got!r} != old {field}={want!r} "
            f"for {key[0]} colony {key[1]}" +
            (f" frame {key[2]}" if len(key) > 2 else "")
        )
    return len(common)


def _read_colonies_tree(root: Path) -> list[dict[str, str]]:
    """Every well's colonies.csv of a segmentation tree, canonically."""
    rows: list[dict[str, str]] = []
    for well_dir in sorted(root.iterdir()):
        csv_path = well_dir / "colonies.csv"
        if well_dir.is_dir() and io.parse_well(well_dir.name) and csv_path.exists():
            rows.extend(io.read_colonies(csv_path))
    return rows


@pytest.mark.integration
def test_full_pipeline_on_real_plate(tmp_path: Path):
    """The standard chain on the real plate (see the module docstring)."""
    data_root = os.environ.get(DATA_ENV, "")
    if not data_root:
        pytest.skip(f"{DATA_ENV} not set — the real plate and the "
                    "prototype's trees are unavailable")
    data_root = Path(data_root)
    if not data_root.is_dir():
        pytest.skip(f"{DATA_ENV}={data_root} is not a directory")
    usable, reason = _cuda_available()
    if not usable:
        pytest.skip(reason)

    reference_text = os.environ.get(REFERENCE_ENV, "5")
    try:
        reference = int(reference_text)
    except ValueError:
        pytest.fail(f"{REFERENCE_ENV}={reference_text!r} is not a frame number")
    if reference not in EXPECTED:
        pytest.fail(
            f"no documented sanity numbers for reference {reference} "
            f"(documented: {sorted(EXPECTED)})"
        )

    # 1. the standard chain into a scratch out root (never inside the data
    # root; the prototype trees are only read)
    out_root = tmp_path / "one-time-check"
    config = PipelineConfig(
        raw_dir=_raw_plate(data_root), out_root=out_root, reference=reference,
    )
    summary = run_pipeline(config)
    assert summary.status == "ok", (
        f"pipeline failed at the {summary.failed_stage} stage "
        "(see the captured log for the stage's error)"
    )
    _, seg_options, analysis_options = config.resolve_stages()
    seg_root = seg_options.out_dir
    analysis_root = analysis_options.out_root / seg_root.name

    # every well complete (the protocol's "all 96 wells with 0 failures")
    well_dirs = [p for p in sorted(seg_root.iterdir())
                 if p.is_dir() and io.parse_well(p.name)]
    undone = [p.name for p in well_dirs if not io.is_done(p)]
    assert not undone, f"wells without a done marker: {', '.join(undone)}"
    assert len(well_dirs) == EXPECTED_WELLS, (
        f"{len(well_dirs)} wells segmented — the plate has {EXPECTED_WELLS}"
    )

    # 2. colony-count sanity against the documented numbers (±5%)
    sums = _marker_sums(seg_root)
    for counter, want in sorted(EXPECTED[reference].items()):
        got = sums.get(counter, 0)
        assert abs(got - want) <= want * TOLERANCE, (
            f"reference {reference}: {counter}={got} summed over all wells — "
            f"documented {want} (tolerance ±{TOLERANCE:.0%})"
        )

    # 3. column equality for rows present in both trees (new tree vs the
    # prototype's published outputs, read-only under the data root)
    proto_seg = data_root / seg_root.name
    if not proto_seg.is_dir():
        pytest.skip(
            f"no prototype tree {seg_root.name}/ under {data_root} — "
            "column equality cannot be checked"
        )
    joined = _assert_join(
        _read_colonies_tree(seg_root), _read_colonies_tree(proto_seg),
        ("colony_id", "frame"), "frame", "colonies.csv",
    )
    proto_analysis = data_root / "analysis" / seg_root.name
    if not proto_analysis.is_dir():
        pytest.skip(
            f"no prototype analysis outputs under {proto_analysis.parent} — "
            "trajectories/growth-rates equality unchecked"
        )
    trajectories = analysis_root / "trajectories.csv"
    growth_rates = analysis_root / "growth_rates.csv"
    assert trajectories.exists() and growth_rates.exists()
    _assert_join(
        io.read_csv_rows(trajectories),
        io.read_csv_rows(proto_analysis / "trajectories.csv"),
        ("colony_id", "frame"), "ancestors", "trajectories.csv",
    )
    _assert_join(
        io.read_csv_rows(growth_rates),
        io.read_csv_rows(proto_analysis / "growth_rates.csv"),
        ("colony_id",), "max_ancestors", "growth_rates.csv",
    )
    assert joined > 0  # the colonies join is the rename check's backbone