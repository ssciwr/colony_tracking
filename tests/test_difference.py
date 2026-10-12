"""Tests for the difference stage (colony_tracking.difference).

Seams under test (per the ticket): the uint8 mapping contract (128 = no
change; a known planted count through a known limit maps to the expected
byte), the polarity flip of frames acquired before the reference, the
per-well ``scale.txt`` contents, resume via the done marker (re-run skips
complete wells, files untouched), the dry run (writes nothing),
worker-count independence of the output bytes, and the ``wells=`` subset.

The mini-plate fixture from conftest provides the input trees. Every plate
here is noise-free and ring-less, so ``cur - ref`` is exact in float32 and
the expected bytes are known literals, not recomputed from the same rule.
The planted disks use a delta of 1024 counts on purpose: 127/1024 is a
power-of-two fraction, so ``128 + diff * 127 / limit`` is exact arithmetic.
"""

import logging

import numpy as np
import pytest
import tifffile

import colony_tracking
from colony_tracking import difference, io
from colony_tracking.difference import (
    DifferenceOptions,
    DifferenceSummary,
    run_difference,
)
from conftest import DiskSpec, build_mini_plate, disk_mask, frame_name


def _diff_name(frame: str) -> str:
    """The difference output file name for an input frame name."""
    return f"{frame[:-4]}_diff_uint8.tif"  # frame names end in ".tif"


def _options(raw, out, **kwargs) -> DifferenceOptions:
    return DifferenceOptions(raw_dir=raw, out_dir=out, **kwargs)


def _snapshot(root) -> dict[str, tuple[bytes, int]]:
    """Every file under root as {relative name: (bytes, mtime_ns)}."""
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree(root) -> dict[str, bytes]:
    """Every file under root as {relative name: bytes}."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- the uint8 contract ------------------------------------------------------


def test_zero_difference_is_exactly_128(tmp_path):
    raw = tmp_path / "raw"
    plate = build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=2, size=256,
        noise_sigma=0, rings={"-A001": None},
        disks={"-A001": [DiskSpec("growing", cy=128.0, cx=128.0, r0=20.0,
                                  delta=1024.0, first_frame=2)]},
    )
    names = plate.wells["-A001"]
    out = tmp_path / "out"
    summary = run_difference(_options(raw, out))
    assert summary.processed == 1

    well_dir = out / "-A001"
    # One output per input frame, 1:1, named after the frame stem.
    assert sorted(p.name for p in well_dir.iterdir() if p.suffix == ".tif") == [
        _diff_name(names[0]),
        _diff_name(names[1]),
    ]

    ref_img = tifffile.imread(well_dir / _diff_name(names[0]))
    # The reference frame itself: an all-128 frame (complete series).
    assert ref_img.dtype == np.uint8
    assert (ref_img == 128).all()

    cur_img = tifffile.imread(well_dir / _diff_name(names[1]))
    # The planted disk's exact painted mask: which pixels changed.
    mask = disk_mask(plate, "-A001", "growing", 2)
    # Zero difference (background, exact for a noise-free plate): exactly 128.
    assert (cur_img[~mask] == 128).all()


def test_known_count_through_known_limit_maps_to_expected_byte(tmp_path):
    raw = tmp_path / "raw"
    plate = build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=2, size=256,
        noise_sigma=0, rings={"-A001": None},
        disks={"-A001": [DiskSpec("growing", cy=128.0, cx=128.0, r0=20.0,
                                  delta=1024.0, first_frame=2)]},
    )
    out = tmp_path / "out"
    summary = run_difference(_options(raw, out))
    assert summary.processed == 1

    # The display limit: the only nonzero |diff| value is the planted 1024
    # counts, and the disk covers more than the 0.5% tail of the stride-8
    # grid (21 of 1024 samples), so the 99.5th percentile is exactly 1024.
    scale = io.read_scale(out / "-A001")
    assert scale["reference"] == plate.wells["-A001"][0]
    assert float(scale["display_limit_int16"]) == 1024.0

    cur_img = tifffile.imread(
        out / "-A001" / _diff_name(plate.wells["-A001"][1])
    )
    mask = disk_mask(plate, "-A001", "growing", 2)
    # 128 + (-1024) * 127/1024 = 1, exactly (127/1024 is a power-of-two
    # fraction); the background stays at 128.
    assert (cur_img[mask] == 1).all()
    assert (cur_img[~mask] == 128).all()


# --- polarity flip -----------------------------------------------------------


def test_frames_before_the_reference_are_polarity_flipped(tmp_path):
    raw = tmp_path / "raw"
    plate = build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=3, size=256,
        noise_sigma=0, rings={"-A001": None},
        disks={"-A001": [DiskSpec("growing", cy=128.0, cx=128.0, r0=15.0,
                                  growth=1.5, delta=1024.0, first_frame=1)]},
    )
    out = tmp_path / "out"
    summary = run_difference(_options(raw, out, reference=3))
    assert summary.processed == 1

    well_dir = out / "-A001"
    names = plate.wells["-A001"]
    f1 = tifffile.imread(well_dir / _diff_name(names[0]))
    f3 = tifffile.imread(well_dir / _diff_name(names[2]))

    # The reference frame itself is all 128.
    assert (f3 == 128).all()
    # scale.txt names the chosen reference frame.
    assert io.read_scale(well_dir)["reference"] == names[2]

    # The disk radii grow 15 -> 22.5 -> 33.75, so frame 1 shows the frame-3
    # disk's grown ring as polarity-flipped: bright (> 128) exactly where
    # colonies later grow, still 128 inside the already-present disk.
    mask1 = disk_mask(plate, "-A001", "growing", 1)
    mask3 = disk_mask(plate, "-A001", "growing", 3)
    grown = mask3 & ~mask1
    assert grown.any()
    assert (f1[grown] > 128).all()
    assert (f1[mask1] == 128).all()
    # A concrete pixel in the grown ring: diff = +1024 -> 128 + 127 = 255.
    assert f1[128, 153] == 255


# --- scale.txt ---------------------------------------------------------------


def test_scale_txt_records_reference_and_limit(tmp_path):
    raw = tmp_path / "raw"
    plate = build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=2, size=256,
        noise_sigma=0, rings={"-A001": None},
        disks={"-A001": [DiskSpec("growing", cy=128.0, cx=128.0, r0=20.0,
                                  delta=1024.0, first_frame=2)]},
    )
    out = tmp_path / "out"
    run_difference(_options(raw, out))

    lines = (out / "-A001" / "scale.txt").read_text().splitlines()
    assert lines == [
        f"reference\t{plate.wells['-A001'][0]}",
        "display_limit_int16\t1024.0",
    ]
    # The new writer writes exactly the two spec keys (the prototype's
    # flatfield/sigma lines are cut).
    assert [line.partition("\t")[0] for line in lines] == [
        "reference", "display_limit_int16",
    ]


# --- resume ------------------------------------------------------------------


def test_rerun_skips_complete_wells_and_touches_nothing(tmp_path):
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=2, size=256,
        noise_sigma=0, rings={"-A001": None},
        disks={"-A001": [DiskSpec("growing", cy=128.0, cx=128.0, r0=20.0,
                                  delta=1024.0, first_frame=2)]},
    )
    out = tmp_path / "out"
    first = run_difference(_options(raw, out))
    assert first.processed == 1 and first.skipped == 0
    assert io.is_done(out / "-A001")

    before = _snapshot(out)
    second = run_difference(_options(raw, out))
    # The complete well is skipped; every file keeps its bytes and mtime.
    assert second.processed == 0 and second.skipped == 1 and second.failed == 0
    assert _snapshot(out) == before

    # The done marker records the frame count.
    assert io.read_done(out / "-A001")["frames"] == 2


# --- dry run -----------------------------------------------------------------


def test_dry_run_writes_nothing_and_reports_plan(tmp_path, caplog):
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=3, size=256,
        noise_sigma=0, rings={"-A001": None},
        disks={"-A001": [DiskSpec("growing", cy=128.0, cx=128.0, r0=20.0,
                                  delta=1024.0, first_frame=2)]},
    )
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="colony_tracking"):
        summary = run_difference(_options(raw, out, dry_run=True))

    assert not out.exists()  # not even created
    assert summary.processed == 0 and summary.failed == 0

    # The plan: wells, frame counts, chosen reference.
    text = caplog.text
    assert "-A001" in text
    assert "3 frames" in text
    assert "LO001" in text  # the "first" reference of this well


# --- parallelism -------------------------------------------------------------


def test_worker_count_does_not_change_output_bytes(tmp_path):
    wells = ("-A001", "-B004")
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=wells, n_frames=3, size=128, noise_sigma=0,
        rings={w: None for w in wells},
        disks={
            w: [DiskSpec("growing", cy=64.0, cx=64.0, r0=10.0, growth=1.4,
                         delta=1024.0, first_frame=2)]
            for w in wells
        },
    )
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    s1 = run_difference(_options(raw, out1, n_workers=1))
    s2 = run_difference(_options(raw, out2, n_workers=4))
    assert s1.processed == s2.processed == 2
    assert _tree(out1) == _tree(out2)  # identical bytes, identical file set


# --- well selection ----------------------------------------------------------


def test_wells_subset_runs_only_named_wells(tmp_path):
    wells = ("-A001", "-B004")
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=wells, n_frames=2, size=128, noise_sigma=0,
        rings={w: None for w in wells},
        disks={w: [DiskSpec("growing", cy=64.0, cx=64.0, r0=10.0,
                            delta=1024.0, first_frame=2)] for w in wells},
    )
    out = tmp_path / "out"
    summary = run_difference(_options(raw, out, wells=["-B004"]))
    assert summary.processed == 1
    assert set(summary.wells) == {"-B004"}
    assert (out / "-B004").is_dir()
    assert not (out / "-A001").exists()


# --- public API and error handling --------------------------------------------


def test_public_api_reexported_from_package_root():
    from colony_tracking import (
        DifferenceOptions as RootOptions,
        DifferenceSummary as RootSummary,
        run_difference as root_run,
    )

    for name in ("DifferenceOptions", "DifferenceSummary", "run_difference"):
        assert name in colony_tracking.__all__
    assert root_run is run_difference
    assert RootOptions is DifferenceOptions
    assert RootSummary is DifferenceSummary


def test_missing_reference_frame_aborts_before_writing(tmp_path):
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=2, size=128,
        noise_sigma=0, rings={"-A001": None}, disks={"-A001": []},
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="-A001"):
        run_difference(_options(raw, out, reference=5))
    assert not out.exists()  # aborted up front, before anything is written


def test_no_frames_found_raises(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="no frames found"):
        run_difference(_options(raw, tmp_path / "out"))


def test_failed_well_is_recorded_and_others_succeed(tmp_path):
    wells = ("-A001", "-B004")
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=wells, n_frames=2, size=128, noise_sigma=0,
        rings={w: None for w in wells},
        disks={w: [DiskSpec("growing", cy=64.0, cx=64.0, r0=10.0,
                            delta=1024.0, first_frame=2)] for w in wells},
    )
    # Corrupt one frame of -B004 after the plate is built.
    bad = raw / frame_name("-B004", 2, 3_600_000)
    bad.write_bytes(b"not a tiff")

    out = tmp_path / "out"
    summary = run_difference(_options(raw, out))
    assert summary.processed == 1 and summary.failed == 1
    assert summary.wells["-B004"].status == "failed"
    assert summary.wells["-B004"].error
    # The failed well is not marked complete; the good one is.
    assert not io.is_done(out / "-B004")
    assert io.is_done(out / "-A001")


# --- degenerate series ---------------------------------------------------------


def test_single_frame_well_stays_a_complete_series(tmp_path):
    raw = tmp_path / "raw"
    build_mini_plate(
        raw, layout="flat", wells=("-A001",), n_frames=1, size=128,
        noise_sigma=0, rings={"-A001": None}, disks={"-A001": []},
    )
    out = tmp_path / "out"
    summary = run_difference(_options(raw, out))
    assert summary.processed == 1

    well_dir = out / "-A001"
    img = tifffile.imread(well_dir / _diff_name(frame_name("-A001", 1, 0)))
    assert (img == 128).all()
    # An unchanged well floors the display limit at 1.0.
    assert float(io.read_scale(well_dir)["display_limit_int16"]) == 1.0
    assert io.read_done(well_dir)["frames"] == 1
