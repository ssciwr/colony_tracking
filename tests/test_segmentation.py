"""Tests for the segment stage's model-free parts.

Seams under test (per the ticket): the pure helpers of
``colony_tracking.segmentation`` — the drift-gate decision, the growth-mask
construction, and the mask-overlap filter — plus ``run_segmentation`` in its
model-free paths (the up-front rejection of a difference reference equal to
the final frame, and the re-ID mode over a synthetic label tree). No model,
no GPU, no network: the model path itself is never entered.
"""

import logging
import sys

import numpy as np
import pytest
import tifffile

from colony_tracking import io
from colony_tracking._options import SegmentationOptions
from colony_tracking.segmentation import (
    build_growth_mask,
    drift_gate_limit,
    drift_gate_skips,
    filter_to_mask,
    run_segmentation,
)
from conftest import T_STEP_MS, build_label_series, build_mini_plate, frame_name


def _model_stack_pulled(modules_before: set[str]) -> set[str]:
    """The model-stack modules pulled since ``modules_before`` — the delta:
    torch, its submodules, and micro-sam. (The delta, because other tests
    may legitimately already have torch in sys.modules.)"""
    return {
        name
        for name in set(sys.modules) - modules_before
        if name == "torch" or name.startswith("torch.")
        or name.startswith("micro_sam")
    }


# --- the drift gate -----------------------------------------------------------


def test_drift_gate_limit_is_ratio_times_reference_floored():
    assert drift_gate_limit(0.1, 1.5) == pytest.approx(0.15)
    # the 0.05 floor keeps the gate meaningful for wells with tiny colonies
    assert drift_gate_limit(0.01, 1.5) == pytest.approx(0.05)


def test_drift_gate_skips_frames_above_the_gate_only():
    # limit = max(1.5 * 0.1, 0.05) = 0.15: strictly above skips
    assert drift_gate_skips(0.2, 0.1, max_fg_ratio=1.5) is True
    assert drift_gate_skips(0.15, 0.1, max_fg_ratio=1.5) is False
    assert drift_gate_skips(0.14, 0.1, max_fg_ratio=1.5) is False


def test_drift_gate_ratio_zero_disables_the_gate():
    # a drift-dominated frame is only skipped when the gate is on
    assert drift_gate_skips(0.9, 0.01, max_fg_ratio=0.0) is False


def test_drift_gate_floored_at_five_percent():
    # tiny reference fraction: the floor dominates the ratio term
    assert drift_gate_skips(0.06, 0.01, max_fg_ratio=1.5) is True
    assert drift_gate_skips(0.05, 0.01, max_fg_ratio=1.5) is False
    assert drift_gate_skips(0.04, 0.01, max_fg_ratio=1.5) is False
    # a zero reference fraction still gets the floor
    assert drift_gate_skips(0.06, 0.0, max_fg_ratio=1.5) is True


# --- growth-mask construction ---------------------------------------------------


def _planted_final_instances() -> np.ndarray:
    """A synthetic final-frame instance segmentation.

    Two solid 20x20 colony instances (disjoint -> two groups), one tiny
    junk instance, and one elongated low-solidity junk instance (the kind
    of well-ring debris the filters exist for).
    """
    img = np.zeros((60, 60), np.uint32)
    img[10:30, 5:25] = 1  # solid 400 px box
    img[10:30, 35:55] = 2  # solid 400 px box, disjoint from instance 1
    img[40:44, 5:9] = 3  # tiny junk: 16 px
    img[40:60, 35:37] = 4  # elongated junk: a 2-px-wide L, 76 px ...
    img[58:60, 35:55] = 4  # ... but solidity ~0.4
    return img


def test_growth_mask_drops_junk_and_splits_the_union_into_groups():
    instances = _planted_final_instances()
    groups, mask = build_growth_mask(
        instances, min_size=50, min_solidity=0.7, mask_margin=3
    )
    assert groups is not None
    assert int(groups.max()) == 2  # the two survivors form two groups
    assert (groups[10:30, 5:25] == 1).all()
    assert (groups[10:30, 35:55] == 2).all()
    # both junk instances are gone (tiny one by size, L by solidity)
    assert groups[42, 7] == 0
    assert groups[50, 36] == 0


def test_growth_mask_is_the_groups_dilated_by_the_margin():
    instances = _planted_final_instances()
    _, mask = build_growth_mask(
        instances, min_size=50, min_solidity=0.7, mask_margin=3
    )
    # box 1 spans rows 10..29 / cols 5..24: dilated rows 7..32 / cols 2..27
    assert mask[7, 15] and not mask[6, 15]
    assert mask[20, 27] and not mask[20, 28]
    # box 2 spans cols 35..54: dilated cols 32..57 — the gap between the
    # dilated boxes stays outside the mask
    assert mask[20, 32] and not mask[20, 31]
    assert (mask[10:30, 5:25]).all() and (mask[10:30, 35:55]).all()
    # the junk pixels stay outside the dilated mask
    assert not mask[42, 7]
    assert not mask[50, 36]


def test_growth_mask_with_no_survivor_is_none_and_empty():
    img = np.zeros((30, 30), np.uint32)
    img[5:8, 5:8] = 1  # 9 px: below min_size
    groups, mask = build_growth_mask(
        img, min_size=50, min_solidity=0.7, mask_margin=3
    )
    assert groups is None
    assert not mask.any()


# --- the mask-overlap filter ----------------------------------------------------


def test_filter_keeps_instances_at_half_mask_overlap_and_drops_below():
    shape = (20, 60)
    mask = np.zeros(shape, bool)
    mask[:, :30] = True  # the growth mask: the left half
    groups = np.ones(shape, np.uint32)  # a single group over everything
    instances = np.zeros(shape, np.uint32)
    instances[2:12, 20:40] = 1  # 10 of 20 columns inside -> exactly 0.5: kept
    instances[14:19, 22:42] = 2  # 8 of 20 columns inside -> 0.4: dropped
    out, discarded = filter_to_mask(instances, mask, groups, min_overlap=0.5)
    assert discarded == 1
    # a kept instance keeps its full extent, relabelled 1..k
    assert (out[2:12, 20:40] == 1).all()
    assert not out[14:19, 22:42].any()


def test_filter_drops_instances_spanning_two_groups():
    shape = (40, 80)
    groups = np.zeros(shape, np.uint32)
    groups[:, 10:30] = 1
    groups[:, 40:60] = 2
    mask = np.zeros(shape, bool)
    mask[:, 5:65] = True  # the dilated groups; the margin lies outside them
    instances = np.zeros(shape, np.uint32)
    # straddles group 1 (6 cols), the margin (10 cols) and group 2 (8 cols):
    # no single group holds >= 0.5 of its pixels -> dropped
    instances[10:20, 24:48] = 1
    # control: fully inside group 1 -> kept
    instances[25:35, 12:28] = 2
    out, discarded = filter_to_mask(instances, mask, groups, min_overlap=0.5)
    assert discarded == 1
    assert (out[25:35, 12:28] == 1).all()  # the control is local ID 1
    assert not out[10:20, 24:48].any()


def test_filter_without_groups_applies_the_mask_rule_only():
    shape = (20, 40)
    mask = np.zeros(shape, bool)
    mask[:, :20] = True
    instances = np.zeros(shape, np.uint32)
    instances[2:12, 10:30] = 1  # 10 of 20 columns inside -> kept
    out, discarded = filter_to_mask(instances, mask, None, min_overlap=0.5)
    assert discarded == 0
    assert (out[2:12, 10:30] == 1).all()


# --- run_segmentation: validation and the up-front reference rejection ----------


def _diff_name(well: str, lo: int, t_ms: int) -> str:
    """The canonical difference-frame file name for one frame."""
    return frame_name(well, lo, t_ms)[:-4] + "__diff_uint8.tif"


def _write_diff_tree(diff_dir, well, los):
    """A minimal difference well: uint8 frames as written (128 = no change)."""
    well_dir = diff_dir / well
    well_dir.mkdir(parents=True)
    names = []
    for lo in los:
        name = _diff_name(well, lo, (lo - 1) * 3_600_000)
        tifffile.imwrite(well_dir / name, np.full((8, 8), 128, np.uint8))
        names.append(name)
    return names


def test_run_segmentation_requires_the_input_directory(tmp_path):
    with pytest.raises(ValueError, match="raw_dir is required"):
        run_segmentation(SegmentationOptions(out_dir=tmp_path / "out"))
    with pytest.raises(ValueError, match="diff_dir is required"):
        run_segmentation(
            SegmentationOptions(input="diff", out_dir=tmp_path / "out")
        )


def test_run_segmentation_rejects_diff_reference_equal_to_final_frame(
    tmp_path, caplog
):
    diff_dir = tmp_path / "difference_LO003"
    names = _write_diff_tree(diff_dir, "-A001", los=(1, 2, 3))
    # the difference reference is the final frame itself: the growth mask
    # would be empty
    io.write_scale(diff_dir / "-A001", reference=names[-1], display_limit=100.0)
    out_dir = tmp_path / "seg"

    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_segmentation(
            SegmentationOptions(input="diff", diff_dir=diff_dir, out_dir=out_dir)
        )

    assert (summary.processed, summary.skipped, summary.failed) == (0, 1, 0)
    assert "growth mask would be empty" in caplog.text
    # no outputs: the well directory is not even created
    assert not (out_dir / "-A001").exists()


def test_run_segmentation_skips_wells_without_a_scale_reference(tmp_path, caplog):
    diff_dir = tmp_path / "difference_LO003"
    _write_diff_tree(diff_dir, "-A001", los=(1, 2, 3))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_segmentation(
            SegmentationOptions(input="diff", diff_dir=diff_dir,
                                out_dir=tmp_path / "seg")
        )
    assert (summary.processed, summary.skipped, summary.failed) == (0, 1, 0)
    assert "no scale.txt with a reference line" in caplog.text
    assert not (tmp_path / "seg" / "-A001").exists()


def test_run_segmentation_resumes_via_the_done_marker_without_the_model(tmp_path):
    plate = build_mini_plate(
        tmp_path / "raw", layout="per_well", wells=("-A001",), n_frames=2,
        size=32, rings={"-A001": None}, disks={"-A001": []}, noise_sigma=0.0,
    )
    out = tmp_path / "seg"
    well_dir = out / "-A001"
    well_dir.mkdir(parents=True)
    io.write_done(well_dir, dict.fromkeys(io.SEGMENT_DONE_KEYS, 0))

    modules_before = set(sys.modules)
    summary = run_segmentation(
        SegmentationOptions(input="raw", raw_dir=plate.root, out_dir=out)
    )

    assert (summary.processed, summary.skipped, summary.failed) == (0, 1, 0)
    # nothing to do: the model was never even loaded (no torch either)
    assert not _model_stack_pulled(modules_before)


def test_run_segmentation_rejects_missing_requested_wells(tmp_path):
    diff_dir = tmp_path / "difference_LO001"
    _write_diff_tree(diff_dir, "-A001", los=(1, 2, 3))
    with pytest.raises(ValueError, match="-Z009"):
        run_segmentation(
            SegmentationOptions(
                input="diff", diff_dir=diff_dir, out_dir=tmp_path / "seg",
                wells=["-Z009"],
            )
        )


# --- re-ID end-to-end on a synthetic label tree -----------------------------------

WELLS = ("-A001", "-B002", "-C003")


def _seed_series(well_dir, well, names, labels, ref=None):
    """Write a completed-looking well output tree (labels, ref, CSV, marker).

    The initial label values are deliberately not colony IDs (and one label
    value is shared by two disjoint objects) — re-ID must re-derive the
    segments and the identity, not copy anything.
    """
    if ref is None:
        ref = np.ones(labels[0].shape, np.uint32)
    build_label_series(well_dir, labels, well=well, names=names)
    io.write_label_image(well_dir / "ref__labels.tif", ref)
    io.write_colonies(well_dir / "colonies.csv", [])
    io.write_done(well_dir, dict.fromkeys(io.SEGMENT_DONE_KEYS, 0))


def _labels_a():
    """Well A, frames 1..5 (40x60): a merge, a newcomer, frame 1 below the cutoff."""
    shape = (40, 60)
    f1 = np.zeros(shape, np.uint32)
    f1[5:15, 5:15] = 1
    f2 = np.zeros(shape, np.uint32)
    f2[5:15, 5:15] = 1  # one value, two components: A and B
    f2[5:15, 25:35] = 1
    f3 = f2.copy()
    f4 = f2.copy()
    f4[25:35, 5:15] = 2  # colony C appears
    f5 = np.zeros(shape, np.uint32)
    f5[5:15, 5:35] = 5  # A and B grown together: one blob
    f5[25:35, 5:15] = 2
    return [f1, f2, f3, f4, f5]


def _labels_b():
    """Well B, frames 1,3,4,5 — the series lacks frame 2."""
    shape = (40, 60)
    f1 = np.zeros(shape, np.uint32)
    f1[10:20, 10:20] = 1
    f3 = np.zeros(shape, np.uint32)
    f3[10:20, 10:20] = 1  # D and E, one value, two components
    f3[10:20, 30:40] = 1
    f4 = f3.copy()
    f5 = np.zeros(shape, np.uint32)
    f5[10:20, 10:40] = 5  # D and E grown together
    return [f1, f3, f4, f5]


def _labels_c():
    """Well C, frames 1..5: nothing was ever segmented (empty reference)."""
    return [np.zeros((40, 60), np.uint32) for _ in range(5)]


def test_reid_retracks_and_rewrites_outputs(tmp_path):
    plate = build_mini_plate(
        tmp_path / "raw", layout="per_well", wells=WELLS, n_frames=5, size=64,
        rings={w: None for w in WELLS}, disks={w: [] for w in WELLS},
        noise_sigma=0.0,
    )
    out = tmp_path / "seg"
    names_a = plate.wells["-A001"]
    _seed_series(out / "-A001", "-A001", names_a, _labels_a())
    # well B: frame 2 is missing from the input tree and the label series
    names_b = plate.wells["-B002"]
    (plate.root / "-B002" / names_b[1]).unlink()
    _seed_series(out / "-B002", "-B002", names_b[:1] + names_b[2:], _labels_b())
    # well C: an empty reference (the growth mask found nothing)
    names_c = plate.wells["-C003"]
    _seed_series(out / "-C003", "-C003", names_c, _labels_c(), ref=np.zeros((40, 60), np.uint32))

    summary = run_segmentation(
        SegmentationOptions(
            input="raw", raw_dir=plate.root, out_dir=out,
            exclude_before=2, reid_only=True,
        )
    )

    modules_before = set(sys.modules)
    summary = run_segmentation(
        SegmentationOptions(
            input="raw", raw_dir=plate.root, out_dir=out,
            exclude_before=2, reid_only=True,
        )
    )

    assert (summary.processed, summary.skipped, summary.failed) == (2, 1, 0)
    assert summary.gate_skipped_frames == 0
    # re-ID never touches the model
    assert not _model_stack_pulled(modules_before)

    # -- well A: IDs continue, the merge is recorded, frame 1 is blanked -----
    a_dir = out / "-A001"
    for name in names_a:  # outputs stay 1:1 with the input frames
        assert (a_dir / (name[:-4] + "__labels.tif")).exists()
    labels = [io.read_label_image(a_dir / (n[:-4] + "__labels.tif"))
              for n in names_a]
    assert not labels[0].any()  # frame 1 < exclude_before: blanked
    assert (labels[1][5:15, 5:15] == 1).all()  # colony 1 continues
    assert (labels[1][5:15, 25:35] == 2).all()  # colony 2
    assert (labels[3][25:35, 5:15] == 3).all()  # colony 3 from frame 4
    assert (labels[4][5:15, 5:35] == 1).all()  # the merge product keeps ID 1
    assert (labels[4][25:35, 5:15] == 3).all()

    rows = io.read_colonies(a_dir / "colonies.csv")
    assert [(r["frame"], r["colony_id"], r["merged_with"], r["pre_existing"])
            for r in rows] == [
        ("2", "1", "", "1"), ("2", "2", "", "1"),  # first detected at 2 <= N
        ("3", "1", "", "1"), ("3", "2", "", "1"),
        ("4", "1", "", "1"), ("4", "2", "", "1"), ("4", "3", "", "0"),
        ("5", "1", "2", "1"),  # the merge: colony 2 absorbed
        ("5", "3", "", "0"),
    ]
    assert rows[0]["well"] == "-A001"
    assert rows[0]["group_id"] == "1"
    assert rows[0]["t_seconds"] == str(T_STEP_MS)  # LO002's T field (ms)
    assert rows[0]["image"] == names_a[1]
    assert io.read_done(a_dir) == {
        "frames": 5, "groups": 1, "colonies": 1, "pre_existing_colonies": 2,
        "merge_events": 1, "forked_segments": 0, "discarded_instances": 0,
    }

    # -- well B: the series lacks frame 2 (N need not exist) ------------------
    b_dir = out / "-B002"
    kept_names = names_b[:1] + names_b[2:]
    for name in kept_names:
        assert (b_dir / (name[:-4] + "__labels.tif")).exists()
    assert not (b_dir / (names_b[1][:-4] + "__labels.tif")).exists()
    labels_b = [io.read_label_image(b_dir / (n[:-4] + "__labels.tif"))
                for n in kept_names]
    assert not labels_b[0].any()  # frame 1 < exclude_before: blanked
    assert (labels_b[1][10:20, 10:20] == 1).all()  # first seen at 3: new growth
    assert (labels_b[1][10:20, 30:40] == 2).all()
    assert (labels_b[3][10:20, 10:40] == 1).all()  # merge product keeps ID 1

    rows_b = io.read_colonies(b_dir / "colonies.csv")
    assert [(r["frame"], r["colony_id"], r["merged_with"], r["pre_existing"])
            for r in rows_b] == [
        ("3", "1", "", "0"), ("3", "2", "", "0"),
        ("4", "1", "", "0"), ("4", "2", "", "0"),
        ("5", "1", "2", "0"),  # the merge: colony 2 absorbed
    ]
    assert io.read_done(b_dir) == {
        "frames": 4, "groups": 1, "colonies": 2, "pre_existing_colonies": 0,
        "merge_events": 1, "forked_segments": 0, "discarded_instances": 0,
    }

    # -- well C: empty reference — nothing to re-assign, marker rewritten -----
    assert io.read_done(out / "-C003") == {
        "frames": 5, "groups": 0, "colonies": 0, "pre_existing_colonies": 0,
        "merge_events": 0, "forked_segments": 0, "discarded_instances": 0,
    }
    assert io.read_colonies(out / "-C003" / "colonies.csv") == []


# --- the module import stays torch/micro-sam free --------------------------------


def test_importing_the_module_does_not_pull_torch_or_micro_sam(monkeypatch):
    import importlib

    monkeypatch.delitem(sys.modules, "colony_tracking.segmentation", raising=False)
    modules_before = set(sys.modules)
    importlib.import_module("colony_tracking.segmentation")
    # The module's own import chain pulls no model stack.
    assert not _model_stack_pulled(modules_before)
