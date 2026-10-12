"""Tests for the segment stage's model-free parts.

Seams under test (per the ticket): the pure helpers of
``colony_tracking.segmentation`` — the drift-gate decision, the growth-mask
construction, and the mask-overlap filter — plus ``run_segmentation`` in its
model-free paths (the up-front rejection of a difference reference equal to
the final frame, and the re-ID mode over a synthetic label tree). No model,
no GPU, no network: the model path itself is never entered.
"""

import sys

import numpy as np
import pytest

from colony_tracking.segmentation import (
    build_growth_mask,
    drift_gate_limit,
    drift_gate_skips,
    filter_to_mask,
)


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


# --- the module import stays torch/micro-sam free --------------------------------


def test_importing_the_module_does_not_pull_torch_or_micro_sam(monkeypatch):
    import importlib

    monkeypatch.delitem(sys.modules, "colony_tracking.segmentation", raising=False)
    importlib.import_module("colony_tracking.segmentation")
    assert not any(name.startswith("micro_sam") for name in sys.modules)
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)