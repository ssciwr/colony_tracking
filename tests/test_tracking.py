"""Tests for the pure tracking module (colony_tracking.tracking).

Seams under test (per the ticket): the tracking entry point
``track_colonies`` (synthetic label series in -> per-frame label images,
CSV-ready rows and done-marker counters out), the touching-segment
pre-merge (``premerge_touching``), and the transitive ancestor count
(``ancestor_counts``). Pure input -> output on small synthetic masks; no
fixture plate, no model.
"""

import numpy as np

from colony_tracking.tracking import track_colonies


def box_image(boxes, shape=(40, 40)):
    """A label image with value v painted at box (y0, x0, y1, x1) per item."""
    img = np.zeros(shape, np.uint32)
    for value, (y0, x0, y1, x1) in boxes.items():
        img[y0:y1, x0:x1] = value
    return img


def one_group(shape=(40, 40)):
    """A single reference group covering the whole (small) frame."""
    return np.ones(shape, np.uint32)


# --- the tracking entry point: outputs and identity rules --------------------


def test_labels_are_one_per_frame_with_colony_ids():
    groups = one_group()
    res = track_colonies(
        [(1, box_image({1: (5, 5, 15, 15)})),
         (2, box_image({1: (5, 5, 15, 15), 2: (20, 20, 30, 30)}))],
        groups,
    )
    assert len(res.labels) == 2  # 1:1 with the input frames
    assert res.labels[0].dtype == np.uint32
    assert (res.labels[0][5:15, 5:15] == 1).all()
    assert (res.labels[1][20:30, 20:30] == 2).all()


def test_rows_carry_the_canonical_columns():
    res = track_colonies([(1, box_image({1: (5, 5, 15, 15)}))], one_group())
    (row,) = res.rows
    assert row["frame"] == 1  # the frame NUMBER, not the series position
    assert row["colony_id"] == 1
    assert row["group_id"] == 1
    assert row["merged_with"] == ""
    assert row["pre_existing"] == 0
    assert row["area_px"] == 100
    assert row["centroid_y"] == 9.5 and row["centroid_x"] == 9.5


def test_counters_carry_the_done_marker_keys():
    res = track_colonies([(1, box_image({1: (5, 5, 15, 15)}))], one_group())
    assert set(res.counters) == {
        "frames", "groups", "colonies", "pre_existing_colonies",
        "merge_events", "forked_segments", "discarded_instances",
    }
    assert res.counters == {
        "frames": 1, "groups": 1, "colonies": 1, "pre_existing_colonies": 0,
        "merge_events": 0, "forked_segments": 0, "discarded_instances": 0,
    }
    # the caller's filtering stage owns the discard count; it is passed through
    res = track_colonies(
        [(1, box_image({1: (5, 5, 15, 15)}))], one_group(), discarded_instances=7,
    )
    assert res.counters["discarded_instances"] == 7


def test_iou_continuation_keeps_the_id():
    groups = one_group()
    # identical mask between frames: IoU 1.0 continues the colony
    res = track_colonies(
        [(10, box_image({1: (5, 5, 15, 15)})),
         (11, box_image({1: (5, 5, 15, 15)}))],
        groups,
    )
    assert [r["colony_id"] for r in res.rows] == [1, 1]
    # shifted 5 of 10 columns: IoU 50/150 = 1/3, exactly at MATCH_IOU -> continues
    res = track_colonies(
        [(10, box_image({1: (5, 5, 15, 15)})),
         (11, box_image({1: (5, 10, 15, 20)}))],
        groups,
    )
    assert [r["colony_id"] for r in res.rows] == [1, 1]


def test_iou_below_threshold_does_not_continue():
    groups = one_group()
    # shifted 6 of 10 columns: IoU 40/160 = 0.25 < MATCH_IOU -> no continuation
    res = track_colonies(
        [(10, box_image({1: (5, 5, 15, 15)})),
         (11, box_image({1: (5, 11, 15, 21)}))],
        groups,
    )
    assert [r["colony_id"] for r in res.rows] == [1, 2]


def test_unmatched_segment_starts_a_new_id():
    groups = one_group()
    res = track_colonies(
        [(1, box_image({1: (5, 5, 15, 15)})),
         (2, box_image({1: (25, 25, 35, 35)}))],  # disjoint from colony 1
        groups,
    )
    assert [r["colony_id"] for r in res.rows] == [1, 2]
    assert res.counters["colonies"] == 2


def test_segments_are_pooled_under_their_dominant_group():
    groups = box_image({1: (0, 0, 40, 20), 2: (0, 20, 40, 40)})
    res = track_colonies(
        [(1, box_image({1: (5, 2, 15, 18), 2: (20, 22, 30, 38)}))], groups,
    )
    # groups are processed by centre of mass (top-left first), so the colony
    # of group 1 gets the lower ID and rows carry the colony's group
    assert [(r["colony_id"], r["group_id"]) for r in res.rows] == [(1, 1), (2, 2)]