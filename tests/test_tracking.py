"""Tests for the pure tracking module (colony_tracking.tracking).

Seams under test (per the ticket): the tracking entry point
``track_colonies`` (synthetic label series in -> per-frame label images,
CSV-ready rows and done-marker counters out), the touching-segment
pre-merge (``premerge_touching``), and the transitive ancestor count
(``ancestor_counts``). Pure input -> output on small synthetic masks; no
fixture plate, no model.
"""

import numpy as np

from colony_tracking.tracking import (
    ancestor_counts,
    extract_segments,
    parse_merged,
    premerge_touching,
    track_colonies,
)


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


# --- absorb/merge and ancestors ----------------------------------------------


def test_absorb_merge_chain_and_transitive_ancestors():
    groups = one_group((40, 64))
    frames = [
        # frame 1: colony A (300 px) -> ID 1
        (1, box_image({1: (10, 2, 20, 32)}, shape=(40, 64))),
        # frame 2: A continues; B (90 px) -> ID 2 and C (72 px) -> ID 3 appear
        (2, box_image({1: (10, 2, 20, 32), 2: (10, 38, 20, 47), 3: (12, 53, 20, 62)},
                      shape=(40, 64))),
        # frame 3: a B+C blob continues colony 2 and absorbs colony 3
        (3, box_image({1: (10, 2, 20, 32), 4: (10, 38, 20, 62)}, shape=(40, 64))),
        # frame 4: one A+B+C blob: colony 1 continues (largest mask), absorbs 2 and 3
        (4, box_image({5: (10, 2, 20, 62)}, shape=(40, 64))),
    ]
    res = track_colonies(frames, groups)
    assert [r["colony_id"] for r in res.rows] == [1, 1, 2, 3, 1, 2, 1]
    assert [r["merged_with"] for r in res.rows] == ["", "", "", "", "", "3", "2;3"]
    assert res.counters["merge_events"] == 2
    # A absorbed B, which had absorbed C: the A+B segment ends at 3 ancestors
    assert ancestor_counts(res.rows) == [1, 1, 1, 1, 1, 2, 3]
    # the survivor keeps its own ID on the merged blob
    assert (res.labels[3][10:20, 2:62] == 1).all()


def test_absorbed_colony_resumes_its_id_after_separating():
    groups = one_group((40, 64))
    frames = [
        (1, box_image({1: (10, 2, 20, 22)}, shape=(40, 64))),  # A -> ID 1
        (2, box_image({1: (10, 2, 20, 22), 2: (10, 30, 20, 42)},
                      shape=(40, 64))),  # B -> ID 2
        # frame 3: the A+B blob continues colony 1 and absorbs colony 2
        (3, box_image({3: (10, 2, 20, 42)}, shape=(40, 64))),
        # frame 4: separated again: each piece matches its own colony
        (4, box_image({1: (10, 2, 20, 22), 2: (10, 30, 20, 42)},
                      shape=(40, 64))),
    ]
    res = track_colonies(frames, groups)
    assert [r["colony_id"] for r in res.rows] == [1, 1, 2, 1, 1, 2]
    assert res.rows[3]["merged_with"] == "2"  # frame 3: colony 2 absorbed
    assert res.counters["merge_events"] == 1
    # no third colony was started: colony 2 resumed its own ID
    assert res.counters["colonies"] == 2
    assert (res.labels[3][10:20, 30:42] == 2).all()


def test_ancestor_counts_identity_direct_transitive_independent():
    # rows deliberately out of order: the helper walks each colony by frame
    rows = [
        {"frame": 4, "colony_id": 1, "merged_with": "2"},
        {"frame": 1, "colony_id": 1, "merged_with": ""},
        {"frame": 3, "colony_id": 2, "merged_with": "3"},
        {"frame": 2, "colony_id": 2, "merged_with": ""},
        {"frame": 2, "colony_id": 3, "merged_with": ""},
        {"frame": 1, "colony_id": 9, "merged_with": ""},
        {"frame": 2, "colony_id": 9, "merged_with": ""},
    ]
    assert ancestor_counts(rows) == [3, 1, 2, 1, 1, 1, 1]
    assert parse_merged("2;3") == {2, 3}
    assert parse_merged("") == set()
    assert parse_merged(None) == set()


# --- fork --------------------------------------------------------------------


def test_fork_continues_the_same_id_and_is_counted():
    groups = one_group()
    frames = [
        # frame 1: colony 1, 30x30
        (1, box_image({1: (5, 5, 35, 35)})),
        # frame 2: the decoder splits the colony: a main body that matches
        # (IoU 400/900) and a detached fragment (> 3 px away, so the pre-merge
        # does not fold it) lying entirely inside the colony's territory
        (2, box_image({1: (5, 5, 25, 25), 2: (28, 28, 35, 35)})),
    ]
    res = track_colonies(frames, groups)
    assert [r["colony_id"] for r in res.rows] == [1, 1, 1]  # fragment keeps the ID
    assert [r["area_px"] for r in res.rows] == [900, 400, 49]
    assert res.counters["forked_segments"] == 1
    assert res.counters["colonies"] == 1  # the fork starts no new colony
    assert (res.labels[1][28:35, 28:35] == 1).all()  # fragment labelled colony 1
    # the continued main body owns the colony's extent for the next frame
    assert res.rows[1]["merged_with"] == ""


# --- pre-merge of touching segments ------------------------------------------


def test_touching_segments_premerge_by_default_and_not_when_disabled():
    groups = one_group()
    f1 = box_image({1: (10, 10, 20, 20)})
    f2 = box_image({1: (10, 10, 20, 20), 2: (10, 22, 20, 30)})  # 2 px apart: touching
    res = track_colonies([(1, f1), (2, f2)], groups)
    # pre-merged into the larger segment, which continues colony 1
    assert [r["colony_id"] for r in res.rows] == [1, 1]
    assert res.rows[1]["area_px"] == 180
    assert res.counters["forked_segments"] == 0
    # without the pre-merge the second segment matches nothing and starts
    # a new colony
    res = track_colonies([(1, f1), (2, f2)], groups, merge_touching=False)
    assert [r["colony_id"] for r in res.rows] == [1, 1, 2]


def test_premerge_touching_merges_chains_but_not_beyond_dilation():
    img = box_image({1: (10, 2, 20, 12), 2: (10, 14, 20, 24), 3: (10, 26, 20, 36)},
                    shape=(30, 40))
    segs = extract_segments(img)
    assert len(segs) == 3
    merged = premerge_touching(segs)
    # chain A-B-C: the pass repeats until stable, so B bridges A and C
    assert len(merged) == 1 and merged[0].area == 300
    # a 4 px gap is beyond TOUCH_DILATE: no merge
    img = box_image({1: (10, 2, 20, 12), 2: (10, 16, 20, 26)}, shape=(30, 40))
    assert len(premerge_touching(extract_segments(img))) == 2
    # the input segments are not mutated (the caller may still hold them)
    assert segs[0].area == 100


# --- the pre-existing filter (exclude_before) --------------------------------


def test_exclude_before_blanks_early_frames_and_flags_pre_existing():
    groups = one_group()
    frames = [
        # frame 3: a colony is present but the frame lies below the cutoff
        (3, box_image({1: (5, 5, 15, 15)})),
        # frame 5 (= N): colony 1 is first detected here -> pre-existing
        (5, box_image({1: (5, 5, 15, 15), 2: (20, 20, 30, 30)})),
        # frame 7: colony 3 appears -> new growth
        (7, box_image({1: (5, 5, 15, 15), 2: (20, 20, 30, 30),
                       3: (30, 2, 38, 10)})),
    ]
    res = track_colonies(frames, groups, exclude_before=5)
    assert len(res.labels) == 3  # outputs stay 1:1 with the input frames
    assert not res.labels[0].any()  # frame 3 (< N): empty label image
    assert (res.labels[1][20:30, 20:30] == 2).all()  # frame 5 tracked normally
    assert [r["frame"] for r in res.rows] == [5, 5, 7, 7, 7]  # no frame-3 rows
    assert [r["colony_id"] for r in res.rows] == [1, 2, 1, 2, 3]
    assert [r["pre_existing"] for r in res.rows] == [1, 1, 1, 1, 0]
    assert res.counters["pre_existing_colonies"] == 2
    assert res.counters["colonies"] == 1  # distinct IDs minus pre-existing


def test_exclude_before_needs_no_matching_frame():
    groups = one_group()
    frames = [
        (3, box_image({1: (5, 5, 15, 15)})),
        (7, box_image({1: (5, 5, 15, 15), 2: (20, 20, 30, 30)})),
    ]
    # N = 5 is not among the well's frame numbers: compared directly, no
    # warning, no disabling
    res = track_colonies(frames, groups, exclude_before=5)
    assert [r["frame"] for r in res.rows] == [7, 7]
    # first detection (7) is after N: flagged as new growth, rows kept
    assert [r["pre_existing"] for r in res.rows] == [0, 0]
    assert res.counters["pre_existing_colonies"] == 0
    assert res.counters["colonies"] == 2
    assert not res.labels[0].any()


def test_empty_reference_produces_nothing_but_keeps_the_frame_count():
    groups = one_group()
    frames = [(1, box_image({1: (5, 5, 15, 15)})), (2, box_image({1: (5, 5, 15, 15)}))]
    for ref in (None, np.zeros((40, 40), np.uint32)):
        res = track_colonies(frames, ref)
        assert len(res.labels) == 2 and not res.labels[0].any()
        assert res.rows == []
        assert res.counters["groups"] == 0 and res.counters["colonies"] == 0