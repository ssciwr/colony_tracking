"""Tests for the analysis stage (ticket 06).

Seams under test (per the ticket and the spec's testing decisions): the
public ``run_analysis``/``AnalysisOptions``/``AnalysisSummary`` API driven by
the synthetic mini-plate and synthetic segmentation trees (no model, no GPU,
no real data), plus the module's pure section functions (well-circle fit,
trajectory building, pre-existing probe, growth rates, bad split, aggregates)
as input → output on synthetic data.
"""

import logging

import numpy as np
import tifffile

from conftest import (
    DiskSpec,
    RingSpec,
    build_label_series,
    build_mini_plate,
    disk_mask,
    frame_name,
)
from colony_tracking import analysis, io
from colony_tracking._options import AnalysisOptions


def _options(tmp_path, **kwargs) -> AnalysisOptions:
    """AnalysisOptions with required paths pointing at tmp_path."""
    base = dict(
        seg_dirs=[tmp_path / "seg"],
        raw_dir=tmp_path / "raw",
        out_root=tmp_path / "out",
    )
    base.update(kwargs)
    return AnalysisOptions(**base)


# --- well circle fit ----------------------------------------------------------


def _ring_only_plate(tmp_path, well="-A001", n_frames=4):
    """A deterministic (noise-free) plate with one ring and no disks."""
    return build_mini_plate(
        tmp_path / "raw",
        layout="flat",
        wells=(well,),
        n_frames=n_frames,
        rings={well: RingSpec()},
        disks={well: []},
        noise_sigma=0.0,
    )


def test_circle_fit_recovers_planted_ring(tmp_path):
    plate = _ring_only_plate(tmp_path)
    cy, cx, R = plate.ring["-A001"]
    paths = [tmp_path / "raw" / name for name in plate.wells["-A001"][:3]]
    fit = analysis.fit_well_circle(paths)
    assert fit is not None
    fcy, fcx, fR = fit
    assert abs(fcy - cy) < 5
    assert abs(fcx - cx) < 5
    assert abs(fR - R) < 5


def test_circle_fit_rejects_frame_without_ring(tmp_path):
    plate = build_mini_plate(
        tmp_path / "raw",
        layout="flat",
        wells=("-A001",),
        n_frames=4,
        rings={"-A001": None},
        disks={"-A001": []},
        noise_sigma=0.0,
    )
    paths = [tmp_path / "raw" / name for name in plate.wells["-A001"][:3]]
    assert analysis.fit_well_circle(paths) is None


def test_circle_fit_rejects_broken_ring(tmp_path):
    """A ring missing three quarters leaves 5 real angular sectors (plus at
    most two boundary-bleed sectors), below the stage-2 coverage gate
    (>= 8 of 12 sectors), so the fit must be rejected."""
    plate = _ring_only_plate(tmp_path)
    cy, cx, R = plate.ring["-A001"]
    yy, xx = np.mgrid[0:plate.size, 0:plate.size].astype(np.float32)
    angle = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    d = np.hypot(yy - cy, xx - cx)
    # Ring missing from 135°..360° (angle convention: 0 = +x, 90 = +y, which
    # is down in image coordinates). The gapped rays are given a constant
    # bright band in the ring zone: wide enough (130 px) that the stage-2
    # window (R ± 70) and its blur neighbourhood lie fully inside the band,
    # so the gapped profiles are exactly flat (no dip points) in stage 2,
    # while the stage-1 window (650–1120) still has a pure background region
    # where the profile minimum lands.
    gap = (angle >= 135) & (np.abs(d - R) < 130)
    for name in plate.wells["-A001"][:3]:
        path = tmp_path / "raw" / name
        img = np.asarray(tifffile.imread(path), dtype=np.float32)
        img[gap] = plate.base + 3000.0
        tifffile.imwrite(path, np.clip(img, 0, 65535).astype(np.uint16))
    paths = [tmp_path / "raw" / name for name in plate.wells["-A001"][:3]]
    assert analysis.fit_well_circle(paths) is None


# --- time, ancestors, rim exclusion, bad flags ---------------------------------


def _row(well, frame, t_seconds, colony_id, area, group_id=None, merged="",
         pre=0):
    """A synthetic colonies.csv row (canonical columns)."""
    return {
        "well": well, "frame": frame, "t_seconds": t_seconds,
        "colony_id": colony_id, "group_id": group_id if group_id is not None else colony_id,
        "merged_with": merged, "pre_existing": pre, "area_px": area,
    }


def test_transitive_ancestors(tmp_path):
    """A+B<-C: colony 2 absorbs colony 3, colony 1 then absorbs colony 2,
    so colony 1's segment combines three original colonies."""
    rows = [
        _row("-A001", 5, 0, 1, 100),
        _row("-A001", 5, 0, 2, 100),
        _row("-A001", 5, 0, 3, 100),
        _row("-A001", 7, 2 * 3_600_000, 1, 300, merged="2"),
        _row("-A001", 7, 2 * 3_600_000, 2, 200, merged="3"),
    ]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    by = {(r["frame"], r["colony_id"]): r for r in traj}
    assert by[(5, 1)]["ancestors"] == 1
    assert by[(5, 2)]["ancestors"] == 1
    assert by[(5, 3)]["ancestors"] == 1
    assert by[(7, 2)]["ancestors"] == 2  # 2 absorbs 3
    assert by[(7, 1)]["ancestors"] == 3  # 1 absorbs 2, which had absorbed 3
    # canonical output fields
    assert set(by[(7, 1)]) == {
        "well", "frame", "t_hours", "colony_id", "group_id", "merged_with",
        "ancestors", "pre_existing", "area_px", "age_h", "rim_frac",
        "rim_excluded", "bad", "bad_reason",
    }
    # none of these colonies is bad
    assert all(r["bad"] == 0 and r["bad_reason"] == "" for r in traj)


def test_rim_exclusion_monotone_from_right_frame(tmp_path):
    """A colony growing into the rim is excluded from the first frame in
    which more than rim_overlap_max of its pixels lie in the rim zone, and
    never un-excluded (monotone over the lifetime)."""
    well = "-A001"
    plate = build_mini_plate(
        tmp_path / "raw",
        layout="flat",
        wells=(well,),
        n_frames=8,
        rings={well: RingSpec()},
        disks={well: [DiskSpec("rim", cy=1024.0, cx=1824.0, r0=40.0,
                               growth=1.35, first_frame=3)]},
        noise_sigma=0.0,
    )
    seg_root = tmp_path / "seg"
    labels, rows = [], []
    for f in range(1, plate.n_frames + 1):
        mask = disk_mask(plate, well, "rim", f)
        lab = np.zeros((plate.size, plate.size), np.uint32)
        if mask is not None:
            lab[mask] = 1
        labels.append(lab)
        if mask is not None:
            ys, xs = np.nonzero(mask)
            rows.append({
                "well": well, "frame": f, "t_seconds": plate.t_values[f - 1],
                "image": plate.wells[well][f - 1], "colony_id": 1, "group_id": 1,
                "merged_with": "", "pre_existing": 0, "area_px": int(mask.sum()),
                "centroid_y": float(ys.mean()), "centroid_x": float(xs.mean()),
            })
    build_label_series(seg_root / well, labels, well=well)

    # Ground truth from the planted ring: first frame with > 0.1 of the
    # colony's pixels within 60 px of the well wall.
    cy, cx, R = plate.ring[well]
    yy, xx = np.mgrid[0:plate.size, 0:plate.size].astype(np.float32)
    zone = np.hypot(yy - cy, xx - cx) > R - 60
    first_excluded = None
    for f in range(1, plate.n_frames + 1):
        mask = disk_mask(plate, well, "rim", f)
        if mask is not None and (mask & zone).sum() / mask.sum() > 0.1:
            first_excluded = f
            break
    assert first_excluded is not None  # the disk does reach the rim

    rim_fracs = analysis.rim_stats(rows, seg_root, (cy, cx, R), 60)
    traj = analysis.build_trajectories(
        rows, rim_fracs, _options(tmp_path), series_start_frame=1)
    excluded = {r["frame"]: r["rim_excluded"] for r in traj}
    for f in range(3, plate.n_frames + 1):
        expected = 0 if f < first_excluded else 1
        assert excluded[f] == expected, f"frame {f}: expected {expected}"
    # the rim fractions themselves are carried (rounded) in the trajectories
    assert all(r["rim_frac"] is not None for r in traj)


def test_bad_flags(tmp_path):
    """from_start (first valid frame at the series start) and no_growth
    (any per-image comparison of a valid frame to the earliest valid frame
    >= 2 images ahead at or below min_growth_per_frame) are flagged; a
    clean grower is not."""
    well = "-A001"
    rows = []
    # colony 1: present from frame 1 (series start), growing 1.25/frame
    for f in range(1, 6):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 1,
                         int(100 * 1.25 ** (f - 1))))
    # colony 2: appears at frame 2, constant area (stationary debris)
    for f in range(2, 7):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 2, 500))
    # colony 3: appears at frame 3, healthy growth 1.35/frame
    for f in range(3, 9):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 3,
                         int(100 * 1.35 ** (f - 3))))
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    by_cid = {}
    for r in traj:
        by_cid.setdefault(r["colony_id"], []).append(r)
    assert {r["bad_reason"] for r in by_cid[1]} == {"from_start"}
    assert all(r["bad"] == 1 for r in by_cid[1])
    assert {r["bad_reason"] for r in by_cid[2]} == {"no_growth"}
    assert all(r["bad"] == 1 for r in by_cid[2])
    assert all(r["bad"] == 0 and r["bad_reason"] == "" for r in by_cid[3])


def test_pre_existing_colonies_are_never_bad(tmp_path):
    """A pre-existing colony gets no bad flags (from_start/no_growth): it is
    excluded from the analysis anyway and must not exclude its well."""
    well = "-A001"
    rows = []
    # colony 1: present from frame 1, stationary -> would be from_start +
    # no_growth, but it is pre-existing
    for f in range(1, 6):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 1, 500, pre=1))
    # colony 2: appears at frame 2, stationary debris -> no_growth applies
    for f in range(2, 7):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 2, 500))
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    by_cid = {}
    for r in traj:
        by_cid.setdefault(r["colony_id"], []).append(r)
    assert all(r["bad"] == 0 and r["bad_reason"] == "" for r in by_cid[1])
    assert all(r["pre_existing"] == 1 for r in by_cid[1])
    assert {r["bad_reason"] for r in by_cid[2]} == {"no_growth"}


def test_no_growth_per_frame_comparisons(tmp_path):
    """The per-frame no_growth criterion: each valid frame is compared once,
    to the earliest valid frame at least 2 images ahead, and any comparison
    growing by min_growth_per_frame (0.1) or less per image flags the
    colony."""
    well = "-A001"
    rows = []
    # colony 1: sparse partial detections far apart and shrinking (the
    # -B001-style artifact): 4784 px at frame 12 -> 2400 px at frame 19;
    # per image ((2400 - 4784) / 4784) / 7 < 0 -> no_growth
    rows.append(_row(well, 12, 11 * 3_600_000, 1, 4784))
    rows.append(_row(well, 19, 18 * 3_600_000, 1, 2400))
    # colony 2: the same shrink across two adjacent frames -> nothing to
    # compare (span < 2 images), exempt
    rows.append(_row(well, 12, 11 * 3_600_000, 2, 4784))
    rows.append(_row(well, 13, 12 * 3_600_000, 2, 2400))
    # colony 3: sparse slow grower, 10% over 5 images -> 0.02 per image,
    # below the minimum
    rows.append(_row(well, 5, 4 * 3_600_000, 3, 1000))
    rows.append(_row(well, 10, 9 * 3_600_000, 3, 1100))
    # colony 4: fast then plateau: the plateau pair (frame 5 -> 7) grows
    # (620 - 529) / 529 / 2 ~ 0.086 per image, below the minimum; the early
    # pairs alone would pass (old rule: at least one window passes)
    for f, area in ((3, 100), (4, 230), (5, 529), (6, 550), (7, 620)):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 4, area))
    # colony 5: healthy exponential grower, 1.35 per frame: every
    # comparison grows (1.35^2 - 1) / 2 ~ 0.41 per image
    for f in range(3, 9):
        rows.append(_row(well, f, (f - 1) * 3_600_000, 5,
                         int(100 * 1.35 ** (f - 3))))
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    by_cid = {}
    for r in traj:
        by_cid.setdefault(r["colony_id"], []).append(r)
    assert {r["bad_reason"] for r in by_cid[1]} == {"no_growth"}
    assert all(r["bad"] == 1 for r in by_cid[1])
    assert {r["bad_reason"] for r in by_cid[2]} == {""}
    assert all(r["bad"] == 0 for r in by_cid[2])
    assert {r["bad_reason"] for r in by_cid[3]} == {"no_growth"}
    assert {r["bad_reason"] for r in by_cid[4]} == {"no_growth"}
    assert all(r["bad"] == 0 and r["bad_reason"] == "" for r in by_cid[5])


def test_no_growth_gap_span_normalized(tmp_path):
    """A gap-spanning comparison is normalized by the image count: the same
    jump spread over more images lowers the per-image demand, so a colony
    that grew only during the gap is not flagged on its absolute jump."""
    well = "-A001"
    rows = [
        # colony 1: 2x growth from frame 3 to 8 -> 0.2 per image, above the
        # 0.1 minimum despite the gap
        _row(well, 3, 2 * 3_600_000, 1, 1000),
        _row(well, 8, 7 * 3_600_000, 1, 2000),
        # colony 2: the same 2x growth, but reached *within* 2 images and
        # stationary since -> the later comparison sees 0 per image
        _row(well, 3, 2 * 3_600_000, 2, 1000),
        _row(well, 5, 4 * 3_600_000, 2, 2000),
        _row(well, 8, 7 * 3_600_000, 2, 2000),
    ]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    by_cid = {}
    for r in traj:
        by_cid.setdefault(r["colony_id"], []).append(r)
    assert all(r["bad"] == 0 and r["bad_reason"] == "" for r in by_cid[1])
    assert {r["bad_reason"] for r in by_cid[2]} == {"no_growth"}


# --- bad-well split ------------------------------------------------------------


def _traj_for_split(tmp_path):
    """Two wells for the bad-well split: -A001 holds two bad colonies
    (from_start, no_growth) plus one clean grower; -A002 holds one bad colony
    plus one clean grower."""
    well_a, well_b = "-A001", "-A002"
    rows = []
    for f in range(1, 6):  # -A001/1: present from the series start
        rows.append(_row(well_a, f, (f - 1) * 3_600_000, 1,
                         int(100 * 1.25 ** (f - 1))))
    for f in range(2, 7):  # -A001/2: stationary debris
        rows.append(_row(well_a, f, (f - 1) * 3_600_000, 2, 500))
    for f in range(3, 9):  # -A001/3: healthy grower
        rows.append(_row(well_a, f, (f - 1) * 3_600_000, 3,
                         int(100 * 1.35 ** (f - 3))))
    for f in range(3, 9):  # -A002/1: healthy grower
        rows.append(_row(well_b, f, (f - 1) * 3_600_000, 1,
                         int(100 * 1.35 ** (f - 3))))
    for f in range(2, 7):  # -A002/2: stationary debris
        rows.append(_row(well_b, f, (f - 1) * 3_600_000, 2, 500))
    return analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)


def test_split_bad_excludes_overloaded_wells(tmp_path):
    """A well with at least ``max_bad_segments`` distinct bad colonies
    (default 2) drops out of the aggregates entirely — its clean colonies
    too (they are suspect as well); a well below the threshold keeps its
    clean colonies, minus the bad rows."""
    traj = _traj_for_split(tmp_path)
    usable, per_well_bad, excluded = analysis.split_bad(traj, _options(tmp_path))
    assert per_well_bad == {"-A001": {1, 2}, "-A002": {2}}
    assert excluded == {"-A001"}
    got = {(r["well"], r["colony_id"]) for r in usable}
    assert got == {("-A002", 1)}  # -A001's clean colony 3 goes with the well


def test_split_bad_keep_bad_wells(tmp_path):
    """With ``keep_bad_wells`` no well is excluded; only the bad rows
    themselves leave the aggregates."""
    traj = _traj_for_split(tmp_path)
    usable, per_well_bad, excluded = analysis.split_bad(
        traj, _options(tmp_path, keep_bad_wells=True))
    assert excluded == set()
    assert per_well_bad == {"-A001": {1, 2}, "-A002": {2}}
    got = {(r["well"], r["colony_id"]) for r in usable}
    assert got == {("-A001", 3), ("-A002", 1)}


# --- pre-existing probe --------------------------------------------------------


def _crow(well, frame, colony_id, cy, cx, pre="0"):
    """A colonies.csv row with string values (as read_colonies returns)."""
    t_ms = (frame - 1) * 3_600_000
    return {
        "well": well, "frame": str(frame), "t_seconds": str(t_ms),
        "image": frame_name(well, frame, t_ms),
        "colony_id": str(colony_id), "group_id": str(colony_id),
        "merged_with": "", "pre_existing": pre, "area_px": "1000",
        "centroid_y": str(cy), "centroid_x": str(cx),
    }


def _probe_setup(tmp_path, root_name="seg_growth_diff_LO005"):
    """A raw root with one colony (centroid 1200,1200, r 20) on the reference
    frame 5, plus an empty diff root directory. Returns (seg_root, raw_seg,
    well)."""
    well = "-A001"
    seg_root = tmp_path / "seg" / root_name
    seg_root.mkdir(parents=True)
    raw_seg = tmp_path / "rawseg"
    well_dir = raw_seg / well
    well_dir.mkdir(parents=True)
    yy, xx = np.mgrid[0:2048, 0:2048]
    lab = np.zeros((2048, 2048), np.uint32)
    lab[np.hypot(yy - 1200.0, xx - 1200.0) <= 20.0] = 1
    name = frame_name(well, 5, 4 * 3_600_000)
    tifffile.imwrite(well_dir / f"{name[:-4]}__labels.tif", lab)
    io.write_colonies(well_dir / "colonies.csv", [{
        "well": well, "frame": 5, "t_seconds": 4 * 3_600_000,
        "image": name, "colony_id": 1, "group_id": 1, "merged_with": "",
        "pre_existing": 0, "area_px": int(np.pi * 20 ** 2),
        "centroid_y": 1200.0, "centroid_x": 1200.0,
    }])
    return seg_root, raw_seg, well


def test_probe_flags_pre_existing_on_base_frame(tmp_path):
    """Colonies whose first-detection centroid lands on base-frame colony
    territory — a label within the 12 px window, or inside a raw colony's
    area-equivalent disk + 15 px slack — are flagged pre_existing on every
    one of their rows; others are untouched, and an existing flag is never
    cleared."""
    seg_root, raw_seg, well = _probe_setup(tmp_path)
    rows = [
        # colony 1: right on the raw colony's centroid -> window hit
        _crow(well, 6, 1, 1200.0, 1200.0),
        _crow(well, 7, 1, 1210.0, 1205.0),
        _crow(well, 8, 1, 1220.0, 1210.0),
        # colony 2: just outside the 12 px window but inside disk + 15 px
        _crow(well, 6, 2, 1200.0, 1234.0),
        # colony 3: far from any base-frame colony -> untouched
        _crow(well, 6, 3, 400.0, 400.0),
        # colony 4: far away, but already flagged by the pipeline -> stays 1
        _crow(well, 6, 4, 400.0, 400.0, pre="1"),
    ]
    analysis.probe_pre_existing(
        well, rows, seg_root, _options(tmp_path, raw_seg_dir=raw_seg))
    by = {(int(r["frame"]), int(r["colony_id"])): r["pre_existing"] for r in rows}
    # the probe writes the int 1 (faithful to the prototype); untouched rows
    # keep the CSV's string values
    assert by[(6, 1)] == 1
    assert by[(7, 1)] == 1
    assert by[(8, 1)] == 1
    assert by[(6, 2)] == 1
    assert by[(6, 3)] == "0"
    assert by[(6, 4)] == "1"


def test_probe_noop_for_raw_root(tmp_path):
    """Roots without a diff_LO<nnn> suffix are raw roots: the probe is a
    no-op even with a raw_seg_dir present."""
    seg_root, raw_seg, well = _probe_setup(tmp_path, root_name="seg_growth")
    rows = [_crow(well, 6, 1, 1200.0, 1200.0)]
    analysis.probe_pre_existing(
        well, rows, seg_root, _options(tmp_path, raw_seg_dir=raw_seg))
    assert rows[0]["pre_existing"] == "0"


def test_probe_falls_back_when_raw_root_label_missing(tmp_path, caplog):
    """A raw_seg_dir without the reference frame's label image falls back to
    the raw-frame probe (warned); here the raw dir is absent too, so the
    fallback also skips (no well circle) and flags nothing."""
    seg_root, raw_seg, well = _probe_setup(tmp_path)
    for p in (raw_seg / well).glob("*--LO005--*__labels.tif"):
        p.unlink()
    rows = [_crow(well, 6, 1, 1200.0, 1200.0)]
    analysis.probe_pre_existing(
        well, rows, seg_root, _options(tmp_path, raw_seg_dir=raw_seg))
    assert rows[0]["pre_existing"] == "0"
    assert any("falling back" in m for m in caplog.messages)
    assert any("raw-frame pre-existing probe skipped" in m
               for m in caplog.messages)


def _fallback_plate(tmp_path, well="-A001", n_frames=6):
    """A noise-free raw plate with a ring, a static dark disk present from
    frame 1 at (1200, 1200) (pre-existing material the difference stream
    cannot see) and plain agar at (400, 400)."""
    return build_mini_plate(
        tmp_path / "raw",
        layout="flat",
        wells=(well,),
        n_frames=n_frames,
        rings={well: RingSpec()},
        disks={well: [DiskSpec("damage", cy=1200.0, cx=1200.0, r0=60.0,
                               delta=4000.0)]},
        noise_sigma=0.0,
    )


def test_probe_falls_back_to_raw_frame_and_flags_dark_centroid(tmp_path):
    """Without raw_seg_dir the raw reference frame itself is probed: a colony
    whose first-detection centroid sits in a clearly dark window (below
    PRE_EXISTING_DARK_FRAC of the well interior's median) is flagged
    pre-existing on all its rows; a colony on agar is untouched, and an
    existing flag is never cleared."""
    well = "-A001"
    plate = _fallback_plate(tmp_path)
    seg_root = tmp_path / "seg" / "seg_growth_diff_LO005"
    seg_root.mkdir(parents=True)
    paths = [tmp_path / "raw" / name for name in plate.wells[well]]
    geo = analysis.fit_well_circle(paths)
    assert geo is not None
    rows = [
        # colony 1: first detected at frame 6 on the dark disk -> flagged
        _crow(well, 6, 1, 1200.0, 1200.0),
        _crow(well, 7, 1, 1210.0, 1205.0),
        # colony 2: on agar (as bright as the background) -> untouched
        _crow(well, 6, 2, 400.0, 400.0),
        # colony 3: already flagged by the pipeline -> stays 1
        _crow(well, 6, 3, 400.0, 400.0, pre="1"),
    ]
    analysis.probe_pre_existing(well, rows, seg_root, _options(tmp_path), geo)
    by = {(int(r["frame"]), int(r["colony_id"])): r["pre_existing"] for r in rows}
    assert by[(6, 1)] == 1
    assert by[(7, 1)] == 1
    assert by[(6, 2)] == "0"
    assert by[(6, 3)] == "1"


def test_probe_fallback_also_runs_with_a_raw_seg_dir(tmp_path):
    """The raw-frame probe is the fallback, not an alternative: with
    raw_seg_dir set but its label image missing, a dark first-detection
    centroid in the raw plate is still flagged."""
    seg_root, raw_seg, well = _probe_setup(tmp_path)
    for p in (raw_seg / well).glob("*--LO005--*__labels.tif"):
        p.unlink()
    plate = _fallback_plate(tmp_path)
    paths = [tmp_path / "raw" / name for name in plate.wells[well]]
    geo = analysis.fit_well_circle(paths)
    rows = [_crow(well, 6, 1, 1200.0, 1200.0)]
    analysis.probe_pre_existing(
        well, rows, seg_root, _options(tmp_path, raw_seg_dir=raw_seg), geo)
    assert rows[0]["pre_existing"] == 1


def test_probe_fallback_skips_without_well_circle(tmp_path, caplog):
    """Without the fitted well circle there is no background anchor: the
    raw-frame probe skips with a warning and flags nothing."""
    well = "-A001"
    _fallback_plate(tmp_path)
    seg_root = tmp_path / "seg" / "seg_growth_diff_LO005"
    seg_root.mkdir(parents=True)
    rows = [_crow(well, 6, 1, 1200.0, 1200.0)]
    analysis.probe_pre_existing(well, rows, seg_root, _options(tmp_path), None)
    assert rows[0]["pre_existing"] == "0"
    assert any("no well circle; raw-frame pre-existing probe skipped" in m
               for m in caplog.messages)


def test_probe_fallback_skips_without_raw_frame(tmp_path, caplog):
    """A raw tree without the reference frame skips the raw-frame probe with
    a warning and flags nothing."""
    well = "-A001"
    plate = _fallback_plate(tmp_path, n_frames=4)  # no LO005
    seg_root = tmp_path / "seg" / "seg_growth_diff_LO005"
    seg_root.mkdir(parents=True)
    paths = [tmp_path / "raw" / name for name in plate.wells[well]]
    geo = analysis.fit_well_circle(paths)
    rows = [_crow(well, 6, 1, 1200.0, 1200.0)]
    analysis.probe_pre_existing(well, rows, seg_root, _options(tmp_path), geo)
    assert rows[0]["pre_existing"] == "0"
    assert any("no raw LO005 frame" in m for m in caplog.messages)


# --- growth rates --------------------------------------------------------------


def test_growth_rates_exponential_fit(tmp_path):
    """The specific rate is the least-squares slope of ln(area) vs hours over
    the valid frames (h^-1); doublings/day = rate * 24 / ln 2; absolute
    growth = (last - first) valid area over elapsed hours."""
    well = "-A001"
    areas = [int(100 * 1.35 ** (f - 1)) for f in range(1, 9)]
    # frames 3..10 (a colony at the series start would be bad/from_start)
    rows = [_row(well, f, (f - 1) * 3_600_000, 1, a)
            for f, a in zip(range(3, 11), areas)]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    (rate,) = analysis.growth_rates(traj)
    # expected: the fit over the exact integer areas the function sees
    k, _ = np.polyfit(np.arange(len(areas), dtype=float), np.log(areas), 1)
    assert abs(k - np.log(1.35)) < 1e-4  # sanity: close to the planted rate
    assert rate["growth_rate_per_h"] == round(float(k), 5)
    assert rate["growth_doublings_per_day"] == round(float(k) * 24 / np.log(2), 3)
    assert rate["abs_growth_px_per_h"] == round(
        (areas[-1] - areas[0]) / (len(areas) - 1), 1)
    assert rate["well"] == well and rate["colony_id"] == 1
    assert rate["group_id"] == 1
    assert rate["first_frame"] == 3 and rate["last_frame"] == 10
    assert rate["n_frames"] == 8 and rate["n_frames_valid"] == 8
    assert rate["area_first"] == areas[0] and rate["area_last"] == areas[-1]
    assert rate["max_ancestors"] == 1
    assert rate["rim_excluded_frame"] is None
    assert rate["bad"] == 0 and rate["bad_reason"] == ""
    assert rate["pre_existing"] == 0


def test_growth_rates_single_valid_frame_gets_nan(tmp_path):
    """A colony with a single valid frame has no rate (None -> empty CSV
    field) but keeps its areas and frame bounds."""
    rows = [_row("-A001", 5, 4 * 3_600_000, 1, 500)]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    (rate,) = analysis.growth_rates(traj)
    assert rate["growth_rate_per_h"] is None
    assert rate["growth_doublings_per_day"] is None
    assert rate["abs_growth_px_per_h"] is None
    assert rate["area_first"] == 500 and rate["area_last"] == 500
    assert rate["first_frame"] == 5 and rate["last_frame"] == 5
    assert rate["n_frames"] == 1 and rate["n_frames_valid"] == 1


def test_growth_rates_adjacent_two_frame_window_gets_nan(tmp_path):
    """A two-valid-frame trajectory whose frames are adjacent is a window
    narrower than the no_growth check's two-image reach — its bare
    two-point slope cannot be vetted against the growth floor, so all
    three rate fields come up empty (areas and bounds still kept)."""
    rows = [_row("-A001", 11, 10 * 3_600_000, 1, 1000),
            _row("-A001", 12, 11 * 3_600_000, 1, 2400)]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    (rate,) = analysis.growth_rates(traj)
    assert rate["growth_rate_per_h"] is None
    assert rate["growth_doublings_per_day"] is None
    assert rate["abs_growth_px_per_h"] is None
    assert rate["area_first"] == 1000 and rate["area_last"] == 2400
    assert rate["first_frame"] == 11 and rate["last_frame"] == 12
    assert rate["n_frames"] == 2 and rate["n_frames_valid"] == 2


def test_growth_rates_valid_window_two_apart_fits(tmp_path):
    """A window exactly two images apart (three frames wide) still fits:
    two valid frames separated by one missing frame span the minimum the
    no_growth comparison can vet, so the two-point slope over the gap is
    kept."""
    rows = [_row("-A001", 11, 10 * 3_600_000, 1, 1000),
            _row("-A001", 13, 12 * 3_600_000, 1, 2400)]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    (rate,) = analysis.growth_rates(traj)
    k = np.log(2400 / 1000) / 2  # two-point slope over 2 h
    assert rate["growth_rate_per_h"] == round(k, 5)
    assert rate["growth_doublings_per_day"] == round(k * 24 / np.log(2), 3)
    assert rate["abs_growth_px_per_h"] == round((2400 - 1000) / 2, 1)
    assert rate["first_frame"] == 11 and rate["last_frame"] == 13


def test_growth_rates_valid_window_narrowed_by_rim_exclusion(tmp_path):
    """Rim exclusion can narrow a colony's valid window below the minimum
    even though the colony saw many frames: the truncated window (here two
    adjacent frames) gets no rates either."""
    well = "-A001"
    rows = [_row(well, f, (f - 1) * 3_600_000, 1, int(100 * 1.35 ** (f - 3)))
            for f in range(3, 9)]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    for r in traj:  # simulate exclusion: only frames 4 and 5 stay valid
        if r["frame"] <= 3 or r["frame"] >= 6:
            r["rim_excluded"] = 1
    (rate,) = analysis.growth_rates(traj)
    assert rate["growth_rate_per_h"] is None
    assert rate["growth_doublings_per_day"] is None
    assert rate["abs_growth_px_per_h"] is None
    assert rate["rim_excluded_frame"] == 3
    assert rate["first_frame"] == 4 and rate["last_frame"] == 5
    assert rate["n_frames"] == 6 and rate["n_frames_valid"] == 2


def test_growth_rates_uses_valid_frames_only(tmp_path):
    """Rim-excluded frames leave the fit and the area bounds; the first
    rim-excluded frame is reported in rim_excluded_frame."""
    well = "-A001"
    rows = [_row(well, f, (f - 1) * 3_600_000, 1, int(100 * 1.35 ** (f - 1)))
            for f in range(1, 9)]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    for r in traj:  # simulate exclusion from frame 7
        if r["frame"] >= 7:
            r["rim_excluded"] = 1
    (rate,) = analysis.growth_rates(traj)
    assert rate["rim_excluded_frame"] == 7
    assert rate["first_frame"] == 1 and rate["last_frame"] == 6
    assert rate["n_frames"] == 8 and rate["n_frames_valid"] == 6
    areas = [int(100 * 1.35 ** (f - 1)) for f in range(1, 7)]
    k, _ = np.polyfit(np.arange(len(areas), dtype=float), np.log(areas), 1)
    assert rate["growth_rate_per_h"] == round(float(k), 5)
    assert rate["area_first"] == areas[0] and rate["area_last"] == areas[-1]
    assert rate["abs_growth_px_per_h"] == round(
        (areas[-1] - areas[0]) / (len(areas) - 1), 1)


def test_growth_rates_carries_flags_and_max_ancestors(tmp_path):
    """bad/bad_reason and pre_existing come from the colony's first row;
    max_ancestors is the largest ancestor count over the lifetime; output
    is ordered by (well, colony_id)."""
    well = "-A001"
    rows = [
        # colony 1: stationary debris -> bad (no_growth)
        _row(well, 5, 4 * 3_600_000, 1, 500),
        _row(well, 6, 5 * 3_600_000, 1, 500),
        _row(well, 7, 6 * 3_600_000, 1, 500),
        # colony 2: healthy, absorbs colony 1 at frame 7
        _row(well, 5, 4 * 3_600_000, 2, 100),
        _row(well, 7, 6 * 3_600_000, 2, 400, merged="1"),
        _row(well, 9, 8 * 3_600_000, 2, 900),
    ]
    traj = analysis.build_trajectories(
        rows, {}, _options(tmp_path), series_start_frame=1)
    rates = analysis.growth_rates(traj)
    assert [r["colony_id"] for r in rates] == [1, 2]
    r1, r2 = rates
    assert r1["bad"] == 1 and r1["bad_reason"] == "no_growth"
    assert r2["bad"] == 0 and r2["bad_reason"] == ""
    assert r2["max_ancestors"] == 2
    assert r2["group_id"] == 2


# --- end-to-end (run_analysis) -------------------------------------------------

import csv  # noqa: E402  (legacy-tree tests write prototype-style CSVs)

E2E_WELLS = ("-A001", "-A002", "-A003")
E2E_DISKS = {
    "-A001": [
        DiskSpec("pair_a", cy=650.0, cx=750.0, r0=30.0, growth=1.3,
                 first_frame=2),
        DiskSpec("pair_b", cy=650.0, cx=950.0, r0=30.0, growth=1.3,
                 first_frame=2),
        DiskSpec("static", cy=1450.0, cx=1450.0, r0=50.0, growth=1.0,
                 first_frame=2),
        DiskSpec("from_start", cy=600.0, cx=1500.0, r0=30.0, growth=1.25,
                 first_frame=1),
        DiskSpec("rim", cy=1024.0, cx=1824.0, r0=40.0, growth=1.35,
                 first_frame=3),
    ],
    "-A002": [
        DiskSpec("growing", cy=1024.0, cx=600.0, r0=40.0, growth=1.35,
                 first_frame=3),
    ],
    "-A003": [
        DiskSpec("pre", cy=1024.0, cx=1200.0, r0=40.0, growth=1.35,
                 first_frame=3),
        DiskSpec("single", cy=1400.0, cx=600.0, r0=30.0, growth=1.0,
                 first_frame=8, last_frame=8),
    ],
}


def _build_e2e(tmp_path, seg_name="seg_growth"):
    """The 3-well e2e fixture: mini plate + synthetic segmentation tree.

    -A001: two disks that grow until they touch (merged into one colony with
    ``merged_with`` at the first overlap frame), a static disk (no_growth),
    one present from frame 1 (from_start), one growing into the rim
    (monotone rim exclusion). -A002: one clean exponential disk. -A003: a
    pre-existing-flagged disk and a single-frame colony. Colony IDs are the
    disk order per well (1..n). Label images exist 1:1 for every frame,
    including empty ones (the segment stage writes them).
    """
    plate = build_mini_plate(
        tmp_path / "raw", layout="flat", wells=E2E_WELLS, n_frames=8,
        rings={w: RingSpec() for w in E2E_WELLS}, disks=E2E_DISKS,
        noise_sigma=0.0)
    seg_root = tmp_path / "seg" / seg_name
    ids = {
        "-A001": {"pair_a": 1, "pair_b": 2, "static": 3, "from_start": 4,
                  "rim": 5},
        "-A002": {"growing": 1},
        "-A003": {"pre": 1, "single": 2},
    }
    for well in E2E_WELLS:
        well_dir = seg_root / well
        well_dir.mkdir(parents=True)
        rows = []
        for f in range(1, plate.n_frames + 1):
            present = {}
            for d in plate.disks[well]:
                m = disk_mask(plate, well, d.name, f)
                if m is not None:
                    present[d.name] = m
            if "pair_a" in present and "pair_b" in present \
                    and (present["pair_a"] & present["pair_b"]).any():
                # first overlap: the two become one colony (pair_a keeps the
                # id) listing the absorbed one, from this frame on
                present["pair_a"] = present["pair_a"] | present["pair_b"]
                merged_with = {1: "2"}
                del present["pair_b"]
            else:
                merged_with = {}
            lab = np.zeros((plate.size, plate.size), np.uint32)
            for name, m in present.items():
                cid = ids[well][name]
                lab[m] = cid
                ys, xs = np.nonzero(m)
                rows.append({
                    "well": well, "frame": f,
                    "t_seconds": plate.t_values[f - 1],
                    "image": plate.wells[well][f - 1],
                    "colony_id": cid, "group_id": cid,
                    "merged_with": merged_with.get(cid, ""),
                    "pre_existing": int((well, cid) == ("-A003", 1)),
                    "area_px": int(m.sum()),
                    "centroid_y": float(ys.mean()),
                    "centroid_x": float(xs.mean()),
                })
            build_label_series(well_dir, [lab], well=well,
                               names=[plate.wells[well][f - 1]])
        io.write_colonies(well_dir / "colonies.csv", rows)
    return plate, seg_root


TRAJ_HEADER = ["well", "frame", "t_hours", "colony_id", "group_id",
               "merged_with", "ancestors", "pre_existing", "area_px", "age_h",
               "rim_frac", "rim_excluded", "bad", "bad_reason"]
RATE_HEADER = ["well", "colony_id", "group_id", "first_frame", "last_frame",
               "n_frames", "n_frames_valid", "area_first", "area_last",
               "max_ancestors", "rim_excluded_frame", "growth_rate_per_h",
               "growth_doublings_per_day", "abs_growth_px_per_h",
               "bad", "bad_reason", "pre_existing"]


def test_run_analysis_end_to_end(tmp_path):
    """One full analyze run over the e2e tree: outputs, flags, exclusions,
    rates and pooled files all come out as the planted ground truth says."""
    _build_e2e(tmp_path)
    summary = analysis.run_analysis(
        _options(tmp_path, seg_dirs=[tmp_path / "seg" / "seg_growth"]))

    # -- the summary -------------------------------------------------------
    (root,) = summary.roots.values()
    assert root.name == "seg_growth"
    assert root.wells == 3
    assert root.colonies == 8          # 5 in -A001 (pair counts once), 1, 2
    assert root.rows == 46             # 33 + 6 + 7
    assert root.rim_excluded == 3      # the rim colony, frames 6-8
    assert root.bad_segments == 2      # static (no_growth), from_start
    assert root.excluded_wells == ["-A001"]
    assert root.pre_existing == 1      # the -A003 pre-flagged colony
    assert summary.failures == []

    # -- the geometry cache --------------------------------------------------
    for well in E2E_WELLS:
        cy, cx, R = summary.geometry[well]
        assert abs(cy - 1024.0) < 5 and abs(cx - 1024.0) < 5
        assert abs(R - 920.0) < 5
    assert set(io.read_geometry(tmp_path / "out" / "well_geometry.csv")) \
        == set(E2E_WELLS)

    # -- the output files ------------------------------------------------------
    out = tmp_path / "out" / "seg_growth"
    for name in ("trajectories.csv", "growth_rates.csv", "bad_segments.csv",
                 "areas_by_ancestors.csv", "rates_by_ancestors.csv",
                 "growth_curves_by_ancestors.png",
                 "mean_area_by_ancestors.png",
                 "growth_rates_by_ancestors.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    for well in E2E_WELLS:
        assert (out / "qa" / f"{well}_rim.png").exists()

    # -- trajectories ------------------------------------------------------------
    traj = io.read_csv_rows(out / "trajectories.csv")
    assert list(traj[0]) == TRAJ_HEADER
    assert len(traj) == 46
    by = {(r["well"], int(r["colony_id"]), int(r["frame"])): r for r in traj}
    for f in range(3, 9):  # the rim colony: monotone exclusion from frame 6
        assert by[("-A001", 5, f)]["rim_excluded"] == ("0" if f < 6 else "1")
        assert by[("-A001", 5, f)]["rim_frac"] != ""
    assert by[("-A001", 1, 6)]["ancestors"] == "1"
    assert by[("-A001", 1, 7)]["ancestors"] == "2"  # the pair merges at f7
    assert by[("-A001", 1, 7)]["merged_with"] == "2"
    assert by[("-A001", 3, 2)]["bad_reason"] == "no_growth"
    assert by[("-A001", 4, 1)]["bad_reason"] == "from_start"
    assert by[("-A003", 1, 3)]["pre_existing"] == "1"
    assert by[("-A003", 2, 8)]["area_px"] != ""
    assert (by[("-A003", 2, 8)]["age_h"]) == "0.0"  # first valid frame

    # -- growth rates ------------------------------------------------------------
    rates = io.read_csv_rows(out / "growth_rates.csv")
    assert list(rates[0]) == RATE_HEADER
    assert len(rates) == 8
    rate_by = {(r["well"], int(r["colony_id"])): r for r in rates}
    # the planted disk grows 1.35x in radius per hour, so its AREA grows
    # 1.35^2: the specific rate is 2 ln(1.35) h^-1
    lam = 2 * float(np.log(1.35))
    grown = rate_by[("-A002", 1)]
    assert abs(float(grown["growth_rate_per_h"]) - lam) / lam < 0.03
    assert abs(float(grown["growth_doublings_per_day"]) - lam * 24 / np.log(2)
               ) / (lam * 24 / np.log(2)) < 0.03
    assert grown["max_ancestors"] == "1"
    assert grown["first_frame"] == "3" and grown["last_frame"] == "8"
    assert rate_by[("-A001", 1)]["max_ancestors"] == "2"
    assert rate_by[("-A001", 5)]["rim_excluded_frame"] == "6"
    assert rate_by[("-A001", 5)]["last_frame"] == "5"  # valid frames only
    assert rate_by[("-A001", 3)]["bad"] == "1"
    assert rate_by[("-A003", 1)]["pre_existing"] == "1"
    single = rate_by[("-A003", 2)]  # one valid frame -> NaN rates
    assert single["growth_rate_per_h"] == ""
    assert single["growth_doublings_per_day"] == ""
    assert single["abs_growth_px_per_h"] == ""
    assert single["area_first"] == single["area_last"] != ""

    # -- bad segments ------------------------------------------------------------
    bads = io.read_csv_rows(out / "bad_segments.csv")
    assert len(bads) == 2
    bad_by = {(r["well"], int(r["colony_id"])): r for r in bads}
    assert bad_by[("-A001", 3)]["bad_reason"] == "no_growth"
    assert bad_by[("-A001", 4)]["bad_reason"] == "from_start"
    assert all(r["n_bad_in_well"] == "2" and r["well_excluded"] == "1"
               for r in bads)

    # -- pooled files -------------------------------------------------------------
    areas = io.read_csv_rows(out / "areas_by_ancestors.csv")
    assert list(areas[0]) == ["frame", "t_hours", "ancestors", "n_colonies",
                              "n_wells", "mean_area_px", "median_area_px"]
    # -A001 is excluded and pre-existing colonies are dropped, so the pool is
    # -A002's grower (f3-f8) plus -A003's single-frame colony (f8).
    got = {(int(r["frame"]), int(r["ancestors"])): int(r["n_colonies"])
           for r in areas}
    assert got == {(3, 1): 1, (4, 1): 1, (5, 1): 1,
                   (6, 1): 1, (7, 1): 1, (8, 1): 2}
    assert len(areas) == len(got)  # one row per (frame, ancestors)

    rates_agg = io.read_csv_rows(out / "rates_by_ancestors.csv")
    assert list(rates_agg[0]) == ["max_ancestors", "n_colonies",
                                  "mean_doublings_per_day",
                                  "median_doublings_per_day",
                                  "mean_abs_growth_px_per_h",
                                  "median_abs_growth_px_per_h"]
    # only -A002's grower has a rate (pre-existing dropped, single-frame none)
    assert len(rates_agg) == 1
    assert int(rates_agg[0]["max_ancestors"]) == 1
    assert int(rates_agg[0]["n_colonies"]) == 1
    assert abs(float(rates_agg[0]["mean_doublings_per_day"])
               - lam * 24 / np.log(2)) < 0.5


def test_run_analysis_keep_bad_wells(tmp_path):
    """keep_bad_wells keeps the bad well's clean colonies in the pool (the
    bad rows themselves still leave)."""
    _build_e2e(tmp_path)
    summary = analysis.run_analysis(_options(
        tmp_path, seg_dirs=[tmp_path / "seg" / "seg_growth"],
        keep_bad_wells=True))
    (root,) = summary.roots.values()
    assert root.excluded_wells == []
    areas = io.read_csv_rows(
        tmp_path / "out" / "seg_growth" / "areas_by_ancestors.csv")
    got = {(int(r["frame"]), int(r["ancestors"])): int(r["n_colonies"])
           for r in areas}
    # -A001: pair (f2+), rim (f3-5 only, excluded from f6), the pair merges
    # at f7 (ancestors 2); -A002 grower f3-8; -A003 single at f8. The two bad
    # -A001 colonies (static, from_start) are still out.
    assert got == {
        (2, 1): 2,  # pair_a, pair_b
        (3, 1): 4,  # + rim, + -A002's grower
        (4, 1): 4,
        (5, 1): 4,
        (6, 1): 3,  # the rim colony is excluded
        (7, 1): 1,  # -A002's grower
        (7, 2): 1,  # the merged pair
        (8, 1): 2,  # grower + -A003's single
        (8, 2): 1,  # the merged pair
    }
    assert len(areas) == len(got)  # one row per (frame, ancestors)
    assert all(int(r["n_wells"]) <= 2 for r in areas)


def test_run_analysis_plots_only_uses_no_image_io(tmp_path, caplog):
    """plots_only re-renders from the existing trajectories CSV: no raw-plate
    reading (no circle fit), no seg-tree reading, no geometry cache write."""
    seg_root = tmp_path / "seg" / "seg_growth_diff_LO005"  # must not exist
    out = tmp_path / "out" / "seg_growth_diff_LO005"
    out.mkdir(parents=True)
    rows = []
    for i, (f, area) in enumerate(((3, 100), (4, 200), (5, 400))):
        rows.append({
            "well": "-A001", "frame": f, "t_hours": float(i),
            "colony_id": 1, "group_id": 1, "merged_with": "",
            "ancestors": 1, "pre_existing": 0, "area_px": area,
            "age_h": float(i), "rim_frac": 0.0, "rim_excluded": 0,
            "bad": 0, "bad_reason": "",
        })
    io.write_csv_rows(out / "trajectories.csv", rows,
                      analysis.TRAJ_FIELDS)
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    options = _options(tmp_path, seg_dirs=[seg_root], raw_dir=empty_raw,
                       plots_only=True)
    summary = analysis.run_analysis(options)

    for name in ("growth_rates.csv", "bad_segments.csv",
                 "areas_by_ancestors.csv", "rates_by_ancestors.csv",
                 "growth_curves_by_ancestors.png",
                 "mean_area_by_ancestors.png",
                 "growth_rates_by_ancestors.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert not any("no plausible well circle" in m for m in caplog.messages)
    assert not (tmp_path / "out" / "well_geometry.csv").exists()
    assert not any("colonies.csv" in m for m in caplog.messages)
    (root,) = summary.roots.values()
    assert root.rows == 3 and root.colonies == 1
    rates = io.read_csv_rows(out / "growth_rates.csv")
    assert len(rates) == 1
    assert rates[0]["growth_rate_per_h"] == f"{np.log(2):.5f}"
    assert abs(float(rates[0]["growth_doublings_per_day"]) - 24.0) < 0.01


def test_run_analysis_reads_legacy_tree(tmp_path):
    """A prototype-written tree (lo/n_merged-style CSVs) is read canonically
    with unchanged values, both for the full run and for plots_only."""
    plate = build_mini_plate(
        tmp_path / "raw", layout="flat", wells=("-A001",), n_frames=4,
        rings={"-A001": RingSpec()}, disks={"-A001": []}, noise_sigma=0.0)
    seg_root = tmp_path / "seg" / "seg_legacy"
    well_dir = seg_root / "-A001"
    well_dir.mkdir(parents=True)
    legacy_cols = ["well", "lo", "t_seconds", "image", "colony_id", "group_id",
                   "merged_with", "pre_existing", "area_px", "centroid_y",
                   "centroid_x"]
    rows = []
    for f in range(2, 5):  # the colony starts at frame 2 (not from_start)
        rows.append({"well": "-A001", "lo": f,
                     "t_seconds": (f - 1) * 3_600_000,
                     "image": plate.wells["-A001"][f - 1], "colony_id": 1,
                     "group_id": 1, "merged_with": "", "pre_existing": 0,
                     "area_px": 100 * 2 ** (f - 2), "centroid_y": 500.0,
                     "centroid_x": 500.0})
    with open(well_dir / "colonies.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=legacy_cols)
        w.writeheader()
        w.writerows(rows)
    lab = np.zeros((plate.size, plate.size), np.uint32)
    build_label_series(well_dir, [lab] * 4, well="-A001",
                       names=plate.wells["-A001"])

    analysis.run_analysis(_options(tmp_path, seg_dirs=[seg_root]))
    out = tmp_path / "out" / "seg_legacy"
    traj = io.read_csv_rows(out / "trajectories.csv")
    assert list(traj[0]) == TRAJ_HEADER
    assert [r["frame"] for r in traj] == ["2", "3", "4"]
    assert [r["area_px"] for r in traj] == ["100", "200", "400"]
    assert all(r["ancestors"] == "1" and r["bad"] == "0" for r in traj)
    rates = io.read_csv_rows(out / "growth_rates.csv")
    assert list(rates[0]) == RATE_HEADER
    assert rates[0]["first_frame"] == "2" and rates[0]["last_frame"] == "4"
    assert abs(float(rates[0]["growth_rate_per_h"]) - np.log(2)) < 1e-5

    # legacy trajectories columns (lo, n_merged) are mapped in plots_only too
    out2 = tmp_path / "out2" / "seg_legacy"
    out2.mkdir(parents=True)
    legacy_traj_cols = ["well", "lo", "t_hours", "colony_id", "group_id",
                        "merged_with", "n_merged", "pre_existing", "area_px",
                        "age_h", "rim_frac", "rim_excluded", "bad",
                        "bad_reason"]
    legacy_rows = []
    for i, (f, area) in enumerate(((2, 100), (3, 200), (4, 400))):
        legacy_rows.append({
            "well": "-A001", "lo": f, "t_hours": float(i + 1),
            "colony_id": 1, "group_id": 1, "merged_with": "",
            "n_merged": 1, "pre_existing": 0, "area_px": area,
            "age_h": float(i), "rim_frac": 0.0, "rim_excluded": 0,
            "bad": 0, "bad_reason": "",
        })
    with open(out2 / "trajectories.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=legacy_traj_cols)
        w.writeheader()
        w.writerows(legacy_rows)
    empty_raw = tmp_path / "empty_raw2"
    empty_raw.mkdir()
    analysis.run_analysis(_options(
        tmp_path, seg_dirs=[seg_root], out_root=tmp_path / "out2",
        raw_dir=empty_raw, plots_only=True))
    rates = io.read_csv_rows(out2 / "growth_rates.csv")
    assert list(rates[0]) == RATE_HEADER
    assert rates[0]["first_frame"] == "2" and rates[0]["last_frame"] == "4"
    assert rates[0]["max_ancestors"] == "1"
    assert abs(float(rates[0]["growth_rate_per_h"]) - np.log(2)) < 1e-5


# --- empty wells (the QA overlay) ------------------------------------------------


def test_process_well_empty_well_writes_rim_only_overlay(tmp_path):
    """A well whose segmentation produced zero colonies but that has a fitted
    well circle must not crash: ``process_well`` returns [] and still writes
    a rim-only QA overlay (circle + rim zone, no colony tinting) so a human
    can confirm the well is truly empty."""
    well = "-A001"
    build_mini_plate(
        tmp_path / "raw", layout="flat", wells=(well,), n_frames=4,
        rings={well: RingSpec()}, disks={well: []}, noise_sigma=0.0)
    seg_root = tmp_path / "seg" / "seg_growth"
    well_dir = seg_root / well
    well_dir.mkdir(parents=True)
    io.write_colonies(well_dir / "colonies.csv", [])
    out_dir = tmp_path / "out" / "seg_growth"
    traj = analysis.process_well(well, seg_root, _options(tmp_path), out_dir, {})
    assert traj == []
    overlay = out_dir / "qa" / f"{well}_rim.png"
    assert overlay.exists() and overlay.stat().st_size > 0


def test_run_analysis_surfaces_per_well_failures(tmp_path, monkeypatch, caplog):
    """A well whose analysis raises is counted, not just logged: the root
    summary and the run summary carry the failed well (the way the
    pipeline's ``fail_detail`` reads them), so a run with one failing well
    is not reported ok."""
    _, seg_root = _build_e2e(tmp_path)
    real = analysis.process_well

    def failing(well, *args, **kwargs):
        if well == "-A002":
            raise RuntimeError("boom")
        return real(well, *args, **kwargs)

    monkeypatch.setattr(analysis, "process_well", failing)
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = analysis.run_analysis(
            _options(tmp_path, seg_dirs=[seg_root]))
    assert summary.failed_wells == {"seg_growth": ["-A002"]}
    (root,) = summary.roots.values()
    assert root.failed_wells == ["-A002"]
    assert any("-A002" in m for m in caplog.messages)
