"""Tests for the analysis stage (ticket 06).

Seams under test (per the ticket and the spec's testing decisions): the
public ``run_analysis``/``AnalysisOptions``/``AnalysisSummary`` API driven by
the synthetic mini-plate and synthetic segmentation trees (no model, no GPU,
no real data), plus the module's pure section functions (well-circle fit,
trajectory building, pre-existing probe, growth rates, bad split, aggregates)
as input → output on synthetic data.
"""

import numpy as np
import tifffile

from conftest import (
    DiskSpec,
    RingSpec,
    build_label_series,
    build_mini_plate,
    disk_mask,
)
from colony_tracking import analysis
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


def _row(well, frame, t_seconds, colony_id, area, group_id=None, merged=""):
    """A synthetic colonies.csv row (canonical columns)."""
    return {
        "well": well, "frame": frame, "t_seconds": t_seconds,
        "colony_id": colony_id, "group_id": group_id if group_id is not None else colony_id,
        "merged_with": merged, "pre_existing": 0, "area_px": area,
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
    (every 2-frame window grows < 1.3) are flagged; a clean grower is not."""
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
