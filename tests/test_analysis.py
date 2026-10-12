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
