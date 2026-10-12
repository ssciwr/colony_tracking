"""Analysis stage: well circle, rim and bad-segment filtering, growth rates,
pooled aggregates, plots, and QA overlays.

The analyze stage consumes one or more segmentation output roots (each with a
per-well ``colonies.csv`` and per-frame label images, as written by the
segment stage) plus the raw plate directory, and derives, per segmentation
root under ``<out_root>/<name of the root>/``:

- ``trajectories.csv`` — one row per colony per frame:
  ``well, frame, t_hours, colony_id, group_id, merged_with, ancestors,
  pre_existing, area_px, age_h, rim_frac, rim_excluded, bad, bad_reason``;
- ``growth_rates.csv`` — one row per colony: specific rate (h⁻¹),
  doublings/day, absolute growth (px/h), first/last frame, areas,
  ``max_ancestors``, first rim-excluded frame, ``bad``/``bad_reason``,
  ``pre_existing``;
- ``bad_segments.csv`` — one row per bad colony: reason, frames, areas,
  per-well bad count, and whether the well was excluded from the aggregates;
- ``areas_by_ancestors.csv`` / ``rates_by_ancestors.csv`` — the pooled
  aggregates over non-bad, non-pre-existing rows of non-excluded wells
  (per (frame, ancestors) area statistics; per max-ancestors rate statistics);
- the three plots ``growth_curves_by_ancestors.png``,
  ``mean_area_by_ancestors.png``, ``growth_rates_by_ancestors.png`` (suppressed
  by ``no_plots``); per-well QA overlays ``qa/<well>_rim.png`` (suppressed by
  ``no_qa``);
- and, directly under ``out_root``, the shared ``well_geometry.csv`` cache of
  the fitted well circles.

Algorithm (constants calibrated to the 2048×2048 acquisition format):

**Well circle.** Fitted per well from up to the three earliest raw frames
(LO001–LO003, no colonies yet), per frame:

1. Gaussian blur (σ 6). Stage 1 samples the darkest point per angle (240
   angles, radii 650–1120 px from the image centre) and keeps only real local
   dips (at least 0.08 × profile range below the mean of the ±60 px radial
   neighbourhood) — illumination gradients create flat-dark regions whose
   minimum is not the ring. < 30 points ⇒ no fit.
2. A coarse Kasa fit with iterative radius-outlier trimming (tol 30, ≥ 20
   points) must land in the plausible radius band (700–1100 px).
3. Stage 2 re-locks every angle in a narrow window (R ± 70 px, dip 0.02,
   ≥ 40 points), trims (tol 25), and accepts only if the radius std < 20 px,
   the centre is within 200–1850 px, and ≥ 8 of the 12 angular sectors (30°)
   are covered.

Frame fits whose R deviates > 40 px from the median are dropped as
mis-locks; the rest are averaged. The fits are cached in ``well_geometry.csv``
(``refresh_geometry`` forces refits). A well with no plausible circle has rim
exclusion disabled (logged).

**Rim.** A colony is rim-excluded from the first frame in which more than
``rim_overlap_max`` (0.1) of its pixels lie within ``rim_width_px`` (60) of
the well circle; exclusion is monotone over the colony's lifetime (colonies
only grow into the rim, never out of it).

**Time.** ``t_hours`` from the colonies CSV ``t_seconds`` (the raw ``T``
field) via the magnitude heuristic: values above 1e6 are read as
milliseconds, smaller ones as seconds.

**Ancestors.** Per colony-frame row, the number of distinct original colonies
combined in the segment, resolved transitively through the ``merged_with``
histories, the colony itself included (an A+B where B had merged with C ends
at 3; unmerged = 1). The prototype computed the same quantity as
``n_merged``; values are unchanged, the name is now ``ancestors``.

**Bad segments.** ``from_start`` — first valid frame within ``start_frames``
(1) of the series start (anchored at the earliest label image); ``no_growth``
— every exact 2-frame window among the valid frames grows by less than
``1 + min_growth_2step`` (1.3; needs ≥ 3 valid frames). A well with ≥
``max_bad_segments`` (2) distinct bad colonies is excluded from
aggregates/plots (``keep_bad_wells`` overrides); bad segments and bad wells
are always reported in the log and ``bad_segments.csv`` (the rows stay in the
CSVs, flagged).

**Pre-existing.** The segment stage's ``pre_existing`` flag excludes a colony
from aggregates/plots over its whole lifetime (``keep_pre_existing``
overrides; the rows stay in the CSVs). For difference roots (roots named
``*diff_LO<nnn>``) the reference frame is invisible to the root, so each
colony's first-detection centroid is additionally probed against the raw
root's (``raw_seg_dir``) label image at the reference frame: a hit at the
centroid ± 12 px, or inside an area-equivalent disk of a raw colony (+ 15 px
slack), flags the colony pre-existing.

**Growth rates.** Over a colony's valid (not rim-excluded) frames with ≥ 2
points and positive area: specific rate = least-squares slope of ln(area) vs
hours (h⁻¹); doublings/day = slope × 24 / ln 2; absolute growth =
(last − first) valid area over elapsed time (px/h). Colonies with a single
valid frame get NaN rates (areas stay in the trajectories).

**Plots.** Growth curves (individual trajectories, absolute and relative
time, coloured by ancestor count); mean area per ancestor group (buckets with
< 3 members not drawn; group opacity fades with member count unless
``no_group_alpha``); rate distributions per group (box + strip, symlog
absolute-growth panel).

``plots_only`` re-renders everything derived from an existing trajectories
CSV with no image IO. Colonies/trajectories CSVs written by the prototype
(legacy column names ``lo``, ``n_merged``, ``max_n_merged``, ``first_lo``,
``last_lo``, ``rim_excluded_lo``) are read via the shared io legacy mapping;
new writes always use the canonical names.
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from . import io, log
from ._options import AnalysisOptions
from .tracking import ancestor_counts

#: Plausible fitted well circles: the wells are ~870–920 px in radius and
#: roughly centred in the 2048×2048 frames; fits outside these bounds are
#: rejected. Calibrated to the 2048×2048 acquisition format.
R_MIN, R_MAX = 700, 1100
CENTER_MIN, CENTER_MAX = 200, 1850


def _radial_minima(
    sm: np.ndarray,
    cy: float,
    cx: float,
    r_lo: float,
    r_hi: float,
    n_ang: int = 240,
    dip: float = 0.08,
) -> np.ndarray:
    """For each angle, the radius of the darkest point in [r_lo, r_hi].

    Points are kept only if that darkest point is a real local dip (darker
    than the mean of its ±60 px radial neighbourhood by at least ``dip``
    times the profile range): the meniscus ring is a dark annulus, but frames
    with strong illumination gradients have large flat-dark regions whose
    minimum is not the ring — those angles are dropped instead of voting a
    wrong radius. Returns an (n, 2) array of (y, x) points.
    """
    h, w = sm.shape
    angs = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rs = np.arange(r_lo, r_hi, 2.0)
    aa, rr = np.meshgrid(angs, rs, indexing="ij")
    ys = (cy + rr * np.sin(aa)).astype(int).ravel()
    xs = (cx + rr * np.cos(aa)).astype(int).ravel()
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    prof = np.full(ys.shape, np.nan)
    prof[inside] = sm[ys[inside], xs[inside]]
    prof = prof.reshape(aa.shape)
    row_ok = inside.reshape(aa.shape).mean(axis=1) > 0.9
    row_mean = np.nanmean(prof, axis=1, keepdims=True)
    prof = np.where(np.isnan(prof), row_mean, prof)
    prof = ndi.uniform_filter1d(prof, 9, axis=1)
    i = np.nanargmin(prof, axis=1)
    rows = np.arange(prof.shape[0])
    lo = np.clip(i - 30, 0, prof.shape[1])
    hi = np.clip(i + 31, 0, prof.shape[1])
    nb_mean = np.array([prof[k, lo[k]:hi[k]].mean() for k in rows])
    depth = np.nanmax(prof, axis=1) - np.nanmin(prof, axis=1)
    keep = row_ok & (prof[rows, i] <= nb_mean - dip * (depth + 1e-9))
    return np.c_[cy + rs[i[keep]] * np.sin(angs[keep]),
                 cx + rs[i[keep]] * np.cos(angs[keep])]


def _circle_fit(pts: np.ndarray) -> tuple[float, float, float]:
    """Kasa (algebraic) least-squares circle fit → (cy, cx, R)."""
    a = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    c, *_ = np.linalg.lstsq(a, pts[:, 0] ** 2 + pts[:, 1] ** 2, rcond=None)
    cy, cx = c[0] / 2, c[1] / 2
    return cy, cx, np.sqrt(c[2] + cy ** 2 + cx ** 2)


def _trim_fit(
    pts: np.ndarray, tol: float = 25, iters: int = 3
) -> tuple[float, float, float, float, int] | None:
    """Kasa circle fit with iterative radius-outlier trimming.

    Returns (cy, cx, R, radius std, n points) over the points that survive
    trimming (never fewer than 20), or None if the set collapses — a wide
    meniscus never converges to keep.all(), so the loop is bounded and the
    remainder is accepted.
    """
    for _ in range(iters):
        cy, cx, r = _circle_fit(pts)
        rr = np.sqrt((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2)
        keep = np.abs(rr - np.median(rr)) < tol
        if keep.all() or keep.sum() < 20:
            break
        pts = pts[keep]
    if len(pts) < 20:
        return None
    cy, cx, r = _circle_fit(pts)
    rr = np.sqrt((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2)
    return cy, cx, r, rr.std(), len(pts)


def _fit_frame(path) -> tuple[float, float, float] | None:
    """Two-stage well-circle fit on one raw frame, or None.

    Stage 1 fits the ring from the clear-dip minima sampled around the image
    centre (some angles drop out on badly illuminated frames, but the
    surviving arc still pins centre and radius). Stage 2 re-locks every angle
    in a narrow window (R ± 70 px) around that coarse ring, where even
    flat-dark angles return the meniscus, and accepts the result only if it
    is tight (radius std < 20 px), covers all angular sectors and has a
    plausible R.
    """
    sm = ndi.gaussian_filter(tifffile.imread(path).astype(np.float32), 6)
    h, w = sm.shape
    pts = _radial_minima(sm, h / 2, w / 2, 650, 1120)
    if len(pts) < 30:
        return None
    coarse = _trim_fit(pts, tol=30)
    if coarse is None or not (R_MIN < coarse[2] < R_MAX):
        return None
    pts = _radial_minima(sm, coarse[0], coarse[1], coarse[2] - 70,
                         coarse[2] + 70, dip=0.02)
    if len(pts) < 40:
        return None
    fit = _trim_fit(pts, tol=25)
    if fit is None:
        return None
    cy, cx, r, std, _ = fit
    ang = (np.degrees(np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)) + 360) % 360
    cov = len(np.unique((ang // 30).astype(int)))
    if not (R_MIN < r < R_MAX and CENTER_MIN < cy < CENTER_MAX
            and CENTER_MIN < cx < CENTER_MAX and std < 20 and cov >= 8):
        return None
    return cy, cx, r


def fit_well_circle(paths: list) -> tuple[float, float, float] | None:
    """Fit the well circle (cy, cx, R) to the dark meniscus ring.

    The wells are imaged down the middle but not identically illuminated, and
    their apparent diameter varies with fill height, so the circle is fitted
    per well from its earliest raw frames (which show no colonies yet): each
    frame is fitted independently (see :func:`_fit_frame`), fits whose R
    deviates more than 40 px from the median are dropped as mis-locks, and
    the rest are averaged. Returns (cy, cx, R) or None if nothing plausible
    is found.
    """
    fits = [f for f in (_fit_frame(p) for p in paths) if f]
    if not fits:
        return None
    med_r = np.median([f[2] for f in fits])
    fits = [f for f in fits if abs(f[2] - med_r) <= 40]
    return tuple(float(np.mean([f[i] for f in fits])) for i in range(3))


# ---------------------------------------------------------------- time ------


def t_hours_of(values) -> np.ndarray:
    """Frame times in hours from the raw ``T`` field (the colonies CSV
    ``t_seconds`` column).

    Values above 1e6 are read as milliseconds (the dataset's format,
    T0028808197 = 8.0 h), smaller ones as seconds.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values / 3600.0
    return values / 3.6e6 if values.max() > 1e6 else values / 3600.0


# ----------------------------------------------------------------- rim ------


def rim_stats(rows, seg_dir, geo, rim_width: int) -> dict:
    """Per (frame, colony_id) fraction of colony pixels inside the rim zone.

    For every frame that has rows, the frame's label image is loaded once
    and binned against a boolean rim-zone image (distance from the fitted
    well centre > R - rim_width). Returns ``{(frame, colony_id): rim_frac}``;
    frames whose label image is missing are warned about once and left out.
    """
    if geo is None:
        return {}
    cy, cx, R = geo
    out = {}
    warned = set()
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)
    zone = None
    for frame, frows in sorted(by_frame.items()):
        # strips ".tif"; works for raw and _diff_uint8 input frames
        stem = frows[0]["image"][:-4]
        path = Path(seg_dir) / frows[0]["well"] / f"{stem}__labels.tif"
        if not path.exists():
            if frame not in warned:
                warned.add(frame)
                log.warning("[%s] LO%03d: missing label image %s; rim fractions "
                            "empty", frows[0]["well"], frame, path.name)
            continue
        lab = io.read_label_image(path)
        if zone is None:
            h, w = lab.shape
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
            zone = d > (R - rim_width)
        mx = int(lab.max())
        if mx == 0:
            continue
        n_all = np.bincount(lab.ravel(), minlength=mx + 1)
        n_zone = np.bincount(lab[zone].ravel(), minlength=mx + 1)
        for r in frows:
            cid = int(r["colony_id"])
            if cid <= mx and n_all[cid]:
                out[(frame, cid)] = float(n_zone[cid]) / float(n_all[cid])
    return out


# ---------------------------------------------------------- trajectories ----


def build_trajectories(rows, rim_fracs, options, series_start_frame=None):
    """Per-frame colony records: ancestors, rim exclusion, bad flags.

    One output row per input row, in (well, frame, colony_id) order. For
    every colony, frames are walked chronologically:

    - ``ancestors`` is the number of distinct original colonies combined in
      the segment so far, the colony itself included (1 = unmerged). It
      grows by the full constituent set of every colony newly listed in
      ``merged_with`` (cumulative union): the pipeline keeps listing
      absorbed colonies whose last-known masks stay covered, so an absorbed
      merge product contributes its own history — A merging with B, which
      had merged with C earlier, ends at 3 (resolved by the shared
      :func:`tracking.ancestor_counts`);
    - ``rim_excluded`` turns 1 at the first frame whose ``rim_frac`` exceeds
      ``rim_overlap_max`` and stays 1 (colonies only grow into the rim);
      rows with no rim fraction (no geometry / missing labels) never
      exclude;
    - ``age_h`` is hours since the colony's first valid (not excluded)
      frame — undefined if the colony was excluded from birth;
    - ``bad`` flags segments to exclude from the analysis: present from the
      beginning of the well's series (first valid frame within
      ``start_frames`` of ``series_start_frame``), or not growing (every
      exact 2-frame window among the valid frames grows by less than
      ``1 + min_growth_2step``; needs >= 3 valid frames). ``bad_reason``
      holds the triggered criteria.
    - ``pre_existing`` is the pipeline's flag, possibly OR-ed with the
      base-image probe result for difference roots (see
      :func:`probe_pre_existing`). Colonies flagged pre_existing are
      dropped from the aggregates and plots over their whole lifetime, not
      from the CSVs.
    """
    t_hours = {int(r["frame"]): t for r, t in
               zip(rows, t_hours_of([r["t_seconds"] for r in rows]))}
    # (well, colony_id) keys: the prototype flow feeds one well at a time,
    # but the grouping must not bleed across wells either way.
    order = sorted(rows, key=lambda r: (str(r["well"]), int(r["frame"]),
                                        int(r["colony_id"])))
    per_colony = {}
    for r in order:
        per_colony.setdefault((r["well"], int(r["colony_id"])), []).append(r)

    # Ancestors: the shared identity machinery
    # (:func:`tracking.ancestor_counts`), resolved one well at a time.
    # Colony IDs are per-well, so the merge graph must not bleed across
    # wells either way — grouping rows by well and keying by colony_id
    # inside the call is exactly the (well, colony_id) keying the prototype
    # used. The pipeline's `merged_with` lists the colonies whose
    # last-known masks a segment covers - including colonies an absorbed
    # colony had taken over earlier, whose stale masks stay covered; an
    # absorbed merge product contributes its own history.
    per_well = {}
    for r in order:
        per_well.setdefault(r["well"], []).append(r)
    for wrows in per_well.values():
        for r, n in zip(wrows, ancestor_counts(wrows)):
            r["ancestors"] = n

    baseline = {}
    for key, crows in per_colony.items():
        well, cid = key
        base = None
        excluded = False  # once in the rim, always in the rim (colonies only grow)
        valid = []  # (frame, area) of frames with usable measurements
        for r in crows:
            frame = int(r["frame"])
            # rim_stats keys by (frame, colony_id) for its per-well input; the
            # (well, frame, colony_id) form is accepted as well.
            frac = rim_fracs.get((well, frame, cid), rim_fracs.get((frame, cid)))
            if frac is not None and frac > options.rim_overlap_max:
                excluded = True
            r["rim_excluded"] = int(excluded)
            r["rim_frac"] = frac
            if not r["rim_excluded"]:
                valid.append((frame, int(r["area_px"])))
                if base is None:
                    base = (frame, t_hours[frame], int(r["area_px"]))
        baseline[key] = base

        reasons = []
        if base and series_start_frame is not None \
                and base[0] < series_start_frame + options.start_frames:
            reasons.append("from_start")
        if len(valid) >= 3:
            windows = [a2 / a1 for i, (l1, a1) in enumerate(valid)
                       for l2, a2 in valid[i + 1:] if l2 - l1 == 2]
            if windows and max(windows) < 1.0 + options.min_growth_2step:
                reasons.append("no_growth")
        for r in crows:
            r["bad"] = int(bool(reasons))
            r["bad_reason"] = ";".join(reasons)

    out = []
    for r in order:
        cid = int(r["colony_id"])
        base = baseline[(r["well"], cid)]
        th = t_hours[int(r["frame"])]
        out.append({
            "well": r["well"], "frame": int(r["frame"]),
            "t_hours": round(th, 3),
            "colony_id": cid, "group_id": int(r["group_id"]),
            "merged_with": r["merged_with"], "ancestors": r["ancestors"],
            "pre_existing": int(r["pre_existing"]),
            "area_px": int(r["area_px"]),
            "age_h": round(th - base[1], 3) if base else None,
            "rim_frac": round(r["rim_frac"], 4) if r["rim_frac"] is not None
            else None,
            "rim_excluded": r["rim_excluded"],
            "bad": r["bad"], "bad_reason": r["bad_reason"],
        })
    return out


# ----------------------------------------------------------------- bad well ---


def split_bad(traj, options):
    """Split trajectories into usable rows and the bad-segment bookkeeping.

    Bad segments (colony-level ``bad`` flag) always leave the aggregates and
    plots. Wells holding at least ``max_bad_segments`` distinct bad segments
    additionally drop out completely (their remaining colonies are suspect
    too) unless ``keep_bad_wells`` is set. Returns ``(usable rows, per-well
    bad colony ids, excluded well names)``.
    """
    per_well_bad = {}
    for r in traj:
        if r["bad"]:
            per_well_bad.setdefault(r["well"], set()).add(r["colony_id"])
    excluded = {w for w, ids in per_well_bad.items()
                if len(ids) >= options.max_bad_segments
                and not options.keep_bad_wells}
    traj_f = [r for r in traj if r["well"] not in excluded and not r["bad"]]
    return traj_f, per_well_bad, excluded


# ---------------------------------------------------- pre-existing probe ------


def probe_pre_existing(well, rows, seg_root, options):
    """Flag colonies that already existed on the root's base image.

    For difference-based roots (name ``...diff_LO<nnn>``): the base image is
    the raw frame at the reference frame. The difference pipeline cannot see
    it (the reference frame is differenced against itself and comes out
    blank), so colonies that already existed there only surface later — first
    detected one or more frames after the reference, often already large.
    This probes the raw root's label image at the reference frame at each
    colony's first-detection centroid: a colony landing on colony territory
    that existed at the base frame (a label at the centroid or within 12 px,
    or inside a raw colony's area-equivalent disk + 15 px slack) is flagged
    ``pre_existing=1`` on all its rows (OR-ed with the pipeline's own flag).
    No-op for raw roots and when the raw root has no matching label image.
    """
    m = re.search(r"diff_LO(\d+)$", Path(seg_root).name)
    if m is None or options.raw_seg_dir is None:
        return
    ref_frame = int(m.group(1))
    well_dir = Path(options.raw_seg_dir) / well
    paths = sorted(well_dir.glob(f"*--LO{ref_frame:03d}--*__labels.tif"))
    if not paths:
        log.warning("[%s] no raw LO%03d label image under %s; pre-existing "
                    "probe skipped", well, ref_frame, options.raw_seg_dir)
        return
    lab = io.read_label_image(paths[0])
    csv_path = well_dir / "colonies.csv"
    raw_rows = io.read_colonies(csv_path) if csv_path.exists() else []
    discs = [(float(r["centroid_y"]), float(r["centroid_x"]),
              (int(r["area_px"]) / np.pi) ** 0.5)
             for r in raw_rows if int(r["frame"]) == ref_frame]
    per_colony = {}
    for r in rows:
        per_colony.setdefault(int(r["colony_id"]), []).append(r)
    flagged = []
    for cid, crows in per_colony.items():
        first = min(crows, key=lambda r: int(r["frame"]))
        cy, cx = float(first["centroid_y"]), float(first["centroid_x"])
        y, x = int(round(cy)), int(round(cx))
        win = lab[max(0, y - 12):y + 13, max(0, x - 12):x + 13]
        hit = bool(win.size) and int(win.max()) > 0
        if not hit:
            hit = any((dcy - cy) ** 2 + (dcx - cx) ** 2 <= (drad + 15) ** 2
                      for dcy, dcx, drad in discs)
        if hit:
            for r in crows:
                r["pre_existing"] = 1
            flagged.append(cid)
    if flagged:
        log.info("[%s] pre-existing on the LO%03d base image: colony ids %s",
                 well, ref_frame, ",".join(str(c) for c in sorted(flagged)))


# ---------------------------------------------------------------- rates ------


def growth_rates(traj):
    """Per-colony growth statistics over the valid (not rim-excluded) frames.

    - specific rate: least-squares slope of ln(area) vs hours over the valid
      frames with positive area (>= 2 frames, dt > 0), reported as h^-1 and
      doublings/day;
    - absolute growth: (last - first) valid area per hour, px/h;
    - ``rim_excluded_frame``: first frame with rim_excluded=1, empty if never.
    Colonies with a single valid frame get NaN rates (they still carry their
    areas in trajectories.csv). Bad segments and pre-existing segments keep
    their rows, flagged ``bad``/``bad_reason`` and ``pre_existing``; the
    aggregates and plots skip both.
    """
    per_colony = {}
    for r in traj:
        per_colony.setdefault((r["well"], r["colony_id"]), []).append(r)
    out = []
    for (well, cid), crows in sorted(per_colony.items()):
        crows.sort(key=lambda r: r["frame"])
        valid = [r for r in crows if not r["rim_excluded"]]
        ex_frame = next((r["frame"] for r in crows if r["rim_excluded"]), None)
        row = {
            "well": well, "colony_id": cid,
            "group_id": crows[0]["group_id"],
            "first_frame": valid[0]["frame"] if valid else crows[0]["frame"],
            "last_frame": valid[-1]["frame"] if valid else crows[-1]["frame"],
            "n_frames": len(crows), "n_frames_valid": len(valid),
            "area_first": valid[0]["area_px"] if valid else None,
            "area_last": valid[-1]["area_px"] if valid else None,
            "max_ancestors": max(r["ancestors"] for r in crows),
            "rim_excluded_frame": ex_frame,
            "bad": crows[0]["bad"], "bad_reason": crows[0]["bad_reason"],
            "pre_existing": crows[0]["pre_existing"],
        }
        rate = abs_rate = None
        if len(valid) >= 2:
            t = np.array([r["t_hours"] for r in valid])
            a = np.array([r["area_px"] for r in valid], dtype=float)
            if t[-1] > t[0] and (a > 0).all():
                k, _ = np.polyfit(t, np.log(a), 1)
                rate = float(k)                     # h^-1
                abs_rate = float((a[-1] - a[0]) / (t[-1] - t[0]))
        row["growth_rate_per_h"] = None if rate is None else round(rate, 5)
        row["growth_doublings_per_day"] = (None if rate is None
                                           else round(rate * 24 / np.log(2), 3))
        row["abs_growth_px_per_h"] = None if abs_rate is None else round(abs_rate, 1)
        out.append(row)
    return out


# -------------------------------------------------------------- aggregates ---


def aggregate_areas(traj):
    """Pooled per (frame, ancestors) colony counts and area statistics.

    One row per (frame, ancestors) bucket over the input rows (which are
    expected to be the usable rows: non-bad, from non-excluded wells, with
    pre-existing colonies dropped): colony count, well count, and the mean /
    median area. Rim-excluded rows and rows without an age are left out.
    """
    groups = {}
    frame_time = {}
    for r in traj:
        if r["rim_excluded"] or r["age_h"] is None:
            continue
        key = (r["frame"], r["ancestors"])
        groups.setdefault(key, []).append(r)
        frame_time.setdefault(r["frame"], []).append(r["t_hours"])
    rows = []
    for (frame, n), rs in sorted(groups.items()):
        areas = np.array([r["area_px"] for r in rs], dtype=float)
        rows.append({
            "frame": frame,
            "t_hours": round(float(np.mean(frame_time[frame])), 3),
            "ancestors": n, "n_colonies": len(rs),
            "n_wells": len({r["well"] for r in rs}),
            "mean_area_px": round(float(areas.mean()), 1),
            "median_area_px": round(float(np.median(areas)), 1),
        })
    return rows


def aggregate_rates(rates):
    """Pooled per (max_ancestors) growth-rate statistics.

    One row per max-ancestors bucket over the input rate rows (expected to be
    the usable ones: non-bad, from non-excluded wells, pre-existing dropped),
    skipping colonies without a rate: count and mean / median doublings/day
    and absolute growth.
    """
    groups = {}
    for r in rates:
        if r["growth_doublings_per_day"] is None:
            continue
        groups.setdefault(r["max_ancestors"], []).append(r)
    rows = []
    for n, rs in sorted(groups.items()):
        dd = np.array([r["growth_doublings_per_day"] for r in rs], dtype=float)
        ab = np.array([r["abs_growth_px_per_h"] for r in rs], dtype=float)
        rows.append({
            "max_ancestors": n, "n_colonies": len(rs),
            "mean_doublings_per_day": round(float(dd.mean()), 3),
            "median_doublings_per_day": round(float(np.median(dd)), 3),
            "mean_abs_growth_px_per_h": round(float(ab.mean()), 1),
            "median_abs_growth_px_per_h": round(float(np.median(ab)), 1),
        })
    return rows


# ---------------------------------------------------------------- plots ------


#: One colour per ancestor-count group; counts above 9 share the last.
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def group_color(n):
    """Colour for an ancestor-count group; counts above 9 share the last."""
    return COLORS[min(int(n), len(COLORS) - 1)]


def group_label(n):
    """Name for an ancestor-count group: '1 ancestor', '3 ancestors', ...

    The count is inclusive of the colony itself, so an unmerged colony
    reads as a single ancestor rather than zero.
    """
    return f"{n} ancestor" if int(n) == 1 else f"{n} ancestors"


def group_alpha(n, n_ref, floor=0.25):
    """Opacity for a group aggregate, faded by how many members back it.

    ``n_ref`` is the largest group size in the figure, so the best-supported
    group is drawn near-full and singleton groups at the floor.
    """
    if not n_ref:
        return 1.0
    return floor + (1.0 - floor) * float(np.sqrt(min(n, n_ref) / n_ref))


def plot_growth_curves(traj, out_path, well_names=""):
    """Individual colony trajectories: absolute area on both time axes.

    Left: area over well time (absolute). Right: area over hours since the
    colony's first valid frame (relative). Curves are split into same-colour
    runs by their ancestor count at that point; the legend maps colours to
    ancestor counts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    per_colony = {}
    for r in traj:
        if r["rim_excluded"] or r["age_h"] is None:
            continue
        per_colony.setdefault((r["well"], r["colony_id"]), []).append(r)

    seen = set()
    for ax, xkey, xlabel in ((axes[0], "t_hours", "well time (h)"),
                             (axes[1], "age_h",
                              "hours since colony's first valid frame")):
        for crows in per_colony.values():
            crows.sort(key=lambda r: r["frame"])
            runs, cur, last = [], [], None
            for r in crows:  # split each curve into same-colour runs
                if last is not None and r["ancestors"] != last:
                    runs.append((last, cur))
                    cur = []
                cur.append(r)
                last = r["ancestors"]
            runs.append((last, cur))
            for n, run in runs:
                seen.add(n)
                ax.plot([r[xkey] for r in run], [r["area_px"] for r in run],
                        color=group_color(n), lw=0.8, alpha=0.35, marker="o",
                        ms=1.8)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("colony area (px)")
        # legend: one entry per ancestor count - a handle per run would mean
        # thousands of entries
        for n in sorted(seen):
            ax.plot([], [], color=group_color(n), lw=2, label=group_label(n))
        ax.legend(fontsize=8)
    axes[0].set_title(f"colony growth, absolute time{well_names}")
    axes[1].set_title("colony growth, time since first appearance")
    fig.suptitle("")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_mean_areas(traj, areas_agg, out_path, well_names="", alpha_on=True):
    """Mean absolute area per ancestor-count group on both time axes.

    Left: over well time (pooled per frame, from areas_by_ancestors). Right:
    over hours since the group members' first valid frame. Buckets with fewer
    than 3 members are not drawn; point opacity fades with the bucket's
    member count (best-supported group near-full, rare groups at the floor)
    so confidence stays readable; connectors use the group's median support.
    The legend shows each group's peak support.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    MIN_MEMBERS = 3

    # left: absolute time, straight from the pooled per-frame aggregate
    ax = axes[0]
    by_n = {}
    for r in areas_agg:
        by_n.setdefault(r["ancestors"], []).append(r)
    n_ref = max((max(r["n_colonies"] for r in rs) for rs in by_n.values()),
                default=1)
    for n, rs in sorted(by_n.items()):
        rs = sorted((r for r in rs if r["n_colonies"] >= MIN_MEMBERS),
                    key=lambda r: r["frame"])
        if not rs:
            continue
        peak = max(r["n_colonies"] for r in rs)
        a_line = (group_alpha(int(np.median([r["n_colonies"] for r in rs])),
                              n_ref) if alpha_on else 0.8)
        ax.plot([r["t_hours"] for r in rs], [r["mean_area_px"] for r in rs],
                color=group_color(n), alpha=a_line, lw=1.2)
        ax.scatter([r["t_hours"] for r in rs], [r["mean_area_px"] for r in rs],
                   s=14, color=group_color(n),
                   alpha=[group_alpha(r["n_colonies"], n_ref) if alpha_on
                          else 0.8 for r in rs],
                   label=f"{group_label(n)} (n up to {peak})")
    ax.set_yscale("log")
    ax.set_xlabel("well time (h)")
    ax.set_ylabel("mean colony area (px)")
    ax.set_title("mean area over time (absolute)")
    ax.legend(fontsize=8)

    # right: relative time (hours since first valid frame), bucketed in-plot
    ax = axes[1]
    buckets = {}
    for r in traj:
        if r["rim_excluded"] or r["age_h"] is None:
            continue
        # frames are hourly; bin to whole hours so T-timestamp jitter
        # (+-2 ms between wells) does not split each age into sub-buckets
        buckets.setdefault((round(r["age_h"]), r["ancestors"]), []).append(
            r["area_px"])
    n_ref = max((len(v) for v in buckets.values()), default=1)
    for n in sorted({k[1] for k in buckets}):
        pts = sorted((age, np.mean(v), len(v)) for (age, nn), v in
                     buckets.items() if nn == n and len(v) >= MIN_MEMBERS)
        if not pts:
            continue
        a_line = (group_alpha(int(np.median([p[2] for p in pts])), n_ref)
                  if alpha_on else 0.8)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=group_color(n),
                alpha=a_line, lw=1.2)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14,
                   color=group_color(n),
                   alpha=[group_alpha(p[2], n_ref) if alpha_on else 0.8
                          for p in pts],
                   label=f"{group_label(n)} (n up to {max(p[2] for p in pts)})")
    ax.set_yscale("log")
    ax.set_xlabel("hours since colony's first valid frame")
    ax.set_ylabel("mean colony area (px)")
    ax.set_title("mean area over time (since first appearance)")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_growth_rates(rates, out_path, well_names="", alpha_on=True):
    """Growth-rate distributions per (max) ancestor count.

    Boxes and strip points fade with the group's member count so rare groups
    read as lower-confidence (disable with ``no_group_alpha``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    groups = {}
    for r in rates:
        if r["growth_doublings_per_day"] is None:
            continue
        groups.setdefault(r["max_ancestors"], []).append(r)
    ns = sorted(groups)
    if not ns:
        log.warning("no growth rates to plot")
        plt.close(fig)
        return
    n_ref = max(len(groups[n]) for n in ns)
    alphas = {n: group_alpha(len(groups[n]), n_ref) if alpha_on else 0.8
              for n in ns}
    data_dd = [[r["growth_doublings_per_day"] for r in groups[n]] for n in ns]
    data_ab = [[r["abs_growth_px_per_h"] for r in groups[n]] for n in ns]

    ax = axes[0]
    bp = ax.boxplot(data_dd, positions=ns, widths=0.6, showfliers=False,
                    patch_artist=True, medianprops={"color": "k"})
    for patch, n in zip(bp["boxes"], ns):
        patch.set_facecolor(group_color(n))
        patch.set_alpha(0.25 + 0.5 * (alphas[n] - 0.25))
    # headroom at the top; the n= labels hang from the top edge (va="top")
    # so they clear the panel title
    dd_min = min(min(v) for v in data_dd)
    dd_top = max(max(v) for v in data_dd) * 1.12
    ax.set_ylim(dd_min * 0.99, dd_top)
    rng = np.random.default_rng(0)
    for n, vals in zip(ns, data_dd):
        ax.plot(rng.uniform(n - 0.15, n + 0.15, len(vals)), vals, ".",
                color=group_color(n), ms=3, alpha=alphas[n])
        ax.annotate(f"n={len(vals)}", (n, dd_top), ha="center", va="top",
                    fontsize=8)
    ax.set_xticks(ns)
    ax.set_xlabel("number of ancestors (max per colony)")
    ax.set_ylabel("growth rate (doublings / day)")
    ax.set_title(f"specific growth rate{well_names}", pad=10)

    ax = axes[1]
    bp = ax.boxplot(data_ab, positions=ns, widths=0.6, showfliers=False,
                    patch_artist=True, medianprops={"color": "k"})
    for patch, n in zip(bp["boxes"], ns):
        patch.set_facecolor(group_color(n))
        patch.set_alpha(0.25 + 0.5 * (alphas[n] - 0.25))
    for n, vals in zip(ns, data_ab):
        ax.plot(rng.uniform(n - 0.15, n + 0.15, len(vals)), vals, ".",
                color=group_color(n), ms=3, alpha=alphas[n])
    ab_min = min(min(v) for v in data_ab)
    ab_max = max(max(v) for v in data_ab)
    # fit the (sym)log axis to the data - the default symlog range wastes
    # half the panel on values the data does not have
    if ab_min > 0:
        ax.set_ylim(ab_min * 0.75, ab_max * 1.4)
    else:
        pad = (ab_max - ab_min) * 0.08
        ax.set_ylim(ab_min - pad, ab_max + pad)
    ax.set_xticks(ns)
    ax.set_yscale("symlog")
    ax.set_xlabel("number of ancestors (max per colony)")
    ax.set_ylabel("absolute growth (px / h)")
    ax.set_title("absolute growth rate")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- QA ------


def qa_overlay(well, rows, seg_root, geo, options, out_path):
    """Overlay of fitted circles and colony states on a raw frame.

    Shows the frame with the highest frame number that has rows: fitted well
    circle (blue), rim-zone boundary (orange), colony edges green (valid),
    orange (pre-existing — excluded over its whole lifetime), red
    (rim-excluded) or magenta (bad segment — from the series start or not
    growing), bad colonies filled translucent magenta. Saved as a 512x512
    composite.
    """
    from PIL import Image

    cy, cx, R = geo
    last = max(r["frame"] for r in rows)
    lrows = [r for r in rows if r["frame"] == last]
    frames = io.discover_wells(options.raw_dir, wells=[well]).get(well, [])
    srcs = [f.path for f in frames if f.frame == last]
    if not srcs:
        log.warning("[%s] LO%03d: no raw frame for the QA overlay", well, last)
        return
    im = tifffile.imread(srcs[0]).astype(np.float32)
    u8 = io.stretch_uint8(im)
    rgb = np.stack([u8] * 3, -1)
    well_dir = Path(seg_root) / well
    labels = sorted(well_dir.glob(f"*--LO{last:03d}--*__labels.tif"))
    lab = io.read_label_image(labels[0]) if labels \
        else np.zeros(im.shape, np.uint32)
    if lab.any():
        for r in lrows:
            cid = r["colony_id"]
            m = lab == cid
            if not m.any():
                continue
            edge = m & ~ndi.binary_erosion(m)
            if r["bad"]:
                sel = np.zeros_like(m, dtype=np.float32)
                sel[m] = 0.45
                rgb = (rgb * (1 - sel[..., None])
                       + np.array([190, 60, 220], np.float32) * sel[..., None]
                       ).astype(np.uint8)
                rgb[edge] = (190, 60, 220)
            elif r["pre_existing"]:
                sel = np.zeros_like(m, dtype=np.float32)
                sel[m] = 0.45
                rgb = (rgb * (1 - sel[..., None])
                       + np.array([255, 160, 0], np.float32) * sel[..., None]
                       ).astype(np.uint8)
                rgb[edge] = (255, 160, 0)
            elif r["rim_excluded"]:
                sel = np.zeros_like(m, dtype=np.float32)
                sel[m] = 0.45
                rgb = (rgb * (1 - sel[..., None])
                       + np.array([255, 40, 40], np.float32) * sel[..., None]
                       ).astype(np.uint8)
                rgb[edge] = (255, 40, 40)
            else:
                rgb[edge] = (40, 220, 40)
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rgb[np.abs(d - R) < 2.5] = (60, 120, 255)
    rgb[np.abs(d - (R - options.rim_width_px)) < 2.5] = (255, 160, 40)
    Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR) \
        .save(out_path)


# ------------------------------------------------------------- pipeline -----


#: The canonical trajectories.csv columns (spec: I/O contract).
TRAJ_FIELDS = ["well", "frame", "t_hours", "colony_id", "group_id",
               "merged_with", "ancestors", "pre_existing", "area_px",
               "age_h", "rim_frac", "rim_excluded", "bad", "bad_reason"]

#: Typed columns when reading a trajectories CSV back (``plots_only``);
#: empty fields come back as None.
TRAJ_TYPES = {"frame": int, "t_hours": float, "colony_id": int,
              "group_id": int, "ancestors": int, "pre_existing": int,
              "area_px": int, "age_h": float, "rim_frac": float,
              "rim_excluded": int, "bad": int}

#: The canonical growth_rates.csv columns (spec: I/O contract).
RATE_FIELDS = ["well", "colony_id", "group_id", "first_frame", "last_frame",
               "n_frames", "n_frames_valid", "area_first", "area_last",
               "max_ancestors", "rim_excluded_frame", "growth_rate_per_h",
               "growth_doublings_per_day", "abs_growth_px_per_h",
               "bad", "bad_reason", "pre_existing"]

#: The canonical bad_segments.csv columns (spec: I/O contract).
BAD_FIELDS = ["well", "colony_id", "bad_reason", "first_frame", "last_frame",
              "n_frames_valid", "area_first", "area_last", "n_bad_in_well",
              "well_excluded"]

#: The canonical areas_by_ancestors.csv columns (spec: I/O contract).
AREAS_FIELDS = ["frame", "t_hours", "ancestors", "n_colonies", "n_wells",
                "mean_area_px", "median_area_px"]

#: The canonical rates_by_ancestors.csv columns (spec: I/O contract).
RATES_FIELDS = ["max_ancestors", "n_colonies", "mean_doublings_per_day",
                "median_doublings_per_day", "mean_abs_growth_px_per_h",
                "median_abs_growth_px_per_h"]


def load_geometry(well, options, cache):
    """Fitted well circle for one well, from the cache or freshly fitted.

    Fits up to the three earliest raw frames and averages them (the stage
    does not move within a well, so the circle is a fixed property; the
    mean dampens the ~10-15 px fit noise). Returns (cy, cx, R) or None
    (rim exclusion is then disabled for the well, logged); a None result is
    cached so later roots do not refit the same well.
    """
    if well in cache:
        return cache[well]
    raw_dir = Path(options.raw_dir)
    frames = (io.discover_wells(raw_dir, wells=[well]).get(well, [])[:3]
              if raw_dir.is_dir() else [])
    g = fit_well_circle([f.path for f in frames])
    if g is None:
        log.warning("[%s] no plausible well circle found in %d frame(s); "
                    "rim exclusion disabled for this well", well, len(frames))
    cache[well] = g
    return g


def process_well(well, seg_root, options, out_dir, geometry):
    """Analyse one well; returns the trajectory rows (empty if skipped)."""
    rows_path = Path(seg_root) / well / "colonies.csv"
    if not rows_path.exists():
        log.info("[%s] no colonies.csv, skipping", well)
        return []
    rows = io.read_colonies(rows_path)
    for r in rows:
        r["frame"] = int(r["frame"])
    probe_pre_existing(well, rows, seg_root, options)
    geo = load_geometry(well, options, geometry)
    rim_fracs = rim_stats(rows, seg_root, geo, options.rim_width_px)
    # the well's own frame series defines "the beginning" (the earliest frame
    # with a label image - colonies.csv rows start later wherever growth does)
    label_frames = [io.parse_frame(p.name)
                    for p in (Path(seg_root) / well).glob("*__labels.tif")]
    series_start_frame = min((f for f in label_frames if f is not None),
                             default=None)
    traj = build_trajectories(rows, rim_fracs, options, series_start_frame)
    n_ex = sum(r["rim_excluded"] for r in traj)
    bad = {}
    for r in traj:
        if r["bad"]:
            bad.setdefault(r["bad_reason"], set()).add(r["colony_id"])
    if geo:
        log.info("[%s] %d rows, %d rim-excluded (R=%.0f, zone>%d px, frac>%g)",
                 well, len(traj), n_ex, geo[2], options.rim_width_px,
                 options.rim_overlap_max)
    else:
        log.info("[%s] %d rows, rim exclusion unavailable (no well circle)",
                 well, len(traj))
    for reason, ids in sorted(bad.items()):
        log.info("[%s] %d bad segment(s) [%s]: ids %s",
                 well, len(ids), reason,
                 ",".join(str(i) for i in sorted(ids)))
    if not options.no_qa and geo:
        qa_dir = Path(out_dir) / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_overlay(well, traj, seg_root, geo, options,
                   qa_dir / f"{well}_rim.png")
    return traj


def write_outputs(traj, options, out_dir):
    """Derive and write the CSVs and plots for one segmentation root.

    trajectories.csv and growth_rates.csv keep every colony (bad ones
    flagged, pre-existing ones via the pre_existing column); the pooled
    aggregates and the plots use only rows from wells that are not excluded
    for repeated bad segments, that are not bad, and that are not
    pre-existing (unless ``keep_pre_existing``). Returns (rates, areas_agg).
    """
    out_dir = Path(out_dir)
    io.write_csv_rows(out_dir / "trajectories.csv", traj, TRAJ_FIELDS)

    rates = growth_rates(traj)
    io.write_csv_rows(out_dir / "growth_rates.csv", rates, RATE_FIELDS)

    traj_f, per_well_bad, excluded = split_bad(traj, options)
    rates_f = [r for r in rates if r["well"] not in excluded and not r["bad"]]
    if not options.keep_pre_existing:
        pre = {(r["well"], r["colony_id"]) for r in traj if r["pre_existing"]}
        traj_f = [r for r in traj_f if (r["well"], r["colony_id"]) not in pre]
        rates_f = [r for r in rates_f
                   if (r["well"], r["colony_id"]) not in pre]
        log.info("%d pre-existing segment(s) excluded from aggregates/plots",
                 len(pre))
    n_bad_total = sum(len(ids) for ids in per_well_bad.values())
    log.info("%d bad segment(s) in %d well(s)%s%s",
             n_bad_total, len(per_well_bad),
             f"; {len(excluded)} well(s) excluded from aggregates/plots: "
             f"{sorted(excluded)}" if excluded else "",
             " (--keep_bad_wells set)" if excluded and options.keep_bad_wells
             else "")

    io.write_csv_rows(
        out_dir / "bad_segments.csv",
        [{"well": r["well"], "colony_id": r["colony_id"],
          "bad_reason": r["bad_reason"], "first_frame": r["first_frame"],
          "last_frame": r["last_frame"],
          "n_frames_valid": r["n_frames_valid"],
          "area_first": r["area_first"], "area_last": r["area_last"],
          "n_bad_in_well": len(per_well_bad[r["well"]]),
          "well_excluded": int(r["well"] in excluded)}
         for r in rates if r["bad"]],
        BAD_FIELDS)

    areas_agg = aggregate_areas(traj_f)
    io.write_csv_rows(out_dir / "areas_by_ancestors.csv", areas_agg,
                      AREAS_FIELDS)
    io.write_csv_rows(out_dir / "rates_by_ancestors.csv",
                      aggregate_rates(rates_f), RATES_FIELDS)

    if not options.no_plots:
        n_wells = len({r["well"] for r in traj_f})
        n_wells_all = len({r["well"] for r in traj})
        n_note = (f" ({n_wells} of {n_wells_all} wells, {len(traj_f)} rows)"
                  if n_wells < n_wells_all
                  else f" ({n_wells} wells, {len(traj_f)} rows)")
        plot_growth_curves(traj_f,
                           out_dir / "growth_curves_by_ancestors.png",
                           well_names=n_note)
        plot_mean_areas(traj_f, areas_agg,
                        out_dir / "mean_area_by_ancestors.png",
                        well_names=n_note,
                        alpha_on=not options.no_group_alpha)
        plot_growth_rates(rates_f,
                          out_dir / "growth_rates_by_ancestors.png",
                          well_names=n_note,
                          alpha_on=not options.no_group_alpha)
    return rates, areas_agg


@dataclass(frozen=True)
class RootSummary:
    """Analyze results for one segmentation root."""

    name: str  # the root's directory name (its output subdirectory)
    wells: int  # wells analysed
    colonies: int  # distinct (well, colony_id) in the trajectories
    rows: int  # trajectory rows
    rim_excluded: int  # rim-excluded trajectory rows
    bad_segments: int  # bad colonies across the wells
    excluded_wells: list[str]  # wells excluded from aggregates/plots
    pre_existing: int  # pre-existing (well, colony_id) segments


@dataclass
class AnalysisSummary:
    """Analyze results for a run (one or more segmentation roots)."""

    #: One entry per root that produced outputs, keyed by root name.
    roots: dict[str, RootSummary] = field(default_factory=dict)
    #: The fitted well circles actually used, well -> (cy, cx, R).
    geometry: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)
    #: Segmentation roots that failed with an exception (by name).
    failures: list[str] = field(default_factory=list)


def _replot(name, out_dir, options):
    """The plots_only path: re-derive everything from the existing
    trajectories CSV — no image IO, no seg-tree reading, no geometry cache.

    Returns a RootSummary, or None when there is no trajectories.csv to
    re-plot from.
    """
    path = Path(out_dir) / "trajectories.csv"
    if not path.exists():
        log.warning("%s: no trajectories.csv, cannot re-plot", name)
        return None
    traj = []
    for r in io.read_csv_rows(path):
        for k, typ in TRAJ_TYPES.items():
            r[k] = typ(r[k]) if r[k] != "" else None
        traj.append(r)
    log.info("%s: re-plotting from %s", name, path)
    write_outputs(traj, options, out_dir)
    _, per_well_bad, excluded = split_bad(traj, options)
    pre = {(r["well"], r["colony_id"]) for r in traj if r["pre_existing"]}
    return RootSummary(
        name=name, wells=len({r["well"] for r in traj}),
        colonies=len({(r["well"], r["colony_id"]) for r in traj}),
        rows=len(traj), rim_excluded=sum(r["rim_excluded"] for r in traj),
        bad_segments=sum(len(ids) for ids in per_well_bad.values()),
        excluded_wells=sorted(excluded), pre_existing=len(pre))


def _process_seg_dir(seg_root, name, out_dir, options, geometry):
    """Analyse one segmentation root: per-well trajectories + pooled outputs.

    Returns a RootSummary, or None when nothing was written (no wells, no
    colony rows).
    """
    t0 = time.time()
    wells = sorted({w for w in (io.parse_well(p.name)
                                for p in seg_root.iterdir() if p.is_dir())
                    if w})
    if options.wells:
        requested = [w for chunk in options.wells for w in chunk.split(",")
                     if w]
        missing = [w for w in requested if w not in wells]
        if missing:
            log.warning("%s: requested wells not found: %s", name, missing)
        wells = [w for w in wells if w in requested]
    if not wells:
        log.warning("%s: no wells to analyse", name)
        return None

    traj = []
    for i, well in enumerate(wells):
        log.info("== %s: well %d/%d ==", name, i + 1, len(wells))
        try:
            traj.extend(process_well(well, seg_root, options, out_dir,
                                     geometry))
        except Exception:
            log.error("[%s] FAILED\n%s", well, traceback.format_exc())

    if not traj:
        log.warning("%s: no colony rows at all", name)
        return None

    write_outputs(traj, options, out_dir)
    _, per_well_bad, excluded = split_bad(traj, options)
    pre = {(r["well"], r["colony_id"]) for r in traj if r["pre_existing"]}
    n_col = len({(r["well"], r["colony_id"]) for r in traj})
    n_ex = sum(r["rim_excluded"] for r in traj)
    log.info("%s done: %d wells, %d colonies, %d rows (%d rim-excluded), "
             "%.1fs", name, len(wells), n_col, len(traj), n_ex,
             time.time() - t0)
    return RootSummary(
        name=name, wells=len(wells), colonies=n_col, rows=len(traj),
        rim_excluded=n_ex,
        bad_segments=sum(len(ids) for ids in per_well_bad.values()),
        excluded_wells=sorted(excluded), pre_existing=len(pre))


def run_analysis(options: AnalysisOptions) -> AnalysisSummary:
    """Run the analyze stage over every segmentation root in options.

    Each root gets its outputs under ``<out_root>/<root-name>/``
    (trajectories, growth rates, bad segments, pooled aggregates, plots,
    per-well QA overlays); the fitted well circles are shared across roots
    and cached in ``<out_root>/well_geometry.csv`` (``refresh_geometry``
    forces refits, ``plots_only`` never fits). A root that fails with an
    exception is recorded in :attr:`AnalysisSummary.failures` and the
    remaining roots are still processed.
    """
    out_root = Path(options.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    geo_path = out_root / "well_geometry.csv"

    # the shared geometry cache (skipped by plots_only, which never fits)
    geometry: dict[str, tuple[float, float, float] | None] = {}
    if geo_path.exists() and not options.refresh_geometry \
            and not options.plots_only:
        geometry.update(io.read_geometry(geo_path))
        log.info("loaded %d cached well circles from %s", len(geometry),
                 geo_path.name)

    summary = AnalysisSummary()
    for seg_root in options.seg_dirs:
        seg_root = Path(seg_root)
        name = seg_root.name
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if options.plots_only:
                root = _replot(name, out_dir, options)
            else:
                root = _process_seg_dir(seg_root, name, out_dir, options,
                                        geometry)
            if root is not None:
                summary.roots[name] = root
        except Exception:
            log.error("%s FAILED\n%s", name, traceback.format_exc())
            summary.failures.append(name)

    if not options.plots_only:
        io.write_geometry(geo_path, {w: g for w, g in geometry.items() if g})
    summary.geometry = {w: g for w, g in geometry.items() if g}
    return summary
