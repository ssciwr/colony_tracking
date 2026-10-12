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

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from . import io, log

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
        # strips ".tif"; works for raw and __diff_uint8 input frames
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


def parse_merged(s: str) -> set:
    """The colony IDs listed in a ``merged_with`` field (empty when none)."""
    return {int(v) for v in s.split(";") if v}


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
      had merged with C earlier, ends at 3;
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

    # Colonies each colony ever absorbed (union over its frames; the
    # pipeline's `merged_with` lists the colonies whose last-known masks a
    # segment covers - including colonies an absorbed colony had taken over
    # earlier, whose stale masks stay covered). Colony IDs are per-well, so
    # every merge-graph key below is (well, colony_id).
    absorbed = {}
    for key, crows in per_colony.items():
        s = set()
        for r in crows:
            s |= parse_merged(r["merged_with"])
        absorbed[key] = s

    # All original colonies combined in each colony's segment, transitively:
    # an absorbed merge product contributes its own constituents (A absorbs
    # B, B had absorbed C -> A's segment comprises A, B and C). Memoised; the
    # tracker follows time, so cycles cannot arise (guarded anyway).
    constituents = {}

    def constituents_of(key, visiting=()):
        if key in constituents:
            return constituents[key]
        if key in visiting:
            return set()
        comp = {key[1]}
        for y in absorbed.get(key, ()):
            comp |= constituents_of((key[0], y), visiting + (key,))
        constituents[key] = comp
        return comp

    for key in per_colony:
        constituents_of(key)

    baseline = {}
    for key, crows in per_colony.items():
        well, cid = key
        seen = set()  # raw IDs listed in merged_with so far
        counted = set()  # constituents of everything absorbed so far
        base = None
        excluded = False  # once in the rim, always in the rim (colonies only grow)
        valid = []  # (frame, area) of frames with usable measurements
        for r in crows:
            frame = int(r["frame"])
            ids = parse_merged(r["merged_with"])
            for y in ids - seen:
                counted |= constituents_of((well, y))
            seen |= ids
            # rim_stats keys by (frame, colony_id) for its per-well input; the
            # (well, frame, colony_id) form is accepted as well.
            frac = rim_fracs.get((well, frame, cid), rim_fracs.get((frame, cid)))
            if frac is not None and frac > options.rim_overlap_max:
                excluded = True
            r["rim_excluded"] = int(excluded)
            r["rim_frac"] = frac
            r["ancestors"] = 1 + len(counted - {cid})
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
