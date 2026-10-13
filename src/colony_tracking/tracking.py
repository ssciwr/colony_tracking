"""Colony identity: forward tracking, merges, forks, and ancestors.

The pure identity logic shared by the full segment run and the re-ID mode:
no model, no file I/O — the inputs are label images and row lists, the
outputs are label images, CSV-ready rows, and counters.

Identity rules (tracked forward in time, per well and per reference group):

- A segment **continues** a colony (same ID) when it matches the colony's
  last-known mask by IoU ≥ ``MATCH_IOU`` (0.3) **or** covers at least
  ``ABSORB_COVERAGE`` (0.5) of it. The containment rule is the growth case
  the IoU rule cannot see: colonies only grow, so a colony's next segment
  contains its previous extent — but with strong growth in one frame
  (more than ~3.3×) the IoU of a perfect continuation drops below the
  threshold, and without the containment rule the segment would start a
  new colony and record a bogus merge. Among matching colonies the best
  IoU wins, ties broken by the larger colony — favouring the up-to-date
  colony over the stale masks of long-absorbed ones, and the dominant
  colony of a merge over the colonies it absorbs (for equal coverage the
  larger colony has the higher IoU).
- A segment that continues no colony but covers at least ``ABSORB_COVERAGE``
  (0.5) of a further colony's last-known mask **absorbs** it: the survivor
  keeps its own ID, the absorbed IDs are recorded in that frame's row as
  ``merged_with`` (';'-joined) and one merge event is counted. Absorbed
  colonies stay tracked with their last-known mask — while it stays
  covered it is listed again in each later frame's ``merged_with`` (the
  analysis unions the histories), and when the colony is segmented
  separately again it resumes its own ID.
- A segment matching no colony that lies mostly (>= ``ABSORB_COVERAGE`` of
  its area) inside one known colony's territory splits that single colony:
  it is a **fork** — the same colony ID continues on it, the event is
  logged and counted in ``forked_segments``.
- Any other unmatched segment starts a **new** colony ID.

Before tracking, segments of one frame and group that touch (within
``TOUCH_DILATE`` = 3 px dilation, tolerating boundary wobble) are
pre-merged into the larger one — colonies that touch have grown together —
unless ``merge_touching`` is off.

Pre-existing filter (``exclude_before = N``): every frame with frame
number < N yields an empty label image and no CSV rows (outputs stay 1:1
with the input frames), and every colony whose first detection is at
frame number <= N is flagged ``pre_existing=1`` (rows kept). Frame numbers
are compared directly, so N need not exist in the well. Because the
earlier frames are blanked, flagging effectively means "first detected
at N". This replaces the prototype's ``--exclude_lo`` and
``--drop_labels_before`` (which always travelled together) — a raw-root
run now blanks too, an intentional behaviour change.

Ancestors (:func:`ancestor_counts`): per colony-frame row, the number of
distinct original colonies combined in the segment — the colony itself
plus the transitive closure of everything its ``merged_with`` histories
ever absorbed (an A+B where B had merged with C ends at 3; unmerged = 1).
The quantity the prototype called ``n_merged``, written as ``ancestors``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label

from . import log

# A segment continues a colony if IoU(segment, colony's last-known mask) is
# at least this; this tolerates area growth between frames while favouring
# the up-to-date colony over stale (long-absorbed) masks. A merge is
# recorded when a segment covers >= ABSORB_COVERAGE of an additional
# colony's mask that no segment continues in this frame.
MATCH_IOU = 0.3
ABSORB_COVERAGE = 0.5

# Two segments of one frame that lie in the same group and touch (within
# this dilation radius) are one physical blob: the colonies have grown
# together, even though the decoder keeps separate instances at their bright
# contact interface in difference images. They are pre-merged into the
# larger segment before tracking, so the merge is recorded like any other.
TOUCH_DILATE = 3


# --- geometry (scipy.ndimage: same 4-connectivity as segmentation.py) -------


def _center(mask: np.ndarray) -> tuple[float, float]:
    """The centre of mass (cy, cx) of a non-empty boolean mask."""
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


@dataclass
class Segment:
    """One connected segment of one frame (a decoder instance's component)."""

    mask: np.ndarray  # boolean mask of the segment's pixels
    area: int  # pixels
    cy: float  # centre of mass
    cx: float


def extract_segments(labels: np.ndarray) -> list[Segment]:
    """Split a label image into connected segments.

    Returns one :class:`Segment` per connected component, so a label value
    shared by disjoint patches yields several segments.
    """
    labels = np.asarray(labels)
    segments: list[Segment] = []
    for value in np.unique(labels):
        if value == 0:
            continue
        cc, n = label(labels == value)
        for i in range(1, n + 1):
            m = cc == i
            cy, cx = _center(m)
            segments.append(Segment(mask=m, area=int(m.sum()), cy=cy, cx=cx))
    return segments


def premerge_touching(
    segments: Sequence[Segment],
    *,
    desc: str = "",
    frame: int | None = None,
    group: int | None = None,
) -> list[Segment]:
    """Merge the segments of one frame/group that touch each other.

    Colonies only grow, so two segments in the same reference group that
    touch (within ``TOUCH_DILATE`` px, tolerating boundary wobble) are one
    physical blob that the decoder split at their contact interface —
    especially in difference images, where the interface lights up because
    both colonies change there. The smaller segment is folded into the
    larger one (union mask); the tracker then sees one segment and records
    the merge through its normal absorb logic. The pass repeats until
    stable, so chains A-B-C merge into one segment even when B only bridges
    A and C. Returns the merged segment list; the input segments are left
    untouched and the survivors keep their own centroid (the prototype's
    behaviour — the merged row reports the largest piece's centre).
    """
    where = f"[{desc}] " if desc else ""
    at = ""
    if frame is not None:
        at = f"frame {frame}"
        if group is not None:
            at += f" group {group}"
        at += ": "
    segs = sorted(segments, key=lambda s: -s.area)
    while True:
        kept: list[Segment] = []
        merged_any = False
        for s in segs:
            for k in kept:
                # iterations is always TOUCH_DILATE >= 1 here: scipy's
                # binary_dilation(iterations=0) dilates until stability,
                # while a 0-iteration cross dilation is a no-op.
                if (binary_dilation(k.mask, iterations=TOUCH_DILATE) & s.mask).any():
                    k.mask |= s.mask
                    k.area += s.area
                    merged_any = True
                    log.info(
                        "%s%smerged touching %d px segment into %d px segment",
                        where, at, s.area, k.area,
                    )
                    break
            else:
                # a copy: folding later segments must not touch the caller's
                kept.append(Segment(mask=s.mask.copy(), area=s.area,
                                    cy=s.cy, cx=s.cx))
        if not merged_any:
            return kept
        segs = sorted(kept, key=lambda s: -s.area)


@dataclass
class TrackingResult:
    """The tracking output for one well.

    ``labels`` holds one colony-ID label image (uint32, 0 = background) per
    input frame, 1:1 — frames blanked by ``exclude_before`` come out empty.
    ``rows`` holds CSV-ready dicts with the tracker-owned ``colonies.csv``
    columns: ``frame`` (the frame number), ``colony_id``, ``group_id``,
    ``merged_with`` (';'-joined absorbed IDs, empty when none),
    ``pre_existing`` (0/1), ``area_px``, ``centroid_y``, ``centroid_x``.
    ``counters`` holds the done-marker counts under the ``SEGMENT_DONE_KEYS``
    names (``frames`` is the input frame count; ``discarded_instances`` is
    the caller's pass-through — the tracker never discards).
    """

    labels: list[np.ndarray]
    rows: list[dict[str, object]]
    counters: dict[str, int]


def track_colonies(
    frames: Sequence[tuple[int, np.ndarray]],
    groups: np.ndarray | None,
    *,
    exclude_before: int | None = None,
    merge_touching: bool = True,
    discarded_instances: int = 0,
    desc: str = "",
) -> TrackingResult:
    """Assign colony IDs to the kept segments of all frames of one well.

    ``frames`` holds ``(frame number, local label image)`` pairs of one
    well, ascending by frame number; local IDs carry no identity yet (as
    produced by the growth-mask filtering, or by re-splitting saved label
    images in the re-ID mode). ``groups`` is the reference group label
    image; segments are pooled under the group most of their pixels lie in
    and tracked forward within each group (see the module docstring for the
    identity rules).

    With ``exclude_before = N``, frames with frame number < N contribute no
    segments and stay empty in the output (1:1 with the input frames), and
    colonies first detected at frame number <= N are flagged
    ``pre_existing=1`` (rows kept).

    Returns a :class:`TrackingResult` (label images, CSV-ready rows,
    done-marker counters).
    """
    frames = sorted(frames, key=lambda f: f[0])
    if groups is not None and frames and groups.shape != frames[0][1].shape:
        raise ValueError(
            f"group image shape {groups.shape} does not match "
            f"frame shape {frames[0][1].shape}"
        )
    labels_out = [np.zeros(k.shape, np.uint32) for _, k in frames]
    counters = {
        "frames": len(frames),
        "groups": 0,
        "colonies": 0,
        "pre_existing_colonies": 0,
        "merge_events": 0,
        "forked_segments": 0,
        "discarded_instances": int(discarded_instances),
    }
    rows: list[dict[str, object]] = []
    if groups is None or not groups.any():
        return TrackingResult(labels_out, rows, counters)
    n_groups = int(groups.max())
    counters["groups"] = n_groups
    where = f"[{desc}] " if desc else ""

    # Pool the segments of every frame under their group (max group overlap).
    per_group: dict[int, list[list[Segment]]] = {
        g: [[] for _ in frames] for g in range(1, n_groups + 1)
    }
    for t, (number, k) in enumerate(frames):
        if exclude_before is not None and number < exclude_before:
            continue  # blanked: no segments, the output frame stays empty
        for s in extract_segments(k):
            hits = np.bincount(groups[s.mask].ravel(), minlength=n_groups + 1)
            if not hits[1:].any():  # only possible inside the dilated margin
                log.warning(
                    "%sframe %d: segment of %d px outside every group",
                    where, number, s.area,
                )
                continue
            per_group[int(np.argmax(hits[1:]) + 1)][t].append(s)

    next_id = 1
    for g in sorted(range(1, n_groups + 1), key=lambda g: _center(groups == g)):
        active: list[dict[str, object]] = []  # colonies: id, mask, area, first

        for t, (number, _) in enumerate(frames):
            if exclude_before is not None and number < exclude_before:
                continue  # blanked: nothing is tracked, the frame stays empty
            segs = sorted(per_group[g][t], key=lambda s: -s.area)
            if merge_touching:
                segs = premerge_touching(segs, desc=desc, frame=number, group=g)
            matched: dict[int, dict[str, object]] = {}  # id(segment) -> colony
            taken: set[int] = set()  # colonies already continued this frame
            # The colonies' territory as of this frame's start: an already
            # continued colony's updated mask no longer contains the pieces
            # the decoder split off it, so forks are judged against this.
            start_masks = [o["mask"] for o in active]
            for s in segs:
                # colonies whose last-known extent this segment matches: by
                # IoU, or by containing at least ABSORB_COVERAGE of the
                # colony's mask (the strong-growth case where the IoU of a
                # perfect continuation drops below MATCH_IOU)
                cands = []
                for o in active:
                    if o["id"] in taken:
                        continue
                    inter = float((s.mask & o["mask"]).sum())
                    iou = inter / float(s.area + o["area"] - inter)
                    covered = inter / float(o["area"])
                    if iou >= MATCH_IOU or covered >= ABSORB_COVERAGE:
                        cands.append((iou, o["area"], o))
                if cands:
                    cands.sort(key=lambda c: (-c[0], -c[1]))
                    o = cands[0][2]
                    taken.add(o["id"])
                    matched[id(s)] = o
                    o["mask"] = s.mask  # the colony grows into this segment
                    o["area"] = s.area
                else:  # no colony continued by this segment
                    # Does it split one known colony (fork: the same ID
                    # continues) or start a new one?
                    host: dict[str, object] | None = None
                    best = 0.0
                    for o, start_mask in zip(active, start_masks):
                        cov = float((s.mask & start_mask).sum()) / s.area
                        if cov > best:
                            host, best = o, cov
                    if host is not None and best >= ABSORB_COVERAGE:
                        matched[id(s)] = host
                        counters["forked_segments"] += 1
                        log.warning(
                            "%sframe %d: segment of %d px in group %d "
                            "continues no colony but splits colony %d; "
                            "forking (the same ID continues)",
                            where, number, s.area, g, host["id"],
                        )
                        if host["id"] not in taken:
                            # the fork is the colony's latest-known extent
                            host["mask"] = s.mask
                            host["area"] = s.area
                    else:  # first appearance: a new colony
                        o = {"id": next_id, "mask": s.mask, "area": s.area,
                             "first": number}
                        next_id += 1
                        active.append(o)
                        matched[id(s)] = o
            # merges: colonies covered by this segment but continued by no
            # segment — the survivor keeps its ID, the covered ones are
            # absorbed (their stale masks stay tracked; see the docstring)
            for s in segs:
                o = matched[id(s)]
                absorbed = sorted({
                    c["id"] for c in active
                    if c is not o and c["id"] not in taken
                    and float((s.mask & c["mask"]).sum()) / c["area"]
                    >= ABSORB_COVERAGE
                })
                if absorbed:
                    counters["merge_events"] += 1
                    log.info(
                        "%sframe %d: segment of %d px in group %d continues "
                        "colony %d; absorbing %s",
                        where, number, s.area, g, o["id"],
                        ", ".join(str(i) for i in absorbed),
                    )
                rows.append({
                    "frame": number,
                    "colony_id": o["id"],
                    "group_id": g,
                    "merged_with": ";".join(str(i) for i in absorbed),
                    "pre_existing": int(exclude_before is not None
                                        and o["first"] <= exclude_before),
                    "area_px": s.area,
                    "centroid_y": round(s.cy, 1),
                    "centroid_x": round(s.cx, 1),
                })
                labels_out[t][s.mask] = o["id"]

    counters["pre_existing_colonies"] = len(
        {r["colony_id"] for r in rows if r["pre_existing"]}
    )
    counters["colonies"] = len({r["colony_id"] for r in rows}) - counters[
        "pre_existing_colonies"
    ]
    return TrackingResult(labels_out, rows, counters)


def parse_merged(value: str | None) -> set[int]:
    """Parse a ``merged_with`` value (';'-separated colony IDs) into a set."""
    return {int(v) for v in str(value or "").split(";") if v}


def ancestor_counts(rows: Sequence[Mapping[str, object]]) -> list[int]:
    """Transitive, self-inclusive ancestor count per row.

    For every colony the count starts at 1 (the colony itself) and grows by
    the full constituent set of every colony newly listed in its
    ``merged_with`` histories, resolved transitively: a segment continuing
    colony A, whose row lists colony B — which had absorbed colony C
    earlier — ends at 3 ancestors; an unmerged colony stays at 1. This is
    the quantity the prototype called ``n_merged``, written as
    ``ancestors``.

    Returns one count per input row, in input order. Rows of one colony are
    walked chronologically (sorted by ``frame`` when every row of the
    colony carries one, otherwise in the order given). The tracker follows
    time, so the histories cannot cycle; the resolution guards anyway.

    Prototype-tolerant on the input: rows without a ``merged_with`` field,
    or with an empty/``None`` one, simply merge nothing.
    """
    per_colony: dict[int, list[tuple[int, Mapping[str, object]]]] = {}
    for pos, r in enumerate(rows):
        per_colony.setdefault(int(r["colony_id"]), []).append((pos, r))

    # Colonies each colony ID ever absorbed: the union over its rows — the
    # tracker keeps listing absorbed colonies whose last-known masks stay
    # covered, so an absorbed merge product contributes its own history.
    absorbed: dict[int, set[int]] = {}
    for cid, items in per_colony.items():
        s: set[int] = set()
        for _, r in items:
            s |= parse_merged(r.get("merged_with"))
        absorbed[cid] = s

    # All original colonies combined in each colony's segment, transitively:
    # an absorbed merge product contributes its own constituents (A absorbs
    # B, B had absorbed C -> A's segment comprises A, B and C). Memoised.
    constituents: dict[int, set[int]] = {}

    def constituents_of(cid: int, visiting: tuple[int, ...] = ()) -> set[int]:
        if cid in constituents:
            return constituents[cid]
        if cid in visiting:
            return set()
        comp = {cid}
        for y in absorbed.get(cid, ()):
            comp |= constituents_of(y, visiting + (cid,))
        constituents[cid] = comp
        return comp

    for cid in per_colony:
        constituents_of(cid)

    counts: list[int] = [1] * len(rows)
    for cid, items in per_colony.items():
        if all(r.get("frame") is not None for _, r in items):
            items = sorted(items, key=lambda p: int(p[1]["frame"]))
        seen: set[int] = set()  # raw IDs listed in merged_with so far
        counted: set[int] = set()  # constituents of everything absorbed so far
        for pos, r in items:
            ids = parse_merged(r.get("merged_with"))
            for y in ids - seen:
                counted |= constituents_of(y)
            seen |= ids
            counts[pos] = 1 + len(counted - {cid})
    return counts