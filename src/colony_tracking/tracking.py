"""Colony identity: forward tracking, merges, forks, and ancestors.

The pure identity logic shared by the full segment run and the re-ID mode:
no model, no file I/O — the inputs are label images and row lists, the
outputs are label images, CSV-ready rows, and counters.

Identity rules (tracked forward in time, per well and per reference group):

- A segment whose IoU with a colony's last-known mask reaches ``MATCH_IOU``
  (0.3) **continues** that colony (same ID). Among matching colonies the
  best IoU wins, ties broken by the larger colony — favouring the
  up-to-date colony over the stale masks of long-absorbed ones.
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


# --- pure numpy geometry (no scipy: this module stays numpy + stdlib) -------


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the 4-connected components of a boolean mask.

    Returns ``(labels, n)`` with components numbered 1..n in row-major order
    of their first pixel — the same connectivity and numbering
    ``scipy.ndimage.label`` produces with its default structure, so local
    segment IDs match the prototype's. Runs of set pixels are extracted per
    row and unified with a small union-find over vertically overlapping
    runs (4-connectivity: a pixel only touches the pixel directly above).
    """
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return labels, 0
    parent: list[int] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    runs: list[tuple[int, int, int, int]] = []  # (y, start, end, run id)
    prev: list[tuple[int, int, int]] = []  # (start, end, run id) of the row above
    for y in range(mask.shape[0]):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        cur: list[tuple[int, int, int]] = []
        first = 0  # prev runs with end <= start can never overlap a later run
        for s, e in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            while first < len(prev) and prev[first][1] <= s:
                first += 1
            j = first
            while j < len(prev) and prev[j][0] < e:
                proot = find(prev[j][2])
                if proot != rid:
                    parent[proot] = rid
                j += 1
            cur.append((s, e, rid))
            runs.append((y, s, e, rid))
        prev = cur

    # Number the components 1..n in row-major order of their first pixel.
    name_of_root: dict[int, int] = {}
    for _, _, _, rid in runs:
        root = find(rid)
        if root not in name_of_root:
            name_of_root[root] = len(name_of_root) + 1
    for y, s, e, rid in runs:
        labels[y, s:e] = name_of_root[find(rid)]
    return labels, len(name_of_root)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a boolean mask with the 4-neighbourhood cross, k iterations."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


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
        cc, n = _label_components(labels == value)
        for i in range(1, n + 1):
            m = cc == i
            cy, cx = _center(m)
            segments.append(Segment(mask=m, area=int(m.sum()), cy=cy, cx=cx))
    return segments


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
            segs = sorted(per_group[g][t], key=lambda s: -s.area)
            matched: dict[int, dict[str, object]] = {}  # id(segment) -> colony
            taken: set[int] = set()  # colonies already continued this frame
            for s in segs:
                # colonies whose last-known extent this segment matches
                cands = []
                for o in active:
                    if o["id"] in taken:
                        continue
                    inter = float((s.mask & o["mask"]).sum())
                    iou = inter / float(s.area + o["area"] - inter)
                    if iou >= MATCH_IOU:
                        cands.append((iou, o["area"], o))
                if cands:
                    cands.sort(key=lambda c: (-c[0], -c[1]))
                    o = cands[0][2]
                    taken.add(o["id"])
                    matched[id(s)] = o
                    o["mask"] = s.mask  # the colony grows into this segment
                    o["area"] = s.area
                else:  # no colony continued by this segment: a new colony
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
                    "pre_existing": 0,
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