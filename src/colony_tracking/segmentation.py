"""Segment stage: growth-mask-gated instance segmentation (ADR 0002).

Colonies in a well only grow, so the final frame sees every colony at its
largest extent. One well is processed like this:

1. **Growth mask from the final frame** — instance segmentation of the
   final frame (highest frame number) at decoder foreground >= ``fg_ref``
   (0.5). Junk instances are dropped (area < ``min_size`` (200 px) or
   solidity < ``min_solidity`` (0.7) — e.g. the dark well ring if it is
   ever segmented); the union of the survivors is split into connected
   components (**groups**; a group may contain several colonies that have
   grown together) and dilated by ``mask_margin`` (15 px) = the **growth
   mask**. ``ref__labels.tif`` holds the group labels of the final frame.
2. **Earlier frames** — foreground >= ``fg_early`` (0.3: the decoder
   under-fires on the faint, small early colonies, and false positives are
   caught by the mask anyway). An instance is kept only if >= ``min_overlap``
   (0.5) of its pixels lie inside the growth mask **and** inside a single
   group — colonies only grow, so a mid-series instance belongs to one
   colony's final extent, and an instance spanning several groups cannot be
   a single colony.
3. **Drift gate** — a frame is skipped when its elevated-foreground
   fraction (the part of the whole image above the frame's foreground
   threshold) exceeds ``max(max_fg_ratio * reference-frame fraction, 0.05)``:
   drift-dominated frames make the decoder hallucinate confident giant
   phantom instances. ``max_fg_ratio = 0`` disables the gate. The final
   frame itself is never gated. Skipped frames still get their (blank)
   label image, so the outputs stay 1:1 with the input frames.
4. **Tracking** — colony IDs are assigned by the shared pure module
   :mod:`colony_tracking.tracking`: touching segments of one group are
   pre-merged, then segments are tracked forward in time with the
   continue / absorb-merge / fork / new-ID rules, and the ``exclude_before``
   pre-existing filter blanks frames before N and flags colonies first
   detected at or before N as ``pre_existing=1``.

Inputs (``input``): with ``"raw"`` each raw frame is 1/99 percentile-
stretched to uint8 (the shared io helper, matching the finetuning data
prep). With ``"diff"`` the uint8 difference frames are fed as written
(128 = no change, ADR 0001 — the colonies' difference polarity flips over
time anyway, so there is no single polarity to enforce) and the reference
frame comes from the well's ``scale.txt``: **a difference reference equal
to the final frame is rejected up front** — the growth mask would be empty.

Outputs, per well, under per-well subdirectories of ``out_dir``:
``<frame-stem>__labels.tif`` per input frame (uint32 colony IDs, 1:1 with
the input frames; all-zero for gate-skipped or ``exclude_before``-blanked
frames), ``ref__labels.tif`` (the group labels of the final frame),
``colonies.csv`` (canonical columns), and the ``_done.txt`` done marker
with the seven canonical keys — a well with a complete marker is skipped
on re-run unless ``redo``.

Re-ID (``reid_only``): no model, no GPU — the segments are re-derived from
the saved label images (each label value re-split into connected
components), tracking is re-run from the first non-blank frame, and the
label images, colonies.csv, and done marker are rewritten. Useful after
changing ``exclude_before`` or the identity logic.

micro-sam (and with it torch) is imported lazily, inside the single model-
loading helper — importing this module never pulls torch, so the
difference, analyze, and train stages do not pay that cost. ``checkpoint``
(a finetuned decoder) replaces the pretrained model when given.

Progress is logged per well and per frame through the package logger;
never printed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

#: Floor of the drift gate (a fraction of the image): wells with tiny
#: colonies still get protection. The prototype's ``MIN_FG_ABS``.
MIN_FG_ABS = 0.05


# --- the drift gate -------------------------------------------------------------


def drift_gate_limit(ref_frac: float, max_fg_ratio: float) -> float:
    """The elevated-foreground fraction a frame may reach before it is skipped.

    ``max(max_fg_ratio * ref_frac, MIN_FG_ABS)``: relative to the reference
    frame's fraction (its legitimate maximum), floored so that wells with
    tiny colonies still get protection.
    """
    return max(max_fg_ratio * ref_frac, MIN_FG_ABS)


def drift_gate_skips(
    elevated: float, ref_frac: float, *, max_fg_ratio: float
) -> bool:
    """The drift-gate decision for one frame.

    True when the frame's elevated-foreground fraction exceeds the gate
    limit (:func:`drift_gate_limit`) — the decoder elevates the foreground
    globally and hallucinates confident giant instances. ``max_fg_ratio``
    of 0 (or less) disables the gate. The reference frame is never gated
    (its fraction equals ``ref_frac`` and can never exceed the limit).
    """
    if max_fg_ratio <= 0:
        return False
    return elevated > drift_gate_limit(ref_frac, max_fg_ratio)


# --- growth-mask construction -----------------------------------------------------


def build_growth_mask(
    instances: np.ndarray,
    *,
    min_size: int,
    min_solidity: float,
    mask_margin: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Junk-filter the final frame's instances and build the growth mask.

    ``instances`` is the final frame's instance segmentation (a label
    image; one positive label value per decoder instance). Instances with
    area < ``min_size`` or solidity < ``min_solidity`` are dropped (well-
    ring junk); the union of the survivors is split into connected
    components — the **groups** (a group label image, uint32, 0 =
    background) — and the groups are dilated by ``mask_margin`` px = the
    **growth mask** (a boolean image).

    Returns ``(groups, mask)``; ``(None, all-False)`` when nothing survives
    the filters (the caller then writes an all-zero ``ref__labels.tif`` and
    keeps no segments).
    """
    instances = np.asarray(instances)
    keep = np.zeros(instances.shape, dtype=bool)
    for region in regionprops(instances):
        if region.area >= min_size and region.solidity >= min_solidity:
            keep |= instances == region.label
    if not keep.any():
        return None, np.zeros(instances.shape, dtype=bool)
    groups = ndi.label(keep)[0].astype(np.uint32)
    return groups, ndi.binary_dilation(groups > 0, iterations=mask_margin)


# --- the mask-overlap filter -------------------------------------------------------


def filter_to_mask(
    instances: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
    min_overlap: float,
) -> tuple[np.ndarray, int]:
    """Keep the instances that lie (mostly) inside the growth mask.

    Two growth-based sanity checks per instance: at least ``min_overlap``
    of its pixels must lie inside the (dilated) growth mask, and at least
    ``min_overlap`` inside its best single reference group — colonies only
    grow, so a mid-series instance belongs to one colony's final extent,
    and an instance spanning several groups cannot be a single colony.

    Returns a label image with the kept instances relabelled 1..k (local
    IDs, no colony identity yet — a kept instance keeps its full extent)
    and the number of discarded instances.
    """
    instances = np.asarray(instances)
    n_groups = 0 if groups is None else int(groups.max())
    out = np.zeros(instances.shape, np.uint32)
    next_id, discarded = 1, 0
    for region in regionprops(instances):
        instance = instances == region.label
        if float((instance & mask).sum()) / region.area >= min_overlap:
            if n_groups:
                hits = np.bincount(
                    groups[instance].ravel(), minlength=n_groups + 1
                )
                if float(hits[1:].max()) / region.area < min_overlap:
                    discarded += 1
                    continue
            out[instance] = next_id
            next_id += 1
        else:
            discarded += 1
    return out, discarded