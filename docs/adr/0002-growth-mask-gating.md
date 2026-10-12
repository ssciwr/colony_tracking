# 0002: Growth-mask gating from the final frame

Status: accepted (2026-10-12)

## Context

The decoder fires on non-colony foreground (debris, dust, illumination
artifacts) throughout a series, so running raw instance segmentation on
every frame yields phantom colonies. The colonies that matter are the ones
that grow, and the ones that grow are all visible together in the final
frame.

## Decision

The segment stage builds a **growth mask** from the **final frame** and
gates all detection with it:

1. Final frame: instance segmentation at decoder foreground ≥ `fg_ref`
   (0.5); instances with area < `min_size` (200 px) or solidity <
   `min_solidity` (0.7) are dropped (well-ring junk).
2. The survivors' union is split into connected components — **groups** —
   and the groups are dilated by `mask_margin` (15 px) to form the growth
   mask.
3. Earlier frames: foreground ≥ `fg_early` (0.3 — the decoder under-fires on
   faint colonies); an instance is kept only if ≥ `min_overlap` (0.5) of its
   pixels lie inside the growth mask **and** inside a single group.

The **drift gate** is part of the gating: a frame is skipped when its
elevated-foreground fraction exceeds `max(max_fg_ratio × reference-frame
fraction, 0.05)` (default `max_fg_ratio = 1.5`; `0` disables the gate).
Drift bursts around the reference frame make the decoder hallucinate giant
instances, so those frames are dropped rather than segmented.

A difference input whose reference frame is the final frame is rejected up
front: the growth mask would be empty.

## Consequences

- Detection is confined to where colonies actually grow; phantom colonies
  from debris are suppressed without a second model pass.
- A colony that appears only early and is gone before the final frame is
  not detected — accepted for this dataset, where colonies grow over the
  series.
- The final frame's segmentation quality bounds the whole run: a group the
  final frame misses makes everything that grows there invisible.
