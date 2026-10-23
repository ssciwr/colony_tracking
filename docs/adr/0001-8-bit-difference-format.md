# 0001: 8-bit difference frames with 128 = no change

Status: accepted (2026-10-12)

## Context

The difference stage writes per-well difference frames (`cur − ref` in
counts) that the segment stage consumes. The on-disk format is the internal
contract between stages and must stay compact for a full 96-well plate of
2048×2048 frames while remaining directly displayable for visual QA.

## Decision

Difference frames are stored between stages as **uint8 previews**. The
per-well display limit is the `display_percentile` (99.5) of |diff|,
estimated on the subsampled grid (stride `subsample`); the preview value is
`128 + diff × 127 / limit`, clamped to 0–255. **128 = no change.** The limit
and the reference frame's file name are recorded per well in `scale.txt`
(`display_limit_int16`), so previews can be converted back to counts.

Considered and rejected:

- **int16 counts.** Exact counts, but 2× the size of the uint8 preview and
  not directly displayable; the per-well scale recorded in `scale.txt` is
  enough to recover counts for the later stages.
- **The `abs` / `abs_percentile` presentations.** Experimental alternative
  presentations from the prototype; cut from the package (listed in the
  spec's Out of Scope).

## Consequences

- The segment stage reads uint8 frames as written; the reference frame and
  the display limit come from the well's `scale.txt`, so a difference tree
  is self-describing per well.
- The previews are approximate counts: downstream decisions must not rely
  on count accuracy beyond the display limit.
- Frames acquired before the reference frame come out polarity-flipped;
  documented, intended, unchanged.
