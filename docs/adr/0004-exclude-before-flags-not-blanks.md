# 0004: `exclude_before` flags pre-existing colonies instead of blanking frames

Status: amended (2026-10-26)

## Context

`exclude_before = N` (segment stage) was implemented as a pre-segmentation
skip: every frame with frame number < N yielded an empty label image and no
CSV rows, and colonies first detected at frame number ≤ N were flagged
`pre_existing = 1`. Because the earlier frames were blanked, flagging
effectively meant "first detected at N", and since the pipeline derives N
from the difference reference (whose own frame is structurally blank), the
flag could never fire in the pipeline chain at all.

The blanking was an intentional behaviour change over the prototype's
`--exclude_lo` + `--drop_labels_before` pair, but it is not the intended
semantics: blank label images before N are not wanted, and segments found
before N should be present in the outputs and kept out of the analysis
through the `pre_existing` flag instead.

## Decision

- Frames below N are **segmented and tracked like any other frame**.
  Nothing is blanked, and their label images and CSV rows are written
  normally.
- Every colony whose first detection is at frame number ≤ N is flagged
  `pre_existing = 1` on all its rows (the boundary is unchanged; it matches
  the base-image probe, which also treats the reference frame itself as
  pre-existing territory).
- Pre-existing colonies are excluded from the analysis **entirely**: no
  aggregates/plots over their whole lifetime (as before, `keep_pre_existing`
  overrides), and never bad-flagged, since `from_start`/`no_growth` do not
  apply and flagged colonies cannot push a well over `max_bad_segments` and
  exclude it.
- Re-ID follows the same rule: saved label images are re-read for all
  frames and tracking re-flags from the first detection.

## Consequences

- In a raw-input chain, everything already growing before N stays visible
  (flagged) in the outputs, and everything first detected from N+1 on is
  the analyzed population, the option's documented intent.
- In the pipeline's difference chain, the frames below the reference are
  polarity-flipped difference images; whatever the decoder finds there
  (growth rings against the reference, drift structure, pre-existing
  damage) is flagged pre-existing and excluded. The reference frame itself
  stays blank (it is differenced against itself), so colonies invisible
  before the reference still surface first at N+1; the analysis stage's
  base-image probe covers that case.
- Empirically (verified on the 19-frame reference run), the pre-reference
  instances of static/deforming objects such as well damage and debris
  shards chain through the blank reference frame into the forward
  detections (the enclosed-gap fill turns their diff rings into full
  extents, which match across the reference), so these objects inherit a
  first detection ≤ N and drop out of the analysis. Real growing colonies
  do not chain: their faint early content stays under the decoder's
  threshold, so they still surface first at N+1 and are analyzed.
- The `pre_existing_colonies` done-marker counter becomes meaningful in
  difference chains.
- Trajectories/growth-rates rows of flagged colonies stay in the CSVs
  (statistics-level exclusion); label images keep their segments.

## Amendment (2026-10-26)

The analysis-stage base-image probe is removed: pre-existing exclusion is
solely the segment-stage first-detection flag above, in both the model path
and the reid path. Colonies that existed at the reference frame but surface
after N in the difference chain are therefore analyzed as new colonies by
definition — the Consequences bullet "the analysis stage's base-image probe
covers that case" is superseded. What still holds: static objects chain
through the blank reference frame, so their first detection stays ≤ N and
they drop out of the analysis; a harder exclusion is not available without
re-running the segment stage with a different `exclude_before` (the reid
path re-flags from the saved label images without re-segmentation).
