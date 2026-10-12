Status: resolved
Blocked by: 02

# 04: `tracking` module — pure colony-identity logic

Goal: the identity/tracking logic as a pure module (numpy only, no model,
no file I/O beyond what io gives you), shared by the full segment run
(ticket 05) and the re-ID mode.

Spec: `.scratch/colony-tracking/spec.md` — "Segment stage → Tracking" and
"→ The pre-existing filter", "I/O contract → colonies.csv / Done marker",
user stories 13–16. Reference implementation:
`/home/kriedmiller/projects/colony_size/prototype/segment_growth_pipeline.py`
— `assign_colony_ids`, `premerge_touching`, and the `--exclude_lo`/
`--drop_labels_before` handling (port the logic; the NEW merged
`exclude_before` semantics below override the prototype's two-flag behavior).

## Build — `src/colony_tracking/tracking.py`

Pure functions, input → output on arrays/row lists (design the exact
signatures for clarity; they must be directly testable without a model):

- **Segment pre-merge (touching)**: segments of one frame that lie in the
  same group and touch (within 3 px dilation, tolerating boundary wobble)
  are merged up front — unless `merge_touching` is off.
- **Forward tracking** across frames (frame numbers ascending), per well:
  - IoU ≥ 0.3 against a colony's last-known mask → **continue** (same ID);
    the up-to-date (most recent) colony wins over stale masks of
    long-absorbed colonies;
  - a segment that (additionally) covers ≥ 0.5 of a further uncontinued
    colony in this frame → **merge/absorb**: the survivor keeps its own ID,
    the absorbed colony IDs are recorded for this frame's row in
    `merged_with` (';'-separated in the CSV), and an absorb event is
    counted; absorbed colonies stay tracked — if they separate again later
    they resume their old ID;
  - a segment matching no colony that lies mostly inside one known colony's
    territory (i.e. it splits a single colony) → **fork**: the same ID
    continues, logged, counted (`forked_segments`);
  - otherwise → **new colony ID**.
- **Pre-existing filter** (the merged `exclude_before` semantics — this
  intentionally changes the prototype's raw-root behavior; see spec Further
  Notes):
  1. every frame with frame number **< N** gets an empty label image and no
     CSV rows (outputs stay 1:1 with the input frames);
  2. colonies whose first detection is **at or before frame N** (frame
     number ≤ N) are flagged `pre_existing=1`; rows are kept.
  Because frames before N are blanked, (2) effectively means "first detected
  at N". Compare frame numbers directly — N need not exist in a particular
  well (no warning, no disabling; this generalizes the prototype).
- **Ancestor counting** (pure helper): per colony-frame row, the number of
  distinct original colonies combined in the segment, resolved transitively
  through the `merged_with` histories, the colony itself included (an A+B
  where B had merged with C ends at 3; unmerged = 1). This is the quantity
  the prototype called `n_merged`; values unchanged, new name
  `ancestors`.
- Output shape: per frame, label arrays (one per input frame, 1:1) + CSV rows
  (`colony_id, group_id, merged_with, pre_existing, area_px, centroid_y,
  centroid_x` per kept segment) + counters for the done marker
  (`groups, colonies, pre_existing_colonies, merge_events, forked_segments,
  discarded_instances`).

Keep it importable without numpy-only (no scipy beyond what io already
pulls; no micro-sam, no torch, ever).

## Tests — `tests/test_tracking.py`

Pure input → output on synthetic masks/row lists (use small arrays; no
fixture plate needed):

- IoU continuation: a segment overlapping ≥ 0.3 of the last mask keeps the ID;
  below → no continue.
- absorb-merge: survivor keeps its ID; `merged_with` records the absorbed ID;
  an A+B←C chain gives transitive ancestor count **3** via the helper.
- an absorbed colony resumes its old ID after separating again.
- fork: same ID continues, `forked_segments` counted.
- unmatched segment → new ID.
- pre-merge of touching segments (on by default; off via `merge_touching`).
- `exclude_before`: frames < N blanked (empty label arrays, no rows);
  first-detection ≤ N flagged `pre_existing=1` with rows kept; a well whose
  series lacks frame N still works (frame-number comparison).
- ancestor helper: identity (1), direct merge (2), transitive (3), independent
  colonies stay 1.

## Acceptance

- `uv run pytest` passes.
- Module docstring carries the identity rules (continue/merge/fork/absorb,
  pre-existing semantics, ancestor definition) per the spec.
- No model imports, no hardcoded paths.

## Comments

- 2026-10-12: Resolved by branch t04-tracking (merge commit 1e4681c509c48f135a3b354866b802171fac2859); 73 passed; API: track_colonies(frames, groups, exclude_before, merge_touching, ...) -> TrackingResult, premerge_touching, ancestor_counts, extract_segments; deviations: fork rule per SPEC prose (same ID continues; prototype used a new ID), per-colony fork-territory check, merged_with re-listed per frame (faithful to prototype), discarded_instances pass-through arg.
