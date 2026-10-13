Status: resolved

# Add `Frame.stem` (v1's .tif/.tiff-safe stem) and route stem derivation through it

Spec: `.scratch/learn-from-v1/spec.md` (item 2). v1 reference repo:
`../colony_tracking` (absolute: `/home/kriedmiller/projects/colony_size/colony_tracking`).

## What

v1's `Frame` carries a `stem` property that strips `.tif` and `.tiff`
correctly (v1 `src/colony_tracking/io.py:63-77`: `.tiff` case-insensitively
loses 5 characters, otherwise 4), and stages build output names from it.
v2's `Frame` (`src/colony_tracking/io.py:64-72`) has no such property, so
stems are derived ad hoc at every output-naming site:

- `src/colony_tracking/segmentation.py:342` and `:513` —
  `f"{frame.path.stem}__labels.tif"` (pathlib stem);
- `src/colony_tracking/difference.py:151` —
  `f"{Path(frame.name).stem}_diff_uint8.tif"` (pathlib stem on the name).

Ad-hoc derivation is exactly what produced the known `[:-4]` bug
(`src/colony_tracking/analysis.py:292`).

## Out of scope

- `src/colony_tracking/analysis.py:292` (`stem = frows[0]["image"][:-4]`) is
  owned by `.scratch/review-fixes/issues/02-rim-stats-stem-breaks-tiff.md` —
  do not touch it here; those rows are CSV dicts, not Frames.
- `src/colony_tracking/finetune/_convert.py:201` pairs CSV rows without Frame
  objects and its `Path(rec["name"]).stem` is already suffix-agnostic.
- `tests/conftest.py:445` slices test-frame names; test scaffolding.

## Fix sketch

1. Add to v2's `Frame` (io.py) the property mirroring v1's, on `self.name`:

   ```python
   @property
   def stem(self) -> str:
       """File name without its ``.tif``/``.tiff`` suffix."""
       return self.name[:-5] if self.name.lower().endswith(".tiff") \
           else self.name[:-4]
   ```

2. Route the three call sites through it: `frame.path.stem` → `frame.stem`
   in segmentation.py (342, 513); `Path(frame.name).stem` → `frame.stem` in
   difference.py (151). On this dataset's names the property and the pathlib
   stem agree (both strip one trailing suffix), so no output name changes.

## Tests

- `tests/test_io.py`: property tests — `.tif`, `.tiff`, and upper-case
  `.TIFF` names yield the same stem.
- The existing segmentation/difference tests already assert the output file
  names; they must stay green unchanged.

## Acceptance criteria

- `Frame.stem` exists and mirrors v1's; the three call sites use it; no
  other production stem derivation of a `Frame` remains (grep for
  `path.stem` / `Path(frame.name)`).
- `uv run pytest` passes; output names unchanged.

## Comments

- 2026-10-13: Resolved by branch t02-frame-stem (commit 1be336c, ff-merged into
  learn-from-v1). `Frame.stem` added to io.py mirroring v1's case-insensitive
  .tif/.tiff strip; the three call sites routed through it (segmentation.py label
  write + re-read, difference.py `<stem>_diff_uint8.tif`); analysis.py:292 and
  _convert.py:201 untouched per ticket out-of-scope. New `test_frame_stem`
  (.tif/.tiff/.TIFF); output names unchanged; 221 passed at the time of the merge.
