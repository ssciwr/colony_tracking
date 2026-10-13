Status: resolved

# Spec: learn-from-v1

Does v1 (`../colony_tracking`) hold anything v2 should adopt? Very little overall: v2 leads
on per-stage failure handling and resume with abort summaries (v1's difference step aborts
the whole run on one bad well), single-source options, test volume, and agent docs.

1. Testable seams in the finetune track. v1 keeps IoU scoring in a module-level function
   with an injectable predictor and bounding-box helper (`finetune/evaluate.py:49-68`),
   tests it hermetically with a FakePredictor (`tests/test_finetune.py:150-205`), and
   smoke-tests the dataloader without a model (`tests/test_finetune.py:125-140`). v2
   inlines `_predict_instances` (`src/colony_tracking/finetune/__init__.py:263-320`, hard
   micro_sam import) and `build_loader` stays a closure in `run_train` (lines 147-168), so
   v2's scoring (tile choice, box-order swap, IoU, clipped flag) and loader are untested.
   Adopt v1's seams and port its three scoring tests and the loader smoke test.
2. `Frame.stem`. v1's `Frame` carries a `stem` property that strips `.tif` and `.tiff`
   correctly (`src/colony_tracking/io.py:63-77`); stages build output names from it. v2's
   `Frame` (`src/colony_tracking/io.py:64-72`) has none, so stems are derived ad hoc, which
   produced the known `[:-4]` bug (`src/colony_tracking/analysis.py:292`, ticketed
   elsewhere). Add a `stem` property to v2's io and route all stem derivation through it.
3. Minor. v1's README documents the in-process Python API with an example
   (`README.md:137-156`); v2's README covers the CLI only, though the v2 spec makes the
   importable API a TUI deliverable (stories 36-37): add a short section. Also consider
   v1's evaluate caps `n_max_images` / `n_max_instances` (`finetune/evaluate.py:30-46`),
   dropped in v2's `EvaluateOptions` (`_options.py:416-445`). (v1's point/box_point
   prompts stay rejected: v2's spec scores the decoder with box prompts only.)

Judged and rejected: v1's finetune split (`train.py` 167, `evaluate.py` 190, a 17-line
re-export `__init__.py`) is mildly cleaner than v2's 463-line `finetune/__init__.py`, but
not worth churn alone; fold it in only if item 1 lands there anyway. v1's docstrings,
fixtures, and options inside stage modules equal or trail v2's.

Verdict: adopt items 1 and 2 plus the small item 3; everything else in v1 is matched or beaten by v2.

## Comments

- 2026-10-13: Implemented on integration branch learn-from-v1 (tickets 01-04 resolved;
  final state ef778c2 after code-review fixes; 233 passed, 1 deselected).

