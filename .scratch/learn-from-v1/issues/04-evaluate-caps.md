Status: resolved
Blocked by: 01

# Adopt v1's evaluate caps: `n_max_images` / `n_max_instances` on `EvaluateOptions`

Spec: `.scratch/learn-from-v1/spec.md` (item 3, caps part). v1 reference
repo: `../colony_tracking` (absolute:
`/home/kriedmiller/projects/colony_size/colony_tracking`).

## What

v1's evaluate carries two optional caps for quick smoke runs on a subset —
`n_max_images` (first N h5 files of the split) and `n_max_instances` (first N
instances per image) — see v1
`src/colony_tracking/finetune/evaluate.py:43-44` (fields) and their use at
`:125-126` and `:73-74`. v2's `EvaluateOptions`
(`src/colony_tracking/_options.py:416-447`) dropped both. Ticket 01 already
gave the extracted `predict_instances` its `n_max_instances` parameter; this
ticket wires the option-level caps on top of it.

## Fix sketch

1. Add two fields to `EvaluateOptions` (after `patch_shape`, before
   `output` — v1's relative order; per-field `help` text is mandatory,
   `tests/test_finetune.py::test_every_finetune_option_field_has_help_text`):

   ```python
   n_max_images: int | None = field(
       default=None,
       metadata={"help": "Evaluate only the first N h5 images of the "
                  "split (a quick smoke run)."},
   )
   n_max_instances: int | None = field(
       default=None,
       metadata={"help": "Score only the first N instances per image."},
   )
   ```

2. Wire them in `run_evaluate`
   (`src/colony_tracking/finetune/__init__.py`): after the split-dir glob
   and empty check, `if options.n_max_images: files =
   files[:options.n_max_images]` (v1 semantics at evaluate.py:125-126 — a
   falsy cap disables the slice); pass
   `n_max_instances=options.n_max_instances` to the `predict_instances`
   call.
3. Do NOT add a `prompt` option — v1's point/box_point prompts stay
   rejected (v2's spec scores the decoder with box prompts only).

## Ripple (contract tests + docs)

- `tests/test_finetune.py::test_evaluate_options_fields_match_contract`
  pins the exact field list — insert the two names.
- `test_evaluate_options_defaults` — assert both default to None.
- Extend the `EvaluateOptions` case in `test_finetune_options_roundtrip`
  with the caps.
- The CLI builds flags from the dataclass fields; regenerate the README's
  `#### evaluate` option table by running
  `uv run python tests/test_readme.py` and pasting the evaluate block
  (`tests/test_readme.py::test_option_table_matches_the_cli` enforces it).
- Check `tests/test_cli.py::test_train_evaluate_flags` and any other
  evaluate-flag assertions for the new flags.

## Tests

- A hermetic run_evaluate-level test: build a tiny real h5 split
  (`_write_split_h5`-style, but 64x64 with a planted disk so instances
  exist), monkeypatch `micro_sam.util.get_sam_model` (imported lazily inside
  `run_evaluate`, so the patch lands) to return a `FakePredictor` (reuse the
  one ticket 01 ports), and assert `n_max_images` caps `summary.n_images`
  / `per_image` (e.g. 3 files written, cap 2 → 2) and `n_max_instances`
  caps the scored instances per image. Skippable fallback if the
  micro_sam import proves unusable in the test env: cover the caps purely at
  the function level (already done by ticket 01's
  `test_predict_instances_clipped_and_capped`) and assert only the
  files-slice logic via the field contract + a direct call of the slice
  expression — but try the monkeypatch route first; micro_sam is installed
  in the venv.

## Acceptance criteria

- The two caps work end to end (option → run_evaluate →
  `predict_instances`), defaults keep current behaviour, README/CLI/contract
  tests consistent, `uv run pytest` passes.

## Comments

- 2026-10-13: Resolved by branch t04-evaluate-caps (commit a1f5544, ff-merged into
  learn-from-v1). `n_max_images` / `n_max_instances` on EvaluateOptions (v1 order,
  per-field help), wired in run_evaluate (falsy image cap disables the files slice;
  instance cap passed to predict_instances); no prompt option added. Contract,
  defaults, roundtrip, CLI-flag, and README-table ripple all done (tables
  regenerated per tests/test_readme.py). End-to-end hermetic caps tests via a
  monkeypatched micro_sam.util.get_sam_model + FakePredictor on a real h5 split;
  _FakePredictor generalized to identify the prompted instance geometrically (size
  + valid tile offset) so un-capped controls with a second instance work. Review
  follow-up (ef778c2): the n_max_images help text now documents "0 = uncapped".
