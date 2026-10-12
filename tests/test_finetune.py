"""Tests for the finetune track (hermetic: the data conversion only).

Seams under test (per the ticket): the finetune option dataclasses (the
CLI contract of the spec's "Options and CLI contract" section) and
``run_prepare`` on a small synthetic labeled mini-plate — the well-based
split (no well crosses the split; greedy instance-richest wells until
``val_fraction``; the ``max_val_wells`` cap; an explicit val-well list),
low-instance label drops, the h5 contents (dtypes, shapes, stretched
values, gzip 4), and the summary counters. The model-dependent
train/evaluate paths are outside the automated suite (no GPU or network —
spec "Out of Scope"); they are covered for their logged
missing-prerequisite errors only.
"""

import dataclasses
import logging
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from colony_tracking import io
from colony_tracking.finetune import (
    EvaluateOptions,
    PrepareOptions,
    PrepareSummary,
    TrainOptions,
    run_evaluate,
    run_prepare,
    run_train,
)
from conftest import DiskSpec, LabeledPlate, build_labeled_plate, frame_name


# --- field contracts (names, order, required-ness) -------------------------


def test_prepare_options_fields_match_contract():
    names = [f.name for f in fields(PrepareOptions)]
    assert names == [
        "raw_dir",
        "label_dir",
        "out_dir",
        "val_fraction",
        "max_val_wells",
        "min_instances",
        "n_workers",
        "val_wells",
    ]


def test_train_options_fields_match_contract():
    names = [f.name for f in fields(TrainOptions)]
    assert names == [
        "data_dir",
        "out_dir",
        "model_type",
        "configuration",
        "samples_per_image",
        "val_samples_per_image",
        "n_objects_per_batch",
        "patch_shape",
        "n_epochs",
        "early_stopping",
        "lr",
        "seed",
    ]


def test_evaluate_options_fields_match_contract():
    names = [f.name for f in fields(EvaluateOptions)]
    assert names == [
        "data_dir",
        "split",
        "checkpoint",
        "model_type",
        "patch_shape",
        "output",
    ]


def test_finetune_required_fields_have_no_default():
    required = {
        PrepareOptions: ["raw_dir", "label_dir", "out_dir"],
        TrainOptions: ["data_dir", "out_dir"],
        EvaluateOptions: ["data_dir"],
    }
    for cls, names in required.items():
        by_name = {f.name: f for f in fields(cls)}
        for name in names:
            assert by_name[name].default is dataclasses.MISSING, (
                f"{cls.__name__}.{name} must be required"
            )


def test_missing_required_field_raises():
    with pytest.raises(TypeError):
        PrepareOptions()
    with pytest.raises(TypeError):
        TrainOptions()
    with pytest.raises(TypeError):
        EvaluateOptions()


# --- defaults, exactly as the spec states -----------------------------------


def test_prepare_options_defaults():
    opts = PrepareOptions(
        raw_dir=Path("raw"), label_dir=Path("labels"), out_dir=Path("out")
    )
    assert opts.val_fraction == 0.2
    assert opts.max_val_wells == 3
    assert opts.min_instances == 1
    assert opts.n_workers == 8
    assert opts.val_wells is None


def test_train_options_defaults():
    opts = TrainOptions(data_dir=Path("data"), out_dir=Path("run"))
    assert opts.model_type == "vit_t"
    assert opts.configuration == "Minimal"
    assert opts.samples_per_image == 3
    assert opts.val_samples_per_image == 2
    assert opts.n_objects_per_batch == 2
    assert opts.patch_shape == (512, 512)
    assert opts.n_epochs == 100
    assert opts.early_stopping == 15
    assert opts.lr == 1e-5
    assert opts.seed == 42


def test_evaluate_options_defaults():
    opts = EvaluateOptions(data_dir=Path("data"))
    assert opts.split == "val"
    assert opts.checkpoint is None
    assert opts.model_type == "vit_t"
    assert opts.patch_shape == (512, 512)
    assert opts.output is None


# --- per-field help text ----------------------------------------------------


def test_every_finetune_option_field_has_help_text():
    for cls in (PrepareOptions, TrainOptions, EvaluateOptions):
        for f in fields(cls):
            assert f.metadata.get("help"), (
                f"{cls.__name__}.{f.name} has no per-field help text"
            )


# --- dict round-tripping (consumed by TOML config and the future TUI) -------


def test_finetune_options_roundtrip():
    opts = TrainOptions(
        data_dir=Path("/data/finetune"),
        out_dir=Path("/data/run"),
        patch_shape=(256, 256),
    )
    data = opts.to_dict()
    assert data["patch_shape"] == [256, 256]  # tuples come out TOML-friendly
    assert TrainOptions.from_dict(data) == opts

    opts = EvaluateOptions(
        data_dir=Path("/data/finetune"),
        split="train",
        checkpoint=Path("/data/run/best.pt"),
        output=Path("/data/report.json"),
    )
    assert EvaluateOptions.from_dict(opts.to_dict()) == opts

    opts = PrepareOptions(
        raw_dir=Path("/raw"),
        label_dir=Path("/labels"),
        out_dir=Path("/h5"),
        val_wells=["-A001", "-A002"],
    )
    assert PrepareOptions.from_dict(opts.to_dict()) == opts


# --- the small labeled plate ------------------------------------------------
#
# 5 wells x 3 frames of 512x512, planted disks:
#   -A001: two disks (the second from frame 2)     -> 5 instances
#   -A002: two disks, both from frame 2 (frame 1   -> 4 instances
#          has an all-zero label)
#   -A003 / -A004: one disk each                    -> 3 instances
#   -A005: raw frames only, no label files


WELLS = ("-A001", "-A002", "-A003", "-A004", "-A005")


def _disks() -> dict[str, list[DiskSpec]]:
    return {
        "-A001": [
            DiskSpec("a1", cy=150.0, cx=130.0, r0=18.0),
            DiskSpec("a2", cy=330.0, cx=370.0, r0=12.0, first_frame=2),
        ],
        "-A002": [
            DiskSpec("b1", cy=150.0, cx=130.0, r0=18.0, first_frame=2),
            DiskSpec("b2", cy=330.0, cx=370.0, r0=12.0, first_frame=2),
        ],
        "-A003": [DiskSpec("c1", cy=256.0, cx=256.0, r0=20.0)],
        "-A004": [DiskSpec("d1", cy=256.0, cx=256.0, r0=16.0)],
    }


@pytest.fixture
def labeled(tmp_path: Path) -> LabeledPlate:
    return build_labeled_plate(tmp_path, wells=WELLS, n_frames=3, disks=_disks())


def _prepare(labeled: LabeledPlate, tmp_path: Path, **overrides) -> PrepareSummary:
    return run_prepare(
        PrepareOptions(
            raw_dir=labeled.raw_root,
            label_dir=labeled.label_root,
            out_dir=tmp_path / "h5",
            **overrides,
        )
    )


def _wells_of(out: Path, tag: str) -> set[str]:
    return {io.parse_well(p.name) for p in (out / tag).glob("*.h5")}


# --- the well-based split ----------------------------------------------------


def test_split_is_by_well_and_never_crosses(labeled, tmp_path):
    out = tmp_path / "h5"
    _prepare(labeled, tmp_path)
    train_wells = _wells_of(out, "train")
    val_wells = _wells_of(out, "val")
    assert train_wells & val_wells == set()
    assert train_wells | val_wells == {"-A001", "-A002", "-A003", "-A004"}


def test_default_split_takes_richest_well_until_val_fraction(labeled, tmp_path):
    # 15 kept instances, val_fraction 0.2 -> target 3.0; the instance-richest
    # well -A001 (5) alone reaches it.
    summary = _prepare(labeled, tmp_path)
    assert summary.val_wells == ["-A001"]
    assert summary.train_wells == ["-A002", "-A003", "-A004"]
    assert summary.n_val == 3 and summary.n_train == 8


def test_greedy_split_across_multiple_wells(labeled, tmp_path):
    # val_fraction 0.5 -> target 7.5: -A001 (5) then -A002 (4) = 9.
    summary = _prepare(labeled, tmp_path, val_fraction=0.5)
    assert summary.val_wells == ["-A001", "-A002"]
    assert summary.train_wells == ["-A003", "-A004"]
    assert summary.n_val == 5 and summary.n_train == 6


def test_max_val_wells_caps_the_split(labeled, tmp_path):
    summary = _prepare(labeled, tmp_path, val_fraction=0.5, max_val_wells=1)
    assert summary.val_wells == ["-A001"]
    assert summary.train_wells == ["-A002", "-A003", "-A004"]


def test_explicit_val_well_list_is_honoured(labeled, tmp_path):
    summary = _prepare(labeled, tmp_path, val_wells=["-A003"])
    assert summary.val_wells == ["-A003"]
    assert summary.train_wells == ["-A001", "-A002", "-A004"]


def test_explicit_val_well_without_labels_fails(labeled, tmp_path, caplog):
    # -A005 has raw frames but no label files.
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = _prepare(labeled, tmp_path, val_wells=["-A005"])
    assert summary.error is not None
    assert "-A005" in caplog.text
    assert not (tmp_path / "h5").exists() or not list((tmp_path / "h5").glob("*"))


def test_explicit_val_wells_listing_an_unknown_well_fails(labeled, tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = _prepare(labeled, tmp_path, val_wells=["-A003", "-A009"])
    assert summary.error is not None
    assert "-A009" in caplog.text


def test_explicit_val_wells_that_empty_the_train_split_fail(labeled, tmp_path, caplog):
    # With min_instances=2 only -A001 and -A002 survive; taking both for val
    # leaves nothing for training.
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = _prepare(
            labeled, tmp_path, min_instances=2, val_wells=["-A001", "-A002"]
        )
    assert summary.error is not None
    assert "empty" in caplog.text.lower()


# --- low-instance label drops -------------------------------------------------


def test_labels_below_min_instances_are_dropped(labeled, tmp_path):
    # min_instances=2 drops: -A001 frame 1 (1 instance), -A002 frame 1 (0),
    # and every label of -A003 and -A004 (1 instance each) -> 8 dropped.
    summary = _prepare(labeled, tmp_path, min_instances=2)
    assert summary.n_dropped == 8
    assert summary.n_pairs == 4  # -A001 and -A002, frames 2 and 3
    assert summary.n_instances == 8
    assert summary.val_wells == ["-A001"]
    assert summary.train_wells == ["-A002"]


def test_well_whose_labels_are_all_dropped_is_logged(labeled, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="colony_tracking"):
        summary = _prepare(labeled, tmp_path, min_instances=2)
    assert summary.error is None  # it must not crash the run
    assert "-A003" in caplog.text
    assert "-A004" in caplog.text


def test_labels_without_raw_are_skipped_and_counted(labeled, tmp_path):
    # An orphan label for a well the raw tree does not have.
    tifffile.imwrite(
        labeled.label_root / frame_name("-A006", 1, 0),
        np.zeros((512, 512), dtype=np.uint32),
    )
    summary = _prepare(labeled, tmp_path)
    assert summary.n_missing_raw == 1
    assert summary.n_pairs == 11  # the orphan changes nothing else


# --- the h5 files -------------------------------------------------------------


def test_h5_contents_match_source(labeled, tmp_path):
    out = tmp_path / "h5"
    _prepare(labeled, tmp_path, val_wells=["-A003"])
    name = labeled.plate.wells["-A003"][0]  # frame 1
    h5_path = out / "val" / (name[:-4] + ".h5")
    assert h5_path.exists()
    raw_src = tifffile.imread(labeled.raw_root / name)
    label_src = tifffile.imread(labeled.label_root / name)
    with h5py.File(h5_path) as h5:
        assert h5["raw"].dtype == np.uint8
        assert h5["label"].dtype == np.int32
        assert h5["raw"].shape == raw_src.shape == (512, 512)
        assert h5["label"].shape == label_src.shape
        # The raw values are the shared io 1/99 stretch of the source frame.
        assert np.array_equal(h5["raw"][:], io.stretch_uint8(raw_src))
        assert np.array_equal(h5["label"][:], label_src.astype(np.int32))
        assert h5["raw"].compression == "gzip"
        assert h5["raw"].compression_opts == 4
        assert h5["label"].compression == "gzip"
        assert h5["label"].compression_opts == 4


def test_per_well_layout(tmp_path):
    plate = build_labeled_plate(
        tmp_path / "pw", wells=WELLS, n_frames=3, disks=_disks(), layout="per_well"
    )
    summary = run_prepare(
        PrepareOptions(
            raw_dir=plate.raw_root, label_dir=plate.label_root, out_dir=tmp_path / "h5"
        )
    )
    assert summary.error is None
    assert summary.val_wells == ["-A001"]  # same greedy split as the flat run
    name = plate.plate.wells["-A003"][0]  # -A003 is in the train split
    h5_path = tmp_path / "h5" / "train" / (name[:-4] + ".h5")
    raw_src = tifffile.imread(plate.raw_root / "-A003" / name)
    label_src = tifffile.imread(plate.label_root / "-A003" / name)
    with h5py.File(h5_path) as h5:
        assert np.array_equal(h5["raw"][:], io.stretch_uint8(raw_src))
        assert np.array_equal(h5["label"][:], label_src.astype(np.int32))


# --- summary counters ----------------------------------------------------------


def test_run_prepare_summary_counters(labeled, tmp_path):
    out = tmp_path / "h5"
    summary = _prepare(labeled, tmp_path)
    assert summary.error is None
    # 15 frames carry labels; -A002's all-zero frame 1 is dropped.
    assert summary.n_pairs == 11
    assert summary.n_train == 8
    assert summary.n_val == 3
    assert summary.n_instances == 15
    assert summary.n_dropped == 1
    assert summary.n_missing_raw == 0
    assert len(list((out / "train").glob("*.h5"))) == 8
    assert len(list((out / "val").glob("*.h5"))) == 3


# --- lazy model imports -------------------------------------------------------


def test_importing_finetune_does_not_import_torch():
    # A fresh interpreter exercises the real import path; the assertion is
    # on that process's sys.modules.
    code = (
        "import sys; import colony_tracking.finetune; "
        "bad = [m for m in ('torch', 'micro_sam', 'torch_em') "
        "if m in sys.modules]; assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- train / evaluate: logged prerequisite errors (no model, no GPU) ----------


def _write_split_h5(split_dir: Path, name: str = "x.h5") -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / name
    with h5py.File(path, "w") as h5:
        h5.create_dataset("raw", data=np.zeros((8, 8), np.uint8))
        h5.create_dataset("label", data=np.zeros((8, 8), np.int32))
    return path


def test_run_train_missing_split_logs_error(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_train(
            TrainOptions(data_dir=tmp_path / "nope", out_dir=tmp_path / "run")
        )
    assert summary.error is not None
    assert "nope" in caplog.text
    assert "torch" not in sys.modules  # validation precedes the lazy import


def test_run_train_missing_val_split_logs_error(tmp_path, caplog):
    _write_split_h5(tmp_path / "data" / "train")
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_train(
            TrainOptions(data_dir=tmp_path / "data", out_dir=tmp_path / "run")
        )
    assert summary.error is not None
    assert "val" in caplog.text
    assert "torch" not in sys.modules


def test_run_evaluate_missing_split_logs_error(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_evaluate(EvaluateOptions(data_dir=tmp_path / "nope"))
    assert summary.error is not None
    assert "nope" in caplog.text
    assert "torch" not in sys.modules


def test_run_evaluate_missing_checkpoint_logs_error(tmp_path, caplog):
    _write_split_h5(tmp_path / "data" / "val")
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_evaluate(
            EvaluateOptions(
                data_dir=tmp_path / "data",
                checkpoint=tmp_path / "missing.pt",
            )
        )
    assert summary.error is not None
    assert "missing.pt" in caplog.text
    assert "torch" not in sys.modules


# --- public API surface ---------------------------------------------------------


def test_package_root_reexports_finetune_api():
    import colony_tracking

    names = (
        "PrepareOptions",
        "TrainOptions",
        "EvaluateOptions",
        "PrepareSummary",
        "TrainSummary",
        "EvaluateSummary",
        "run_prepare",
        "run_train",
        "run_evaluate",
    )
    for name in names:
        assert name in colony_tracking.__all__
        assert getattr(colony_tracking, name) is getattr(
            colony_tracking.finetune, name
        )
    # The conversion stays private.
    assert "_convert" not in colony_tracking.__all__
    assert not hasattr(colony_tracking, "_convert")
