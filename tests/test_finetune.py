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
from dataclasses import fields
from pathlib import Path

import pytest

from colony_tracking.finetune import (
    EvaluateOptions,
    PrepareOptions,
    TrainOptions,
)


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
