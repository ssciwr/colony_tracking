"""The finetune steps' flat config documents, shared by the CLI and the TUI.

Seams under test: the shared reader/writer
(:mod:`colony_tracking.finetune.config`) and ``colony_tracking.cli.main``
with ``train prepare|train|evaluate --config``. A step's config is a flat
TOML document — one key per option of its dataclass — written whole and
read back unchanged. ``--config`` may stand alone, explicit flags override
the file, and unknown keys, missing required options, and mistyped values
are usage errors (exit 2) in the pipeline config's wordings.
"""

import logging
from pathlib import Path

import pytest

import colony_tracking.cli as cli
from colony_tracking import (
    EvaluateSummary,
    PrepareSummary,
    TrainSummary,
)
from colony_tracking.finetune.config import (
    STEP_OPTIONS,
    read_step_options,
    resolve_step_document,
    write_step_config,
)
from colony_tracking.pipeline import ConfigError

#: A complete, tuned document per step (required keys plus one tune).
_DOCS = {
    "prepare": {
        "image_dir": "/data/images",
        "label_dir": "/data/labels",
        "out_dir": "/data/h5",
        "val_fraction": 0.3,
        "max_val_wells": 2,
        "min_instances": 2,
        "n_workers": 4,
        "val_wells": ["-C006", "-D007"],
    },
    "train": {
        "data_dir": "/data/h5",
        "out_dir": "/data/model",
        "n_epochs": 3,
        "lr": 2e-5,
        "patch_shape": [448, 448],
    },
    "evaluate": {
        "data_dir": "/data/h5",
        "split": "train",
        "checkpoint": "/data/model/best.pt",
        "n_max_images": 3,
        "output": "/data/report.json",
    },
}

_DATACLASS = STEP_OPTIONS

#: One representative document per error kind (unknown, missing, mistyped).
_ERROR_CASES = [
    (
        "train",
        {"data_dir": "/d", "out_dir": "/o", "nonsense": 1},
        "unknown key(s) ['nonsense'] — expected one key per name: ",
    ),
    (
        "train",
        {"data_dir": "/d"},
        "missing required key(s): ['out_dir']",
    ),
    (
        "train",
        {"data_dir": "/d", "out_dir": "/o", "n_epochs": "many"},
        "train: key 'n_epochs' must be an integer (got 'many')",
    ),
]


@pytest.mark.parametrize("step", list(_DOCS), ids=list(_DOCS))
def test_written_step_config_reads_back_unchanged(tmp_path, step):
    """The shared writer's file loads through the shared reader to exactly
    the options it wrote — defaults filled, None values omitted."""
    options = resolve_step_document(step, _DOCS[step])
    path = write_step_config(step, options, tmp_path / "configs" / f"{step}.toml")
    assert read_step_options(step, path) == options


@pytest.mark.parametrize("step", list(_DOCS), ids=list(_DOCS))
def test_resolve_step_document_builds_the_dataclass(step):
    """A complete flat document resolves to its step's options dataclass,
    with omitted keys taking their defaults."""
    options = resolve_step_document(step, _DOCS[step])
    assert isinstance(options, _DATACLASS[step])
    assert options == _DATACLASS[step].from_dict(_DOCS[step])


@pytest.mark.parametrize(
    ("step", "doc", "message"),
    _ERROR_CASES,
    ids=["unknown", "missing", "mistyped"],
)
def test_resolve_step_document_reports_the_config_errors(step, doc, message):
    """The three error kinds refuse with the pipeline's own wordings."""
    with pytest.raises(ConfigError) as info:
        resolve_step_document(step, doc)
    assert message in str(info.value)


# --- the CLI's --config flag ---------------------------------------------------


def _capture(monkeypatch, step: str) -> list:
    """Patch ``run_<step>`` to record its options and succeed."""
    calls: list = []
    summary = {
        "prepare": PrepareSummary(),
        "train": TrainSummary(),
        "evaluate": EvaluateSummary(),
    }[step]
    monkeypatch.setattr(
        cli, f"run_{step}",
        lambda options: (calls.append(options), summary)[1],
    )
    return calls


def _config_file(step: str, doc: dict, tmp_path) -> Path:
    """A config file written by the shared writer."""
    path = tmp_path / f"{step}.toml"
    write_step_config(step, resolve_step_document(step, doc), path)
    return path


def test_prepare_runs_from_the_config_file_alone(monkeypatch, tmp_path):
    """``train prepare --config`` needs no flags: the file supplies every
    option, required ones included."""
    calls = _capture(monkeypatch, "prepare")
    path = _config_file("prepare", _DOCS["prepare"], tmp_path)

    assert cli.main(["train", "prepare", "--config", str(path)]) == 0
    (options,) = calls
    assert options == resolve_step_document("prepare", _DOCS["prepare"])


def test_train_flag_overrides_the_config_file_and_nothing_else(
    monkeypatch, tmp_path
):
    """``--n-epochs 3`` wins over the file's value; every other option still
    comes from the file."""
    calls = _capture(monkeypatch, "train")
    path = _config_file("train", _DOCS["train"], tmp_path)

    code = cli.main([
        "train", "train", "--config", str(path), "--n-epochs", "7",
    ])

    assert code == 0
    (options,) = calls
    assert options.n_epochs == 7  # the explicit flag
    assert options.lr == 2e-5  # still from the file
    assert options.patch_shape == (448, 448)
    assert options.data_dir == Path("/data/h5")
    assert options.out_dir == Path("/data/model")


def test_evaluate_runs_from_a_config_and_keeps_flags_only_behaviour(
    monkeypatch, tmp_path
):
    """``--config`` may stand alone; without it, the flags drive the step
    exactly as before."""
    calls = _capture(monkeypatch, "evaluate")
    path = _config_file("evaluate", _DOCS["evaluate"], tmp_path)

    assert cli.main(["train", "evaluate", "--config", str(path)]) == 0
    (from_file,) = calls
    assert from_file == resolve_step_document("evaluate", _DOCS["evaluate"])

    code = cli.main([
        "train", "evaluate", "--data-dir", "/flag/h5", "--split", "val",
        "--n-max-images", "1",
    ])
    assert code == 0
    (flags_only,) = calls[-1:]
    assert flags_only.data_dir == Path("/flag/h5")
    assert flags_only.split == "val"
    assert flags_only.n_max_images == 1
    assert flags_only.checkpoint is None


@pytest.mark.parametrize(
    ("step", "doc", "message"),
    _ERROR_CASES,
    ids=["unknown", "missing", "mistyped"],
)
def test_config_errors_are_usage_errors(
    monkeypatch, tmp_path, caplog, step, doc, message
):
    """An unknown key, a missing required option, and a mistyped value each
    exit 2 and log the pipeline config's wording."""
    calls = _capture(monkeypatch, step)
    path = tmp_path / f"{step}.toml"
    path.write_text(
        "\n".join(f"{key} = {_toml(value)}" for key, value in doc.items()),
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["train", step, "--config", str(path)])

    assert code == 2
    assert message in caplog.text
    assert calls == []


def _toml(value) -> str:
    """A minimal TOML literal for the error-kind fixtures."""
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)
