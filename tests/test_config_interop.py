"""Config interop: the stage CLIs consume the TOML the TUI writes.

Seams under test: ``colony_tracking.cli.main`` with ``--config`` on the
``difference``, ``segment``, and ``analyze`` subcommands. Each stage accepts
both shapes — its own flat document (what ``ConfigDraft`` writes for a stage)
and a sectioned ``pipeline.toml`` (resolved with the chain wiring and
run-level values). The ``run_*`` functions are monkeypatched to record the
constructed options. Explicit flags override the file; invalid or unreadable
files keep the exit-code contract (2 for a config error, 1 for a missing
file).
"""

import logging
from pathlib import Path

import pytest

import colony_tracking.cli as cli
from colony_tracking import (
    AnalysisOptions,
    AnalysisSummary,
    DifferenceOptions,
    DifferenceSummary,
    SegmentationOptions,
    SegmentationSummary,
)
from colony_tracking.tui.draft import ConfigDraft

#: Run type → the CLI subcommand that drives it.
_COMMAND = {
    "difference": "difference",
    "segmentation": "segment",
    "analysis": "analyze",
}

_STAGE_OPTIONS = {
    "difference": DifferenceOptions,
    "segmentation": SegmentationOptions,
    "analysis": AnalysisOptions,
}

#: A complete flat document per stage (required keys plus one tuned option).
_FLAT = {
    "difference": {
        "raw_dir": "/data/raw",
        "out_dir": "/data/diff",
        "reference": 3,
        "subsample": 4,
    },
    "segmentation": {
        "input": "raw",
        "raw_dir": "/data/raw",
        "out_dir": "/data/seg",
        "min_size": 300,
    },
    "analysis": {
        "seg_dirs": ["/data/seg"],
        "raw_dir": "/data/raw",
        "out_root": "/data/analysis",
        "rim_width_px": 40,
    },
}


def _summary(run_type: str):
    if run_type == "difference":
        return DifferenceSummary(wells={}, processed=0, skipped=0, failed=0)
    if run_type == "segmentation":
        return SegmentationSummary(wells=0, processed=0, skipped=0, failed=0)
    return AnalysisSummary()


def _capture(monkeypatch, run_type: str) -> list:
    """Patch ``run_<run_type>`` to record its options and succeed."""
    calls: list = []
    summary = _summary(run_type)
    monkeypatch.setattr(
        cli, f"run_{run_type}",
        lambda options: (calls.append(options), summary)[1],
    )
    return calls


def _stage_draft(run_type: str) -> ConfigDraft:
    """The TUI's draft for one stage, complete and tuned."""
    draft = ConfigDraft(run_type)
    for key, value in _FLAT[run_type].items():
        draft.set(run_type, key, value)
    return draft


def _pipeline_draft() -> ConfigDraft:
    """A complete pipeline draft: run level, wired chain, one tune per stage."""
    draft = ConfigDraft()
    draft.set("pipeline", "raw_dir", "/data/raw")
    draft.set("pipeline", "out_root", "/data/out")
    draft.set("pipeline", "reference", 3)
    draft.set("pipeline", "wells", ["-A001", "-B002"])
    draft.set("difference", "subsample", 4)
    draft.set("segmentation", "min_size", 300)
    draft.set("analysis", "rim_width_px", 40)
    return draft


@pytest.mark.parametrize("run_type", list(_FLAT), ids=list(_FLAT))
def test_stage_cli_consumes_the_tui_flat_config(monkeypatch, tmp_path, run_type):
    """A stage written by ``ConfigDraft`` runs with no flags and builds
    exactly the options the draft resolved."""
    calls = _capture(monkeypatch, run_type)
    draft = _stage_draft(run_type)
    path = tmp_path / f"{run_type}.toml"
    draft.write(path)

    assert cli.main([_COMMAND[run_type], "--config", str(path)]) == 0
    (options,) = calls
    assert isinstance(options, _STAGE_OPTIONS[run_type])
    assert options == draft.resolve()


def test_explicit_flags_override_config_values(monkeypatch, tmp_path):
    """A flag wins over the file — for a required path and a tuning option;
    the file still supplies everything the command line omits."""
    calls = _capture(monkeypatch, "difference")
    path = tmp_path / "difference.toml"
    _stage_draft("difference").write(path)

    code = cli.main([
        "difference", "--config", str(path),
        "--raw-dir", "/override/raw", "--subsample", "2",
    ])

    assert code == 0
    (options,) = calls
    assert options.raw_dir == Path("/override/raw")
    assert options.subsample == 2
    assert options.out_dir == Path("/data/diff")  # still from the file
    assert options.reference == 3


#: The chain wiring a stage sees in a pipeline config with reference 3 and
#: the run-level wells.
_PIPELINE_EXPECTED = {
    "difference": {
        "raw_dir": Path("/data/raw"),
        "reference": 3,
        "out_dir": Path("/data/out/difference_LO003"),
        "wells": ["-A001", "-B002"],
        "subsample": 4,
    },
    "segmentation": {
        "input": "diff",
        "diff_dir": Path("/data/out/difference_LO003"),
        "out_dir": Path("/data/out/seg_growth_diff_LO003"),
        "exclude_before": 3,
        "wells": ["-A001", "-B002"],
        "min_size": 300,
    },
    "analysis": {
        "raw_dir": Path("/data/raw"),
        "out_root": Path("/data/out/analysis"),
        "seg_dirs": [Path("/data/out/seg_growth_diff_LO003")],
        "wells": ["-A001", "-B002"],
        "rim_width_px": 40,
    },
}


@pytest.mark.parametrize("run_type", list(_FLAT), ids=list(_FLAT))
def test_stage_cli_consumes_a_pipeline_config(monkeypatch, tmp_path, run_type):
    """A ``pipeline.toml`` resolves to this stage's chain-wired options —
    the run-level raw dir/reference/wells, the derived trees, and the
    stage's own tuning — exactly as the pipeline itself would resolve them."""
    calls = _capture(monkeypatch, run_type)
    draft = _pipeline_draft()
    path = tmp_path / "pipeline.toml"
    resolved = draft.write(path)

    assert cli.main([_COMMAND[run_type], "--config", str(path)]) == 0
    (options,) = calls
    for attr, value in _PIPELINE_EXPECTED[run_type].items():
        assert getattr(options, attr) == value
    assert options == getattr(resolved, run_type)


def test_abbreviated_flags_stay_explicit(monkeypatch, tmp_path):
    """argparse's unambiguous abbreviations (``--sub`` / ``--raw``) are
    still explicit — both alone and as overrides over a config file."""
    calls = _capture(monkeypatch, "difference")

    assert cli.main([
        "difference", "--raw", "/flag/raw", "--out-dir", "/flag/out",
        "--sub", "4",
    ]) == 0
    (flag_only,) = calls
    assert flag_only.raw_dir == Path("/flag/raw")
    assert flag_only.out_dir == Path("/flag/out")
    assert flag_only.subsample == 4

    draft = _stage_draft("difference")
    path = tmp_path / "difference.toml"
    draft.write(path)
    assert cli.main([
        "difference", "--config", str(path),
        "--raw", "/override/raw", "--sub", "2",
    ]) == 0
    overridden = calls[-1]
    assert overridden.raw_dir == Path("/override/raw")
    assert overridden.subsample == 2
    assert overridden.out_dir == Path("/data/diff")  # still from the file


def test_missing_required_key_after_the_merge_is_a_usage_error(tmp_path, caplog):
    path = tmp_path / "partial.toml"
    path.write_text('raw_dir = "/data/raw"\n', encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["difference", "--config", str(path)])

    assert code == 2
    assert "out_dir" in caplog.text


def test_invalid_config_is_a_usage_error_and_a_missing_file_is_not(
    tmp_path, caplog
):
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(
        'raw_dir = "/data/raw"\nout_dir = "/data/diff"\nnonsense = 1\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        assert cli.main(["difference", "--config", str(unknown)]) == 2
    assert "nonsense" in caplog.text

    owned = tmp_path / "owned.toml"
    owned.write_text(
        '[pipeline]\nraw_dir = "/data/raw"\nout_root = "/data/out"\n'
        '[difference]\nraw_dir = "/elsewhere"\n',
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        assert cli.main(["difference", "--config", str(owned)]) == 2
    assert "owned by the pipeline" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        assert cli.main(["analyze", "--config", str(tmp_path / "nope.toml")]) == 1
    assert "nope.toml" in caplog.text
