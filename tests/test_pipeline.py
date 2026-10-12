"""Tests for the pipeline module (colony_tracking.pipeline).

Seams under test (per the ticket): the TOML config contract (defaults for
omitted keys, run-level + per-stage values, round-trip equality, owned keys),
the derived tree names and the segment ``exclude_before`` derivation, the
orchestration of the standard chain (difference → segment on the difference
tree → analyze) with the three stage functions monkeypatched to recording
fakes, failure abort semantics, resume on re-entry (a real difference stage
against fake later stages — no model involved), and the usage errors for
invalid configs.
"""

import logging
import tomllib
from pathlib import Path
from textwrap import dedent

import pytest

from colony_tracking import pipeline
from colony_tracking.analysis import AnalysisSummary
from colony_tracking.difference import DifferenceSummary
from colony_tracking.pipeline import (
    ConfigError,
    PipelineConfig,
    PipelineSummary,
    read_config,
    run_pipeline,
    write_config,
)
from colony_tracking.segmentation import SegmentationSummary
from conftest import build_mini_plate


def _write_config(tmp_path, text: str) -> Path:
    """Write a TOML config text and return its path."""
    path = tmp_path / "config.toml"
    path.write_text(dedent(text), encoding="utf-8")
    return path


# --- orchestration fakes --------------------------------------------------------

#: The canned stage summaries the fakes return (all healthy unless told
#: otherwise): the pipeline must react to the real summary fields.


def _diff_summary(failed: int = 0) -> DifferenceSummary:
    return DifferenceSummary(wells={}, processed=2, skipped=0, failed=failed)


def _seg_summary(failed: int = 0) -> SegmentationSummary:
    return SegmentationSummary(
        wells=2, processed=2 - failed, skipped=0, failed=failed
    )


def _ana_summary(failures: list[str] | None = None) -> AnalysisSummary:
    return AnalysisSummary(failures=list(failures or []))


def _install_fakes(
    monkeypatch,
    *,
    diff_summary: DifferenceSummary | None = None,
    seg_summary: SegmentationSummary | None = None,
    ana_summary: AnalysisSummary | None = None,
    diff_error: Exception | None = None,
    seg_error: Exception | None = None,
    ana_error: Exception | None = None,
) -> list[tuple[str, object]]:
    """Replace the three stage functions with recording fakes.

    Returns the list of (stage name, options) calls in call order.
    """
    calls: list[tuple[str, object]] = []

    def fake_difference(options):
        calls.append(("difference", options))
        if diff_error is not None:
            raise diff_error
        return diff_summary if diff_summary is not None else _diff_summary()

    def fake_segmentation(options):
        calls.append(("segmentation", options))
        if seg_error is not None:
            raise seg_error
        return seg_summary if seg_summary is not None else _seg_summary()

    def fake_analysis(options):
        calls.append(("analysis", options))
        if ana_error is not None:
            raise ana_error
        return ana_summary if ana_summary is not None else _ana_summary()

    monkeypatch.setattr(pipeline, "run_difference", fake_difference)
    monkeypatch.setattr(pipeline, "run_segmentation", fake_segmentation)
    monkeypatch.setattr(pipeline, "run_analysis", fake_analysis)
    return calls


# --- config parse: defaults for omitted keys ---------------------------------


def test_from_toml_defaults_for_omitted_keys(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        """,
    )
    config = read_config(path)
    assert config.raw_dir == Path("/data/plate")
    assert config.out_root == Path("/data/run")
    assert config.reference == "first"
    assert config.wells is None
    # every omitted stage option takes its spec default
    assert config.difference.display_percentile == 99.5
    assert config.difference.subsample == 8
    assert config.difference.n_workers == 6
    assert config.difference.dry_run is False
    assert config.segmentation.model_type == "vit_b_lm"
    assert config.segmentation.fg_ref == 0.5
    assert config.segmentation.min_size == 200
    assert config.segmentation.merge_touching is True
    assert config.segmentation.reid_only is False
    assert config.analysis.rim_width_px == 60
    assert config.analysis.max_bad_segments == 2
    assert config.analysis.keep_bad_wells is False
    assert config.analysis.raw_seg_dir is None


def test_from_toml_run_level_and_per_stage_values(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5
        wells = ["-A001", "-A002"]

        [difference]
        display_percentile = 99.0
        n_workers = 2

        [segmentation]
        min_size = 300
        exclude_before = 7

        [analysis]
        rim_width_px = 80
        keep_pre_existing = true
        raw_seg_dir = "/data/run/seg_growth_raw_LO001"
        """,
    )
    config = read_config(path)
    assert config.reference == 5
    assert config.wells == ["-A001", "-A002"]
    assert config.difference.display_percentile == 99.0
    assert config.difference.n_workers == 2
    assert config.segmentation.min_size == 300
    assert config.segmentation.exclude_before == 7
    assert config.analysis.rim_width_px == 80
    assert config.analysis.keep_pre_existing is True
    assert config.analysis.raw_seg_dir == Path("/data/run/seg_growth_raw_LO001")
    # untouched options in a present section still take their defaults
    assert config.difference.subsample == 8
    assert config.segmentation.model_type == "vit_b_lm"
    assert config.analysis.rim_overlap_max == 0.1


# --- derived names ------------------------------------------------------------


def test_derived_names_numeric_reference(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5
        """,
    )
    config = read_config(path)
    assert config.difference.out_dir == Path("/data/run/difference_LO005")
    assert config.segmentation.out_dir == Path("/data/run/seg_growth_diff_LO005")
    assert config.analysis.out_root == Path("/data/run/analysis")
    assert config.analysis.seg_dirs == [Path("/data/run/seg_growth_diff_LO005")]


def test_derived_names_first_frame_reference(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = "first"
        """,
    )
    config = read_config(path)
    assert config.difference.out_dir == Path("/data/run/difference")
    assert config.segmentation.out_dir == Path("/data/run/seg_growth_diff")
    assert config.analysis.seg_dirs == [Path("/data/run/seg_growth_diff")]


def test_explicit_overrides_win_over_derived_names(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5

        [difference]
        out_dir = "/elsewhere/diff"

        [segmentation]
        out_dir = "/elsewhere/seg"

        [analysis]
        out_root = "/elsewhere/analysis"
        seg_dirs = ["/elsewhere/seg", "/other/seg"]
        """,
    )
    config = read_config(path)
    assert config.difference.out_dir == Path("/elsewhere/diff")
    assert config.segmentation.out_dir == Path("/elsewhere/seg")
    assert config.analysis.out_root == Path("/elsewhere/analysis")
    assert config.analysis.seg_dirs == [Path("/elsewhere/seg"), Path("/other/seg")]


def test_analysis_seg_dirs_follows_segment_override(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5

        [segmentation]
        out_dir = "/data/run/custom_seg"
        """,
    )
    config = read_config(path)
    assert config.segmentation.out_dir == Path("/data/run/custom_seg")
    # no explicit [analysis].seg_dirs: it tracks the (overridden) segment root
    assert config.analysis.seg_dirs == [Path("/data/run/custom_seg")]


# --- exclude_before derivation --------------------------------------------------


def test_exclude_before_derives_from_numeric_reference(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5
        """,
    )
    config = read_config(path)
    assert config.segmentation.exclude_before == 5


def test_explicit_exclude_before_wins(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5

        [segmentation]
        exclude_before = 7
        """,
    )
    config = read_config(path)
    assert config.segmentation.exclude_before == 7


def test_exclude_before_stays_unset_for_first_reference(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = "first"
        """,
    )
    config = read_config(path)
    assert config.segmentation.exclude_before is None


# --- TOML round-trip ----------------------------------------------------------


def test_toml_round_trip_equality(tmp_path):
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5
        wells = ["-A001"]

        [difference]
        display_percentile = 99.0

        [segmentation]
        exclude_before = 7
        min_solidity = 0.6

        [analysis]
        raw_seg_dir = "/data/run/seg_growth_raw_LO001"
        """,
    )
    config = read_config(path)
    written = write_config(config, tmp_path / "round.toml")
    assert read_config(written) == config


def test_toml_round_trip_minimal_programmatic_config(tmp_path):
    config = PipelineConfig(
        raw_dir=Path("/data/plate"),
        out_root=Path("/data/run"),
        reference=5,
        wells=["-A001"],
    )
    written = write_config(config, tmp_path / "config.toml")
    assert read_config(written) == config


def test_written_toml_omits_owned_keys(tmp_path):
    config = PipelineConfig(
        raw_dir=Path("/data/plate"), out_root=Path("/data/run"), reference=5
    )
    written = write_config(config, tmp_path / "config.toml")
    doc = tomllib.loads(written.read_text(encoding="utf-8"))
    assert set(doc) == {"pipeline", "difference", "segmentation", "analysis"}
    assert doc["pipeline"]["reference"] == 5
    # owned by the pipeline: taken from [pipeline] or the chain wiring
    assert "raw_dir" not in doc["difference"]
    assert "reference" not in doc["difference"]
    assert "input" not in doc["segmentation"]
    assert "diff_dir" not in doc["segmentation"]
    assert "raw_dir" not in doc["analysis"]


# --- orchestration: the standard chain (recording fakes, no model) --------------


def _minimal_config(tmp_path) -> PipelineConfig:
    path = _write_config(
        tmp_path,
        """\
        [pipeline]
        raw_dir = "/data/plate"
        out_root = "/data/run"
        reference = 5
        wells = ["-A001"]
        """,
    )
    return read_config(path)


def test_stages_run_in_order_with_wired_options(tmp_path, monkeypatch):
    config = _minimal_config(tmp_path)
    calls = _install_fakes(monkeypatch)
    summary = run_pipeline(config)
    assert [name for name, _ in calls] == [
        "difference",
        "segmentation",
        "analysis",
    ]
    (diff_opts,) = [opts for name, opts in calls if name == "difference"]
    assert diff_opts.raw_dir == Path("/data/plate")
    assert diff_opts.out_dir == Path("/data/run/difference_LO005")
    assert diff_opts.reference == 5
    assert diff_opts.wells == ["-A001"]
    (seg_opts,) = [opts for name, opts in calls if name == "segmentation"]
    # the segment stage consumes the difference tree just produced
    assert seg_opts.input == "diff"
    assert seg_opts.diff_dir == Path("/data/run/difference_LO005")
    assert seg_opts.out_dir == Path("/data/run/seg_growth_diff_LO005")
    assert seg_opts.exclude_before == 5
    (ana_opts,) = [opts for name, opts in calls if name == "analysis"]
    # the analyze stage consumes the segment root plus the run-level raw dir
    assert ana_opts.seg_dirs == [Path("/data/run/seg_growth_diff_LO005")]
    assert ana_opts.raw_dir == Path("/data/plate")
    assert ana_opts.out_root == Path("/data/run/analysis")
    assert ana_opts.wells == ["-A001"]
    # no raw-input segmentation root is ever invented
    assert ana_opts.raw_seg_dir is None
    assert summary.status == "ok"
    assert summary.failed_stage is None
    assert summary.difference is not None and summary.difference.processed == 2
    assert summary.segmentation is not None and summary.segmentation.failed == 0
    assert summary.analysis is not None and summary.analysis.failures == []


def test_stage_exception_aborts_the_chain(tmp_path, monkeypatch, caplog):
    config = _minimal_config(tmp_path)
    calls = _install_fakes(
        monkeypatch, diff_error=FileNotFoundError("no such raw tree")
    )
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_pipeline(config)
    # later stages are not called
    assert [name for name, _ in calls] == ["difference"]
    assert summary.status == "failed"
    assert summary.failed_stage == "difference"
    assert summary.difference is None
    assert summary.segmentation is None
    assert summary.analysis is None
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("difference stage failed" in r.getMessage() for r in errors)


def test_stage_summary_with_failed_wells_aborts_the_chain(
    tmp_path, monkeypatch, caplog
):
    config = _minimal_config(tmp_path)
    calls = _install_fakes(monkeypatch, diff_summary=_diff_summary(failed=1))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_pipeline(config)
    assert [name for name, _ in calls] == ["difference"]
    assert summary.status == "failed"
    assert summary.failed_stage == "difference"
    # the failed stage's summary is still reported
    assert summary.difference is not None and summary.difference.failed == 1
    assert any(
        "difference stage failed for 1 well(s)" in r.getMessage()
        for r in caplog.records
    )


def test_segment_failure_skips_analysis(tmp_path, monkeypatch, caplog):
    config = _minimal_config(tmp_path)
    calls = _install_fakes(monkeypatch, seg_summary=_seg_summary(failed=2))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_pipeline(config)
    assert [name for name, _ in calls] == ["difference", "segmentation"]
    assert summary.status == "failed"
    assert summary.failed_stage == "segmentation"
    assert summary.segmentation is not None and summary.segmentation.failed == 2
    assert summary.analysis is None


def test_analysis_failures_fail_the_summary(tmp_path, monkeypatch, caplog):
    config = _minimal_config(tmp_path)
    calls = _install_fakes(
        monkeypatch, ana_summary=_ana_summary(failures=["seg_growth_diff_LO005"])
    )
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        summary = run_pipeline(config)
    assert [name for name, _ in calls] == [
        "difference",
        "segmentation",
        "analysis",
    ]
    assert summary.status == "failed"
    assert summary.failed_stage == "analysis"
    assert summary.analysis is not None
    assert summary.analysis.failures == ["seg_growth_diff_LO005"]