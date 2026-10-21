"""Tests for the TUI's ``StageRunner`` and app skeleton (ticket 01).

Seams under test:

- ``runner.start`` calls the in-process ``run_*`` functions (monkeypatched
  here — the suite is hermetic, no data/GPU) with the resolved stage
  options the pipeline's wiring would give them, one run at a time;
- log records route from the package logger to the run's ``on_record``,
  via ``call_from_thread`` onto the UI thread when a textual app drives
  the run, and the handler is detached (exactly what was attached) when
  the run completes;
- an exception or a failed summary is captured on the ``RunHandle`` and
  reported through ``on_done`` without killing anything;
- the app skeleton launches headlessly (textual's ``Pilot``): the home
  screen first, the menu navigates into the registered screens and back,
  and the planned Train entry is disabled.

Async ``Pilot`` scenarios run through ``asyncio.run`` — the suite has no
async test plugin, and this keeps the tests dependency-free.
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest
from textual.widgets import OptionList

import colony_tracking.cli as cli
import colony_tracking.tui.runner as runner
from colony_tracking import (
    AnalysisSummary,
    DifferenceSummary,
    PipelineConfig,
    PipelineSummary,
    SegmentationSummary,
    log,
)
from colony_tracking.difference import WellResult
from colony_tracking.tui import run_tui
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.screens.home import HomeScreen

#: How long a test waits for a (fake, fast) run to finish.
_TIMEOUT = 10.0

#: The ``run_*`` function each kind runs (monkeypatch target).
_RUN_FUNCTION = {
    "difference": "run_difference",
    "segmentation": "run_segmentation",
    "analysis": "run_analysis",
    "pipeline": "run_pipeline",
}


def _config(tmp_path: Path) -> PipelineConfig:
    """A minimal resolved config: run level only, derived stages."""
    return PipelineConfig.from_dict(
        {
            "pipeline": {
                "raw_dir": str(tmp_path / "raw"),
                "out_root": str(tmp_path / "out"),
            },
        }
    )


def _diff_summary(failed: int = 0) -> DifferenceSummary:
    return DifferenceSummary(wells={}, processed=1, skipped=0, failed=failed)


# --- start: kinds and wiring ------------------------------------------------------


@pytest.mark.parametrize("kind", list(runner.KINDS))
def test_start_runs_the_kind_on_the_resolved_config(monkeypatch, tmp_path, kind):
    config = _config(tmp_path)
    expected_arg = {
        "difference": config.difference,
        "segmentation": config.segmentation,
        "analysis": config.analysis,
        "pipeline": config,
    }[kind]
    summaries = {
        "difference": _diff_summary(),
        "segmentation": SegmentationSummary(wells=1, processed=1, skipped=0),
        "analysis": AnalysisSummary(),
        "pipeline": PipelineSummary(status="ok"),
    }
    calls = []

    def fake(options):
        calls.append(options)
        return summaries[kind]

    monkeypatch.setattr(runner, _RUN_FUNCTION[kind], fake)
    done = []
    handle = runner.start(kind, config, on_done=done.append)

    assert handle.done.wait(_TIMEOUT)
    assert calls == [expected_arg]
    if kind == "segmentation":  # the wiring a pipeline run would use
        assert expected_arg.input == "diff"
        assert expected_arg.diff_dir == config.difference.out_dir
    assert handle.summary == summaries[kind]
    assert handle.error is None
    assert not handle.failed
    assert done == [handle]
    assert runner.active() is None


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown run kind 'train'"):
        runner.start("train", _config(tmp_path), on_done=lambda handle: None)


# --- one run at a time ------------------------------------------------------------


def test_second_start_while_a_run_is_active_is_rejected(monkeypatch, tmp_path):
    config = _config(tmp_path)
    release = threading.Event()

    def slow(options):
        release.wait(_TIMEOUT)
        return _diff_summary()

    monkeypatch.setattr(runner, "run_difference", slow)
    handle = runner.start("difference", config, on_done=lambda h: None)

    assert runner.active() is handle
    with pytest.raises(RuntimeError, match="already active"):
        runner.start("segmentation", config, on_done=lambda h: None)

    release.set()
    assert handle.done.wait(_TIMEOUT)
    # The runner is free again: the next start succeeds (a resume).
    monkeypatch.setattr(runner, "run_difference", lambda options: _diff_summary())
    second = runner.start("difference", config, on_done=lambda h: None)
    assert second.done.wait(_TIMEOUT)
    assert runner.active() is None


# --- failures are captured, never fatal --------------------------------------------


def test_a_stage_exception_is_caught_and_reported(monkeypatch, tmp_path, caplog):
    def boom(options):
        raise RuntimeError("the difference stage exploded")

    monkeypatch.setattr(runner, "run_difference", boom)
    done = []
    handle = runner.start("difference", _config(tmp_path), on_done=done.append)

    assert handle.done.wait(_TIMEOUT)
    assert isinstance(handle.error, RuntimeError)
    assert handle.summary is None
    assert handle.failed
    assert done == [handle]
    assert runner.active() is None
    assert "the difference stage exploded" in caplog.text  # logged, not fatal


def test_summary_failed_maps_the_cli_exit_1_semantics():
    assert not runner.summary_failed(_diff_summary())
    assert runner.summary_failed(_diff_summary(failed=2))
    assert not runner.summary_failed(
        SegmentationSummary(wells=1, processed=1, skipped=0)
    )
    assert runner.summary_failed(
        SegmentationSummary(
            wells=1,
            processed=0,
            skipped=0,
            failed=1,
            failed_wells=["-A001"],
        )
    )
    assert not runner.summary_failed(AnalysisSummary())
    assert runner.summary_failed(AnalysisSummary(failures=["seg_growth_diff"]))
    assert runner.summary_failed(
        AnalysisSummary(failed_wells={"seg_growth_diff": ["-A001"]})
    )
    assert not runner.summary_failed(PipelineSummary(status="ok"))
    assert runner.summary_failed(
        PipelineSummary(status="failed", failed_stage="difference")
    )


def test_failed_summary_lines_are_the_cli_wording():
    """The failed-summary lines — the single source the runner logs and
    the run screen's error panel shows — are the CLI's ``_dispatch``
    ERROR lines, verbatim; empty when nothing failed."""
    difference = DifferenceSummary(
        wells={
            "-A001": WellResult(
                well="-A001",
                status="failed",
                n_frames=3,
                error="no frames",
            )
        },
        processed=0,
        skipped=0,
        failed=1,
    )
    assert runner.failed_summary_lines("difference", difference) == [
        "difference: 1 well(s) failed: -A001 (no frames)"
    ]
    segmentation = SegmentationSummary(
        wells=2,
        processed=0,
        skipped=0,
        failed=2,
        failed_wells=["-A001", "-B002"],
    )
    assert runner.failed_summary_lines("segmentation", segmentation) == [
        "segment: 2 well(s) failed: -A001, -B002"
    ]
    analysis = AnalysisSummary(
        failures=["seg_growth_diff"],
        failed_wells={"other": ["-C003"]},
    )
    assert runner.failed_summary_lines("analysis", analysis) == [
        "analyze: 1 root(s) failed: seg_growth_diff",
        "analyze: 1 well(s) failed: -C003",
    ]
    pipeline = PipelineSummary(status="failed", failed_stage="difference")
    assert runner.failed_summary_lines("pipeline", pipeline) == [
        "pipeline: the difference stage failed"
    ]
    assert runner.failed_summary_lines("difference", _diff_summary()) == []


def test_a_failed_summary_is_reported_and_logged_like_the_cli(
    monkeypatch, tmp_path, caplog
):
    summary = DifferenceSummary(
        wells={
            "-A001": WellResult(
                well="-A001",
                status="failed",
                n_frames=3,
                error="no frames",
            )
        },
        processed=0,
        skipped=0,
        failed=1,
    )
    monkeypatch.setattr(runner, "run_difference", lambda options: summary)
    done = []
    handle = runner.start("difference", _config(tmp_path), on_done=done.append)

    assert handle.done.wait(_TIMEOUT)
    assert handle.error is None
    assert handle.summary is summary
    assert handle.failed  # the exit-1 semantics: the UI shows an error panel
    assert done == [handle]
    assert "difference: 1 well(s) failed: -A001 (no frames)" in caplog.text


# --- log routing -------------------------------------------------------------------


def test_log_records_are_forwarded_only_while_the_run_is_active(monkeypatch, tmp_path):
    def fake(options):
        log.info("tui test: difference started")
        return _diff_summary()

    monkeypatch.setattr(runner, "run_difference", fake)
    records = []
    idents = []
    handle = runner.start(
        "difference",
        _config(tmp_path),
        on_done=lambda h: idents.append(threading.get_ident()),
        on_record=records.append,
    )

    assert handle.done.wait(_TIMEOUT)
    assert "tui test: difference started" in [record.getMessage() for record in records]
    # Without an app the callbacks run on the worker thread, not the
    # caller's (the app path below delivers on the UI thread instead).
    assert idents and set(idents) != {threading.get_ident()}
    # Detached on completion: later records do not reach the pane.
    n_before = len(records)
    log.info("tui test: after the run")
    assert len(records) == n_before


def test_runs_never_stack_handlers(monkeypatch, tmp_path):
    handlers_before = list(log.handlers)
    for _ in range(2):
        handle = runner.start(
            "difference",
            _config(tmp_path),
            on_done=lambda h: None,
        )
        assert handle.done.wait(_TIMEOUT)
    assert list(log.handlers) == handlers_before


def test_through_a_textual_app_callbacks_arrive_on_the_ui_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_difference",
        lambda options: (log.info("tui test: difference ran"), _diff_summary())[1],
    )
    delivered = []
    records = []

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            handle = runner.start(
                "difference",
                _config(tmp_path),
                on_done=lambda h: delivered.append(threading.get_ident()),
                on_record=lambda r: records.append(
                    (threading.get_ident(), r.getMessage())
                ),
                app=app,
            )
            deadline = time.monotonic() + _TIMEOUT
            while not handle.done.is_set() and time.monotonic() < deadline:
                await pilot.pause(0.05)
            assert handle.done.is_set()

    asyncio.run(scenario())
    # The test coroutine and textual's message loop share the main thread,
    # so delivery through ``call_from_thread`` lands on this thread id.
    ui_thread = threading.get_ident()
    assert delivered == [ui_thread]
    assert [text for _ident, text in records] == ["tui test: difference ran"]
    assert {ident for ident, _text in records} == {ui_thread}
    assert runner.active() is None


# --- the app skeleton (headless) ----------------------------------------------------


def test_the_app_launches_and_navigates():
    """Headless app smoke: the home screen first, the menu navigates into
    a registered screen and back, and every registered name opens."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            assert isinstance(app.draft, ConfigDraft)
            home = app.screen
            assert isinstance(home, HomeScreen)
            menu = home.query_one(OptionList)
            # The planned train entry is disabled (spec requirement 3)…
            assert menu.get_option("train").disabled
            # …the real entries are not.
            assert not menu.get_option("quick_start").disabled
            menu.focus()
            await pilot.pause()
            await pilot.press("enter")  # select the first entry
            await pilot.pause()
            assert app.screen is not home  # a registered screen opened
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is home
            for name in ColonyTuiApp.SCREENS:  # the registry opens cleanly
                app.push_screen(name)
                await pilot.pause()
                assert app.screen is not home
                app.pop_screen()
                await pilot.pause()
                assert app.screen is home

    asyncio.run(scenario())


def test_run_tui_rejects_arguments(caplog):
    assert run_tui(["--bogus"]) == 2  # the tui command takes no flags
    assert "usage error" in caplog.text


def test_cli_dispatches_tui(monkeypatch):
    monkeypatch.setattr(cli, "run_tui", lambda: 7)
    assert cli.main(["tui"]) == 7


def test_cli_rejects_tui_flags():
    assert cli.main(["tui", "--raw-dir", "x"]) == 2  # a usage error
