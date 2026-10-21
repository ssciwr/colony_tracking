"""Tests for the run screens and the full-auto quick start (ticket 03).

Seams under test:

- the quick start's *Run now*: the fields sync into the shared draft, the
  shared save dialog writes the TOML *before* the run starts (recorded at
  the mocked ``run_pipeline``'s call time), and the run screen ends with
  the chain's summaries displayed;
- the quick start's *Write TOML…*: writes the config, never runs;
- the quick start's action gate (the stage screens' mechanism): the
  actions stay disabled until the fields sync and the draft resolves —
  the pipeline's exact ``ConfigError`` (or the reference parser's) on the
  status line — and a press on a closed gate writes and runs nothing;
- the run screen: log records reach the pane while the run is active,
  the current-stage line comes from the pipeline's structured stage-start
  markers (never parsed log text), the final summary block comes from the
  summary objects, and a failed summary or an exception shows the error
  panel — the app survives every outcome;
- the direct-start entry points: ``open_stage_run``'s segmentation
  precondition (the difference tree must exist when the draft has no
  ``[difference]`` section) and wiring, and the one-run-at-a-time rule
  reported politely by both layers;
- one real in-process difference run on a conftest mini plate (the
  un-mocked path), run twice: the re-run resumes via the done marker
  (processed 0, skipped 1).

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free). Assertions on
widget state run inside the headless session — the screen stack is gone
once it ends.
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest
from conftest import DiskSpec, build_mini_plate
from textual.widgets import Button, Input, RichLog, Static

from colony_tracking import (
    AnalysisSummary,
    DifferenceSummary,
    PipelineConfig,
    PipelineSummary,
    SegmentationSummary,
    log,
)
from colony_tracking.difference import WellResult
from colony_tracking.pipeline import read_config
from colony_tracking.tui import runner
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.screens._save import SaveConfigDialog
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.quick_start import QuickStartScreen
from colony_tracking.tui.screens.run import RunScreen, open_pipeline_run, open_stage_run

#: How long a test waits for a (fake or small, fast) run to finish.
_TIMEOUT = 10.0


def _config(raw: Path, out: Path) -> PipelineConfig:
    """A minimal resolved config: the run level only, derived stages."""
    return PipelineConfig.from_dict(
        {
            "pipeline": {
                "raw_dir": str(raw),
                "out_root": str(out),
            },
        }
    )


def _draft(raw: Path, out: Path) -> ConfigDraft:
    """A draft with the run level filled."""
    draft = ConfigDraft()
    draft.set("pipeline", "raw_dir", raw)
    draft.set("pipeline", "out_root", out)
    return draft


def _diff_summary() -> DifferenceSummary:
    return DifferenceSummary(wells={}, processed=1, skipped=0, failed=0)


def _log_text(screen: RunScreen) -> str:
    """The run screen's log pane text so far."""
    rich_log = screen.query_one("#run-log", RichLog)
    return "\n".join(strip.text for strip in rich_log.lines)


async def _await_run_screen(pilot, app: ColonyTuiApp) -> RunScreen:
    """The app's current screen once it is a run screen whose run has
    started (a dialog's write pushes it; the run starts on mount)."""
    deadline = time.monotonic() + _TIMEOUT
    while (
        not isinstance(app.screen, RunScreen) or app.screen.handle is None
    ) and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert isinstance(app.screen, RunScreen)
    assert app.screen.handle is not None, "the run screen never started"
    return app.screen


async def _await_done(pilot, screen: RunScreen):
    """Wait for the screen's run to finish (the UI keeps pumping) and
    return its handle."""
    deadline = time.monotonic() + _TIMEOUT
    while screen.handle is None and time.monotonic() < deadline:
        await pilot.pause(0.05)
    handle = screen.handle
    assert handle is not None, "the run screen never started its run"
    while not handle.done.is_set() and time.monotonic() < deadline:
        await pilot.pause(0.05)
    await pilot.pause()
    assert handle.done.is_set()
    return handle


# --- the quick start: Run now (full-auto) -----------------------------------------


def test_quick_start_run_now_writes_the_toml_before_running(monkeypatch, tmp_path):
    """Run now: the shared dialog writes the TOML first (the mocked
    run_pipeline sees it on disk), the chain runs on the resolved config,
    and the run screen ends with the summaries displayed."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    toml_path = out / "pipeline.toml"
    at_run: dict[str, object] = {}

    def fake_run_pipeline(config):
        at_run["toml"] = toml_path.read_text() if toml_path.exists() else None
        at_run["raw_dir"] = str(config.raw_dir)
        return PipelineSummary(
            status="ok",
            difference=DifferenceSummary(wells={}, processed=2, skipped=1, failed=0),
            segmentation=SegmentationSummary(wells=2, processed=2, skipped=0, failed=0),
            analysis=AnalysisSummary(),
        )

    monkeypatch.setattr(runner, "run_pipeline", fake_run_pipeline)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            screen.query_one("#qs-raw-dir", Input).value = str(raw)
            screen.query_one("#qs-out-root", Input).value = str(out)
            await pilot.pause()
            assert await pilot.click("#qs-run-now")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            path_input = dialog.query_one("#toml-path", Input)
            assert path_input.value == str(toml_path)
            await pilot.press("enter")
            run_screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, run_screen)
            status = str(run_screen.query_one("#run-status", Static).render())
            assert "all three stages ok" in status
            summary = str(run_screen.query_one("#run-summary", Static).render())
            assert "difference" in summary
            assert "processed 2 · skipped 1" in summary
            assert "segmentation" in summary
            assert "analysis" in summary
            assert not run_screen.query_one("#run-error", Static).display
            # the run went out with the dialog's resolved config
            assert run_screen.config is handle.config
            assert str(handle.config.raw_dir) == str(raw)
            assert run_screen.toml_path == toml_path

    asyncio.run(scenario())
    # the TOML was written before the run started, and it round-trips
    assert at_run["toml"] is not None
    assert at_run["raw_dir"] == str(raw)
    assert read_config(toml_path).raw_dir == raw


def test_quick_start_write_toml_writes_but_never_runs(monkeypatch, tmp_path):
    """*Write TOML…* writes the config through the shared dialog and
    starts no run."""
    calls = []
    monkeypatch.setattr(runner, "run_pipeline", lambda config: calls.append(config))
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            screen.query_one("#qs-raw-dir", Input).value = str(raw)
            screen.query_one("#qs-out-root", Input).value = str(out)
            await pilot.pause()
            assert await pilot.click("#qs-write")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfigDialog)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            # back on the quick start, nothing running
            assert isinstance(app.screen, QuickStartScreen)
            assert runner.active() is None
            assert (out / "pipeline.toml").exists()

    asyncio.run(scenario())
    assert calls == []


def test_quick_start_actions_are_gated_on_the_draft(monkeypatch, tmp_path):
    """The stage screens' gate: *Write TOML…* and *Run now* start disabled
    with the pipeline's exact ``ConfigError`` on the status line, stay
    disabled while the draft does not resolve, and open once it does; a
    press on a closed gate does nothing."""
    calls = []
    monkeypatch.setattr(runner, "run_pipeline", lambda config: calls.append(config))

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            # closed on the empty draft, with the pipeline's exact message
            assert not screen.gate_open
            for button_id in ("qs-write", "qs-run-now"):
                assert screen.query_one(f"#{button_id}", Button).disabled
            error = screen.query_one("#qs-error", Static)
            assert error.display
            assert (
                str(error.render())
                == "missing required [pipeline] key(s): ['raw_dir', 'out_root']"
            )
            await pilot.click("#qs-run-now")  # disabled: nothing happens
            await pilot.pause()
            assert isinstance(app.screen, QuickStartScreen)  # no dialog
            assert runner.active() is None
            # filling the required fields opens the gate
            screen.query_one("#qs-raw-dir", Input).value = str(tmp_path)
            await pilot.pause()
            assert not screen.gate_open  # out_root still missing
            assert (
                str(error.render())
                == "missing required [pipeline] key(s): ['out_root']"
            )
            screen.query_one("#qs-out-root", Input).value = str(tmp_path / "out")
            await pilot.pause()
            assert screen.gate_open
            for button_id in ("qs-write", "qs-run-now"):
                assert not screen.query_one(f"#{button_id}", Button).disabled
            assert not error.display

    asyncio.run(scenario())
    assert calls == []
    assert list(tmp_path.rglob("*.toml")) == []


def test_quick_start_reports_a_bad_reference(monkeypatch, tmp_path):
    """A reference the shared parser rejects closes the gate with the
    parser's message (the CLI's) and nothing is written or run; the other
    fields keep syncing, and the CLI's LO<nnn> form opens the gate with
    the parsed frame number."""
    calls = []
    monkeypatch.setattr(runner, "run_pipeline", lambda config: calls.append(config))

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            screen.query_one("#qs-raw-dir", Input).value = str(tmp_path)
            screen.query_one("#qs-out-root", Input).value = str(tmp_path)
            await pilot.pause()
            assert screen.gate_open
            screen.query_one("#qs-reference", Input).value = "abc"
            await pilot.pause()
            assert not screen.gate_open
            for button_id in ("qs-write", "qs-run-now"):
                assert screen.query_one(f"#{button_id}", Button).disabled
            error = screen.query_one("#qs-error", Static)
            assert error.display
            assert str(error.render()) == (
                "invalid reference 'abc': expected 'first', a 1-based frame "
                "number, or LO<nnn>"
            )
            # the other fields kept syncing; the reference key did not
            assert app.draft.document() == {
                "pipeline": {"raw_dir": str(tmp_path), "out_root": str(tmp_path)}
            }
            await pilot.click("#qs-run-now")  # disabled: nothing happens
            await pilot.pause()
            assert isinstance(app.screen, QuickStartScreen)  # no dialog
            assert runner.active() is None
            # the CLI's LO<nnn> form parses like in the CLI (frame 5)
            screen.query_one("#qs-reference", Input).value = "LO005"
            await pilot.pause()
            assert screen.gate_open
            assert app.draft.get("pipeline", "reference") == 5

    asyncio.run(scenario())
    assert calls == []
    assert list(tmp_path.rglob("*.toml")) == []


def test_quick_start_reference_and_wells_reach_the_written_toml(monkeypatch, tmp_path):
    """The optional fields parse (a numeric reference, a comma-separated
    wells list) and land in the written TOML; a reopened screen prefills
    from the draft."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            screen.query_one("#qs-raw-dir", Input).value = str(raw)
            screen.query_one("#qs-out-root", Input).value = str(out)
            screen.query_one("#qs-reference", Input).value = "2"
            screen.query_one("#qs-wells", Input).value = "-A001, -A002"
            await pilot.pause()
            assert await pilot.click("#qs-write")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfigDialog)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            # a reopened quick start prefills from the shared draft
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            reopened = app.screen
            assert isinstance(reopened, QuickStartScreen)
            assert reopened.query_one("#qs-raw-dir", Input).value == str(raw)
            assert reopened.query_one("#qs-reference", Input).value == "2"
            assert reopened.query_one("#qs-wells", Input).value == "-A001, -A002"

    asyncio.run(scenario())
    config = read_config(out / "pipeline.toml")
    assert config.reference == 2
    assert config.wells == ["-A001", "-A002"]


def test_quick_start_clearing_a_field_is_back_to_the_default(tmp_path):
    """Emptying a field clears its key (including a wells input of only
    separators): the next write carries the defaults again."""
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            screen.query_one("#qs-raw-dir", Input).value = str(tmp_path)
            screen.query_one("#qs-out-root", Input).value = str(out)
            screen.query_one("#qs-reference", Input).value = "2"
            await pilot.pause()
            assert await pilot.click("#qs-write")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            # empty the optional fields (wells: separators only) and rewrite
            reopened = app.screen
            assert isinstance(reopened, QuickStartScreen)
            assert reopened.query_one("#qs-reference", Input).value == "2"
            reopened.query_one("#qs-reference", Input).value = ""
            reopened.query_one("#qs-wells", Input).value = " , , "
            await pilot.pause()
            assert await pilot.click("#qs-write")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

    asyncio.run(scenario())
    config = read_config(out / "pipeline.toml")
    assert config.reference == "first"
    assert config.wells is None


# --- the run screen: outcome display -----------------------------------------------


def test_a_failed_pipeline_shows_the_error_panel_and_the_app_survives(
    monkeypatch, tmp_path
):
    """A failed stage summary: the error panel names the stage and the
    failed wells (from the summary objects); the app stays alive."""
    summary = PipelineSummary(
        status="failed",
        failed_stage="difference",
        difference=DifferenceSummary(
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
        ),
    )
    monkeypatch.setattr(runner, "run_pipeline", lambda config: summary)
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(RunScreen("pipeline", _config(raw, out)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            await _await_done(pilot, screen)
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            text = str(panel.render())
            assert "stopped at the difference stage" in text
            assert "1 well(s) failed: -A001 (no frames)" in text
            assert app.screen is screen  # the app survived
            status = str(screen.query_one("#run-status", Static).render())
            assert "stopped at the difference stage" in status
            # the summary block: the failed stage's totals, the rest not run
            summary_text = str(screen.query_one("#run-summary", Static).render())
            assert "difference" in summary_text
            assert summary_text.count("not reached") == 2

    asyncio.run(scenario())


def test_a_raising_stage_shows_the_error_panel_and_the_app_survives(
    monkeypatch, tmp_path
):
    """An exception escaping the run: the error panel shows it and the
    full-traceback pointer; no summary block appears; the app survives."""

    def boom(config):
        raise RuntimeError("the analysis stage exploded")

    monkeypatch.setattr(runner, "run_pipeline", boom)
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(RunScreen("pipeline", _config(raw, out)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            await _await_done(pilot, screen)
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            text = str(panel.render())
            assert "RuntimeError: the analysis stage exploded" in text
            assert "full traceback" in text
            assert not screen.query_one("#run-summary", Static).display
            status = str(screen.query_one("#run-status", Static).render())
            assert "pipeline failed: RuntimeError" in status
            assert app.screen is screen  # the app survived

    asyncio.run(scenario())


# --- the run screen: live log and current stage ------------------------------------


def test_log_records_reach_the_pane_while_the_run_is_active(monkeypatch, tmp_path):
    """Records stream into the pane while the run is active; the status
    line shows what is running and the no-cancellation note; the final
    status replaces them when the run is done."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    release = threading.Event()

    def fake(options):
        log.info("tui test: the difference stage is running")
        release.wait(_TIMEOUT)
        return _diff_summary()

    monkeypatch.setattr(runner, "run_difference", fake)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(RunScreen("difference", _config(raw, out)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            needle = "tui test: the difference stage is running"
            deadline = time.monotonic() + _TIMEOUT
            while needle not in _log_text(screen) and time.monotonic() < deadline:
                await pilot.pause(0.05)
            assert needle in _log_text(screen)
            handle = screen.handle
            assert handle is not None and not handle.done.is_set()
            status = str(screen.query_one("#run-status", Static).render())
            assert "running difference" in status
            assert "no cancellation" in status
            release.set()
            await _await_done(pilot, screen)
            assert "finished — ok" in str(
                screen.query_one("#run-status", Static).render()
            )

    asyncio.run(scenario())


def test_the_status_line_names_the_pipeline_stage_running_now(monkeypatch, tmp_path):
    """During a pipeline run the status line follows the chain via the
    structured stage-start markers (never parsed log text); the final
    outcome replaces it."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    release = threading.Event()

    def fake(config):
        log.info("pipeline: %s → %s", "[1/3] difference", out / "difference")
        log.info(
            "pipeline: %s → %s",
            "[2/3] segment (input=diff)",
            out / "seg_growth_diff",
        )
        release.wait(_TIMEOUT)
        return PipelineSummary(
            status="ok",
            difference=_diff_summary(),
            segmentation=SegmentationSummary(wells=1, processed=1, skipped=0, failed=0),
            analysis=AnalysisSummary(),
        )

    monkeypatch.setattr(runner, "run_pipeline", fake)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(RunScreen("pipeline", _config(raw, out)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            deadline = time.monotonic() + _TIMEOUT
            status_widget = screen.query_one("#run-status", Static)
            while (
                "now: [2/3] segment (input=diff)" not in str(status_widget.render())
                and time.monotonic() < deadline
            ):
                await pilot.pause(0.05)
            assert "now: [2/3] segment (input=diff)" in str(status_widget.render())
            release.set()
            await _await_done(pilot, screen)
            status = str(status_widget.render())
            assert "all three stages ok" in status
            assert "now:" not in status  # the hint is gone once finished
            summary = str(screen.query_one("#run-summary", Static).render())
            for name in ("difference", "segmentation", "analysis"):
                assert name in summary

    asyncio.run(scenario())


# --- the direct-start entry points --------------------------------------------------


def test_single_stage_run_and_restart_resumes_via_done_markers(tmp_path):
    """One real in-process difference run through ``open_stage_run`` (the
    un-mocked path, on a mini plate), then a restart: the done marker
    makes the re-run a resume (processed 0, skipped 1)."""
    raw = tmp_path / "raw"
    build_mini_plate(
        raw,
        layout="flat",
        wells=("-A001",),
        n_frames=2,
        size=256,
        noise_sigma=0,
        rings={"-A001": None},
        disks={
            "-A001": [
                DiskSpec(
                    "growing",
                    cy=128.0,
                    cx=128.0,
                    r0=20.0,
                    delta=1024.0,
                    first_frame=2,
                )
            ]
        },
    )
    out = tmp_path / "out"
    draft = _draft(raw, out)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            assert open_stage_run(app, draft, "difference") is None
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            await pilot.press("enter")
            screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, screen)
            summary = handle.summary
            assert isinstance(summary, DifferenceSummary)
            assert (summary.processed, summary.skipped, summary.failed) == (
                1,
                0,
                0,
            )
            # the wiring run_pipeline would use: the resolved stage options
            assert handle.config.difference.raw_dir == raw
            assert handle.config.difference.out_dir == out / "difference"
            # the real run's records reached the pane, the TOML round-trips
            assert "-A001" in _log_text(screen)
            assert "finished — ok" in str(
                screen.query_one("#run-status", Static).render()
            )
            assert read_config(out / "pipeline.toml") == handle.config
            await pilot.press("escape")
            await pilot.pause()
            # restart: the same draft re-enters the stage as a resume
            assert open_stage_run(app, draft, "difference") is None
            await pilot.pause()
            await pilot.press("enter")
            screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, screen)
            summary = handle.summary
            assert isinstance(summary, DifferenceSummary)
            assert (summary.processed, summary.skipped) == (0, 1)
            assert "processed 0 · skipped 1" in str(
                screen.query_one("#run-summary", Static).render()
            )

    asyncio.run(scenario())


def test_segmentation_alone_needs_the_difference_tree(monkeypatch, tmp_path):
    """Running segmentation from a draft with no [difference] section
    requires the difference tree to exist — reported before anything is
    written or started; once it exists, the run goes out with the chain
    wiring."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    draft = _draft(raw, out)
    calls = []

    def fake(options):
        calls.append(options)
        return SegmentationSummary(wells=1, processed=1, skipped=0, failed=0)

    monkeypatch.setattr(runner, "run_segmentation", fake)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            message = open_stage_run(app, draft, "segmentation")
            assert message is not None
            assert "difference tree" in message
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)  # nothing pushed
            assert calls == []
            assert list(tmp_path.rglob("*.toml")) == []
            # the tree exists now: the run starts, wired like the chain's
            (out / "difference").mkdir(parents=True)
            assert open_stage_run(app, draft, "segmentation") is None
            await pilot.pause()
            assert isinstance(app.screen, SaveConfigDialog)
            await pilot.press("enter")
            screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, screen)
            (options,) = calls
            assert options.input == "diff"
            assert options.diff_dir == out / "difference"
            assert options.out_dir == out / "seg_growth_diff"
            assert isinstance(handle.summary, SegmentationSummary)
            assert "processed 1" in str(
                screen.query_one("#run-summary", Static).render()
            )

    asyncio.run(scenario())


def test_a_second_run_cannot_start_while_one_is_active(monkeypatch, tmp_path):
    """One run at a time, reported politely at both layers: the entry
    point declines (nothing pushed, nothing written) and a run screen
    pushed anyway shows the problem instead of starting."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    config = _config(raw, out)
    release = threading.Event()

    def slow(options):
        release.wait(_TIMEOUT)
        return _diff_summary()

    monkeypatch.setattr(runner, "run_difference", slow)
    first = runner.start("difference", config, on_done=lambda handle: None)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            message = open_pipeline_run(app, _draft(raw, out))
            assert message is not None
            assert "already active" in message
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)  # nothing pushed
            app.push_screen(RunScreen("pipeline", config))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            assert screen.handle is None  # the guard started nothing
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            assert "already active" in str(panel.render())
            assert not (out / "pipeline.toml").exists()

    asyncio.run(scenario())
    release.set()
    assert first.done.wait(_TIMEOUT)


def test_open_stage_run_rejects_unknown_stages():
    """An unknown stage name is a programming error, raised up front."""
    with pytest.raises(ValueError, match="unknown stage 'train'"):
        open_stage_run(ColonyTuiApp(), ConfigDraft(), "train")


def test_the_run_screen_rejects_unknown_kinds(tmp_path):
    """An unknown run kind is a programming error, raised up front."""
    with pytest.raises(ValueError, match="unknown run kind 'train'"):
        RunScreen("train", _config(tmp_path, tmp_path))
