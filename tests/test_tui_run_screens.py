"""Tests for the run screens and the direct-start flows.

Seams under test:

- the quick start's *Run now* (the whole-chain pipeline flow): the fields
  sync into the pipeline draft, the shared save dialog writes the
  ``pipeline.toml`` *before* the run starts (recorded at the mocked
  ``run_pipeline``'s call time), and the run screen ends with the chain's
  summaries displayed;
- the quick start's *Write TOML…*: writes the config, never runs;
- the quick start's action gate (the stage screens' mechanism): the
  actions stay disabled until the fields sync and the draft resolves —
  the pipeline's exact ``ConfigError`` (or the reference parser's) on the
  status line — and a press on a closed gate writes and runs nothing;
- the run screen: log records reach the pane while the run is active, the
  current-stage line comes from the pipeline's structured stage-start
  markers (never parsed log text), the final summary block comes from the
  summary objects, and a failed summary or an exception shows the error
  panel — the app survives every outcome;
- the direct-start entry points: ``open_stage_run``'s two checks (the
  runner's one-at-a-time rule, the stage draft's own ``resolve``), and
  the absence of every other gate — there is NO cross-run pre-flight: a
  stage run starts on its independent config as given, and an error the
  stage itself raises on its inputs surfaces in the run screen's error
  panel (the config's inputs are the config's own content);
- one real in-process difference run on a conftest mini plate (the
  un-mocked path), run twice: the re-run resumes via the done marker
  (processed 0, skipped 1).

A stage kind's ``RunScreen`` carries the stage's own resolved Options;
the ``pipeline`` kind the chain's PipelineConfig — the tests pass each
kind what it runs on. Async ``Pilot`` scenarios run through
``asyncio.run`` (no async test plugin in the dev group; the suite stays
dependency-free). Assertions on widget state run inside the headless
session — the screen stack is gone once it ends.
"""

import asyncio
import threading
import time
import tomllib
from pathlib import Path

import pytest
from conftest import DiskSpec, build_mini_plate
from textual.widgets import Button, Input, Static

from colony_tracking import (
    AnalysisSummary,
    DifferenceOptions,
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

#: The gate wording each stage draft shows while it is empty (the
#: ``open_stage_run`` draft check reports these verbatim).
_MISSING_REQUIRED = {
    "difference": "missing required key(s): ['raw_dir', 'out_dir']",
    "segmentation": "missing required key(s): ['out_dir']",
    "analysis": "missing required key(s): ['seg_dirs', 'raw_dir', 'out_root']",
}


def _config(raw: Path, out: Path) -> PipelineConfig:
    """A minimal resolved chain config: the run level only, derived stages."""
    return PipelineConfig.from_dict(
        {
            "pipeline": {
                "raw_dir": str(raw),
                "out_root": str(out),
            },
        }
    )


def _diff_options(raw: Path, out: Path) -> DifferenceOptions:
    """A stage run's own resolved options: fully explicit, nothing derived."""
    return DifferenceOptions(raw_dir=raw, out_dir=out)


def _draft(raw: Path, out: Path) -> ConfigDraft:
    """A pipeline draft with the run level filled."""
    draft = ConfigDraft()
    draft.set("pipeline", "raw_dir", raw)
    draft.set("pipeline", "out_root", out)
    return draft


def _stage_draft(run_type: str, **values: str) -> ConfigDraft:
    """A stage draft of ``run_type`` with its own keys filled (the plain
    strings a form would store)."""
    draft = ConfigDraft(run_type)
    for key, value in values.items():
        draft.set(run_type, key, value)
    return draft


def _mini_plate(raw: Path) -> None:
    """The one-well mini plate the real difference stage runs on (tiny,
    deterministic, no ring)."""
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


def _diff_summary() -> DifferenceSummary:
    return DifferenceSummary(wells={}, processed=1, skipped=0, failed=0)


def _log_text(screen: RunScreen) -> str:
    """The run screen's log pane text so far."""
    from textual.widgets import RichLog

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


# --- the quick start: Run now (the chain's full-auto flow) -------------------------


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


def test_arrow_keys_move_focus_between_quick_start_fields():
    """Down/up step through the quick start's run-level fields (wrapping)
    — the same field navigation the form screens get; tab keeps working."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickStartScreen)
            raw_dir = screen.query_one("#qs-raw-dir", Input)
            raw_dir.focus()
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is screen.query_one("#qs-out-root", Input)
            await pilot.press("up")
            await pilot.pause()
            assert app.focused is raw_dir
            await pilot.press("tab")
            await pilot.pause()
            assert app.focused is screen.query_one("#qs-out-root", Input)

    asyncio.run(scenario())


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
            assert app.drafts["pipeline"].document() == {
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
            assert app.drafts["pipeline"].get("pipeline", "reference") == 5

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


def test_a_failed_stage_panel_reads_like_the_runners_logged_lines(
    monkeypatch, tmp_path
):
    """Behavior parity: the error panel's failure line is the runner's
    logged ERROR line, verbatim (the shared
    :func:`~colony_tracking.tui.runner.failed_summary_lines`). A stage
    run carries the stage's own Options."""
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
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(RunScreen("difference", _diff_options(raw, out)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            await _await_done(pilot, screen)
            line = "difference: 1 well(s) failed: -A001 (no frames)"
            assert line in _log_text(screen)  # the runner logged it
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            assert line in str(panel.render())  # the panel shows the same
            assert app.screen is screen  # the app survived

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
            app.push_screen(RunScreen("difference", _diff_options(raw, out)))
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
    un-mocked path, on a mini plate, on the difference draft's own
    options), then a restart: the done marker makes the re-run a resume
    (processed 0, skipped 1)."""
    raw = tmp_path / "raw"
    _mini_plate(raw)
    out_dir = tmp_path / "diff_out"
    draft = ConfigDraft("difference")
    draft.set("difference", "raw_dir", raw)
    draft.set("difference", "out_dir", out_dir)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            assert open_stage_run(app, draft, "difference") is None
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(
                tmp_path / "difference.toml"  # <out_dir parent>/difference.toml
            )
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
            # the stage's own fully-explicit options, exactly as given
            options = handle.config
            assert isinstance(options, DifferenceOptions)
            assert options.raw_dir == raw
            assert options.out_dir == out_dir
            # the real run's records reached the pane, the TOML round-trips
            assert "-A001" in _log_text(screen)
            assert "finished — ok" in str(
                screen.query_one("#run-status", Static).render()
            )
            with open(tmp_path / "difference.toml", "rb") as file_handle:
                assert DifferenceOptions.from_dict(tomllib.load(file_handle)) == options
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


def test_a_stage_run_starts_on_its_inputs_as_given(monkeypatch, tmp_path):
    """There is no cross-run pre-flight: a segmentation run with an
    explicit diff_dir starts whether that directory exists or not — the
    config's inputs are taken as given. When the stage itself errors on a
    missing tree, the run screen's error panel shows it and the app
    survives; with the tree present the same draft runs to its summary.
    The stage function received exactly the typed options, both times."""
    raw = tmp_path / "raw"
    raw.mkdir()
    diff_dir = tmp_path / "difference"  # does not exist (yet)
    out_dir = tmp_path / "seg_out"
    draft = ConfigDraft("segmentation")
    draft.set("segmentation", "input", "diff")
    draft.set("segmentation", "diff_dir", diff_dir)
    draft.set("segmentation", "out_dir", out_dir)
    calls = []
    outcomes: list[Exception | object] = [FileNotFoundError(
        "no difference tree at " + str(diff_dir)
    )]

    def fake(options):
        calls.append(options)
        outcome = outcomes[0]
        outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(runner, "run_segmentation", fake)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            # missing tree: the run still starts (nothing pre-flights it)
            assert open_stage_run(app, draft, "segmentation") is None
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(
                tmp_path / "segmentation.toml"  # <out_dir parent>
            )
            await pilot.press("enter")
            screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, screen)
            (options,) = calls
            assert options.input == "diff"
            assert options.diff_dir == diff_dir  # taken as given
            assert options.out_dir == out_dir
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            text = str(panel.render())
            assert "the segmentation run raised FileNotFoundError" in text
            assert str(diff_dir) in text
            assert app.screen is screen  # the app survived
            # the tree exists now: the same draft runs to its summary
            diff_dir.mkdir()
            calls.clear()
            outcomes.append(
                SegmentationSummary(wells=1, processed=1, skipped=0, failed=0)
            )
            assert open_stage_run(app, draft, "segmentation") is None
            await pilot.pause()
            await pilot.press("enter")
            screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, screen)
            (options,) = calls
            assert options.diff_dir == diff_dir
            assert not screen.query_one("#run-error", Static).display
            assert "processed 1" in str(
                screen.query_one("#run-summary", Static).render()
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_open_stage_run_reports_the_stages_own_invalid_draft(stage):
    """The only config check ``open_stage_run`` makes is the draft's own
    ``resolve``: an empty stage draft reports that run type's exact
    ConfigError (its required keys, declaration order) — reported in the
    UI too, and nothing is pushed or written."""
    raw = tmp_paths()
    out = tmp_paths()

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            message = open_stage_run(app, ConfigDraft(stage), stage)
            assert message == _MISSING_REQUIRED[stage]
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)  # nothing pushed
            assert runner.active() is None

    asyncio.run(scenario())
    assert list(out.rglob("*.toml")) == []


def tmp_paths() -> Path:
    """A fresh throwaway Path (the wording tests write nothing anywhere)."""
    import tempfile

    return Path(tempfile.mkdtemp())


def test_open_stage_run_declines_while_a_run_is_active(monkeypatch, tmp_path):
    """The runner's one-at-a-time rule at the stage entry point: with a
    difference run active, ``open_stage_run`` for segmentation declines —
    the busy message, reported in the UI, nothing pushed."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    release = threading.Event()

    def slow(options):
        release.wait(_TIMEOUT)
        return _diff_summary()

    monkeypatch.setattr(runner, "run_difference", slow)
    first = runner.start("difference", _diff_options(raw, out), on_done=lambda h: None)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            message = open_stage_run(
                app, _stage_draft("segmentation", out_dir=str(out / "seg")), "segmentation"
            )
            assert message is not None
            assert "already active" in message
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)  # nothing pushed
            assert runner.active() is not None
            assert isinstance(runner.active().config, DifferenceOptions)

    asyncio.run(scenario())
    release.set()
    assert first.done.wait(_TIMEOUT)


def test_a_second_run_cannot_start_while_one_is_active(monkeypatch, tmp_path):
    """One run at a time, reported politely at both layers: the pipeline
    entry point declines (nothing pushed, nothing written) and a run
    screen pushed anyway shows the problem instead of starting."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    config = _config(raw, out)
    release = threading.Event()

    def slow(options):
        release.wait(_TIMEOUT)
        return _diff_summary()

    monkeypatch.setattr(runner, "run_difference", slow)
    first = runner.start("difference", _diff_options(raw, out), on_done=lambda h: None)

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
