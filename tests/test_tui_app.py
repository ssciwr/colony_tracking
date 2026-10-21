"""End-to-end: the wired app (ticket 04).

The registry-driven flows, launched as the real app would be: the home
menu opens the quick start and the three stage screens (through the app's
``SCREENS`` registry — the tests never push those screens directly), the
quick start's *Run now* runs the real chain in-process, and the stage
screens' run actions reach the run screen through the shipped default run
flow.

The headline scenario runs the real, unmocked difference stage on a
conftest mini plate; only the GPU-bound later stages
(``pipeline.run_segmentation`` / ``pipeline.run_analysis`` — the functions
``run_pipeline`` itself calls) are fast fakes, so the chain completes
hermetically. The TOML written by the shared save dialog is proven to
predate the run (recorded when the run reaches its stages) and to load
via ``read_config`` unchanged.

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free). Assertions on
widget state run inside the headless session — the screen stack is gone
once it ends.
"""

import asyncio
import time
from pathlib import Path

from conftest import DiskSpec, build_mini_plate
from textual.widgets import Input, OptionList, RichLog, Static

from colony_tracking import (
    AnalysisSummary,
    DifferenceSummary,
    SegmentationSummary,
)
from colony_tracking import pipeline as pipeline_module
from colony_tracking.pipeline import PipelineConfig, PipelineSummary, read_config
from colony_tracking.tui import runner
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.screens._save import SaveConfigDialog
from colony_tracking.tui.screens.config import ConfigScreen
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.quick_start import QuickStartScreen
from colony_tracking.tui.screens.run import RunScreen
from colony_tracking.tui.screens.stage import StageScreen

#: How long a test waits for navigation or a (small, fast) run to finish.
_TIMEOUT = 10.0

#: The home menu's row order (the ENTRIES tuple in screens/home.py).
_QUICK_START, _DIFFERENCE, _SEGMENTATION, _ANALYSIS, _CONFIG = range(5)


def _mini_plate(raw: Path) -> None:
    """The one-well mini plate the real difference stage runs on (the
    recipe ticket 03's un-mocked test uses: tiny, deterministic, no ring)."""
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


def _log_text(screen: RunScreen) -> str:
    """The run screen's log pane text so far."""
    rich_log = screen.query_one("#run-log", RichLog)
    return "\n".join(strip.text for strip in rich_log.lines)


async def _open_from_home(pilot, app: ColonyTuiApp, index: int, screen_type):
    """Select one home-menu entry through the real menu (focus, highlight,
    enter) and return the screen once it is on top."""
    menu = app.screen.query_one("#home-menu", OptionList)
    menu.focus()
    menu.highlighted = index
    await pilot.pause()
    await pilot.press("enter")
    deadline = time.monotonic() + _TIMEOUT
    while not isinstance(app.screen, screen_type) and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert isinstance(app.screen, screen_type), (
        f"the home menu did not open the {screen_type.__name__} screen"
    )
    return app.screen


async def _await_run_screen(pilot, app: ColonyTuiApp) -> RunScreen:
    """The app's current screen once it is a run screen whose run has
    started (the dialog's write pushes it; the run starts on mount)."""
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
    handle = screen.handle
    while (
        handle is not None and not handle.done.is_set() and time.monotonic() < deadline
    ):
        await pilot.pause(0.05)
    assert handle is not None and handle.done.is_set(), "the run never finished"
    await pilot.pause()
    return handle


# --- the headline flow: home → quick start → the real chain -------------------------


def test_quick_start_runs_the_real_chain_end_to_end(monkeypatch, tmp_path):
    """Launch the app, navigate home → quick start through the menu, fill
    raw_dir/out_root on a conftest mini plate, *Run now*, accept the
    suggested TOML path: the real difference stage runs in-process inside
    the real ``run_pipeline`` (the GPU-bound later stages are fast fakes),
    the run screen shows the status and the per-stage summaries, and the
    TOML was written before the run and loads via ``read_config``."""
    raw = tmp_path / "raw"
    _mini_plate(raw)
    out = tmp_path / "out"
    toml_path = out / "pipeline.toml"

    seg_calls: list = []
    ana_calls: list = []

    def fake_segmentation(options):
        seg_calls.append(options)
        return SegmentationSummary(wells=1, processed=1, skipped=0, failed=0)

    def fake_analysis(options):
        ana_calls.append(options)
        return AnalysisSummary()

    monkeypatch.setattr(pipeline_module, "run_segmentation", fake_segmentation)
    monkeypatch.setattr(pipeline_module, "run_analysis", fake_analysis)

    real_run_pipeline = pipeline_module.run_pipeline
    at_run: dict[str, object] = {}

    def recording_run_pipeline(config: PipelineConfig):
        # the runner resolves this global at call time, so wrapping it
        # keeps the real chain while proving the write predates the run
        at_run["toml"] = toml_path.read_text() if toml_path.exists() else None
        return real_run_pipeline(config)

    monkeypatch.setattr(runner, "run_pipeline", recording_run_pipeline)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            quick_start = await _open_from_home(
                pilot, app, _QUICK_START, QuickStartScreen
            )
            quick_start.query_one("#qs-raw-dir", Input).value = str(raw)
            quick_start.query_one("#qs-out-root", Input).value = str(out)
            await pilot.pause()
            assert await pilot.click("#qs-run-now")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(toml_path)
            await pilot.press("enter")
            run_screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, run_screen)

            # the chain's outcome, driven by the real summary objects
            summary = handle.summary
            assert isinstance(summary, PipelineSummary)
            assert summary.status == "ok"
            status = str(run_screen.query_one("#run-status", Static).render())
            assert "all three stages ok" in status
            block = str(run_screen.query_one("#run-summary", Static).render())
            assert "difference" in block and "processed 1 · skipped 0" in block
            assert "segmentation" in block and "analysis" in block
            assert not run_screen.query_one("#run-error", Static).display
            # the real difference stage produced its tree (unmocked)
            assert handle.config.difference.raw_dir == raw
            assert handle.config.difference.out_dir == out / "difference"
            assert (out / "difference" / "-A001").is_dir()
            # the fakes ran on the wiring run_pipeline gives the stages
            (seg_options,) = seg_calls
            assert seg_options.input == "diff"
            assert seg_options.diff_dir == out / "difference"
            assert seg_options.out_dir == out / "seg_growth_diff"
            (ana_options,) = ana_calls
            assert ana_options.raw_dir == raw
            assert ana_options.seg_dirs == [out / "seg_growth_diff"]
            assert ana_options.out_root == out / "analysis"
            # the real chain's records streamed into the pane
            assert "pipeline: reference first" in _log_text(run_screen)

            # escape back through the stack; the draft persists on the app
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, QuickStartScreen)
            await pilot.press("escape")
            await pilot.pause()
            config_screen = await _open_from_home(pilot, app, _CONFIG, ConfigScreen)
            preview = str(config_screen.query_one("#toml-preview", Static).render())
            assert f'raw_dir = "{raw}"' in preview
            assert f'out_root = "{out}"' in preview

    asyncio.run(scenario())
    # the TOML was written before the run started and round-trips exactly
    assert at_run["toml"] is not None
    config = read_config(toml_path)
    assert config.raw_dir == raw
    assert config.out_root == out
    assert config.reference == "first"


# --- the registry-driven stage screens ----------------------------------------------


def test_stage_screens_open_from_home_gated_with_derived_values(tmp_path):
    """home → each stage screen through the menu: the gate is closed on
    the empty draft with the pipeline's exact ``ConfigError``, the owned
    fields show their derived values (— while the run level is empty),
    and *Back* returns home. After the run level is filled on the config
    screen, the gate is open and the derived values and greyed defaults
    name the chain's trees."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    missing = "missing required [pipeline] key(s): ['raw_dir', 'out_root']"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:

            def owned_text() -> str:
                return str(app.screen.query_one("#owned-values", Static).render())

            # every stage opens gated, shows its owned fields, and backs out
            for index, stage, labels in (
                (_DIFFERENCE, "difference", ("raw-dir = —", "reference = first")),
                (_SEGMENTATION, "segmentation", ("input = diff", "diff-dir = —")),
                (_ANALYSIS, "analysis", ("raw-dir = —",)),
            ):
                screen = await _open_from_home(pilot, app, index, StageScreen)
                assert screen.stage == stage
                assert not screen.gate_open
                assert (
                    str(app.screen.query_one("#resolve-status", Static).render())
                    == missing
                )
                for button_id in ("run-stage", "write-toml", "run-pipeline"):
                    assert app.screen.query_one(f"#{button_id}").disabled
                text = owned_text()
                for label in labels:
                    assert label in text
                await pilot.click("#back")
                await pilot.pause()
                assert isinstance(app.screen, HomeScreen)

            # the run level is filled on the config screen (reference 2:
            # the derived tree names carry the LO002 suffix)
            config_screen = await _open_from_home(pilot, app, _CONFIG, ConfigScreen)
            config_screen.query_one("#pipeline-raw-dir", Input).value = str(raw)
            config_screen.query_one("#pipeline-out-root", Input).value = str(out)
            config_screen.query_one("#pipeline-reference", Input).value = "2"
            await pilot.pause()
            assert config_screen.gate_open
            await pilot.click("#back")
            await pilot.pause()

            # the stages now show the derived values, the gate is open,
            # and the free fields grey their derived defaults
            difference = await _open_from_home(pilot, app, _DIFFERENCE, StageScreen)
            assert difference.gate_open
            text = owned_text()
            assert f"raw-dir = {raw}" in text
            assert "reference = 2" in text
            assert (
                difference.form.query_one("#difference-out-dir", Input).placeholder
                == str(out / "difference_LO002")
            )
            await pilot.click("#back")
            await pilot.pause()

            segmentation = await _open_from_home(pilot, app, _SEGMENTATION, StageScreen)
            assert segmentation.gate_open
            text = owned_text()
            assert "input = diff" in text
            assert f"diff-dir = {out / 'difference_LO002'}" in text
            assert (
                segmentation.form.query_one("#segmentation-out-dir", Input).placeholder
                == str(out / "seg_growth_diff_LO002")
            )
            await pilot.click("#back")
            await pilot.pause()

            analysis = await _open_from_home(pilot, app, _ANALYSIS, StageScreen)
            assert analysis.gate_open
            assert f"raw-dir = {raw}" in owned_text()
            assert (
                analysis.form.query_one("#analysis-seg-dir", Input).placeholder
                == str(out / "seg_growth_diff_LO002")
            )
            assert (
                analysis.form.query_one("#analysis-out-root", Input).placeholder
                == str(out / "analysis")
            )
            await pilot.click("#back")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_a_stage_screen_run_action_runs_through_the_wired_flow(
    monkeypatch, tmp_path
):
    """*Run this stage* on a registry-opened stage screen (no explicit
    ``start_run`` — the shipped default): the shared dialog writes the
    TOML first, the run screen runs the difference stage with the wiring
    ``run_pipeline`` would give it, and escape returns to the same stage
    screen with the app alive."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    toml_path = out / "pipeline.toml"
    calls: list = []
    at_call: dict[str, object] = {}

    def fake_difference(options):
        calls.append(options)
        at_call["toml"] = toml_path.read_text() if toml_path.exists() else None
        return DifferenceSummary(wells={}, processed=1, skipped=0, failed=0)

    monkeypatch.setattr(runner, "run_difference", fake_difference)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.draft.set("pipeline", "raw_dir", str(raw))
            app.draft.set("pipeline", "out_root", str(out))
            difference = await _open_from_home(pilot, app, _DIFFERENCE, StageScreen)
            assert difference.gate_open
            await pilot.click("#run-stage")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfigDialog)
            await pilot.press("enter")
            run_screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, run_screen)
            (options,) = calls
            # the single-stage wiring: the resolved stage options, owned
            # fields forced exactly as the chain would
            assert options.raw_dir == raw
            assert options.out_dir == out / "difference"
            assert options.reference == "first"
            assert "processed 1 · skipped 0" in str(
                run_screen.query_one("#run-summary", Static).render()
            )
            assert "finished — ok" in str(
                run_screen.query_one("#run-status", Static).render()
            )
            # the written TOML is exactly the draft's resolution
            assert read_config(toml_path) == handle.config
            await pilot.press("escape")
            await pilot.pause()
            # the registry-cached stage screen again, draft intact
            screen = app.screen
            assert isinstance(screen, StageScreen)
            assert screen.stage == "difference"
            assert screen.gate_open

    asyncio.run(scenario())
    assert at_call["toml"] is not None  # written before the run started
