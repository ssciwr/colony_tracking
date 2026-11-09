"""The viewer screen: pick wells from the sources and open napari.

Seams under test, all headless through textual's ``Pilot``:

- the well list the screen offers: exactly the wells the pipeline
  draft's resolved sources hold (:func:`colony_tracking.viewer.available_wells`
  over the draft's ``PipelineConfig``) — a well present in none of the
  four trees, and a well directory without frames, are never offered;
- the multi-selection: two chosen wells reach the launcher together;
- the exact argument vector handed to the launcher: one comma-separated
  ``--wells=`` value (dash-leading names stay inside the flag) plus the
  draft's four resolved paths, run as ``python -m colony_tracking view``;
- the failure paths: a child exiting non-zero (napari missing, no
  display) and a spawn raising — each a log-pane line and a status line,
  with the TUI still alive and navigable;
- the app never imports napari: the TUI launch path is a subprocess.

``launch.SPAWN`` is stubbed everywhere, so no viewer process is ever
started and no display is needed. Async ``Pilot`` scenarios run through
``asyncio.run`` (no async test plugin in the dev group).
"""

import asyncio
import sys
import time
from pathlib import Path

from conftest import build_mini_plate
from textual.widgets import Button, Input, OptionList, RichLog, SelectionList, Static

from colony_tracking import cli
from colony_tracking._options import ViewOptions
from colony_tracking.pipeline import PipelineConfig
from colony_tracking.tui import launch
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.screens import viewer as viewer_module
from colony_tracking.tui.screens._save import LoadConfigDialog
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.viewer import ViewerScreen

#: How long a test waits for navigation or a (stubbed) launch to finish.
_TIMEOUT = 10.0

#: The home menu's row order (the ENTRIES tuple in screens/home.py):
#: ``View wells in napari`` sits after the config entry, before Train.
_VIEW = 5

#: The raw plate's wells and the difference tree's extra well.
_RAW_WELLS = ("-A001", "-B004")
_DIFF_WELL = "-C006"
_EMPTY_WELL = "-C007"


def _plate(root: Path, wells, layout: str) -> None:
    """A tiny synthetic tree (one frame, no ring, no disks): discovery is
    all these tests need."""
    build_mini_plate(
        root,
        layout=layout,
        wells=wells,
        n_frames=1,
        size=64,
        noise_sigma=0,
        rings={well: None for well in wells},
        disks={},
    )


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    """A raw plate and a difference tree under the chain's derived out
    root; the difference tree also holds a well directory without frames
    (``-C007``), which must never be offered."""
    raw = tmp_path / "raw"
    _plate(raw, _RAW_WELLS, "flat")
    out = tmp_path / "out"
    difference = out / "difference"
    _plate(difference, (_DIFF_WELL,), "per_well")
    (difference / _EMPTY_WELL).mkdir()
    return raw, out


def _fill_draft(app: ColonyTuiApp, raw: Path, out: Path) -> None:
    app.drafts["pipeline"].set("pipeline", "raw_dir", raw)
    app.drafts["pipeline"].set("pipeline", "out_root", out)


class _StubProcess:
    """A :class:`subprocess.Popen` stand-in: ``communicate`` returns the
    captured streams and the child reports its exit status."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def communicate(self):
        return self._stdout, self._stderr


def _stub_spawn(monkeypatch, process: _StubProcess | None = None, raises=None):
    """Replace :data:`launch.SPAWN`; returns the recorded calls as
    ``(argv, kwargs)`` pairs. No viewer is ever started."""
    calls: list[tuple[list[str], dict]] = []

    def spawn(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if raises is not None:
            raise raises
        return process if process is not None else _StubProcess()

    monkeypatch.setattr(launch, "SPAWN", spawn)
    return calls


async def _open_viewer(pilot, app: ColonyTuiApp) -> ViewerScreen:
    """Open the viewer screen through the real home menu."""
    menu = app.screen.query_one("#home-menu", OptionList)
    menu.focus()
    menu.highlighted = _VIEW
    await pilot.pause()
    await pilot.press("enter")
    deadline = time.monotonic() + _TIMEOUT
    while not isinstance(app.screen, ViewerScreen) and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert isinstance(app.screen, ViewerScreen), "the menu did not open the viewer"
    return app.screen


async def _await_launch(pilot, screen: ViewerScreen) -> launch.ViewHandle:
    """Wait for the screen's last launch to finish (the UI keeps pumping)
    and return its handle."""
    deadline = time.monotonic() + _TIMEOUT
    while (
        not screen.handles or not screen.handles[-1].done.is_set()
    ) and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert screen.handles and screen.handles[-1].done.is_set(), "the launch never finished"
    await pilot.pause()
    return screen.handles[-1]


def _log_text(screen: ViewerScreen) -> str:
    """The viewer screen's log pane text so far."""
    rich_log = screen.query_one("#view-log", RichLog)
    return "\n".join(strip.text for strip in rich_log.lines)


def _status(screen: ViewerScreen) -> str:
    return str(screen.query_one("#view-status", Static).render())


# --- the well list the screen offers ------------------------------------------------


def test_the_viewer_action_lists_exactly_the_wells_the_sources_hold(tmp_path):
    """home → *View wells in napari*: the selection holds every well of the
    draft's resolved sources (raw plus the derived difference tree), sorted;
    the frame-less well directory is not offered, nothing is preselected,
    and *Open in napari* stays disabled until a well is chosen."""
    raw, out = _sources(tmp_path)

    async def scenario() -> None:
        app = ColonyTuiApp()
        _fill_draft(app, raw, out)
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            selection = screen.query_one("#view-wells", SelectionList)
            assert [option.value for option in selection.options] == [
                "-A001",
                "-B004",
                _DIFF_WELL,
            ]
            assert _EMPTY_WELL not in screen.wells
            assert screen.selected_wells == []
            assert screen.query_one("#open-viewer", Button).disabled
            assert _log_text(screen) == ""
            # the TUI's launch path carries no napari import
            assert "napari" not in vars(launch)
            assert "napari" not in vars(viewer_module)

    asyncio.run(scenario())


def test_an_unresolved_draft_shows_the_exact_config_error(tmp_path):
    """With the pipeline draft empty the screen offers nothing and reports
    the draft's exact ``ConfigError`` — on the status line and in the log —
    and the app stays alive."""
    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            expected = "missing required [pipeline] key(s): ['raw_dir', 'out_root']"
            assert screen.options is None
            assert screen.wells == []
            assert _status(screen) == expected
            assert expected in _log_text(screen)
            assert screen.query_one("#view-wells", SelectionList).option_count == 0
            assert screen.query_one("#open-viewer", Button).disabled
            await pilot.click("#view-back")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_loading_a_relative_pipeline_toml_fills_the_well_list(
    monkeypatch, tmp_path
):
    """An unresolved draft is not a dead end, and a config saved from
    another directory still works: *Load pipeline TOML…* opens the shared
    load dialog on the same pipeline draft, resolves the file's relative
    paths against the directory it was written from, and the wells of its
    sources appear when the dialog closes."""
    project = tmp_path / "project"
    raw, out = _sources(project)
    config = out / "pipeline.toml"
    config.write_text('[pipeline]\nraw_dir = "raw"\nout_root = "out"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            assert screen.wells == []

            await pilot.click("#load-viewer-toml")
            deadline = time.monotonic() + _TIMEOUT
            while (
                not isinstance(app.screen, LoadConfigDialog)
                and time.monotonic() < deadline
            ):
                await pilot.pause(0.05)
            dialog = app.screen
            assert isinstance(dialog, LoadConfigDialog)
            assert dialog.draft is app.drafts["pipeline"]

            path_input = dialog.query_one("#toml-path", Input)
            path_input.focus()
            path_input.value = str(config)
            await pilot.press("enter")
            deadline = time.monotonic() + _TIMEOUT
            while (
                not isinstance(app.screen, ViewerScreen)
                and time.monotonic() < deadline
            ):
                await pilot.pause(0.05)
            assert app.screen is screen
            assert screen.options is not None
            assert screen.draft.get("pipeline", "raw_dir") == str(raw.resolve())
            assert screen.draft.get("pipeline", "out_root") == str(out.resolve())
            assert screen.wells == sorted([*_RAW_WELLS, _DIFF_WELL])
            assert screen.query_one("#view-wells", SelectionList).option_count == 3
            assert _status(screen) == ""

    asyncio.run(scenario())


def test_a_resolved_draft_with_no_sources_offers_no_wells(tmp_path):
    """A draft that resolves but points at missing trees offers nothing
    (discovery never raises), leaves the action disabled, and names the
    directories it checked so the empty list explains itself."""
    async def scenario() -> None:
        app = ColonyTuiApp()
        missing = tmp_path / "does-not-exist"
        _fill_draft(app, missing, tmp_path / "out")
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            assert screen.options is not None
            assert screen.wells == []
            assert _status(screen) == ""
            assert screen.query_one("#open-viewer", Button).disabled
            hint = str(screen.query_one("#view-hint", Static).render())
            assert "No wells found" in hint
            assert str(missing) in hint
            assert isinstance(app.screen, ViewerScreen)

    asyncio.run(scenario())


def test_the_well_box_is_a_keyboard_selectable_list(tmp_path):
    """The bordered box is a multi-selection list, not a text input: it
    labels itself as one (so no one tries to type a path into it), and
    the keyboard drives it — down moves the highlight, space toggles the
    well, which opens the action."""
    raw, out = _sources(tmp_path)

    async def scenario() -> None:
        app = ColonyTuiApp()
        _fill_draft(app, raw, out)
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            wells = screen.query_one("#view-wells", SelectionList)
            assert wells.border_title == "wells"
            assert wells.border_subtitle == "space toggles"
            assert app.focused is wells
            await pilot.press("down", "space")
            await pilot.pause()
            assert screen.selected_wells == ["-A001"]
            assert not screen.query_one("#open-viewer", Button).disabled

    asyncio.run(scenario())


# --- the selection and the exact command --------------------------------------------


def test_confirming_starts_the_viewer_process_with_the_wells_and_paths(
    monkeypatch, tmp_path
):
    """Select two wells (multi-select), *Open in napari*: the launcher gets
    one comma-separated ``--wells=`` value and the draft's four resolved
    paths, as ``python -m colony_tracking view``; the launch lines land in
    the log pane, and the app is still interactive afterwards."""
    raw, out = _sources(tmp_path)
    calls = _stub_spawn(monkeypatch)

    async def scenario() -> None:
        app = ColonyTuiApp()
        _fill_draft(app, raw, out)
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            selection = screen.query_one("#view-wells", SelectionList)
            selection.select("-A001")
            selection.select(_DIFF_WELL)
            await pilot.pause()
            assert screen.selected_wells == ["-A001", _DIFF_WELL]
            assert not screen.query_one("#open-viewer", Button).disabled
            await pilot.click("#open-viewer")
            handle = await _await_launch(pilot, screen)

            assert not handle.failed
            assert handle.returncode == 0
            assert len(calls) == 1
            argv, kwargs = calls[0]
            assert argv == [
                sys.executable,
                "-m",
                "colony_tracking",
                "view",
                "--wells=-A001,-C006",
                f"--raw-dir={raw}",
                f"--diff-dir={out / 'difference'}",
                f"--seg-dir={out / 'seg_growth_diff'}",
                f"--analysis-dir={out / 'analysis'}",
            ]
            # the child's streams are captured (its lines go to the pane)
            assert kwargs["stdout"] is not None
            assert kwargs["stderr"] is not None
            text = _log_text(screen)
            assert "opening -A001, -C006 in a separate process" in text
            assert "colony_tracking view --wells=-A001,-C006" in text
            assert "the viewer process closed" in text
            assert "the viewer closed" in _status(screen)

            # the TUI kept running: navigate back home
            await pilot.click("#view-back")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_a_failing_viewer_process_is_a_log_line_and_the_tui_keeps_running(
    monkeypatch, tmp_path
):
    """A child that exits non-zero (napari missing, no display) reports its
    captured error and exit status in the log pane, the status line says
    the TUI keeps running, and the screen stays usable."""
    raw, out = _sources(tmp_path)
    _stub_spawn(
        monkeypatch,
        _StubProcess(
            returncode=1,
            stderr="napari is not installed: the view command needs it",
        ),
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        _fill_draft(app, raw, out)
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            screen.query_one("#view-wells", SelectionList).select("-A001")
            await pilot.pause()
            await pilot.click("#open-viewer")
            handle = await _await_launch(pilot, screen)

            assert handle.failed
            assert handle.returncode == 1
            text = _log_text(screen)
            assert "napari is not installed" in text
            assert "exited with status 1" in text
            assert "exited with status 1" in _status(screen)
            assert "the TUI keeps running" in _status(screen)
            # still the viewer screen, still navigable
            assert app.screen is screen
            await pilot.click("#view-back")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_a_spawn_failure_is_a_log_line_and_the_tui_keeps_running(
    monkeypatch, tmp_path
):
    """A spawn that raises is never fatal: the reason is a log-pane line
    and a status line, and the screen keeps running."""
    raw, out = _sources(tmp_path)
    _stub_spawn(monkeypatch, raises=OSError("no display available"))

    async def scenario() -> None:
        app = ColonyTuiApp()
        _fill_draft(app, raw, out)
        async with app.run_test(size=(100, 44)) as pilot:
            screen = await _open_viewer(pilot, app)
            screen.query_one("#view-wells", SelectionList).select("-B004")
            await pilot.pause()
            await pilot.click("#open-viewer")
            handle = await _await_launch(pilot, screen)

            assert handle.failed
            assert isinstance(handle.error, OSError)
            assert handle.returncode is None
            text = _log_text(screen)
            assert "could not start the viewer process: no display" in text
            assert "could not be started" in _status(screen)
            assert app.screen is screen

    asyncio.run(scenario())


# --- the command and the session mapping (unit seams) --------------------------------


def test_the_command_is_one_wells_flag_and_every_set_path():
    options = ViewOptions(raw_dir=Path("/raw"), seg_dir=Path("/seg"))
    assert launch.command(["-A001", "-B004"], options) == [
        sys.executable,
        "-m",
        "colony_tracking",
        "view",
        "--wells=-A001,-B004",
        "--raw-dir=/raw",
        "--seg-dir=/seg",
    ]


def test_the_session_is_the_chains_resolved_paths(tmp_path):
    """``view_options`` maps the chain config's resolved trees — never a
    re-derived name — onto the viewer's four sources."""
    config = PipelineConfig.from_dict(
        {"pipeline": {"raw_dir": str(tmp_path / "raw"), "out_root": str(tmp_path / "out")}}
    )
    options = launch.view_options(config)
    assert options == ViewOptions(
        raw_dir=tmp_path / "raw",
        diff_dir=tmp_path / "out" / "difference",
        seg_dir=tmp_path / "out" / "seg_growth_diff",
        analysis_dir=tmp_path / "out" / "analysis",
    )


def test_the_command_is_the_cli_session_the_draft_describes(monkeypatch, tmp_path):
    """The launched vector is a valid ``view`` invocation for exactly the
    draft's session: the CLI parses it, resolves the same
    :class:`ViewOptions`, and accepts the well — the same layers the
    command by hand would open."""
    raw, out = _sources(tmp_path)
    config = PipelineConfig.from_dict(
        {"pipeline": {"raw_dir": str(raw), "out_root": str(out)}}
    )
    argv = launch.command(["-A001"], launch.view_options(config))
    monkeypatch.chdir(tmp_path)  # no stray pipeline.toml changes the session
    args = cli._build_parser().parse_args(argv[3:])
    parsed = cli._view_options(args)
    assert parsed == ViewOptions(
        wells=["-A001"],
        raw_dir=raw,
        diff_dir=out / "difference",
        seg_dir=out / "seg_growth_diff",
        analysis_dir=out / "analysis",
    )
    cli._check_view_wells(parsed)  # the well exists: no usage error
