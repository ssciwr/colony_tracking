"""Tests for the file dialogs: the config doors and file-valued fields.

Both config-file doors carry the forms' *browse* button in their path
row — the shared ``BrowseButton``, click-only like every path field's
button. The press opens textual-fspicker's file dialog (``FileOpen`` for
loading, ``FileSave`` for writing) at the typed path's nearest existing
directory, filtered to ``*.toml`` with an all-files fallback. Choosing a
file is the confirmation: the path lands in the dialog's input and the
draft is loaded or written at once. Cancelling the picker changes
nothing.

The option forms' file-valued path fields — those whose declared type
carries :class:`~colony_tracking._options.FilePath`, i.e. the segment and
evaluate checkpoints (open) and the evaluate JSON report (save) — get the
same treatment with their own suffix filter, while directory-valued path
fields keep the directory picker.

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group); the file pickers are driven headlessly, so no
real terminal or file dialog is needed.
"""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conftest import click_widget
from textual.widgets import Input

from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.pickers import BROWSE_LABEL, BrowseButton
from colony_tracking.tui.screens._save import (
    TOML_FILTERS,
    LoadConfigDialog,
    SaveConfigDialog,
)
from colony_tracking.tui.screens.stage import StageScreen
from colony_tracking.tui.screens.step import StepScreen
from textual_fspicker import FileOpen, FileSave, SelectDirectory

#: How long a test waits for a picker screen or a dialog result.
_TIMEOUT = 10.0


async def _wait_until(pilot, check: Callable[[], bool]) -> None:
    """Pump the UI until ``check`` is true."""
    deadline = time.monotonic() + _TIMEOUT
    while not check() and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert check()


async def _choose(pilot, app: ColonyTuiApp, name: str) -> None:
    """Choose ``name`` in the open file picker: type it into the picker's
    input and press its select button."""
    app.screen.query_one("Input").value = name
    await pilot.pause()
    assert await pilot.click("#select"), "#select"
    await pilot.pause()


def _run(
    dialog: LoadConfigDialog | SaveConfigDialog,
    observe: Callable[..., Any],
) -> None:
    """Mount ``dialog`` on the real app and hand the app, dialog, pilot,
    and dismissed-result list to ``observe`` inside the session."""

    async def scenario() -> None:
        results: list[object] = []
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 40)) as pilot:
            app.push_screen(dialog, results.append)
            await pilot.pause()
            await observe(app, dialog, pilot, results)

    asyncio.run(scenario())


def _dialogs() -> tuple[LoadConfigDialog, SaveConfigDialog]:
    """One load dialog and one save dialog on independent drafts, with
    everything the save dialog needs to write."""
    load_draft = ConfigDraft("pipeline")
    save_draft = ConfigDraft("difference")
    save_draft.set("difference", "raw_dir", "/data/raw")
    save_draft.set("difference", "out_dir", "/data/out/difference")
    return LoadConfigDialog(load_draft), SaveConfigDialog(save_draft)


def test_the_pickers_prefer_toml_but_keep_every_file_reachable():
    """The picker's first filter is ``*.toml`` (the config files), with an
    all-files fallback for a config without the conventional suffix."""
    assert TOML_FILTERS[0].name == "TOML files (*.toml)"
    assert TOML_FILTERS[0](Path("pipeline.toml"))
    assert not TOML_FILTERS[0](Path("pipeline.txt"))
    assert TOML_FILTERS[1](Path("pipeline.txt"))


def test_both_dialogs_carry_the_shared_click_only_browse_button():
    """The path row has the forms' browse button, labelled and click-only
    so it never becomes a tab stop between the input and the dialog's
    action."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 40)) as pilot:
            for dialog in _dialogs():
                app.push_screen(dialog)
                await pilot.pause()
                button = app.screen.query_one("#browse-toml-path", BrowseButton)
                assert button.label == BROWSE_LABEL
                assert not button.can_focus
                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(scenario())


def test_load_browse_picks_a_toml_and_loads_it_at_once(tmp_path: Path) -> None:
    """Load: the browse button opens ``FileOpen``; the picked file lands
    in the path input and is loaded into the draft without a second
    Enter."""
    config = tmp_path / "pipeline.toml"
    config.write_text('[pipeline]\nraw_dir = "/data/raw"\nout_root = "/data/out"\n')
    dialog, _save = _dialogs()

    async def observe(app, dialog, pilot, results):
        await pilot.click("#browse-toml-path")
        await _wait_until(pilot, lambda: isinstance(app.screen, FileOpen))
        assert app.screen.query_one("DirectoryNavigation").location == Path.cwd()
        app.screen.query_one("DirectoryNavigation").location = tmp_path
        await pilot.pause()
        await _choose(pilot, app, config.name)
        await _wait_until(pilot, lambda: results != [])
        assert results == [config]
        assert dialog.draft.document() == {
            "pipeline": {"raw_dir": "/data/raw", "out_root": "/data/out"}
        }

    _run(dialog, observe)


def test_save_browse_picks_a_name_and_writes_it_at_once(tmp_path: Path) -> None:
    """Write: the browse button opens ``FileSave`` at the suggested
    config's directory with its file name pre-filled; the picked path
    lands in the input and is written without a second Enter."""
    _load, dialog = _dialogs()
    dialog.draft.set("difference", "out_dir", str(tmp_path / "difference"))

    async def observe(app, dialog, pilot, results):
        await pilot.click("#browse-toml-path")
        await _wait_until(pilot, lambda: isinstance(app.screen, FileSave))
        picker = app.screen
        assert picker.query_one("DirectoryNavigation").location == tmp_path
        assert picker.query_one("Input").value == "difference.toml"
        await _choose(pilot, app, "picked.toml")
        picked = tmp_path / "picked.toml"
        await _wait_until(pilot, picked.is_file)
        await _wait_until(pilot, lambda: results != [])
        assert results[0][0] == picked

    _run(dialog, observe)


# --- the option forms: file-valued fields get the file dialogs ----------------------


def _run_screen(screen, observe) -> None:
    """Mount ``screen`` on the real app and hand the app, screen, and pilot
    to ``observe`` inside the session."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await observe(app, screen, pilot)

    asyncio.run(scenario())


def test_the_evaluate_checkpoint_browses_with_the_file_open_dialog(
    tmp_path: Path,
) -> None:
    """``evaluate.checkpoint`` is a file to read: its browse button opens
    ``FileOpen`` (filtered to ``*.pt`` with the all-files fallback), and the
    picked file lands in the input and the draft."""
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")

    async def observe(app, screen, pilot):
        screen.form.query_one("#evaluate-data-dir", Input).value = str(tmp_path)
        checkpoint_field = screen.form.query_one("#evaluate-checkpoint", Input)
        checkpoint_field.value = str(tmp_path)
        await pilot.pause()
        await click_widget(pilot, app, "#browse-evaluate-checkpoint")
        await _wait_until(pilot, lambda: isinstance(app.screen, FileOpen))
        picker = app.screen
        nav = picker.query_one("DirectoryNavigation")
        assert nav.location == tmp_path
        assert nav.file_filter.name == "Checkpoints (*.pt)"
        assert nav.file_filter(Path("best.pt"))
        assert not nav.file_filter(Path("report.json"))
        await _choose(pilot, app, checkpoint.name)
        await _wait_until(pilot, lambda: app.screen is screen)
        assert checkpoint_field.value == str(checkpoint)
        assert app.drafts["evaluate"].get("evaluate", "checkpoint") == str(checkpoint)
        assert screen.gate_open

    _run_screen(StepScreen("evaluate"), observe)


def test_the_evaluate_report_browses_with_the_file_save_dialog(tmp_path: Path) -> None:
    """``evaluate.output`` is a file to write: its browse button opens
    ``FileSave`` (filtered to ``*.json`` with the all-files fallback), and
    the chosen path lands in the input and the draft."""
    report = tmp_path / "report.json"

    async def observe(app, screen, pilot):
        screen.form.query_one("#evaluate-data-dir", Input).value = str(tmp_path)
        output_field = screen.form.query_one("#evaluate-output", Input)
        output_field.value = str(tmp_path)
        await pilot.pause()
        await click_widget(pilot, app, "#browse-evaluate-output")
        await _wait_until(pilot, lambda: isinstance(app.screen, FileSave))
        picker = app.screen
        nav = picker.query_one("DirectoryNavigation")
        assert nav.location == tmp_path
        assert nav.file_filter.name == "JSON reports (*.json)"
        assert nav.file_filter(Path("report.json"))
        assert not nav.file_filter(Path("best.pt"))
        await _choose(pilot, app, report.name)
        await _wait_until(pilot, lambda: app.screen is screen)
        assert output_field.value == str(report)
        assert app.drafts["evaluate"].get("evaluate", "output") == str(report)
        assert screen.gate_open

    _run_screen(StepScreen("evaluate"), observe)


def test_the_evaluate_data_dir_keeps_the_directory_picker(tmp_path: Path) -> None:
    """A directory-valued path field keeps the directory picker: the
    evaluate ``data_dir`` browse opens ``SelectDirectory``, not a file
    dialog."""
    (picked := tmp_path / "h5").mkdir()

    async def observe(app, screen, pilot):
        data_dir = screen.form.query_one("#evaluate-data-dir", Input)
        data_dir.value = str(tmp_path)
        await pilot.pause()
        await click_widget(pilot, app, "#browse-evaluate-data-dir")
        await _wait_until(pilot, lambda: isinstance(app.screen, SelectDirectory))
        nav = app.screen.query_one("DirectoryNavigation")
        nav.location = tmp_path
        await pilot.pause()
        nav.highlighted = 1
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert nav.location == picked
        assert await pilot.click("#select")
        await _wait_until(pilot, lambda: app.screen is screen)
        assert data_dir.value == str(picked)
        assert app.drafts["evaluate"].get("evaluate", "data_dir") == str(picked)

    _run_screen(StepScreen("evaluate"), observe)


def test_the_segmentation_checkpoint_browses_with_the_file_open_dialog(
    tmp_path: Path,
) -> None:
    """The existing ``segmentation.checkpoint`` field is file-valued too:
    its browse button now opens ``FileOpen`` instead of the directory
    picker, and the pick lands in the stage's own draft."""
    checkpoint = tmp_path / "sam_colony.pt"
    checkpoint.write_bytes(b"weights")

    async def observe(app, screen, pilot):
        checkpoint_field = screen.form.query_one("#segmentation-checkpoint", Input)
        checkpoint_field.value = str(tmp_path)
        await pilot.pause()
        await click_widget(pilot, app, "#browse-segmentation-checkpoint")
        await _wait_until(pilot, lambda: isinstance(app.screen, FileOpen))
        await _choose(pilot, app, checkpoint.name)
        await _wait_until(pilot, lambda: app.screen is screen)
        assert checkpoint_field.value == str(checkpoint)
        assert app.drafts["segmentation"].get("segmentation", "checkpoint") == str(
            checkpoint
        )

    _run_screen(StageScreen("segmentation"), observe)


def test_cancelling_the_picker_changes_nothing() -> None:
    """Escape in the file picker returns to the dialog with its typed
    path untouched and nothing loaded or written."""

    async def scenario() -> None:
        load, _save = _dialogs()
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 40)) as pilot:
            app.push_screen(load)
            await pilot.pause()
            before = load.query_one("#toml-path", Input).value
            await pilot.click("#browse-toml-path")
            await _wait_until(pilot, lambda: isinstance(app.screen, FileOpen))
            await pilot.press("escape")
            await _wait_until(pilot, lambda: isinstance(app.screen, LoadConfigDialog))
            assert load.query_one("#toml-path", Input).value == before
            assert load.draft.document() == {"pipeline": {}}

    asyncio.run(scenario())
