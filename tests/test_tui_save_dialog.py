"""Tests for the shared save-config dialog (the "write the TOML" prompt).

The dialog is the one component every write-the-config action shares: it
suggests ``<out_root>/pipeline.toml``, writes the draft through
``ConfigDraft.write`` on submit, keeps itself open with the pipeline's own
error message when the draft is invalid or the path is unwritable, and
dismisses with ``(path, config)`` on success / ``None`` on cancel. The
written file must load unchanged through ``read_config`` — the
config-only flow (write now, ``colony-tracking pipeline --config`` later)
depends on that.

Pilot scenarios run through ``asyncio.run`` (no async test plugin in the
dev group; the suite stays dependency-free). Assertions run inside the
headless session — the app's screen stack is gone once it ends.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from textual.widgets import Input, Label

from colony_tracking.pipeline import read_config
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.screens._save import SaveConfigDialog, default_toml_path


def _draft(tmp_path: Path, *, complete: bool) -> ConfigDraft:
    """A draft with the run level filled (``complete``) or empty."""
    draft = ConfigDraft()
    if complete:
        draft.set("pipeline", "raw_dir", tmp_path / "raw")
        draft.set("pipeline", "out_root", tmp_path / "out")
    return draft


def _run(
    dialog: SaveConfigDialog,
    keys: tuple[str, ...],
    observe: Callable[[ColonyTuiApp, object], None],
    *,
    before: Callable[[SaveConfigDialog], None] | None = None,
) -> None:
    """Push ``dialog`` on a fresh headless app, run ``before`` (e.g. to
    clear the suggested path), press ``keys`` one key at a time, and hand
    the app and the dialog's dismissed result (or ``None``) to
    ``observe`` inside the still-running session."""

    async def scenario() -> None:
        results: list[object] = []
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(dialog, results.append)
            await pilot.pause()
            if before is not None:
                before(dialog)
            await pilot.press(*keys)
            await pilot.pause()
            observe(app, results[0] if results else None)

    asyncio.run(scenario())


def _clear(dialog: SaveConfigDialog) -> None:
    """Empty the path input (it starts with the suggested default)."""
    dialog.query_one("#toml-path", Input).value = ""


def test_suggests_out_root_pipeline_toml(tmp_path: Path) -> None:
    """The suggested path is <out_root>/pipeline.toml, and none without an
    out root."""
    draft = _draft(tmp_path, complete=True)
    assert default_toml_path(draft) == tmp_path / "out" / "pipeline.toml"
    assert default_toml_path(_draft(tmp_path, complete=False)) is None


def test_write_round_trips_through_read_config(tmp_path: Path) -> None:
    """Submit writes the TOML; read_config loads it unchanged; the dialog
    dismisses with (path, config)."""
    draft = _draft(tmp_path, complete=True)
    path = tmp_path / "run.toml"

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert isinstance(result, tuple)
        written, config = result
        assert Path(written) == path
        assert read_config(path) == config
        assert config == draft.resolve()

    _run(
        SaveConfigDialog(draft),
        (*str(path), "enter"),
        observe,
        before=_clear,
    )


def test_invalid_draft_reports_and_writes_nothing(tmp_path: Path) -> None:
    """A draft without raw_dir/out_root stays open with the ConfigError
    message and writes no file."""
    draft = _draft(tmp_path, complete=False)
    path = tmp_path / "run.toml"

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, SaveConfigDialog)  # still open
        message = dialog.query_one("#toml-error", Label).render()
        assert "raw_dir" in str(message)
        assert not path.exists()

    _run(SaveConfigDialog(draft), (*str(path), "enter"), observe)


def test_error_message_shows_bracketed_sections_verbatim(tmp_path: Path) -> None:
    """The ConfigError text renders unchanged — section names like
    ``[pipeline]`` must not be eaten as rich markup."""
    draft = _draft(tmp_path, complete=False)
    path = tmp_path / "run.toml"

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, SaveConfigDialog)
        message = str(dialog.query_one("#toml-error", Label).render())
        assert message == "missing required [pipeline] key(s): ['raw_dir', 'out_root']"

    _run(SaveConfigDialog(draft), (*str(path), "enter"), observe)


def test_cancel_writes_nothing(tmp_path: Path) -> None:
    """Escape dismisses with None and writes no file."""
    draft = _draft(tmp_path, complete=True)
    path = tmp_path / "run.toml"

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        assert not path.exists()

    _run(SaveConfigDialog(draft), ("escape",), observe)


def test_empty_path_reports(tmp_path: Path) -> None:
    """Submitting an empty path asks for a path and writes nothing."""
    draft = _draft(tmp_path, complete=True)

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, SaveConfigDialog)
        error = dialog.query_one("#toml-error", Label).render()
        assert "Enter a path" in str(error)
        assert dialog.query_one("#toml-path", Input).value == ""

    _run(SaveConfigDialog(draft), ("enter",), observe, before=_clear)


def test_write_success_toast_renders_and_expires_without_erroring(
    tmp_path: Path,
) -> None:
    """The success toast renders and expires without crashing the app:
    ``App.notify`` takes a plain markup-free string — it re-processes the
    content itself and a rich ``Text`` (or bracket-bearing text rendered as
    markup) breaks the toast."""
    draft = _draft(tmp_path, complete=True)
    path = tmp_path / "run.toml"

    async def scenario() -> None:
        results: list[object] = []
        app = ColonyTuiApp()
        app.NOTIFICATION_TIMEOUT = 0.2  # shrink the toast's lifetime
        async with app.run_test(notifications=True) as pilot:
            app.push_screen(SaveConfigDialog(draft), results.append)
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            dialog.query_one("#toml-path", Input).value = str(path)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert path.exists()
            assert isinstance(results[0], tuple)
            toasts = list(app.screen.query("Toast"))
            assert toasts  # the success toast rendered
            assert "Config written to" in str(toasts[0].render().plain)
            await pilot.pause(0.6)  # the toast expires; nothing raises
            assert not list(app.screen.query("Toast"))

    asyncio.run(scenario())


def test_unwritable_path_reports(tmp_path: Path) -> None:
    """A path under a file (not a directory) reports the OSError and stays
    open."""
    draft = _draft(tmp_path, complete=True)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    path = blocker / "run.toml"

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, SaveConfigDialog)
        error = dialog.query_one("#toml-error", Label).render()
        assert "could not write" in str(error)
        assert not path.exists()

    _run(SaveConfigDialog(draft), (*str(path), "enter"), observe)
