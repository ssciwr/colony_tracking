"""Tests for the shared save- and load-config dialogs (the config-file doors).

``SaveConfigDialog`` is the one component every write-the-config action
shares: it suggests its draft's run type's own config path
(:func:`default_toml_path`), writes the draft through ``ConfigDraft.write``
on submit, keeps itself open with the machinery's own error message when
the draft is invalid or the path is unwritable, and dismisses with
``(path, config)`` on success / ``None`` on cancel. Each run type saves
its own file name next to its own output setting — the sectioned
``pipeline.toml`` for the chain, the flat stage TOMLs for the standalone
stage runs. The written pipeline file must load unchanged through
``read_config`` (the config-only flow — write now,
``colony-tracking pipeline --config`` later — depends on that); a
written stage file's flat document round-trips through the stage
dataclass's ``from_dict``.

``LoadConfigDialog`` is the read door: it asks for a path and loads that
file into its draft (:meth:`~colony_tracking.tui.draft.ConfigDraft.load`
— a load resets, it never merges). An unreadable or structurally wrong
file reports plain-Text inside the dialog, which stays open, and the
draft stays untouched; a success notifies markup-free and dismisses with
the loaded path.

Pilot scenarios run through ``asyncio.run`` (no async test plugin in the
dev group; the suite stays dependency-free). Assertions run inside the
headless session — the app's screen stack is gone once it ends.
"""

import asyncio
import tomllib
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest
from textual.widgets import Input, Label

from colony_tracking._options import DifferenceOptions
from colony_tracking.pipeline import read_config
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.screens._save import (
    LoadConfigDialog,
    SaveConfigDialog,
    default_toml_path,
)


def _draft(tmp_path: Path, *, complete: bool) -> ConfigDraft:
    """A draft with the run level filled (``complete``) or empty."""
    draft = ConfigDraft()
    if complete:
        draft.set("pipeline", "raw_dir", tmp_path / "raw")
        draft.set("pipeline", "out_root", tmp_path / "out")
    return draft


def _stage_draft(tmp_path: Path, run_type: str, *, complete: bool) -> ConfigDraft:
    """A stage draft with the run's own required keys (``complete``) or
    empty — only its own paths, nothing derived from another run."""
    draft = ConfigDraft(run_type)
    if not complete:
        return draft
    if run_type == "difference":
        draft.set("difference", "raw_dir", tmp_path / "raw")
        draft.set("difference", "out_dir", tmp_path / "out" / "difference_LO003")
    elif run_type == "segmentation":
        draft.set("segmentation", "out_dir", tmp_path / "out" / "seg_growth_diff")
    else:
        draft.set("analysis", "seg_dirs", [tmp_path / "out" / "seg_growth_diff"])
        draft.set("analysis", "raw_dir", tmp_path / "raw")
        draft.set("analysis", "out_root", tmp_path / "out" / "analysis")
    return draft


def _any_draft(tmp_path: Path, run_type: str) -> ConfigDraft:
    """``run_type``'s own draft, resolvable (its required paths filled)."""
    if run_type == "pipeline":
        return _draft(tmp_path, complete=True)
    return _stage_draft(tmp_path, run_type, complete=True)


def _difference_fixture(tmp_path: Path, text: str) -> Path:
    """A ``difference.toml`` fixture carrying exactly ``text``."""
    fixture = tmp_path / "difference.toml"
    fixture.write_text(text)
    return fixture


def _run(
    dialog: SaveConfigDialog | LoadConfigDialog,
    keys: tuple[str, ...],
    observe: Callable[[ColonyTuiApp, object], None],
    *,
    before: Callable[[SaveConfigDialog], None] | None = None,
) -> None:
    """Push ``dialog`` (its draft constructed explicitly by the test; the
    app is just the harness) on a fresh headless app, run ``before``
    (e.g. to clear the suggested path), press ``keys`` one key at a time,
    and hand the app and the dialog's dismissed result (or ``None``) to
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


def test_default_toml_path_is_per_run_type(tmp_path: Path) -> None:
    """The stage suggestions consult only the draft's own output setting:
    difference/segmentation suggest the out_dir's parent's own TOML file
    (``None`` without an out_dir), analysis its out_root's (``None``
    without an out_root) — the pipeline's suggestion is pinned by
    ``test_suggests_out_root_pipeline_toml`` above."""
    suggested = {
        "difference": tmp_path / "out" / "difference.toml",
        "segmentation": tmp_path / "out" / "segmentation.toml",
        "analysis": tmp_path / "out" / "analysis" / "analysis.toml",
    }
    for run_type, expected in suggested.items():
        draft = _stage_draft(tmp_path, run_type, complete=True)
        assert default_toml_path(draft) == expected
        empty = _stage_draft(tmp_path, run_type, complete=False)
        assert default_toml_path(empty) is None


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


@pytest.mark.parametrize(
    ("run_type", "caption"),
    [
        ("pipeline", "Write the config TOML to (runs later with: colony-tracking "
         "pipeline --config):"),
        ("difference", "Write the difference run's config TOML to:"),
        ("segmentation", "Write the segment run's config TOML to:"),
        ("analysis", "Write the analyze run's config TOML to:"),
    ],
)
def test_save_captions_the_run_type(
    tmp_path: Path, run_type: str, caption: str
) -> None:
    """The dialog captions and starts from the run type's own TOML: the
    chain's caption unchanged, a stage run's own, and the input holds the
    run type's own suggested path while the placeholder names its file."""
    draft = _any_draft(tmp_path, run_type)
    suggested = default_toml_path(draft)
    assert suggested is not None

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, SaveConfigDialog)
        assert str(list(dialog.query("Label"))[0].render()) == caption
        path_input = dialog.query_one("#toml-path", Input)
        assert path_input.placeholder == f"/path/to/{run_type}.toml"
        assert path_input.value == str(suggested)

    _run(SaveConfigDialog(draft), (), observe)


def test_stage_save_writes_the_flat_toml(tmp_path: Path) -> None:
    """Enter with the suggested default writes the difference run's flat
    TOML — no [pipeline] section, the stage dataclass's round-trip shape —
    and dismisses with (path, the stage's own resolved Options)."""
    draft = _stage_draft(tmp_path, "difference", complete=True)
    path = tmp_path / "out" / "difference.toml"  # the suggested default

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert isinstance(result, tuple)
        written, options = result
        assert Path(written) == path
        assert options == draft.resolve()
        assert isinstance(options, DifferenceOptions)
        assert options.reference == "first"  # nothing derived from a run level
        with open(path, "rb") as handle:
            doc = tomllib.load(handle)
        assert "pipeline" not in doc  # flat: no sectioned chain shape
        assert doc == {k: v for k, v in options.to_dict().items() if v is not None}
        assert DifferenceOptions.from_dict(doc) == options

    _run(SaveConfigDialog(draft), ("enter",), observe)


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


# --- the load dialog: the config TOML read back into a draft ----------------------


@pytest.mark.parametrize(
    ("run_type", "caption"),
    [
        ("pipeline", "Load the pipeline config TOML from:"),
        ("difference", "Load the difference config TOML from:"),
        ("segmentation", "Load the segment config TOML from:"),
        ("analysis", "Load the analyze config TOML from:"),
    ],
)
def test_load_captions_the_run_type(
    tmp_path: Path, run_type: str, caption: str
) -> None:
    """The load dialog captions the run type's own TOML and starts from an
    empty input with the run type's own placeholder (the `a analyze`
    reading is the machinery's f-string per the brief), nothing loaded
    while it waits."""
    draft = _any_draft(tmp_path, run_type)
    before_document = draft.document()

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, LoadConfigDialog)
        assert str(list(dialog.query("Label"))[0].render()) == caption
        path_input = dialog.query_one("#toml-path", Input)
        assert path_input.value == ""
        assert path_input.placeholder == f"/path/to/{run_type}.toml"
        assert draft.document() == before_document  # escape never loaded

    _run(LoadConfigDialog(draft), (), observe)


def test_load_loads_and_resets_the_draft(tmp_path: Path) -> None:
    """Enter loads the file into the draft — the draft is reset to the
    file's content (a load resets, it never merges; keys absent from the
    file are gone) — and the dialog dismisses with the loaded path."""
    fixture = _difference_fixture(
        tmp_path, 'raw_dir = "/data/raw"\nout_dir = "/data/out/difference_LO003"\n'
    )
    draft = ConfigDraft("difference")
    draft.set("difference", "display_percentile", 42.0)  # absent from the file

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result == fixture
        assert draft.document() == {
            "raw_dir": "/data/raw",
            "out_dir": "/data/out/difference_LO003",
        }
        draft.resolve()  # the loaded document is resolvable as-is

    _run(LoadConfigDialog(draft), (*str(fixture), "enter"), observe)


def test_load_cancel_dismisses_none(tmp_path: Path) -> None:
    """Escape dismisses with None and the draft keeps its content."""
    draft = ConfigDraft("difference")
    draft.set("difference", "display_percentile", 42.0)

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        assert draft.document() == {"display_percentile": 42.0}

    _run(LoadConfigDialog(draft), ("escape",), observe)


def test_load_empty_path_asks(tmp_path: Path) -> None:
    """Submitting an empty path asks for a path, stays open, and the draft
    keeps its content."""
    draft = ConfigDraft("difference")
    draft.set("difference", "display_percentile", 42.0)

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, LoadConfigDialog)  # still open
        assert "Enter a path" in str(dialog.query_one("#toml-error", Label).render())
        assert draft.document() == {"display_percentile": 42.0}

    _run(LoadConfigDialog(draft), ("enter",), observe)


def test_load_reports_unreadable_path(tmp_path: Path) -> None:
    """A path that cannot be opened (under a file, not a directory)
    reports the OSError in the dialog and stays open."""
    draft = ConfigDraft("difference")
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    path = blocker / "difference.toml"

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, LoadConfigDialog)  # still open
        message = dialog.query_one("#toml-error", Label).render()
        assert "could not read" in str(message)
        assert draft.document() == {}

    _run(LoadConfigDialog(draft), (*str(path), "enter"), observe)


def test_load_reports_malformed_toml(tmp_path: Path) -> None:
    """A file that is not TOML reports the decode wording inside the
    dialog and stays open, draft untouched."""
    draft = ConfigDraft("difference")
    fixture = _difference_fixture(tmp_path, "this is ] not toml =\n")

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, LoadConfigDialog)  # still open
        message = dialog.query_one("#toml-error", Label).render()
        assert f"{fixture}: not valid TOML: " in str(message)
        assert draft.document() == {}  # nothing was loaded into the draft

    _run(LoadConfigDialog(draft), (*str(fixture), "enter"), observe)


def test_load_error_message_shows_bracketed_keys_verbatim(tmp_path: Path) -> None:
    """A structurally wrong file reports the machinery's exact wording
    unchanged — brackets like ``['nonsense']`` must not be eaten as rich
    markup (the wording renders as plain text) — and nothing enters the
    draft."""
    draft = ConfigDraft("difference")
    fixture = _difference_fixture(
        tmp_path, 'raw_dir = "/data/raw"\nout_dir = "/data/out/diff"\nnonsense = 1\n'
    )
    names = sorted(f.name for f in fields(DifferenceOptions))

    def observe(app: ColonyTuiApp, result: object) -> None:
        assert result is None
        dialog = app.screen
        assert isinstance(dialog, LoadConfigDialog)  # still open
        message = str(dialog.query_one("#toml-error", Label).render())
        assert message == (
            f"unknown key(s) ['nonsense'] — expected one key per name: {names}"
        )
        assert draft.document() == {}

    _run(LoadConfigDialog(draft), (*str(fixture), "enter"), observe)


def test_load_success_toast_renders_and_expires_without_erroring(
    tmp_path: Path,
) -> None:
    """The success toast renders over the reset draft and expires without
    crashing the app — the same markup-free ``App.notify`` contract the
    write test pins."""
    fixture = _difference_fixture(
        tmp_path, 'raw_dir = "/data/raw"\nout_dir = "/data/out/difference_LO003"\n'
    )
    draft = ConfigDraft("difference")

    async def scenario() -> None:
        results: list[object] = []
        app = ColonyTuiApp()
        app.NOTIFICATION_TIMEOUT = 0.2  # shrink the toast's lifetime
        async with app.run_test(notifications=True) as pilot:
            app.push_screen(LoadConfigDialog(draft), results.append)
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, LoadConfigDialog)
            dialog.query_one("#toml-path", Input).value = str(fixture)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert results[0] == fixture
            assert draft.document() == {
                "raw_dir": "/data/raw",
                "out_dir": "/data/out/difference_LO003",
            }
            toasts = list(app.screen.query("Toast"))
            assert toasts  # the success toast rendered
            assert "Config loaded from" in str(toasts[0].render().plain)
            await pilot.pause(0.6)  # the toast expires; nothing raises
            assert not list(app.screen.query("Toast"))

    asyncio.run(scenario())
