"""Tests for the stage screens and the config screen (ticket 02).

Screens under test: the per-stage :class:`StageScreen` — generated from
the stage registry (``STAGES``), showing the pipeline-owned fields
read-only with their derived values, its actions gated on
``ConfigDraft.resolve()`` — and the :class:`ConfigScreen` (the run-level
form plus the draft's effective TOML preview). The flows the spec fixes:
fill a stage's options, then write the TOML through the shared save
dialog (the written file loads unchanged through ``read_config``), or hand
the resolved config to the app's run flow (the seam the run-screens
ticket wires).

Pilot scenarios run through ``asyncio.run`` (no async test plugin in the
dev group; the suite stays dependency-free). Assertions run inside the
headless session — the app's screen stack is gone once it ends. These
tests mostly push the real screens directly (the registry-driven
navigation home → stage screen → config screen is covered end-to-end in
``tests/test_tui_app.py``); the navigation they exercise is exactly what
the registry drives.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_origin, get_type_hints

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from colony_tracking._options import (
    AnalysisOptions,
    DifferenceOptions,
    Options,
    SegmentationOptions,
    _union_members,
)
from colony_tracking.pipeline import _OWNED, _STAGES, read_config
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import RUN_LEVEL, ConfigDraft
from colony_tracking.tui.forms import kebab
from colony_tracking.tui.screens._save import SaveConfigDialog
from colony_tracking.tui.screens.config import ConfigScreen
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.stage import STAGES, StageScreen, owned_values


@dataclass(kw_only=True)
class FutureStageOptions(Options):
    """A future stage's options (the shape the registry must accommodate):
    a Literal select and a tuple among the usual kinds — the kinds the
    three pipeline stages' editable fields don't exercise."""

    mode: Literal["a", "b"] = field(default="a", metadata={"help": "Mode."})
    shape: tuple[int, int] = field(default=(4, 4), metadata={"help": "Shape."})
    label_dir: Path = field(metadata={"help": "Ground-truth labels."})


def _expected_widget(hint) -> type:
    """The spec table's widget for a hint — the test's own copy of the
    mapping, pinning the forms module to the spec."""
    members = _union_members(hint)
    if members is not None and len(members) == 1:
        return _expected_widget(members[0])
    if hint is bool:
        return Checkbox
    if get_origin(hint) is Literal:
        return Select
    return Input


def _set_run_level(app: ColonyTuiApp, tmp_path: Path) -> None:
    """Fill the run level of the app's draft (the config screen's job in
    the real flow)."""
    app.draft.set(RUN_LEVEL, "raw_dir", tmp_path / "raw")
    app.draft.set(RUN_LEVEL, "out_root", tmp_path / "out")


# --- the registry ---------------------------------------------------------------


def test_registry_covers_the_pipeline_stages():
    """The registry has one entry per pipeline stage, pointing at that
    stage's options dataclass — the screens are generated, not hardcoded."""
    assert set(STAGES) == set(_STAGES)
    assert STAGES["difference"].options is DifferenceOptions
    assert STAGES["segmentation"].options is SegmentationOptions
    assert STAGES["analysis"].options is AnalysisOptions


def test_future_stage_slots_into_the_registry(monkeypatch):
    """Adding one registry entry (plus a draft section) is all a future
    stage needs: the same screen class renders its form, including the
    select/tuple kinds."""
    from colony_tracking.tui.screens.stage import StageSpec

    monkeypatch.setitem(
        STAGES, "train", StageSpec("train", "Train", FutureStageOptions)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.draft.stages["train"] = {}
            stage = StageScreen("train")
            app.push_screen(stage)
            await pilot.pause()
            assert isinstance(stage.form.query_one("#train-mode"), Select)
            assert isinstance(stage.form.query_one("#train-shape"), Input)
            assert isinstance(stage.form.query_one("#train-label-dir"), Input)
            stage.form.query_one("#train-mode", Select).value = "b"
            await pilot.pause()
            assert app.draft.get("train", "mode") == "b"

    asyncio.run(scenario())


# --- rendering ------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_every_option_field_renders_as_its_widget(stage):
    """Every non-owned field of the stage renders as the spec table's
    widget; the owned fields render read-only in the header instead (they
    are derived, never edited)."""
    hints = get_type_hints(STAGES[stage].options)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            for key, hint in hints.items():
                field_id = f"#{stage}-{kebab(key)}"
                if key in _OWNED[stage]:
                    with pytest.raises(NoMatches):
                        screen.form.query_one(field_id)
                else:
                    assert type(screen.form.query_one(field_id)) is _expected_widget(
                        hint
                    ), key
            owned_text = str(app.screen.query_one("#owned-values", Static).render())
            for key in _OWNED[stage]:
                assert kebab(key) in owned_text

    asyncio.run(scenario())


def test_run_level_fields_render_on_the_config_screen():
    """The config screen's form is the run level: the four ``[pipeline]``
    fields, as inputs."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            for key in ("raw_dir", "out_root", "reference", "wells"):
                assert isinstance(
                    screen.form.query_one(f"#pipeline-{kebab(key)}"), Input
                )

    asyncio.run(scenario())


def test_owned_values_derive_from_the_run_level():
    """The owned fields' shown values are the ones resolve_stages derives:
    raw dir and reference from [pipeline], input forced to diff, the
    difference tree behind diff_dir (the difference section's own out_dir
    when set); underivable values show as —."""
    draft = ConfigDraft()
    assert owned_values(draft, "difference") == (
        ("raw-dir", "—"),
        ("reference", "first"),
    )
    draft.set(RUN_LEVEL, "raw_dir", "/data/raw")
    draft.set(RUN_LEVEL, "out_root", "/data/out")
    draft.set(RUN_LEVEL, "reference", 3)
    assert owned_values(draft, "difference") == (
        ("raw-dir", "/data/raw"),
        ("reference", "3"),
    )
    assert owned_values(draft, "segmentation") == (
        ("input", "diff"),
        ("diff-dir", "/data/out/difference_LO003"),
    )
    draft.set("difference", "out_dir", "/custom/tree")
    assert owned_values(draft, "segmentation") == (
        ("input", "diff"),
        ("diff-dir", "/custom/tree"),
    )
    assert owned_values(draft, "analysis") == (("raw-dir", "/data/raw"),)


def test_derived_defaults_show_as_placeholders(tmp_path):
    """The fields that default to a derived name grey the derived default
    in their empty input — the chain's tree names for the out root."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            _set_run_level(app, tmp_path)
            app.draft.set(RUN_LEVEL, "reference", 2)
            stage = StageScreen("analysis")
            app.push_screen(stage)
            await pilot.pause()
            out_root = tmp_path / "out"
            assert (
                stage.form.query_one("#analysis-seg-dir", Input).placeholder
                == f"{out_root}/seg_growth_diff_LO002"
            )
            assert (
                stage.form.query_one("#analysis-out-root", Input).placeholder
                == f"{out_root}/analysis"
            )

    asyncio.run(scenario())


# --- the action gate ------------------------------------------------------------


def test_actions_gate_on_the_whole_draft(tmp_path):
    """The gated actions start disabled with the pipeline's exact
    ConfigError message on screen; filling the run level on the config
    screen and coming back re-gates them (the resume refresh); *Back* is
    never gated and a disabled action does nothing."""
    expected_error = "missing required [pipeline] key(s): ['raw_dir', 'out_root']"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            status = app.screen.query_one("#resolve-status", Static)
            assert str(status.render()) == expected_error  # verbatim
            for button_id in ("run-stage", "write-toml", "run-pipeline"):
                assert app.screen.query_one(f"#{button_id}", Button).disabled
            assert not app.screen.query_one("#back", Button).disabled
            await pilot.click("#write-toml")  # disabled: nothing happens
            await pilot.pause()
            assert not isinstance(app.screen, SaveConfigDialog)
            # The run level is filled on the config screen; coming back
            # re-gates the stage screen.
            app.push_screen(ConfigScreen())
            await pilot.pause()
            raw = app.screen.query_one("#pipeline-raw-dir", Input)
            out = app.screen.query_one("#pipeline-out-root", Input)
            raw.focus()
            await pilot.press(*"/data/raw")
            out.focus()
            await pilot.press(*"/data/out")
            await pilot.pause()
            assert app.draft.get(RUN_LEVEL, "raw_dir") == "/data/raw"
            await pilot.click("#back")
            await pilot.pause()
            assert app.screen is stage
            assert stage.gate_open
            for button_id in ("run-stage", "write-toml", "run-pipeline"):
                assert not app.screen.query_one(f"#{button_id}", Button).disabled
            assert str(app.screen.query_one("#resolve-status", Static).render()) == ""

    asyncio.run(scenario())


def test_toggling_dry_run_with_an_incomplete_draft_only_updates_the_status_line():
    """The global draft error stays global: toggling ``dry_run`` while the
    run level is incomplete updates only the ``#resolve-status`` line — no
    Input's value or placeholder and no field's inline error ever receives
    the ``ConfigError`` message."""
    expected = "missing required [pipeline] key(s): ['raw_dir', 'out_root']"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            out_dir = stage.form.query_one("#difference-out-dir", Input)
            out_dir.focus()
            await pilot.press(*"/data/diff")
            await pilot.pause()
            inputs_before = {
                widget.id: (widget.value, widget.placeholder)
                for widget in stage.form.query(Input)
            }
            checkbox = stage.form.query_one("#difference-dry-run", Checkbox)
            stage.form.scroll_to_widget(checkbox, animate=False)
            await pilot.pause()
            checkbox.focus()
            await pilot.press("space")
            await pilot.pause()
            assert app.draft.get("difference", "dry_run") is True  # toggle landed
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == expected
            )
            for widget in stage.form.query(Input):
                assert (widget.value, widget.placeholder) == inputs_before[widget.id], (
                    widget.id
                )
            for label in stage.form.query(Label):
                if label.id and label.id.startswith("error-"):
                    assert str(label.render()) == "", label.id

    asyncio.run(scenario())


def test_gate_blocks_on_an_unparseable_field(tmp_path):
    """A field whose text does not parse disables the actions with the
    fix-fields note (the field's own error is inline) even though the
    draft itself would still resolve; fixing the text re-opens the gate."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            _set_run_level(app, tmp_path)
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            assert stage.gate_open
            percent = stage.form.query_one("#difference-display-percentile", Input)
            percent.value = "abc"
            await pilot.pause()
            assert not stage.gate_open
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == "Fix the marked fields to enable the actions."
            )
            assert (
                stage.form.error("display_percentile") == "must be a number (got 'abc')"
            )
            percent.value = "97.5"
            await pilot.pause()
            assert stage.gate_open

    asyncio.run(scenario())


# --- the flows ------------------------------------------------------------------


def test_edit_field_and_write_toml_round_trips(tmp_path):
    """The spec's per-stage decision point: edit a stage field (real
    keystrokes), then *Write TOML…* opens the shared dialog — which
    suggests <out_root>/pipeline.toml, writes on Enter, and the file loads
    unchanged through read_config, equal to the draft's resolution."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.draft
            _set_run_level(app, tmp_path)
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            percent = stage.form.query_one("#difference-display-percentile", Input)
            percent.focus()
            await pilot.press(*"97.5")
            await pilot.pause()
            assert draft.get("difference", "display_percentile") == 97.5
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            await pilot.press("enter")  # accept the suggested path
            await pilot.pause()
            path = tmp_path / "out" / "pipeline.toml"
            assert path.exists()
            config = read_config(path)
            assert config == draft.resolve()
            assert config.difference.display_percentile == 97.5
            assert app.screen is stage  # back on the stage screen

    asyncio.run(scenario())


def test_config_screen_writes_a_config_the_cli_accepts(tmp_path):
    """Type the whole run level; the preview shows the draft's effective
    TOML (markup-safe — the section header shows as typed); *Write TOML…*
    through the dialog writes a file read_config accepts unchanged."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press(*"/raw")  # raw_dir (auto-focused)
            await pilot.press("tab")
            await pilot.press(*"/out")  # out_root
            await pilot.press("tab")
            await pilot.press(*"2")  # reference
            await pilot.press("tab")
            await pilot.press(*"-A001,-B002")  # wells
            await pilot.pause()
            draft = app.draft
            assert draft.get(RUN_LEVEL, "raw_dir") == "/raw"
            assert draft.get(RUN_LEVEL, "out_root") == "/out"
            assert draft.get(RUN_LEVEL, "reference") == 2
            assert draft.get(RUN_LEVEL, "wells") == ["-A001", "-B002"]
            preview = str(app.screen.query_one("#toml-preview", Static).render())
            assert "[pipeline]" in preview
            assert 'raw_dir = "/raw"' in preview
            assert "reference = 2" in preview
            assert screen.gate_open
            path = tmp_path / "run.toml"
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            dialog.query_one("#toml-path", Input).value = str(path)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert path.exists()
            assert read_config(path) == draft.resolve()

    asyncio.run(scenario())


def test_arrow_keys_move_focus_between_fields():
    """Down/up move the focus between the form's fields (wrapping) while
    tab keeps its own course and left/right stay inside the focused
    input."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            out_dir = stage.form.query_one("#difference-out-dir", Input)
            assert app.focused is out_dir  # the entry auto-focus
            await pilot.press("down")
            await pilot.pause()
            wells = stage.form.query_one("#difference-wells")
            assert app.focused is wells
            await pilot.press("down")
            await pilot.pause()
            percent = stage.form.query_one("#difference-display-percentile")
            assert app.focused is percent
            await pilot.press("up")
            await pilot.pause()
            assert app.focused is wells
            # tab still moves on its own (wells -> display-percentile)
            await pilot.press("tab")
            await pilot.pause()
            assert app.focused is percent
            # left/right stay inside the focused input
            await pilot.press(*"97.5")
            await pilot.pause()
            await pilot.press("left")
            await pilot.pause()
            assert app.focused is percent
            assert percent.cursor_position == 3
            # down from the last field wraps back to the first
            dry_run = stage.form.query_one("#difference-dry-run", Checkbox)
            dry_run.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is out_dir

    asyncio.run(scenario())


def test_select_overlay_keeps_arrow_precedence(monkeypatch):
    """A focused Select keeps its own arrow bindings: down opens its
    dropdown, further downs navigate the overlay, and the screen's
    field-stepping binding never steals the key."""
    from colony_tracking.tui.screens.stage import StageSpec

    monkeypatch.setitem(
        STAGES, "train", StageSpec("train", "Train", FutureStageOptions)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.draft.stages["train"] = {}
            stage = StageScreen("train")
            app.push_screen(stage)
            await pilot.pause()
            mode = stage.form.query_one("#train-mode", Select)
            mode.focus()
            await pilot.pause()
            await pilot.press("down")  # the Select's own binding: open
            await pilot.pause()
            assert mode.expanded
            await pilot.press("down")  # the overlay's highlight moves
            await pilot.pause()
            shape = stage.form.query_one("#train-shape")
            assert app.focused is not shape  # the focus stayed with the select
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")  # pick the highlighted literal
            await pilot.pause()
            assert not mode.expanded
            assert app.draft.get("train", "mode") == "b"
            assert app.focused is mode  # the focus came back to the field

    asyncio.run(scenario())


def test_run_actions_hand_the_resolved_config_to_the_run_flow(tmp_path):
    """*Run this stage* / *Run whole pipeline from draft* resolve the
    draft and hand (kind, config) to the app's run flow — the seam the
    run-screens ticket wires."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.draft
            _set_run_level(app, tmp_path)
            runs: list[tuple[str, object]] = []
            stage = StageScreen(
                "difference", start_run=lambda kind, config: runs.append((kind, config))
            )
            app.push_screen(stage)
            await pilot.pause()
            await pilot.click("#run-stage")
            await pilot.pause()
            assert [kind for kind, _config in runs] == ["difference"]
            assert runs[0][1] == draft.resolve()
            await pilot.click("#run-pipeline")
            await pilot.pause()
            assert [kind for kind, _config in runs] == ["difference", "pipeline"]

    asyncio.run(scenario())


def test_run_action_opens_the_save_dialog_by_default(tmp_path):
    """The shipped default run flow (no explicit ``start_run``): *Run
    this stage* opens the shared save dialog — the reproducibility write —
    and nothing runs until that write succeeds (escaping it runs
    nothing)."""
    from colony_tracking.tui import runner
    from colony_tracking.tui.screens._save import SaveConfigDialog

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            _set_run_level(app, tmp_path)
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            await pilot.click("#run-stage")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfigDialog)
            assert runner.active() is None
            await pilot.press("escape")  # cancel: nothing written, nothing run
            await pilot.pause()
            assert runner.active() is None
            assert not (tmp_path / "out" / "pipeline.toml").exists()

    asyncio.run(scenario())


def test_back_button_returns_home():
    """*Back* pops to the home screen (the app's bottom of the stack)."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            app.push_screen(StageScreen("analysis"))
            await pilot.pause()
            await pilot.click("#back")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())
