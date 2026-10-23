"""Tests for the stage screens: each stage's own independent draft (the
independent-configs redesign, spec story 36 and §2 of the addendum brief).

Screens under test: the per-stage :class:`StageScreen` — generated from
the stage registry (``STAGES``), rendering ONE form with every field of
the stage's options dataclass (all of them ordinary and editable — the
pipeline-owned ones included: difference ``raw_dir``/``reference``,
segmentation ``input``/``diff_dir``, analysis ``raw_dir``; no mirrors, no
owned lines anywhere) — its actions gated on its own draft's
``ConfigDraft.resolve()`` with the stage machinery's own wordings, and
its independent config file: *Write TOML…* suggests
``<out_dir parent>/difference.toml`` (each run type its own name, see
``TOML_NAMES``) and the written flat file loads through the stage
dataclass's ``from_dict`` to the draft's exact resolution.

Independence is asserted both ways: a stage screen's edits touch its own
run type's draft (``app.drafts[stage]``) and nothing of the pipeline's,
and edits on the pipeline document (filled on the config screen) touch
nothing of a stage's. Pilot scenarios run through ``asyncio.run`` (no
async test plugin in the dev group; the suite stays dependency-free);
assertions run inside the headless session — the app's screen stack is
gone once it ends. These tests mostly push the real screens directly
(the registry-driven navigation home → stage screen → config screen is
covered end-to-end in ``tests/test_tui_app.py``); the navigation they
exercise is exactly what the registry drives.
"""

import asyncio
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_origin, get_type_hints

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Select, Static

from colony_tracking._options import (
    AnalysisOptions,
    DifferenceOptions,
    Options,
    SegmentationOptions,
    _union_members,
)
from colony_tracking.pipeline import _STAGES
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.forms import kebab
from colony_tracking.tui.screens._save import LoadConfigDialog, SaveConfigDialog
from colony_tracking.tui.screens.config import ConfigScreen
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.stage import STAGES, StageScreen, StageSpec

#: The gate wording each stage's own draft shows while incomplete
#: (declaration order of the dataclass's required fields).
_MISSING_REQUIRED = {
    "difference": "missing required key(s): ['raw_dir', 'out_dir']",
    "segmentation": "missing required key(s): ['out_dir']",
    "analysis": "missing required key(s): ['seg_dirs', 'raw_dir', 'out_root']",
}

#: The fields that used to be pipeline-owned (read-only everywhere); the
#: redesign makes them ordinary editable widgets on their stage's screen.
_FORMERLY_OWNED = {
    "difference": ("raw_dir", "reference"),
    "segmentation": ("input", "diff_dir"),
    "analysis": ("raw_dir",),
}


@dataclass(kw_only=True)
class FutureStageOptions(Options):
    """A future stage's options (the shape the registry must accommodate):
    a Literal select and a tuple among the usual kinds — the kinds the
    three pipeline stages' fields don't all exercise."""

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


def _section_of(app: ColonyTuiApp, run_type: str) -> dict:
    """A run type's current flat document section (the stage drafts also
    carry a run type key; only a stage draft's own section exists)."""
    draft = app.drafts[run_type]
    return {} if draft.is_pipeline else draft.stages[run_type]


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
    monkeypatch.setitem(
        STAGES, "train", StageSpec("train", "Train", FutureStageOptions)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["pipeline"]
            draft.stages["train"] = {}
            stage = StageScreen("train", draft)
            app.push_screen(stage)
            await pilot.pause()
            assert isinstance(stage.form.query_one("#train-mode"), Select)
            assert isinstance(stage.form.query_one("#train-shape"), Input)
            assert isinstance(stage.form.query_one("#train-label-dir"), Input)
            stage.form.query_one("#train-mode", Select).value = "b"
            await pilot.pause()
            assert draft.get("train", "mode") == "b"

    asyncio.run(scenario())


# --- rendering: every field of the dataclass, none owned ---------------------------


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_every_option_field_renders_as_its_widget(stage):
    """EVERY field of the stage's dataclass renders as the spec table's
    widget on its screen — the formerly pipeline-owned ones as ordinary
    editable widgets now, exactly like the rest; no excluding split."""
    hints = get_type_hints(STAGES[stage].options)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            for key, hint in hints.items():
                field_id = f"#{stage}-{kebab(key)}"
                assert type(screen.form.query_one(field_id)) is _expected_widget(
                    hint
                ), key

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_formerly_owned_fields_are_editable_widgets_and_nothing_owned_remains(
    stage,
):
    """The fields the old chain wired read-only (difference raw_dir and
    reference, segmentation input/diff_dir, analysis raw_dir) render as
    ordinary widgets on the stage's own form: the segmentation ``input``
    is a Select, every other one an Input with a browse button where a
    path hint asks for one. The old mirror/owned machinery is gone from
    the screen: no mirror form, no ``#owned-values`` line, no
    ``run-level:`` labels anywhere."""
    from colony_tracking.tui.forms import BrowseButton

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            for key in _FORMERLY_OWNED[stage]:
                widget_id = f"#{stage}-{kebab(key)}"
                widget = screen.form.query_one(widget_id)
                if key == "input":
                    assert isinstance(widget, Select), key
                else:
                    assert isinstance(widget, Input), key
                if not isinstance(widget, Select):  # a path field browses too
                    if key.endswith("_dir") or key == "raw_dir":
                        browse = screen.form.query_one(
                            f"#browse-{stage}-{kebab(key)}", BrowseButton
                        )
                        assert not browse.can_focus, key
            # nothing of the old screen remains
            for absent in ("#mirror-title", "#owned-values"):
                with pytest.raises(NoMatches, match=absent):
                    app.screen.query_one(absent)
            labels = [
                str(label.render()) for label in screen.form.query(Static)
            ]
            assert not any("run-level" in text for text in labels)
            assert str(app.screen.query_one("#stage-title", Static).render()) == (
                f"{STAGES[stage].title} — options"
            )

    asyncio.run(scenario())


def test_stage_screen_edits_reach_only_the_stages_own_draft(tmp_path):
    """Independence, stage-screen direction: an edit on the difference
    screen stores it in the difference draft's own flat section and
    touches nothing of the pipeline's document. A key the old chain
    refused everywhere downstream (an owned key) is an ordinary key on
    the stage draft: ``reference`` stores like any other field — the
    pipeline draft's own refusal of owned keys is its chain's business
    (draft-level, see ``tests/test_tui_draft.py``)."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen("difference")
            app.push_screen(screen)
            await pilot.pause()
            raw = screen.form.query_one("#difference-raw-dir", Input)
            out = screen.form.query_one("#difference-out-dir", Input)
            raw.focus()
            await pilot.press(*"/data/raw")
            out.focus()
            await pilot.press(*str(tmp_path / "diff"))
            await pilot.pause()
            reference = screen.form.query_one("#difference-reference", Input)
            reference.focus()
            await pilot.press(*"3")
            await pilot.pause()
            assert _section_of(app, "difference") == {
                "raw_dir": "/data/raw",
                "out_dir": str(tmp_path / "diff"),
                "reference": 3,
            }
            assert app.drafts["pipeline"].document() == {"pipeline": {}}
            # and the pipeline draft never received a difference-owned key:
            # with the shared draft this write used to refuse; here it was
            # the difference draft's own all along
            app.drafts["pipeline"].set("difference", "out_dir", str(tmp_path / "x"))
            assert app.drafts["pipeline"].get("difference", "out_dir") == str(
                tmp_path / "x"
            )

    asyncio.run(scenario())


def test_pipeline_edits_reach_nothing_of_the_stage_drafts(tmp_path):
    """Independence, pipeline direction: filling the run level on the
    config screen leaves every stage draft where it was (and a stage
    screen entered after shows its gate still closed on its own empty
    draft — no run-level mirror fills it in)."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            raw = screen.form.query_one("#pipeline-raw-dir", Input)
            out = screen.form.query_one("#pipeline-out-root", Input)
            raw.focus()
            await pilot.press(*str(tmp_path / "raw"))
            out.focus()
            await pilot.press(*str(tmp_path / "out"))
            await pilot.pause()
            assert app.drafts["pipeline"].get("pipeline", "raw_dir") == str(
                tmp_path / "raw"
            )
            for stage in ("difference", "segmentation", "analysis"):
                assert _section_of(app, stage) == {}, stage
            stage_screen = StageScreen("difference")
            app.push_screen(stage_screen)
            await pilot.pause()
            assert not stage_screen.gate_open
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == _MISSING_REQUIRED["difference"]
            )
            assert isinstance(app.screen, StageScreen)

    asyncio.run(scenario())


# --- the action gate -------------------------------------------------------------


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_actions_gate_on_the_stages_own_draft(stage):
    """The gated actions (*Run this stage*, *Write TOML…*) start disabled
    with the stage machinery's exact ConfigError message on screen —
    missing required keys in the dataclass's declaration order, one
    wording per run type. *Load TOML…* (never gated) and *Back* stay
    enabled, and a disabled action press does nothing."""
    expected_error = _MISSING_REQUIRED[stage]

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(StageScreen(stage))
            await pilot.pause()
            status = app.screen.query_one("#resolve-status", Static)
            assert str(status.render()) == expected_error  # verbatim
            assert app.screen.query_one("#run-stage", Button).disabled
            assert app.screen.query_one("#write-toml", Button).disabled
            assert not app.screen.query_one("#load-toml", Button).disabled
            assert not app.screen.query_one("#back", Button).disabled
            await pilot.click("#write-toml")  # disabled: nothing happens
            await pilot.pause()
            assert not isinstance(app.screen, SaveConfigDialog)
            await pilot.click("#load-toml")  # ungated: the load dialog opens
            await pilot.pause()
            assert isinstance(app.screen, LoadConfigDialog)
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_filling_the_stages_own_fields_opens_the_gate(stage):
    """Filling the stage's own required fields opens the gate — the
    (difference) screen needs its raw_dir and out_dir, nothing else: no
    run level, no other stage's settings."""
    required = {
        "difference": ("raw_dir", "out_dir"),
        "segmentation": ("out_dir",),
        "analysis": ("seg_dirs", "raw_dir", "out_root"),
    }[stage]

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            assert not screen.gate_open
            for key in required:
                widget = screen.form.query_one(
                    f"#{stage}-{kebab(key)}", Input
                )
                if screen.form._kinds.get(key) == "select":  # an unexpected kind
                    widget.value = "raw"
                else:
                    widget.value = f"/data/{key}"
                await pilot.pause()
                if key != required[-1]:
                    assert not screen.gate_open  # one still missing
            assert screen.gate_open
            assert str(app.screen.query_one("#resolve-status", Static).render()) == ""

    asyncio.run(scenario())


def test_toggling_dry_run_with_an_incomplete_draft_only_updates_the_status_line():
    """The stage draft's error stays on the status line: toggling
    ``dry_run`` while the stage's own required fields are incomplete
    updates only the ``#resolve-status`` line — no Input's value or
    placeholder and no field's inline error ever receives the ConfigError
    message, and the toggle itself lands in the stage draft."""
    expected = "missing required key(s): ['raw_dir']"  # out_dir was typed

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
            assert (
                app.drafts["difference"].get("difference", "dry_run") is True
            )  # toggle landed in the stage's own draft
            app.drafts["pipeline"].set("pipeline", "raw_dir", "/reuse")
            await pilot.pause()
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == expected
            )
            for widget in stage.form.query(Input):
                assert (widget.value, widget.placeholder) == inputs_before[widget.id], (
                    widget.id
                )
            for label in stage.form.query(Static):
                if label.id and label.id.startswith("error-"):
                    assert str(label.render()) == "", label.id

    asyncio.run(scenario())


def test_gate_blocks_on_an_unparseable_field(tmp_path):
    """A field whose text does not parse disables the actions with the
    fix-fields note (the field's own error is inline) even though the
    draft itself would still resolve; fixing the text re-opens the gate."""
    raw = tmp_path / "raw"
    out = tmp_path / "diff"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            stage.form.query_one("#difference-raw-dir", Input).value = str(raw)
            stage.form.query_one("#difference-out-dir", Input).value = str(out)
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
    suggests ``<out_dir parent>/difference.toml`` (the run type's own
    file name, next to its own out setting), writes on Enter, and the
    flat file loads through ``DifferenceOptions.from_dict`` equal to the
    draft's resolution — a difference config, not a pipeline.toml."""
    out_dir = tmp_path / "out" / "diff_tree"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["difference"]
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            stage.form.query_one("#difference-raw-dir", Input).value = str(
                tmp_path / "raw"
            )
            stage.form.query_one("#difference-out-dir", Input).value = str(out_dir)
            await pilot.pause()
            assert stage.gate_open
            percent = stage.form.query_one("#difference-display-percentile", Input)
            percent.focus()
            await pilot.press(*"97.5")
            await pilot.pause()
            assert draft.get("difference", "display_percentile") == 97.5
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert (
                str(dialog.query_one("#toml-path", Input).value)
                == str(tmp_path / "out" / "difference.toml")  # <out_dir parent>/
            )
            assert str(dialog.query_one("#toml-path", Input).placeholder) == (
                "/path/to/difference.toml"
            )
            await pilot.press("enter")  # accept the suggested path
            await pilot.pause()
            path = tmp_path / "out" / "difference.toml"
            assert path.exists()
            with open(path, "rb") as handle:
                file_doc = tomllib.load(handle)
            assert "pipeline" not in file_doc  # a flat difference config
            assert DifferenceOptions.from_dict(file_doc) == draft.resolve()
            assert file_doc["display_percentile"] == 97.5
            assert app.screen is stage  # back on the stage screen

    asyncio.run(scenario())


def test_load_toml_opens_the_dialog_and_escape_cancels():
    """*Load TOML…* opens the shared load-config dialog (it addresses the
    stage's own run type) and escape cancels it — the draft is untouched
    and nothing is loaded."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["difference"]
            draft.set("difference", "raw_dir", "/kept")
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            await pilot.click("#load-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, LoadConfigDialog)
            assert (
                str(dialog.query_one("#toml-path", Input).placeholder)
                == "/path/to/difference.toml"
            )
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is stage  # back, nothing loaded
            assert _section_of(app, "difference") == {"raw_dir": "/kept"}
            assert not stage.gate_open  # out_dir still missing
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == "missing required key(s): ['out_dir']"
            )

    asyncio.run(scenario())


def test_run_action_hands_the_resolved_stage_options_to_the_run_flow():
    """*Run this stage* — the stage screens' only run action — resolves
    the draft and hands (stage, config) to the app's run flow: the config
    is the stage's own resolved options, everything the draft holds,
    nothing derived. There is no pipeline button on a stage screen."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["difference"]
            draft.set("difference", "raw_dir", "/data/raw")
            draft.set("difference", "out_dir", "/data/diff")
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
            assert runs[0][1].reference == "first"  # nothing derived
            with pytest.raises(NoMatches, match="#run-pipeline"):
                app.screen.query_one("#run-pipeline")  # one run action only

    asyncio.run(scenario())


def test_run_action_opens_the_save_dialog_by_default(tmp_path):
    """The shipped default run flow (no explicit ``start_run``): *Run
    this stage* opens the shared save dialog — it suggests the stage's
    own difference.toml path — and nothing runs until that write succeeds
    (escaping it runs nothing)."""
    from colony_tracking.tui import runner

    out_dir = tmp_path / "diff_tree"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["difference"]
            draft.set("difference", "raw_dir", str(tmp_path / "raw"))
            draft.set("difference", "out_dir", str(out_dir))
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            await pilot.click("#run-stage")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(
                tmp_path / "difference.toml"  # <out_dir parent>/difference.toml
            )
            assert runner.active() is None
            await pilot.press("escape")  # cancel: nothing written, nothing run
            await pilot.pause()
            assert runner.active() is None
            assert not (tmp_path / "difference.toml").exists()

    asyncio.run(scenario())


def test_arrow_keys_move_focus_between_fields():
    """Down/up move the focus between the form's fields (wrapping) while
    tab keeps its own course and left/right stay inside the focused
    input. The difference form leads with its own raw_dir now — the
    stage's own raw_dir is the first field and auto-focus target."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            raw_dir = stage.form.query_one("#difference-raw-dir", Input)
            assert app.focused is raw_dir  # the entry auto-focus
            await pilot.press("down")
            await pilot.pause()
            out_dir = stage.form.query_one("#difference-out-dir")
            assert app.focused is out_dir
            await pilot.press("down")
            await pilot.pause()
            reference = stage.form.query_one("#difference-reference")
            assert app.focused is reference
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
            # down from the last field (the dry-run checkbox) wraps back to
            # the first — the stage's own raw_dir the screen leads with
            dry_run = stage.form.query_one("#difference-dry-run", Checkbox)
            dry_run.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is raw_dir

    asyncio.run(scenario())


def test_select_overlay_keeps_arrow_precedence(monkeypatch):
    """A focused Select keeps its own arrow bindings: down opens its
    dropdown, further downs navigate the overlay, and the screen's
    field-stepping binding never steals the key."""
    monkeypatch.setitem(
        STAGES, "train", StageSpec("train", "Train", FutureStageOptions)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["pipeline"]
            draft.stages["train"] = {}
            stage = StageScreen("train", draft)
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
            assert draft.get("train", "mode") == "b"
            assert app.focused is mode  # the focus came back to the field

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
