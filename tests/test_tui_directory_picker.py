"""Tests for ticket 07: the path fields' browse buttons and the picker.

Every single-``Path`` field renders with a *browse* button in its field
row — the reset button's click-only precedent (``can_focus`` off), so tab
and the arrow-key field walk stay field-to-field. The press opens the
directory-picker modal, textual-fspicker's ``SelectDirectory`` (spec
addendum decision 6's preferred solution), driven headless here: picking
a directory writes it into the field's input, and the normal on-change
flow stores it — on a stage screen into that stage's OWN draft (each run
type holds an independent config: the chain's ``[pipeline]`` draft is
untouched, and the stage's pipeline-owned fields pick into the stage's
own section like every other field); on the config screen and the quick
start into the chain's pipeline draft. Cancelling changes nothing.
List-of-path fields (``[analysis].seg_dirs``) keep manual entry, and the
config screen's within-chain owned lines render no widgets at all.

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free); assertions run
inside the headless session. The picker's directory list loads in a
background worker, so the driving helpers wait for it.
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, Input

from colony_tracking._options import DifferenceOptions, Options, _union_members
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import RUN_LEVEL, ConfigDraft
from colony_tracking.tui.forms import (
    BROWSE_LABEL,
    BrowseButton,
    OptionForm,
    kebab,
    picker_location,
    run_level_specs,
    specs_of,
)
from colony_tracking.tui.screens.config import ConfigScreen, editable_specs
from colony_tracking.tui.screens.quick_start import QuickStartScreen
from colony_tracking.tui.screens.stage import STAGES, StageSpec, StageScreen
from textual_fspicker import SelectDirectory

#: How long a test waits for the picker's background directory load.
_TIMEOUT = 10.0


def _is_single_path(hint) -> bool:
    """The test's own copy of the browse predicate — pinning the forms
    module to the spec (the completeness tests' idiom): exactly the
    single-``Path`` hints (``Path``, ``Path | None``) get a button."""
    members = _union_members(hint)
    if members is not None:
        if len(members) != 1:  # e.g. Reference: "first" | int
            return False
        hint = members[0]
    return hint is Path


@dataclass(kw_only=True)
class FutureStageOptions(Options):
    """A future stage's options (the completeness tests' shape): one path
    field and one non-path field, nothing registry-specific."""

    mode: Literal["a", "b"] = field(default="a", metadata={"help": "Mode."})
    label_dir: Path = field(metadata={"help": "Ground-truth labels."})


class _FormHost(App[None]):
    """The minimal host for a bare OptionForm (the forms tests' idiom:
    mounted alone, with the draft handed to it at construction)."""

    def __init__(self, form: OptionForm) -> None:
        super().__init__()
        self._form = form

    def compose(self) -> ComposeResult:
        yield self._form


def _run_form(form: OptionForm, observe) -> None:
    """Mount ``form`` headless and hand app + form + pilot to the async
    ``observe`` inside the still-running session (the widget tree is gone
    once it ends)."""

    async def scenario() -> None:
        app = _FormHost(form)
        async with app.run_test() as pilot:
            await pilot.pause()
            await observe(app, form, pilot)

    asyncio.run(scenario())


async def _click(pilot, app: App, selector: str) -> None:
    """Scroll a (possibly folded-away) widget into view and click it."""
    app.screen.query_one(selector).scroll_visible(animate=False)
    await pilot.pause()
    assert await pilot.click(selector), selector
    await pilot.pause()


async def _wait_loaded(pilot, nav) -> None:
    """Wait for the picker's directory list to populate (it loads in a
    worker thread): the ``..`` entry plus at least one real directory."""
    deadline = time.monotonic() + _TIMEOUT
    while nav.option_count < 2 and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert nav.option_count >= 2, "the picker's directory tree never loaded"


async def _pick_subdir(pilot, app: App, target: Path) -> None:
    """Drive the open picker to ``target`` like a user would: step down to
    the one subdirectory its parent holds (the list leads with ``..``),
    open it with enter, press the dialog's *Select*."""
    nav = app.screen.query_one("DirectoryNavigation")
    await _wait_loaded(pilot, nav)
    nav.highlighted = 1
    await pilot.pause()
    await pilot.press("enter")  # open the highlighted directory
    await pilot.pause()
    assert nav.location == target
    assert await pilot.click("#select")
    await pilot.pause()


def test_browse_buttons_follow_the_path_hints_on_every_screen():
    """Every rendered field's browse button matches its hint — single-Path
    hints have one (click-only, labelled like the reset), everything else
    has none: the stage screens' own forms (every field of the dataclass,
    the pipeline-owned ones included), the config screen's run level and
    per-stage sections (owned fields excluded — they render as read-only
    lines, no widgets), and the quick start's two path fields."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            screens = {stage: StageScreen(stage) for stage in STAGES}
            for screen in screens.values():
                app.push_screen(screen)
            config = ConfigScreen()
            app.push_screen(config)
            app.push_screen(QuickStartScreen())
            await pilot.pause()

            def check(root, prefix: str, specs) -> None:
                for spec in specs:
                    selector = f"#browse-{prefix}-{kebab(spec.key)}"
                    if _is_single_path(spec.hint):
                        button = root.query_one(selector, Button)
                        assert isinstance(button, BrowseButton), spec.key
                        assert not button.can_focus, spec.key
                        assert button.label == BROWSE_LABEL, spec.key
                    else:
                        with pytest.raises(NoMatches):
                            root.query_one(selector)

            check(config, "pipeline", run_level_specs())
            for stage in STAGES:
                check(screens[stage], stage, specs_of(STAGES[stage].options))
                check(config, f"cfg-{stage}", editable_specs(stage))
            # the owned lines show values, not widgets — no browse there
            for absent in (
                "#browse-cfg-segmentation-input",
                "#browse-cfg-segmentation-diff-dir",
                "#browse-cfg-analysis-raw-dir",
                "#browse-cfg-difference-raw-dir",
                "#browse-cfg-difference-reference",
            ):
                with pytest.raises(NoMatches):
                    config.query_one(absent)
            quick = app.screen
            assert isinstance(quick, QuickStartScreen)
            assert isinstance(quick.query_one("#browse-qs-raw-dir"), BrowseButton)
            assert isinstance(quick.query_one("#browse-qs-out-root"), BrowseButton)
            for absent in ("#browse-qs-reference", "#browse-qs-wells"):
                with pytest.raises(NoMatches):
                    quick.query_one(absent)

    asyncio.run(scenario())


def test_a_future_stage_gets_browse_buttons_for_its_path_fields(monkeypatch):
    """One registry entry adds the whole section — its Path field's browse
    button included (none on the non-path field)."""
    monkeypatch.setitem(
        STAGES, "train", StageSpec("train", "Train", FutureStageOptions)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            app.drafts["pipeline"].stages["train"] = {}
            app.push_screen(ConfigScreen())
            await pilot.pause()
            assert isinstance(
                app.screen.query_one("#browse-cfg-train-label-dir"), BrowseButton
            )
            with pytest.raises(NoMatches):
                app.screen.query_one("#browse-cfg-train-mode")

    asyncio.run(scenario())


def test_browse_pick_fills_the_input_and_the_draft(tmp_path):
    """Pressing a field's browse button opens the picker headlessly — the
    directory tree shows the start directory's entries — and picking one
    writes it into the input: the normal change flow stores it in the
    draft and enables the field's reset."""
    (target := tmp_path / "picked").mkdir()
    draft = ConfigDraft("difference")
    draft.set("difference", "out_dir", str(tmp_path))
    form = OptionForm("difference", specs_of(DifferenceOptions), draft)

    async def observe(app, form, pilot):
        out_dir = form.query_one("#difference-out-dir", Input)
        await pilot.pause()
        assert await pilot.click("#browse-difference-out-dir")
        await pilot.pause()
        assert isinstance(app.screen, SelectDirectory)
        await _pick_subdir(pilot, app, target)
        assert out_dir.value == str(target)
        assert draft.get("difference", "out_dir") == str(target)
        assert form.valid
        assert not form.query_one("#reset-difference-out-dir", Button).disabled

    _run_form(form, observe)


def test_browse_cancel_changes_nothing(tmp_path):
    """Escape closes the picker: the input and the draft keep their values,
    no error appears, and the field stays valid."""
    draft = ConfigDraft("difference")
    draft.set("difference", "out_dir", "/kept")
    form = OptionForm("difference", specs_of(DifferenceOptions), draft)

    async def observe(app, form, pilot):
        out_dir = form.query_one("#difference-out-dir", Input)
        await pilot.pause()
        assert out_dir.value == "/kept"
        assert await pilot.click("#browse-difference-out-dir")
        await pilot.pause()
        assert isinstance(app.screen, SelectDirectory)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SelectDirectory)
        assert out_dir.value == "/kept"
        assert draft.get("difference", "out_dir") == "/kept"
        assert form.error("out_dir") == ""
        assert form.valid

    _run_form(form, observe)


@pytest.mark.parametrize(
    ("stage", "key"),
    [
        ("difference", "raw_dir"),
        ("segmentation", "diff_dir"),
        ("analysis", "raw_dir"),
    ],
)
def test_stage_browse_picks_into_the_stages_own_draft(stage, key, tmp_path):
    """A stage screen's path fields browse into the stage's OWN draft: the
    pick lands in the field's input and — through the normal change flow —
    in that run type's section (its pipeline-owned ``diff_dir`` included:
    on a stage screen everything is ordinary and editable). The picker
    starts at the field's current value, and the chain's ``[pipeline]``
    draft stays untouched — each run type holds an independent config."""
    (target := tmp_path / "picked").mkdir()

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            field_id = f"{stage}-{kebab(key)}"
            field = screen.form.query_one(f"#{field_id}", Input)
            field.value = str(tmp_path)
            await pilot.pause()
            await _click(pilot, app, f"#browse-{field_id}")
            assert isinstance(app.screen, SelectDirectory)
            assert app.screen.query_one("DirectoryNavigation").location == tmp_path
            await _pick_subdir(pilot, app, target)
            assert app.screen is screen  # back on the stage screen
            assert screen.form.query_one(f"#{field_id}", Input).value == str(target)
            assert app.drafts[stage].get(stage, key) == str(target)
            # the chain's run level is untouched: the pick went only here
            assert app.drafts["pipeline"].document() == {"pipeline": {}}

    asyncio.run(scenario())


def test_quick_start_browse_picks_into_the_draft_and_gate(tmp_path):
    """The quick start's two path fields browse: a pick lands in the input
    and the chain's pipeline draft (the normal change flow), and the gate
    tracks it — still closed with only the raw directory picked, open once
    the out root is too; cancelling a later browse changes nothing."""
    (raw := tmp_path / "raw").mkdir()
    (wells := raw / "wells").mkdir()

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(QuickStartScreen())
            await pilot.pause()
            assert not app.screen.gate_open
            app.screen.query_one("#qs-raw-dir", Input).value = str(raw)
            await pilot.pause()
            await _click(pilot, app, "#browse-qs-raw-dir")
            assert isinstance(app.screen, SelectDirectory)
            assert app.screen.query_one("DirectoryNavigation").location == raw
            await _pick_subdir(pilot, app, wells)
            quick = app.screen
            assert isinstance(quick, QuickStartScreen)
            assert quick.query_one("#qs-raw-dir", Input).value == str(wells)
            assert app.drafts["pipeline"].get(RUN_LEVEL, "raw_dir") == str(wells)
            assert not quick.gate_open  # the out root is still missing
            # the out root: an empty input starts the picker at the working
            # directory; drive it to the tmp tree and select that
            await _click(pilot, app, "#browse-qs-out-root")
            nav = app.screen.query_one("DirectoryNavigation")
            nav.location = tmp_path
            await pilot.pause()
            assert await pilot.click("#select")
            await pilot.pause()
            assert app.drafts["pipeline"].get(RUN_LEVEL, "out_root") == str(tmp_path)
            assert quick.gate_open
            # a later browse, cancelled: nothing changes
            await _click(pilot, app, "#browse-qs-out-root")
            assert isinstance(app.screen, SelectDirectory)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, QuickStartScreen)
            assert app.drafts["pipeline"].get(RUN_LEVEL, "out_root") == str(tmp_path)

    asyncio.run(scenario())


def test_browse_buttons_stay_out_of_the_focus_walk():
    """The browse buttons are click-only like the resets: none appears in
    the screen's focus chain, so tab (and the arrow-key field walk) go
    field-to-field unchanged — tab from the raw-dir input lands on the
    next field, not on a button."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen("difference")
            app.push_screen(screen)
            await pilot.pause()
            buttons = list(app.screen.query(BrowseButton))
            assert buttons  # the path fields rendered theirs
            assert all(not button.can_focus for button in buttons)
            chain = app.screen.focus_chain
            assert not [w for w in chain if isinstance(w, BrowseButton)]
            raw_dir = screen.form.query_one("#difference-raw-dir", Input)
            raw_dir.focus()
            await pilot.press("tab")
            await pilot.pause()
            out_dir = screen.form.query_one("#difference-out-dir", Input)
            assert app.focused is out_dir

    asyncio.run(scenario())


def test_picker_location_walks_to_an_existing_directory(tmp_path):
    """The picker starts at the value's nearest existing directory: the
    value itself when it exists, its first existing ancestor otherwise,
    the working directory for an empty value."""
    deep = tmp_path / "not" / "yet"
    assert picker_location(str(tmp_path)) == tmp_path
    assert picker_location(str(deep)) == tmp_path
    assert picker_location("") == Path.cwd()
    assert picker_location("  ") == Path.cwd()
