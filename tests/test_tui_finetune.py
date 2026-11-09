"""Tests for the finetune track's TUI entry: the submenu and the prepare step.

Seams under test:

- navigation: the home menu's ``Finetune`` row opens the registered
  submenu, whose rows are ``Prepare data`` / ``Train decoder`` /
  ``Evaluate checkpoint`` — only the steps with a registered screen are
  live in this ticket (``Prepare data``);
- the prepare step screen: every field of
  :class:`~colony_tracking._options.PrepareOptions` renders as its widget
  with the declared gray, the actions gate on the prepare draft's own
  ``ConfigError`` (resolved through
  :mod:`colony_tracking.finetune.config`), *Write TOML…* suggests
  ``prepare.toml`` beside the h5 output directory, and *Load TOML…* fills
  the form back;
- the run flow: *Run this step* writes the TOML first, then runs
  ``run_prepare`` in-process with the live log; the summary block reports
  the prepare counters, and a summary carrying an ``error`` becomes the
  error panel with the CLI's exit-1 wording (``train prepare: …``).

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free). Assertions run
inside the headless session — the app's screen stack is gone once it
ends.
"""

import asyncio
import time
import tomllib
from pathlib import Path
from typing import get_type_hints

import pytest
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static

from colony_tracking._options import Options, PrepareOptions, _union_members
from colony_tracking.finetune import PrepareSummary
from colony_tracking.finetune.config import read_step_options
from colony_tracking.tui import runner
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import (
    FINETUNE_RUN_TYPES,
    TITLES,
    TOML_NAMES,
    ConfigDraft,
)
from colony_tracking.tui.forms import NO_DEFAULT, kebab, specs_of
from colony_tracking.tui.screens._save import LoadConfigDialog, SaveConfigDialog
from colony_tracking.tui.screens.finetune import ENTRIES, FinetuneScreen
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.run import RunScreen, open_step_run
from colony_tracking.tui.screens.step import STEPS, StepScreen, StepSpec

#: How long a test waits for a (fake, fast) run to finish.
_TIMEOUT = 10.0

#: The prepare step's required fields, in declaration order.
_MISSING_REQUIRED = "missing required key(s): ['image_dir', 'label_dir', 'out_dir']"

#: The required fields' example texts (the forms module's copy table).
_REQUIRED = {
    "image_dir": "/path/to/input-frames",
    "label_dir": "/path/to/label-images",
    "out_dir": "/path/to/output-dir",
}


def _prepare_options(tmp_path: Path) -> PrepareOptions:
    """A minimal resolved prepare config (its own flat document)."""
    return PrepareOptions(
        image_dir=tmp_path / "images",
        label_dir=tmp_path / "labels",
        out_dir=tmp_path / "h5",
    )


def _prepare_summary(**overrides) -> PrepareSummary:
    """A representative prepare summary: both splits, wells, and counters."""
    values = {
        "n_pairs": 4,
        "n_train": 3,
        "n_val": 1,
        "n_instances": 42,
        "train_wells": ["-A001", "-B002"],
        "val_wells": ["-C003"],
        "n_dropped": 2,
        "n_missing_images": 1,
    }
    values.update(overrides)
    return PrepareSummary(**values)


def _expected_widget(hint) -> type:
    """The widget kind for a hint — the test's own copy of the forms
    module's mapping."""
    members = _union_members(hint)
    if members is not None and len(members) == 1:
        return _expected_widget(members[0])
    if hint is bool:
        return Checkbox
    from typing import Literal, get_origin

    if get_origin(hint) is Literal:
        return Select
    return Input


def _gray(spec) -> str:
    """A field's expected gray text off the declared default alone."""
    if spec.default is NO_DEFAULT:
        return f"required: {_REQUIRED.get(spec.key, spec.key)}"
    if spec.default is None:
        return "(default: unset)"
    if isinstance(spec.default, (list, tuple)):
        return ", ".join(str(item) for item in spec.default)
    return str(spec.default)


def _log_text(screen: RunScreen) -> str:
    """The run screen's log pane text so far."""
    from textual.widgets import RichLog

    rich_log = screen.query_one("#run-log", RichLog)
    return "\n".join(strip.text for strip in rich_log.lines)


async def _await_run_screen(pilot, app: ColonyTuiApp) -> RunScreen:
    """The app's current screen once it is a run screen whose run started."""
    deadline = time.monotonic() + _TIMEOUT
    while (
        not isinstance(app.screen, RunScreen) or app.screen.handle is None
    ) and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert isinstance(app.screen, RunScreen)
    assert app.screen.handle is not None, "the run screen never started"
    return app.screen


async def _await_done(pilot, screen: RunScreen):
    """Wait for the screen's run to finish and return its handle."""
    deadline = time.monotonic() + _TIMEOUT
    handle = screen.handle
    while (
        handle is not None and not handle.done.is_set() and time.monotonic() < deadline
    ):
        await pilot.pause(0.05)
    handle = screen.handle
    assert handle is not None and handle.done.is_set(), "the run never finished"
    await pilot.pause()
    return handle


# --- navigation: home → Finetune → Prepare data -------------------------------------


def test_home_opens_the_finetune_submenu_with_the_prepare_row_live():
    """The home menu's Finetune row opens a registered submenu whose rows
    are the spec's three names; only the step with a registered screen is
    live in this ticket."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            home = app.screen
            assert isinstance(home, HomeScreen)
            menu = home.query_one("#home-menu", OptionList)
            assert not menu.get_option("finetune").disabled
            menu.focus()
            menu.highlighted = [
                option.id for option in menu.options
            ].index("finetune")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, FinetuneScreen)
            submenu = app.screen.query_one("#finetune-menu", OptionList)
            assert [option.prompt for option in submenu.options] == [
                "Prepare data",
                "Train decoder",
                "Evaluate checkpoint",
            ]
            assert not submenu.get_option("prepare").disabled
            assert submenu.get_option("train").disabled
            assert submenu.get_option("evaluate").disabled
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_prepare_row_opens_the_step_screen():
    """Selecting the live Prepare data row opens the prepare step screen,
    generated from the finetune step registry."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(FinetuneScreen())
            await pilot.pause()
            submenu = app.screen.query_one("#finetune-menu", OptionList)
            submenu.focus()
            submenu.highlighted = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, StepScreen)
            assert app.screen.step == "prepare"

    asyncio.run(scenario())


# --- the step registry --------------------------------------------------------------


def test_registry_covers_prepare():
    """One registry entry for the live step, pointing at the prepare
    options dataclass — the screen is generated, not hardcoded. Tickets
    03/04 extend this registry."""
    assert set(STEPS) == {"prepare"}
    assert STEPS["prepare"].options is PrepareOptions
    assert STEPS["prepare"].title == "Prepare data"


def test_the_finetune_registries_agree_on_the_live_steps():
    """The live finetune run types, the step-screen registry, the submenu
    rows, the titles/TOML names, and the runner kinds agree — so a step
    cannot be half-registered (the drift the repeatable pattern guards
    against)."""
    live = set(FINETUNE_RUN_TYPES)
    assert live == set(STEPS)
    assert live <= set(runner.KINDS)
    assert live <= set(TITLES)
    assert live <= set(TOML_NAMES)
    labels = dict(ENTRIES)
    assert live <= set(labels)
    assert labels["prepare"] == STEPS["prepare"].title
    assert TOML_NAMES["prepare"] == "prepare.toml"


def test_a_new_registry_entry_slots_in(monkeypatch):
    """A second registry entry renders through the same screen class; the
    submenu row becomes live without any per-step code."""
    from dataclasses import dataclass, field
    from typing import Literal

    @dataclass(kw_only=True)
    class FutureStepOptions(Options):
        mode: Literal["a", "b"] = field(default="a", metadata={"help": "Mode."})
        image_dir: Path = field(metadata={"help": "Input frames."})

    monkeypatch.setitem(
        STEPS,
        "evaluate",
        StepSpec("evaluate", "Evaluate checkpoint", FutureStepOptions),
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(FinetuneScreen())
            await pilot.pause()
            submenu = app.screen.query_one("#finetune-menu", OptionList)
            assert not submenu.get_option("evaluate").disabled

    asyncio.run(scenario())


# --- the prepare screen: every field, its gray, the gate ----------------------------


def test_the_prepare_screen_renders_every_option_field():
    """One widget per field of ``PrepareOptions``, in declaration order,
    with the form's ids prefixed by the step name."""
    hints = get_type_hints(PrepareOptions)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            shown = {
                widget.id: type(widget)
                for widget in screen.form.query("Input, Select, Checkbox")
            }
            assert set(shown) == {
                f"prepare-{kebab(key)}" for key in hints
            }
            for key, hint in hints.items():
                assert shown[f"prepare-{kebab(key)}"] is _expected_widget(hint), key

    asyncio.run(scenario())


def test_the_prepare_screen_grays_every_field():
    """Every widget shows what a blank resolves to, from the declared
    defaults: required examples for the three paths, the bare defaults for
    the split/filtering options, ``(default: unset)`` for ``val_wells``."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            for spec in specs_of(PrepareOptions):
                widget = screen.form.query_one(
                    f"#prepare-{kebab(spec.key)}"
                )
                if isinstance(widget, Checkbox):
                    continue
                shown = (
                    widget.prompt if isinstance(widget, Select) else widget.placeholder
                )
                assert shown == _gray(spec), spec.key
            assert (
                screen.form.query_one("#prepare-image-dir", Input).placeholder
                == "required: /path/to/input-frames"
            )
            assert (
                screen.form.query_one("#prepare-val-wells", Input).placeholder
                == "(default: unset)"
            )

    asyncio.run(scenario())


def test_the_gate_uses_the_prepare_drafts_exact_message():
    """The actions start disabled with the finetune config machinery's
    own missing-required wording; filling only the required fields opens
    them, and *Load TOML…* / *Back* are never gated."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            status = app.screen.query_one("#resolve-status", Static)
            assert str(status.render()) == _MISSING_REQUIRED
            assert app.screen.query_one("#run-step", Button).disabled
            assert app.screen.query_one("#write-toml", Button).disabled
            assert not app.screen.query_one("#load-toml", Button).disabled
            assert not app.screen.query_one("#back", Button).disabled
            await pilot.click("#write-toml")  # disabled: nothing happens
            await pilot.pause()
            assert not isinstance(app.screen, SaveConfigDialog)
            screen.form.query_one("#prepare-image-dir", Input).value = "/data/images"
            screen.form.query_one("#prepare-label-dir", Input).value = "/data/labels"
            await pilot.pause()
            assert not screen.gate_open  # out_dir still missing
            screen.form.query_one("#prepare-out-dir", Input).value = "/data/h5"
            await pilot.pause()
            assert screen.gate_open
            assert str(status.render()) == ""

    asyncio.run(scenario())


def test_an_unparseable_field_closes_the_gate():
    """A field that does not parse disables the actions with the fix-fields
    note (the field's error is inline); fixing it re-opens the gate."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#prepare-image-dir", Input).value = "/data/images"
            screen.form.query_one("#prepare-label-dir", Input).value = "/data/labels"
            screen.form.query_one("#prepare-out-dir", Input).value = "/data/h5"
            await pilot.pause()
            assert screen.gate_open
            fraction = screen.form.query_one("#prepare-val-fraction", Input)
            fraction.value = "many"
            await pilot.pause()
            assert not screen.gate_open
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == "Fix the marked fields to enable the actions."
            )
            assert fraction.has_class("invalid")
            fraction.value = "0.3"
            await pilot.pause()
            assert screen.gate_open
            assert app.drafts["prepare"].get("prepare", "val_fraction") == 0.3

    asyncio.run(scenario())


# --- write / load doors -------------------------------------------------------------


def test_write_toml_suggests_prepare_toml_beside_the_h5_directory(tmp_path):
    """*Write TOML…* suggests ``<out_dir parent>/prepare.toml`` and the
    written file is the flat document the CLI's ``train prepare --config``
    accepts."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["prepare"]
            draft.set("prepare", "image_dir", str(tmp_path / "images"))
            draft.set("prepare", "label_dir", str(tmp_path / "labels"))
            draft.set("prepare", "out_dir", str(tmp_path / "h5"))
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert (
                dialog.query_one("#toml-path", Input).value
                == str(tmp_path / "prepare.toml")
            )
            assert (
                dialog.query_one("#toml-path", Input).placeholder
                == "/path/to/prepare.toml"
            )
            await pilot.press("enter")
            await pilot.pause()
            path = tmp_path / "prepare.toml"
            assert path.exists()
            options = read_step_options("prepare", path)
            assert options == draft.resolve()

    asyncio.run(scenario())


def test_load_toml_fills_the_prepare_form_back(tmp_path):
    """A written prepare TOML loads back through *Load TOML…*: the draft
    is replaced by the file's flat document and the form shows its
    values."""
    path = tmp_path / "prepare.toml"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["prepare"]
            draft.set("prepare", "image_dir", str(tmp_path / "images"))
            draft.set("prepare", "label_dir", str(tmp_path / "labels"))
            draft.set("prepare", "out_dir", str(tmp_path / "h5"))
            draft.set("prepare", "val_fraction", 0.3)
            draft.write(path)
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#prepare-val-fraction", Input).value = "0.5"
            await pilot.pause()
            assert draft.get("prepare", "val_fraction") == 0.5
            await pilot.click("#load-toml")
            await pilot.pause()
            loader = app.screen
            assert isinstance(loader, LoadConfigDialog)
            assert (
                loader.query_one("#toml-path", Input).placeholder
                == "/path/to/prepare.toml"
            )
            loader.query_one("#toml-path", Input).value = str(path)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.screen is screen
            with open(path, "rb") as handle:
                file_doc = tomllib.load(handle)
            assert draft.document() == file_doc  # a load resets, never merges
            assert (
                screen.form.query_one("#prepare-val-fraction", Input).value
                == "0.3"
            )
            assert screen.gate_open

    asyncio.run(scenario())


# --- run flow: TOML first, live log, prepare summary --------------------------------


def test_run_this_step_writes_the_toml_then_runs_prepare_with_the_live_log(
    monkeypatch, tmp_path
):
    """*Run this step*: the shared dialog writes ``prepare.toml`` before
    the run; the run screen streams the prepare step's log records, runs
    with exactly the typed options, and shows the prepare counters."""
    toml_path = tmp_path / "prepare.toml"
    at_call: dict[str, object] = {}

    def recording_run_prepare(options):
        at_call["toml"] = (
            tomllib.loads(toml_path.read_text()) if toml_path.exists() else None
        )
        from colony_tracking import log

        log.info("finetune test: preparing pairs")
        return _prepare_summary()

    monkeypatch.setattr(runner, "run_prepare", recording_run_prepare)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("prepare")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#prepare-image-dir", Input).value = str(
                tmp_path / "images"
            )
            screen.form.query_one("#prepare-label-dir", Input).value = str(
                tmp_path / "labels"
            )
            screen.form.query_one("#prepare-out-dir", Input).value = str(
                tmp_path / "h5"
            )
            await pilot.pause()
            assert screen.gate_open
            await pilot.click("#run-step")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(toml_path)
            await pilot.press("enter")
            run_screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, run_screen)

            assert isinstance(handle.config, PrepareOptions)
            assert handle.config.image_dir == tmp_path / "images"
            assert handle.config.label_dir == tmp_path / "labels"
            assert handle.config.out_dir == tmp_path / "h5"
            assert run_screen.toml_path == toml_path
            assert "finetune test: preparing pairs" in _log_text(run_screen)
            assert "finished — ok" in str(
                run_screen.query_one("#run-status", Static).render()
            )
            block = str(run_screen.query_one("#run-summary", Static).render())
            assert "pairs 4" in block
            assert "train 3 (2 well(s))" in block
            assert "val 1 (1 well(s))" in block
            assert "instances 42" in block
            assert "dropped labels 2" in block
            assert "labels without input frame 1" in block
            assert "train wells: -A001, -B002" in block
            assert "val wells: -C003" in block
            assert not run_screen.query_one("#run-error", Static).display

    asyncio.run(scenario())
    assert at_call["toml"] is not None  # written before the run started
    written = read_step_options("prepare", toml_path)
    assert written.image_dir == tmp_path / "images"


def test_a_prepare_summary_error_becomes_the_error_panel_with_the_cli_wording(
    monkeypatch, tmp_path
):
    """A summary carrying ``error`` is a failed run: the panel shows the
    CLI's exit-1 line (``train prepare: …``), the runner logged the same
    line, and the app stays alive."""
    summary = PrepareSummary(error="no such label directory: /nowhere")
    monkeypatch.setattr(runner, "run_prepare", lambda options: summary)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(RunScreen("prepare", _prepare_options(tmp_path)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            await _await_done(pilot, screen)
            line = "train prepare: no such label directory: /nowhere"
            assert line in _log_text(screen)  # the runner logged it
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            assert line in str(panel.render())
            status = str(screen.query_one("#run-status", Static).render())
            assert "prepare finished with failures" in status
            assert app.screen is screen  # the app survived

    asyncio.run(scenario())


def test_open_step_run_reports_the_prepare_drafts_invalid_message(tmp_path):
    """``open_step_run``'s own config check is the prepare draft's
    ``resolve`` — the finetune machinery's wording, reported in the UI,
    and nothing is pushed or written."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test() as pilot:
            message = open_step_run(app, ConfigDraft("prepare"), "prepare")
            assert message == _MISSING_REQUIRED
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)
            assert not (tmp_path / "prepare.toml").exists()

    asyncio.run(scenario())


def test_open_step_run_rejects_steps_without_a_live_screen():
    """``train`` / ``evaluate`` are registered finetune configs but not
    live TUI run types yet: passing one is a programming error up front,
    not a save dialog followed by an unknown run kind."""
    with pytest.raises(ValueError, match="unknown finetune step 'train'"):
        open_step_run(ColonyTuiApp(), ConfigDraft("prepare"), "train")
