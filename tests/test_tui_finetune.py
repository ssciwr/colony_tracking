"""Tests for the finetune track's TUI entry: the submenu and the prepare step.

Seams under test:

- navigation: the home menu's ``Finetune`` row opens the registered
  submenu, whose rows are ``Prepare data`` / ``Train decoder`` /
  ``Evaluate checkpoint`` — every step with a registered screen is live;
- the prepare, train, and evaluate step screens: every field of
  :class:`~colony_tracking._options.PrepareOptions` /
  :class:`~colony_tracking._options.TrainOptions` /
  :class:`~colony_tracking._options.EvaluateOptions` renders as its
  widget with the declared gray, the actions gate on the draft's own
  ``ConfigError`` (resolved through
  :mod:`colony_tracking.finetune.config`), *Write TOML…* suggests the
  step's TOML beside its output (evaluate: beside the JSON report), and
  *Load TOML…* fills the form back;
- the run flow: *Run this step* writes the TOML first, then runs the
  step in-process with the live log; a finetune step's printed output and
  tqdm repaints (stdout/stderr) reach the pane as plain unprefixed lines,
  collapsing every carriage-return repaint to its last segment — even
  when the writer flushes between repaints, as tqdm does — while the
  stage path keeps its logger-only routing; the summary blocks
  report the step's numbers, and a summary carrying an ``error`` becomes
  the error panel with the CLI's exit-1 wording (``train prepare: …`` /
  ``train train: …`` / ``train evaluate: …``).

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free). Assertions run
inside the headless session — the app's screen stack is gone once it
ends.
"""

import asyncio
import sys
import time
import tomllib
from pathlib import Path
from typing import get_type_hints

import pytest
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static

from colony_tracking import DifferenceOptions, DifferenceSummary, log
from colony_tracking._options import (
    EvaluateOptions,
    Options,
    PrepareOptions,
    TrainOptions,
    _union_members,
)
from colony_tracking.finetune import EvaluateSummary, PrepareSummary, TrainSummary
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
from colony_tracking.tui.records import deliverer, output_capture
from colony_tracking.tui.screens._save import LoadConfigDialog, SaveConfigDialog
from colony_tracking.tui.screens.finetune import ENTRIES, FinetuneScreen
from colony_tracking.tui.screens.home import HomeScreen
from colony_tracking.tui.screens.run import RunScreen, open_step_run
from colony_tracking.tui.screens.step import STEPS, StepScreen, StepSpec

#: How long a test waits for a (fake, fast) run to finish.
_TIMEOUT = 10.0

#: The prepare step's required fields, in declaration order.
_MISSING_REQUIRED = "missing required key(s): ['image_dir', 'label_dir', 'out_dir']"

#: The train step's required fields, in declaration order.
_TRAIN_MISSING_REQUIRED = "missing required key(s): ['data_dir', 'out_dir']"

#: The evaluate step's required fields, in declaration order.
_EVALUATE_MISSING_REQUIRED = "missing required key(s): ['data_dir']"

#: The required fields' example texts (the forms module's copy table).
_REQUIRED = {
    "image_dir": "/path/to/input-frames",
    "label_dir": "/path/to/label-images",
    "out_dir": "/path/to/output-dir",
    "data_dir": "/path/to/h5-dataset",
}


def _prepare_options(tmp_path: Path) -> PrepareOptions:
    """A minimal resolved prepare config (its own flat document)."""
    return PrepareOptions(
        image_dir=tmp_path / "images",
        label_dir=tmp_path / "labels",
        out_dir=tmp_path / "h5",
    )


def _train_options(tmp_path: Path) -> TrainOptions:
    """A minimal resolved train config (its own flat document)."""
    return TrainOptions(
        data_dir=tmp_path / "h5",
        out_dir=tmp_path / "train_out",
    )


def _evaluate_options(tmp_path: Path) -> EvaluateOptions:
    """A minimal resolved evaluate config (its own flat document)."""
    return EvaluateOptions(data_dir=tmp_path / "h5")


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


def _train_summary(**overrides) -> TrainSummary:
    """A representative train summary: both splits, smoke shapes, ckpt."""
    values = {
        "smoke_test": True,
        "smoke_raw_shape": (1, 3, 8, 8),
        "smoke_label_shape": (1, 1, 8, 8),
        "n_train_images": 4,
        "n_val_images": 2,
        "checkpoint": Path("/models/sam_colony/best.pt"),
    }
    values.update(overrides)
    return TrainSummary(**values)


def _evaluate_summary(**overrides) -> EvaluateSummary:
    """A representative evaluate summary: the benchmark's own numbers."""
    values = {
        "n_images": 6,
        "n_instances": 120,
        "mean_iou": 0.712,
        "median_iou": 0.804,
        "iou_at_05": 0.833,
        "iou_at_075": 0.625,
        "iou_at_09": 0.417,
        "n_clipped": 3,
        "split": "val",
        "checkpoint": Path("/models/sam_colony/best.pt"),
    }
    values.update(overrides)
    return EvaluateSummary(**values)


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


def test_home_opens_the_finetune_submenu_with_the_live_rows_enabled():
    """The home menu's Finetune row opens a registered submenu whose rows
    are the spec's three names, all live."""

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
            assert not submenu.get_option("train").disabled
            assert not submenu.get_option("evaluate").disabled
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


def test_train_row_opens_the_step_screen():
    """Selecting the live Train decoder row opens the train step screen,
    generated from the same registry-driven screen class."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(FinetuneScreen())
            await pilot.pause()
            submenu = app.screen.query_one("#finetune-menu", OptionList)
            submenu.focus()
            submenu.highlighted = 1
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, StepScreen)
            assert app.screen.step == "train"

    asyncio.run(scenario())


def test_evaluate_row_opens_the_step_screen():
    """Selecting the live Evaluate checkpoint row opens the evaluate step
    screen, generated from the same registry-driven screen class."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(FinetuneScreen())
            await pilot.pause()
            submenu = app.screen.query_one("#finetune-menu", OptionList)
            submenu.focus()
            submenu.highlighted = 2
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, StepScreen)
            assert app.screen.step == "evaluate"

    asyncio.run(scenario())


# --- the step registry --------------------------------------------------------------


def test_registry_covers_the_live_steps():
    """One registry entry per live step, pointing at its options dataclass
    — the screens are generated, not hardcoded."""
    assert set(STEPS) == {"prepare", "train", "evaluate"}
    assert STEPS["prepare"].options is PrepareOptions
    assert STEPS["prepare"].title == "Prepare data"
    assert STEPS["train"].options is TrainOptions
    assert STEPS["train"].title == "Train decoder"
    assert STEPS["evaluate"].options is EvaluateOptions
    assert STEPS["evaluate"].title == "Evaluate checkpoint"


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
    for step in live:
        assert labels[step] == STEPS[step].title
        assert TOML_NAMES[step] == f"{step}.toml"


def test_a_new_registry_entry_slots_in(monkeypatch):
    """A new registry entry renders through the same screen class; the
    submenu row becomes live without any per-step code."""
    from dataclasses import dataclass, field
    from typing import Literal

    @dataclass(kw_only=True)
    class FutureStepOptions(Options):
        mode: Literal["a", "b"] = field(default="a", metadata={"help": "Mode."})
        image_dir: Path = field(metadata={"help": "Input frames."})

    monkeypatch.setitem(
        STEPS,
        "future",
        StepSpec("future", "Future step", FutureStepOptions),
    )
    monkeypatch.setattr(
        "colony_tracking.tui.screens.finetune.ENTRIES",
        ENTRIES + (("future", "Future step"),),
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(FinetuneScreen())
            await pilot.pause()
            submenu = app.screen.query_one("#finetune-menu", OptionList)
            assert not submenu.get_option("future").disabled

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


# --- the train screen: every field, the gate, the resume note -----------------------


def test_the_train_screen_renders_every_option_field():
    """One widget per field of ``TrainOptions``, in declaration order,
    with the form's ids prefixed by the step name."""
    hints = get_type_hints(TrainOptions)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("train")
            app.push_screen(screen)
            await pilot.pause()
            shown = {
                widget.id: type(widget)
                for widget in screen.form.query("Input, Select, Checkbox")
            }
            assert set(shown) == {f"train-{kebab(key)}" for key in hints}
            for key, hint in hints.items():
                assert shown[f"train-{kebab(key)}"] is _expected_widget(hint), key

    asyncio.run(scenario())


def test_the_train_gate_uses_the_train_drafts_exact_message():
    """The actions start disabled with the finetune machinery's own
    missing-required wording for the train document; filling only the two
    required directories opens them."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("train")
            app.push_screen(screen)
            await pilot.pause()
            status = app.screen.query_one("#resolve-status", Static)
            assert str(status.render()) == _TRAIN_MISSING_REQUIRED
            assert app.screen.query_one("#run-step", Button).disabled
            assert app.screen.query_one("#write-toml", Button).disabled
            screen.form.query_one("#train-data-dir", Input).value = "/data/h5"
            await pilot.pause()
            assert not screen.gate_open
            screen.form.query_one("#train-out-dir", Input).value = "/data/train"
            await pilot.pause()
            assert screen.gate_open
            assert str(status.render()) == ""

    asyncio.run(scenario())


def test_the_train_screen_explains_resume_and_has_no_start_over_control():
    """The train screen states the CLI's resume behaviour: the same output
    directory resumes, or returns cleanly when training already finished.
    There is no confirmation prompt and no start-over action."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("train")
            app.push_screen(screen)
            await pilot.pause()
            note = str(screen.query_one("#train-note", Static).render())
            assert "resume" in note.lower()
            assert "already reached" in note.lower()
            assert "n_epochs" in note
            labels = [
                str(button.label).lower()
                for button in screen.query(Button)
            ]
            assert not any("start over" in label for label in labels)
            assert not any("overwrite" in label for label in labels)

    asyncio.run(scenario())


# --- the evaluate screen: every field, its gray, the gate ---------------------------


def test_the_evaluate_screen_renders_every_option_field():
    """One widget per field of ``EvaluateOptions``, in declaration order,
    with the form's ids prefixed by the step name."""
    hints = get_type_hints(EvaluateOptions)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            shown = {
                widget.id: type(widget)
                for widget in screen.form.query("Input, Select, Checkbox")
            }
            assert set(shown) == {f"evaluate-{kebab(key)}" for key in hints}
            for key, hint in hints.items():
                assert shown[f"evaluate-{kebab(key)}"] is _expected_widget(hint), key

    asyncio.run(scenario())


def test_the_evaluate_screen_grays_every_field():
    """Every widget shows what a blank resolves to from the declared
    defaults: ``split`` is ``val``, the two files and the caps are
    ``(default: unset)``, ``data_dir`` is the required example, and the
    rest are the bare defaults."""
    hints = get_type_hints(EvaluateOptions)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            for spec in specs_of(EvaluateOptions):
                widget = screen.form.query_one(f"#evaluate-{kebab(spec.key)}")
                if isinstance(widget, Checkbox):
                    continue
                shown = (
                    widget.prompt if isinstance(widget, Select) else widget.placeholder
                )
                assert shown == _gray(spec), spec.key
            assert (
                screen.form.query_one("#evaluate-data-dir", Input).placeholder
                == "required: /path/to/h5-dataset"
            )
            assert (
                screen.form.query_one("#evaluate-split", Input).placeholder
                == "val"
            )
            assert (
                screen.form.query_one("#evaluate-checkpoint", Input).placeholder
                == "(default: unset)"
            )
            assert (
                screen.form.query_one("#evaluate-output", Input).placeholder
                == "(default: unset)"
            )
            assert set(hints) == {spec.key for spec in specs_of(EvaluateOptions)}

    asyncio.run(scenario())


def test_the_evaluate_gate_uses_the_evaluate_drafts_exact_message():
    """The actions start disabled with the finetune machinery's own
    missing-required wording for the evaluate document (``data_dir``
    alone); filling it opens them."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            status = app.screen.query_one("#resolve-status", Static)
            assert str(status.render()) == _EVALUATE_MISSING_REQUIRED
            assert app.screen.query_one("#run-step", Button).disabled
            assert app.screen.query_one("#write-toml", Button).disabled
            assert not app.screen.query_one("#load-toml", Button).disabled
            screen.form.query_one("#evaluate-data-dir", Input).value = "/data/h5"
            await pilot.pause()
            assert screen.gate_open
            assert str(status.render()) == ""

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


def test_write_toml_suggests_train_toml_beside_the_checkpoint_directory(tmp_path):
    """*Write TOML…* suggests ``<out_dir parent>/train.toml`` — the
    checkpoint/log directory's neighbour — and the file is the flat
    document the CLI's ``train train --config`` accepts."""
    out_dir = tmp_path / "run" / "checkpoints"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["train"]
            draft.set("train", "data_dir", str(tmp_path / "h5"))
            draft.set("train", "out_dir", str(out_dir))
            screen = StepScreen("train")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            expected = out_dir.parent / "train.toml"
            assert dialog.query_one("#toml-path", Input).value == str(expected)
            assert (
                dialog.query_one("#toml-path", Input).placeholder
                == "/path/to/train.toml"
            )
            await pilot.press("enter")
            await pilot.pause()
            assert expected.exists()
            options = read_step_options("train", expected)
            assert isinstance(options, TrainOptions)
            assert options == draft.resolve()

    asyncio.run(scenario())


def test_write_toml_suggests_evaluate_toml_beside_the_report(tmp_path):
    """*Write TOML…* suggests ``<report parent>/evaluate.toml`` — the JSON
    report's neighbour — and the file is the flat document the CLI's
    ``train evaluate --config`` accepts."""
    report = tmp_path / "reports" / "bench.json"
    expected = report.parent / "evaluate.toml"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["evaluate"]
            draft.set("evaluate", "data_dir", str(tmp_path / "h5"))
            draft.set("evaluate", "output", str(report))
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(expected)
            assert (
                dialog.query_one("#toml-path", Input).placeholder
                == "/path/to/evaluate.toml"
            )
            await pilot.press("enter")
            await pilot.pause()
            assert expected.exists()
            options = read_step_options("evaluate", expected)
            assert isinstance(options, EvaluateOptions)
            assert options == draft.resolve()

    asyncio.run(scenario())


def test_write_toml_starts_empty_without_an_evaluate_report(tmp_path):
    """Without a report path there is no output to sit beside: the save
    dialog starts empty (the user types a path), even with ``data_dir``
    set."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["evaluate"]
            draft.set("evaluate", "data_dir", str(tmp_path / "h5"))
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == ""
            assert (
                dialog.query_one("#toml-path", Input).placeholder
                == "/path/to/evaluate.toml"
            )
            # a directory-shaped report path names no file to sit beside
            await pilot.press("escape")
            await pilot.pause()
            for directory in (str(tmp_path), str(tmp_path / "not-yet") + "/"):
                draft.set("evaluate", "output", directory)
                await pilot.click("#write-toml")
                await pilot.pause()
                dialog = app.screen
                assert isinstance(dialog, SaveConfigDialog)
                assert dialog.query_one("#toml-path", Input).value == ""
                await pilot.press("escape")
                await pilot.pause()

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


def test_load_toml_fills_the_train_form_back(tmp_path):
    """A written train TOML loads back through *Load TOML…*: the draft is
    replaced by the file's flat document and the form shows its values."""
    path = tmp_path / "train.toml"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["train"]
            draft.set("train", "data_dir", str(tmp_path / "h5"))
            draft.set("train", "out_dir", str(tmp_path / "train_out"))
            draft.set("train", "n_epochs", 7)
            draft.write(path)
            screen = StepScreen("train")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#train-n-epochs", Input).value = "99"
            await pilot.pause()
            assert draft.get("train", "n_epochs") == 99
            await pilot.click("#load-toml")
            await pilot.pause()
            loader = app.screen
            assert isinstance(loader, LoadConfigDialog)
            loader.query_one("#toml-path", Input).value = str(path)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.screen is screen
            with open(path, "rb") as handle:
                file_doc = tomllib.load(handle)
            assert draft.document() == file_doc  # a load resets, never merges
            assert screen.form.query_one("#train-n-epochs", Input).value == "7"
            assert screen.gate_open

    asyncio.run(scenario())


def test_load_toml_fills_the_evaluate_form_back(tmp_path):
    """A written evaluate TOML loads back through *Load TOML…*: the draft
    is replaced by the file's flat document and the form shows its
    values."""
    path = tmp_path / "evaluate.toml"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["evaluate"]
            draft.set("evaluate", "data_dir", str(tmp_path / "h5"))
            draft.set("evaluate", "split", "train")
            draft.set("evaluate", "n_max_images", 3)
            draft.write(path)
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#evaluate-n-max-images", Input).value = "9"
            await pilot.pause()
            assert draft.get("evaluate", "n_max_images") == 9
            await pilot.click("#load-toml")
            await pilot.pause()
            loader = app.screen
            assert isinstance(loader, LoadConfigDialog)
            loader.query_one("#toml-path", Input).value = str(path)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.screen is screen
            with open(path, "rb") as handle:
                file_doc = tomllib.load(handle)
            assert draft.document() == file_doc  # a load resets, never merges
            assert screen.form.query_one("#evaluate-n-max-images", Input).value == "3"
            assert screen.form.query_one("#evaluate-split", Input).value == "train"
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


def test_run_this_step_writes_the_toml_then_runs_train_with_the_live_log(
    monkeypatch, tmp_path
):
    """*Run this step* on the train screen: ``train.toml`` is written
    first; the run screen streams both the package logger and the step's
    own stdout/stderr, collapsing one write's carriage-return repaints to
    their last segment, and shows the train counters, smoke shapes, and
    checkpoint."""
    toml_path = tmp_path / "train.toml"
    at_call: dict[str, object] = {}

    def recording_run_train(options):
        at_call["toml"] = (
            tomllib.loads(toml_path.read_text()) if toml_path.exists() else None
        )
        log.info("finetune test: training")
        sys.stdout.write("loading dataset...\n")
        # The shape real tqdm writes: one repaint per write, then a
        # newline when the bar closes — only the last segment may reach
        # the pane.
        for percent, bar in ((10, "#"), (20, "##"), (30, "###")):
            sys.stdout.write(f"\r {percent}%|{bar} |")
            sys.stdout.flush()  # as tqdm does after every repaint
        sys.stdout.write("\n")
        sys.stderr.write("val loader ready\n")
        return _train_summary()

    monkeypatch.setattr(runner, "run_train", recording_run_train)

    async def scenario() -> None:
        from textual.widgets import RichLog

        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("train")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#train-data-dir", Input).value = str(
                tmp_path / "h5"
            )
            screen.form.query_one("#train-out-dir", Input).value = str(
                tmp_path / "train_out"
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

            assert isinstance(handle.config, TrainOptions)
            assert handle.config.data_dir == tmp_path / "h5"
            assert handle.config.out_dir == tmp_path / "train_out"
            assert run_screen.toml_path == toml_path
            lines = [
                strip.text
                for strip in run_screen.query_one("#run-log", RichLog).lines
            ]
            assert "finetune test: training" in "\n".join(lines)
            # printed stdout/stderr lines are plain: no level tag prefix
            assert "loading dataset..." in lines
            assert "val loader ready" in lines
            assert " 10%|# |" in lines  # immediate first snapshot
            assert not any("20%" in line for line in lines)  # throttled
            assert " 30%|### |" in lines  # final frame at the newline
            assert "finished — ok" in str(
                run_screen.query_one("#run-status", Static).render()
            )
            block = str(run_screen.query_one("#run-summary", Static).render())
            assert "train 4 images" in block
            assert "val 2 images" in block
            assert "smoke test" in block
            assert "raw (1, 3, 8, 8)" in block
            assert "label (1, 1, 8, 8)" in block
            assert "checkpoint: /models/sam_colony/best.pt" in block
            assert not run_screen.query_one("#run-error", Static).display

    asyncio.run(scenario())
    assert at_call["toml"] is not None  # written before the run started
    written = read_step_options("train", toml_path)
    assert written.data_dir == tmp_path / "h5"


def test_a_train_summary_error_becomes_the_error_panel_with_the_cli_wording(
    monkeypatch, tmp_path
):
    """A train summary carrying ``error`` is a failed run: the panel shows
    the CLI's exit-1 line (``train train: …``), the runner logged the same
    line, and the app stays alive."""
    message = "no training files in /nowhere/train"
    monkeypatch.setattr(
        runner, "run_train", lambda options: TrainSummary(error=message)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(RunScreen("train", _train_options(tmp_path)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            await _await_done(pilot, screen)
            line = f"train train: {message}"
            assert line in _log_text(screen)  # the runner logged it
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            assert line in str(panel.render())
            status = str(screen.query_one("#run-status", Static).render())
            assert "train finished with failures" in status
            assert app.screen is screen  # the app survived

    asyncio.run(scenario())


def test_run_this_step_writes_the_toml_then_runs_evaluate_with_the_live_log(
    monkeypatch, tmp_path
):
    """*Run this step* on the evaluate screen: ``evaluate.toml`` is written
    first; the run screen streams the step's log and shows the benchmark's
    numbers — images, instances, mean/median IoU, the threshold fractions,
    the clipped count, and the checkpoint."""
    report = tmp_path / "reports" / "bench.json"
    toml_path = report.parent / "evaluate.toml"
    at_call: dict[str, object] = {}

    def recording_run_evaluate(options):
        at_call["toml"] = (
            tomllib.loads(toml_path.read_text()) if toml_path.exists() else None
        )
        log.info("finetune test: scoring the val split")
        return _evaluate_summary()

    monkeypatch.setattr(runner, "run_evaluate", recording_run_evaluate)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StepScreen("evaluate")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#evaluate-data-dir", Input).value = str(
                tmp_path / "h5"
            )
            screen.form.query_one("#evaluate-output", Input).value = str(report)
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

            assert isinstance(handle.config, EvaluateOptions)
            assert handle.config.data_dir == tmp_path / "h5"
            assert handle.config.output == report
            assert run_screen.toml_path == toml_path
            assert "finetune test: scoring the val split" in _log_text(run_screen)
            assert "finished — ok" in str(
                run_screen.query_one("#run-status", Static).render()
            )
            block = str(run_screen.query_one("#run-summary", Static).render())
            assert "images 6" in block
            assert "instances 120" in block
            assert "mean IoU 0.712" in block
            assert "median IoU 0.804" in block
            assert "IoU>=0.5 0.833" in block
            assert "IoU>=0.75 0.625" in block
            assert "IoU>=0.9 0.417" in block
            assert "larger than a tile 3" in block
            assert "checkpoint: /models/sam_colony/best.pt" in block
            assert not run_screen.query_one("#run-error", Static).display

    asyncio.run(scenario())
    assert at_call["toml"] is not None  # written before the run started
    written = read_step_options("evaluate", toml_path)
    assert written.data_dir == tmp_path / "h5"
    assert written.output == report


def test_an_evaluate_summary_error_becomes_the_error_panel_with_the_cli_wording(
    monkeypatch, tmp_path
):
    """An evaluate summary carrying ``error`` is a failed run: the panel
    shows the CLI's exit-1 line (``train evaluate: …``), the runner logged
    the same line, and the app stays alive."""
    message = "no images in /nowhere/val — run train prepare first"
    monkeypatch.setattr(
        runner, "run_evaluate", lambda options: EvaluateSummary(error=message)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.push_screen(RunScreen("evaluate", _evaluate_options(tmp_path)))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunScreen)
            await _await_done(pilot, screen)
            line = f"train evaluate: {message}"
            assert line in _log_text(screen)  # the runner logged it
            panel = screen.query_one("#run-error", Static)
            assert panel.display
            assert line in str(panel.render())
            status = str(screen.query_one("#run-status", Static).render())
            assert "evaluate finished with failures" in status
            assert app.screen is screen  # the app survived

    asyncio.run(scenario())


def test_printed_output_capture_is_finetune_only(monkeypatch, tmp_path):
    """The stdout/stderr capture is a finetune-step behaviour: a stage run
    keeps the logger-only routing (its ``sys.stdout`` untouched, no output
    callback), while a train run forwards printed lines and restores the
    streams afterwards."""
    original_stdout = sys.stdout
    seen: dict[str, object] = {}

    def fake_difference(options):
        seen["difference_stdout"] = sys.stdout
        print("difference printed line")
        log.info("difference logged line")
        return DifferenceSummary(wells={}, processed=1, skipped=0, failed=0)

    monkeypatch.setattr(runner, "run_difference", fake_difference)
    stage_output: list[str] = []
    stage_records = []
    stage = runner.start(
        "difference",
        DifferenceOptions(raw_dir=tmp_path / "raw", out_dir=tmp_path / "diff"),
        on_done=lambda handle: None,
        on_record=stage_records.append,
        on_output=stage_output.append,
    )
    assert stage.done.wait(_TIMEOUT)
    assert seen["difference_stdout"] is original_stdout
    assert stage_output == []
    assert "difference logged line" in [r.getMessage() for r in stage_records]
    assert sys.stdout is original_stdout

    def fake_train(options):
        seen["train_stdout"] = sys.stdout
        print("train printed line")
        return TrainSummary()

    monkeypatch.setattr(runner, "run_train", fake_train)
    train_output: list[str] = []
    train = runner.start(
        "train",
        _train_options(tmp_path),
        on_done=lambda handle: None,
        on_output=train_output.append,
    )
    assert train.done.wait(_TIMEOUT)
    assert seen["train_stdout"] is not original_stdout
    assert train_output == ["train printed line"]
    assert sys.stdout is original_stdout  # restored after the run


def test_stream_flushes_do_not_end_a_line():
    """tqdm flushes after every repaint write; a flush is not a line
    boundary. The first frame is the immediate snapshot, the rest are
    throttled, and end-of-run emits the last frame exactly once."""
    lines: list[str] = []

    def on_output(line: str) -> None:
        lines.append(line)

    with output_capture(deliverer(None), on_output):
        for percent in (10, 50, 100):
            sys.stdout.write(f"\r{percent}%")
            sys.stdout.flush()

    assert lines == ["10%", "100%"]


def test_a_tqdm_like_write_flush_sequence_ending_in_a_newline_is_one_line():
    """The real bar's shape: repaint, flush, repaint, flush, … and a final
    newline when it closes. The pane sees the immediate first snapshot
    and the closed line, not one line per repaint."""
    lines: list[str] = []

    def on_output(line: str) -> None:
        lines.append(line)

    with output_capture(deliverer(None), on_output):
        for percent in (10, 50, 100):
            sys.stdout.write(f"\r{percent}%|{'#' * percent} |")
            sys.stdout.flush()
        sys.stdout.write("\n")

    assert lines == [
        "10%|" + "#" * 10 + " |",
        "100%|" + "#" * 100 + " |",
    ]


def test_repaint_snapshots_are_throttled_to_the_interval():
    """Repaints are snapshots, not lines: one is emitted immediately, then
    only after ``min_interval`` has elapsed. The final frame always comes
    out at the end."""
    from colony_tracking.tui.records import _LineWriter

    ticks = iter([0.0, 1.0, 11.0])
    lines: list[str] = []
    writer = _LineWriter(
        lines.append, min_interval=10.0, clock=lambda: next(ticks)
    )
    writer.write("\r0%\r50%\r75%\r100%")
    writer.finish()

    assert lines == ["0%", "75%", "100%"]


def test_a_crlf_split_across_writes_is_still_a_line_end():
    """A CRLF can arrive as ``"hello\\r"`` then ``"\\n"``; the line still
    ends and is emitted once (a trailing CR is deferred, not consumed)."""
    lines: list[str] = []

    def on_output(line: str) -> None:
        lines.append(line)

    with output_capture(deliverer(None), on_output):
        sys.stdout.write("hello\r")
        sys.stdout.flush()
        sys.stdout.write("\n")

    assert lines == ["hello"]


def test_a_real_tqdm_bar_becomes_a_bounded_number_of_lines():
    """A real tqdm loop cannot flood the pane: with a repaint per step it
    still emits only a small bounded number of lines, the last at 100%."""
    import tqdm

    lines: list[str] = []

    def on_output(line: str) -> None:
        lines.append(line)

    with output_capture(deliverer(None), on_output):
        for _ in tqdm.tqdm(range(6), miniters=1, mininterval=0, file=None):
            pass

    assert 1 <= len(lines) <= 2, lines
    assert "100%" in lines[-1]


def test_a_real_tqdm_bar_announces_progress_before_it_closes():
    """A long training run must show the bar while it is still running:
    tqdm ends its line only when the bar closes (torch_em never closes it
    before ``fit`` returns), so waiting for a newline leaves the pane
    silent for the whole epoch. Progress must surface as periodic
    snapshots between repaints."""
    import tqdm

    lines: list[str] = []

    def on_output(line: str) -> None:
        lines.append(line)

    with output_capture(deliverer(None), on_output):
        # torch_em drives the bar with ``update`` (it iterates its data
        # loader), so tqdm's iterator auto-close never runs; the bar is
        # still open here, exactly like during ``fit``.
        bar = tqdm.tqdm(total=6, miniters=1, mininterval=0, file=None)
        for _ in range(6):
            bar.update(1)
        assert lines, "no progress reached the pane during the run"
        assert "0%" in lines[0]

    # The stream's end emits the last frame, even though torch_em never
    # closes the bar itself.
    assert "100%" in lines[-1]


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


def test_open_step_run_rejects_unregistered_steps():
    """A step name that is not a live TUI run type is a programming error
    up front, not a save dialog followed by an unknown run kind."""
    with pytest.raises(ValueError, match="unknown finetune step 'quantify'"):
        open_step_run(ColonyTuiApp(), ConfigDraft("prepare"), "quantify")
