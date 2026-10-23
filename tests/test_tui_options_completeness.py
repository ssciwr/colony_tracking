"""Tests for the completeness story: every option visible, editable on
its own run type (the independent-configs redesign; spec story 36 and
§2 of the brief).

A run type's config is fully explicit, so its stage screen shows EVERY
field of its options dataclass as its ordinary widget (a parameterized
check, nothing excluded). The pipeline chain keeps its sectioned document
and its config screen: the ``[pipeline]`` run level, one form per stage
section with the stage's non-owned fields as widgets (each with its own
reset button — enabled only while its key is set), and the pipeline-owned
fields shown read-only on the ``wired within the chain:`` lines, every
entry marked ``(derived)`` — the within-run wiring a chain run is, no
longer styled as run-level links. The fields that default to a derived
name still grey that default in their empty inputs (the chain's own
within-run wiring).

The headline: a difference run starts from its stage screen alone — its
own raw_dir and out_dir typed, *Run this stage*, the run type's own
``difference.toml`` suggested and written, the stage's resolved options
driving the run and nothing derived. Also the config-file doors both
ways: a written stage TOML loads back into the stage screen's form (and a
written ``pipeline.toml`` into the config screen's run-level form).

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free). Assertions run
inside the headless session — the app's screen stack is gone once it
ends. Buttons may sit below the test viewport's fold (the config
screen's sections scroll), so those clicks go through a helper that
scrolls the widget into view first.
"""

import asyncio
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_origin

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Static

from colony_tracking import DifferenceSummary
from colony_tracking._options import DifferenceOptions, Options, _union_members
from colony_tracking.pipeline import _OWNED, read_config
from colony_tracking.tui import runner
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.forms import kebab, run_level_specs, specs_of
from colony_tracking.tui.screens._save import LoadConfigDialog, SaveConfigDialog
from colony_tracking.tui.screens.config import ConfigScreen, editable_specs
from colony_tracking.tui.screens.run import RunScreen
from colony_tracking.tui.screens.stage import STAGES, StageSpec, StageScreen

#: How long a test waits for a (fake) run to finish.
_TIMEOUT = 10.0


@dataclass(kw_only=True)
class FutureStageOptions(Options):
    """A future stage's options (the shape the registry must accommodate)."""

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
        from textual.widgets import Select

        return Select
    return Input


async def _await_run_screen(pilot, app: ColonyTuiApp) -> RunScreen:
    """The app's current screen once it is a run screen whose run has
    started (a dialog's write pushes it; the run starts on mount)."""
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
        handle is not None
        and not handle.done.is_set()
        and (time.monotonic() < deadline)
    ):
        await pilot.pause(0.05)
    handle = screen.handle
    assert handle is not None and handle.done.is_set()
    await pilot.pause()
    return handle


async def _click(pilot, app: ColonyTuiApp, selector: str) -> None:
    """Scroll a (possibly folded-away) widget into view and click it."""
    app.screen.query_one(selector).scroll_visible(animate=False)
    await pilot.pause()
    assert await pilot.click(selector), selector
    await pilot.pause()


# --- the stage screens: every field, nothing excluded ---------------------------------


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_the_stage_screens_show_every_field_of_their_dataclass(stage):
    """One widget per field of the stage's options dataclass, in
    declaration order, on the stage's own screen — no owned exclusions, a
    stage run type's config is fully explicit."""
    hints = {spec.key: spec.hint for spec in specs_of(STAGES[stage].options)}

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            shown = {widget.id: type(widget) for widget in screen.form.query(
                "Input, Select, Checkbox"
            )}
            assert shown, stage  # the form rendered
            assert set(shown) == {f"{stage}-{kebab(key)}" for key in hints}, stage
            for key, hint in hints.items():
                assert shown[f"{stage}-{kebab(key)}"] is _expected_widget(hint), key

    asyncio.run(scenario())


# --- the config screen: every field of every section ---------------------------------


def test_the_config_screen_renders_every_field_of_every_section():
    """Every option is on screen: the run level's four inputs; per stage
    section its non-owned fields as widgets (each with its reset button)
    and its owned fields in the ``wired within the chain:`` line — every
    entry marked ``(derived)`` (no run-level markers any more)."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            for key in ("raw_dir", "out_root", "reference", "wells"):
                assert isinstance(
                    screen.form.query_one(f"#pipeline-{kebab(key)}"), Input
                ), key
                assert app.screen.query_one(f"#reset-pipeline-{kebab(key)}", Button)
            for stage, entry in STAGES.items():
                assert app.screen.query_one(f"#{stage}-section-title", Static)
                owned_text = str(
                    app.screen.query_one(f"#owned-{stage}", Static).render()
                )
                assert owned_text.startswith("wired within the chain:")
                form = screen.stage_forms[stage]
                for spec in specs_of(entry.options):
                    reset_id = f"#reset-cfg-{stage}-{kebab(spec.key)}"
                    if spec.key in _OWNED[stage]:
                        assert f"{kebab(spec.key)} (derived)" in owned_text, (
                            stage,
                            spec.key,
                        )
                        # read-only on this screen: no widget of its own
                        with pytest.raises(NoMatches):
                            form.query_one(f"#cfg-{stage}-{kebab(spec.key)}")
                    else:
                        widget = form.query_one(f"#cfg-{stage}-{kebab(spec.key)}")
                        assert type(widget) is _expected_widget(spec.hint), spec.key
                        assert app.screen.query_one(reset_id, Button)
                label_texts = [str(label.render()) for label in form.query(Static)]
                for spec in editable_specs(stage):
                    if spec.hint is not bool:  # a checkbox labels itself
                        assert kebab(spec.key) in label_texts, (stage, spec.key)

    asyncio.run(scenario())


def test_the_owned_lines_derive_from_the_pipeline_draft_alone():
    """The owned lines' values are the chain's derivation from the
    pipeline draft's run level and the chain wiring — read-only, marked
    ``(derived)``, never widget-editable on this screen; nothing is taken
    from another run type's configuration or outputs."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        draft = app.drafts["pipeline"]
        draft.set("pipeline", "raw_dir", "/data/raw")
        draft.set("pipeline", "out_root", "/data/out")
        draft.set("pipeline", "reference", 3)
        draft.set("difference", "out_dir", "/data/out/diff_tree")
        async with app.run_test(size=(100, 60)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            lines = {
                stage: str(app.screen.query_one(f"#owned-{stage}", Static).render())
                for stage in STAGES
            }
            assert (
                "raw-dir (derived) = /data/raw  reference (derived) = 3"
                in lines["difference"]
            )
            assert (
                "input (derived) = diff  diff-dir (derived) = /data/out/diff_tree"
            ) in lines["segmentation"]
            assert "raw-dir (derived) = /data/raw" in lines["analysis"]
            for text in lines.values():
                assert "(run-level)" not in text  # the old marker is gone
            preview = str(app.screen.query_one("#toml-preview", Static).render())
            assert "[pipeline]" in preview
            assert 'raw_dir = "/data/raw"' in preview

    asyncio.run(scenario())


def test_derived_defaults_show_as_placeholders(tmp_path):
    """The config screen's fields that default to a derived name grey the
    derived default in their empty input — the chain's tree names for the
    out root. The stage screens show no such greys (a stage run type
    derives nothing)."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        draft = app.drafts["pipeline"]
        draft.set("pipeline", "raw_dir", tmp_path / "raw")
        draft.set("pipeline", "out_root", tmp_path / "out")
        draft.set("pipeline", "reference", 2)
        async with app.run_test(size=(100, 60)) as pilot:
            app.push_screen(ConfigScreen())
            await pilot.pause()
            screen = app.screen
            out_root = tmp_path / "out"
            assert (
                screen.stage_forms["difference"]
                .query_one("#cfg-difference-out-dir", Input)
                .placeholder
                == f"{out_root}/difference_LO002"
            )
            assert (
                screen.stage_forms["segmentation"]
                .query_one("#cfg-segmentation-out-dir", Input)
                .placeholder
                == f"{out_root}/seg_growth_diff_LO002"
            )
            assert (
                screen.stage_forms["analysis"]
                .query_one("#cfg-analysis-seg-dir", Input)
                .placeholder
                == f"{out_root}/seg_growth_diff_LO002"
            )
            assert (
                screen.stage_forms["analysis"]
                .query_one("#cfg-analysis-out-root", Input)
                .placeholder
                == f"{out_root}/analysis"
            )

    asyncio.run(scenario())


def test_editable_specs_exclude_the_owned_fields():
    """The config screen's stage forms are exactly each stage's non-owned
    fields — for every registry entry (the chain-wired owned fields are
    not among them; the stage screens do keep them, see the stage-screen
    module's rendering tests)."""
    for stage, entry in STAGES.items():
        keys = [spec.key for spec in editable_specs(stage)]
        assert not set(keys) & set(_OWNED[stage]), stage
        expected = [
            spec.key for spec in specs_of(entry.options) if spec.key not in _OWNED[stage]
        ]
        assert keys == expected, stage


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
            draft = app.drafts["pipeline"]
            assert draft.get("pipeline", "raw_dir") == "/raw"
            assert draft.get("pipeline", "out_root") == "/out"
            assert draft.get("pipeline", "reference") == 2
            assert draft.get("pipeline", "wells") == ["-A001", "-B002"]
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


# --- reset buttons (per-field, the config screen's forms) -----------------------------


def test_reset_clears_exactly_its_key_and_updates_the_preview():
    """A reset press drops exactly its own key back to the default: the
    draft document loses that key alone, the field re-shows the default,
    the TOML preview drops the key, and the other sections' keys
    survive."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            draft = app.drafts["pipeline"]
            draft.set("pipeline", "raw_dir", "/data/raw")
            draft.set("pipeline", "reference", 2)
            draft.set("difference", "display_percentile", 97.5)
            draft.set("analysis", "rim_width_px", 90)
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            preview = str(app.screen.query_one("#toml-preview", Static).render())
            assert "reference = 2" in preview
            assert "display_percentile = 97.5" in preview
            assert "rim_width_px = 90" in preview
            await _click(pilot, app, "#reset-cfg-analysis-rim-width-px")
            assert draft.document() == {
                "pipeline": {"raw_dir": "/data/raw", "reference": 2},
                "difference": {"display_percentile": 97.5},
            }
            assert (
                screen.stage_forms["analysis"]
                .query_one("#cfg-analysis-rim-width-px", Input)
                .value
                == ""
            )
            preview = str(app.screen.query_one("#toml-preview", Static).render())
            assert "rim_width_px" not in preview
            assert "display_percentile = 97.5" in preview
            # the run-level form resets the same way
            await _click(pilot, app, "#reset-pipeline-reference")
            assert draft.document() == {
                "pipeline": {"raw_dir": "/data/raw"},
                "difference": {"display_percentile": 97.5},
            }
            assert screen.form.query_one("#pipeline-reference", Input).value == ""
            preview = str(app.screen.query_one("#toml-preview", Static).render())
            assert "reference" not in preview

    asyncio.run(scenario())


def test_reset_enabled_only_for_set_fields():
    """The set-vs-at-default state is visible on the reset buttons: every
    reset starts disabled on the empty draft, an edit enables exactly its
    field's button, a checkbox back at its default disables it again, and
    a reset disables its own button."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            resets = list(app.screen.query(".field-reset"))
            expected = len(run_level_specs()) + sum(
                len(editable_specs(stage)) for stage in STAGES
            )
            assert len(resets) == expected  # one per rendered field
            assert all(button.disabled for button in resets)
            raw = screen.form.query_one("#pipeline-raw-dir", Input)
            raw.value = "/data/raw"
            await pilot.pause()
            assert not app.screen.query_one("#reset-pipeline-raw-dir", Button).disabled
            others = [
                button for button in resets if button.id != "reset-pipeline-raw-dir"
            ]
            assert all(button.disabled for button in others)
            # a checkbox: set away from the default enables, back disables
            dry = screen.stage_forms["difference"].query_one(
                "#cfg-difference-dry-run", Checkbox
            )
            dry.value = True
            await pilot.pause()
            assert not app.screen.query_one(
                "#reset-cfg-difference-dry-run", Button
            ).disabled
            dry.value = False
            await pilot.pause()
            assert app.screen.query_one(
                "#reset-cfg-difference-dry-run", Button
            ).disabled
            # a reset returns the field to default and disables its button
            await _click(pilot, app, "#reset-pipeline-raw-dir")
            assert app.drafts["pipeline"].get("pipeline", "raw_dir") is None
            assert app.screen.query_one("#reset-pipeline-raw-dir", Button).disabled

    asyncio.run(scenario())


def test_a_future_stage_slots_into_the_config_screen(monkeypatch):
    """One registry entry (plus the draft section) adds the config
    screen's whole train section: its form, its owned line, its reset
    buttons — no config-screen code per stage."""
    monkeypatch.setitem(
        STAGES, "train", StageSpec("train", "Train", FutureStageOptions)
    )

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            app.drafts["pipeline"].stages["train"] = {}
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            assert app.screen.query_one("#train-section-title", Static)
            form = screen.stage_forms["train"]
            assert isinstance(form.query_one("#cfg-train-mode"), _expected_widget(
                FutureStageOptions.__annotations__["mode"]
            ))
            assert isinstance(form.query_one("#cfg-train-shape"), Input)
            assert isinstance(form.query_one("#cfg-train-label-dir"), Input)
            assert app.screen.query_one("#reset-cfg-train-mode", Button)
            assert "none" in str(app.screen.query_one("#owned-train", Static).render())
            form.query_one("#cfg-train-mode").value = "b"
            await pilot.pause()
            assert app.drafts["pipeline"].get("train", "mode") == "b"
            await _click(pilot, app, "#reset-cfg-train-mode")
            assert app.drafts["pipeline"].get("train", "mode") is None

    asyncio.run(scenario())


# --- the headline: a difference run from its stage screen alone ------------------------


def test_a_difference_run_starts_from_its_stage_screen_alone(monkeypatch, tmp_path):
    """The headline: a fresh app, the difference screen only — its OWN
    raw_dir and out_dir typed (no mirrors, no run level), *Run this
    stage* writes the draft's own ``difference.toml`` (the save dialog
    suggests ``<out_dir parent>/difference.toml``) and runs the stage
    with exactly what was typed. The written file is a difference config
    (loads through ``DifferenceOptions.from_dict``), not a pipeline.toml;
    the pipeline draft is untouched."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out_dir = tmp_path / "out" / "difference"
    calls: list = []

    def fake_difference(options):
        calls.append(options)
        return DifferenceSummary(wells={}, processed=1, skipped=0, failed=0)

    monkeypatch.setattr(runner, "run_difference", fake_difference)

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen("difference")
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#difference-raw-dir", Input).value = str(raw)
            screen.form.query_one("#difference-out-dir", Input).value = str(out_dir)
            await pilot.pause()
            assert screen.gate_open
            await pilot.click("#run-stage")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(
                tmp_path / "out" / "difference.toml"
            )
            await pilot.press("enter")  # accept the suggested path
            run_screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, run_screen)
            (options,) = calls
            assert options.raw_dir == raw
            assert options.out_dir == out_dir
            assert options.reference == "first"  # nothing derived
            assert "finished — ok" in str(
                run_screen.query_one("#run-status", Static).render()
            )
            # the written TOML is a flat difference config, not a pipeline one
            toml_path = tmp_path / "out" / "difference.toml"
            assert run_screen.toml_path == toml_path
            with open(toml_path, "rb") as file_handle:
                file_doc = tomllib.load(file_handle)
            assert "pipeline" not in file_doc
            assert "difference" not in file_doc  # flat: fields at the top level
            assert DifferenceOptions.from_dict(file_doc) == handle.config
            assert run_screen.config is options  # the dialog's resolved Options
            # no other run type was touched: the pipeline draft stayed empty
            assert app.drafts["pipeline"].document() == {"pipeline": {}}

    asyncio.run(scenario())


# --- the config-file doors: write, then load back into the forms ------------------------


def test_load_toml_restores_a_written_stage_config_into_the_form(tmp_path):
    """A written stage TOML loads back: *Write TOML…* writes the
    difference config, an edit moves the form away from it, *Load
    TOML…* (never gated) replaces the draft with the file's content and
    the form shows the file's values again."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            draft = app.drafts["difference"]
            draft.set("difference", "raw_dir", str(tmp_path / "raw"))
            draft.set("difference", "out_dir", str(tmp_path / "diff"))
            stage = StageScreen("difference")
            app.push_screen(stage)
            await pilot.pause()
            path = tmp_path / "difference.toml"
            await pilot.click("#write-toml")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            assert dialog.query_one("#toml-path", Input).value == str(path)
            await pilot.press("enter")  # write the suggested path
            await pilot.pause()
            assert path.exists()
            assert app.screen is stage
            # the form moves away from the file: reference 2 while the
            # file carries the draft's default "first"
            stage.form.query_one("#difference-reference", Input).value = "2"
            await pilot.pause()
            assert draft.get("difference", "reference") == 2
            with open(path, "rb") as file_handle:
                file_doc = tomllib.load(file_handle)
            assert file_doc["reference"] == "first"
            # *Load TOML…* replaces the draft with the file's content
            await pilot.click("#load-toml")
            await pilot.pause()
            loader = app.screen
            assert isinstance(loader, LoadConfigDialog)
            assert loader.query_one("#toml-path", Input).placeholder == (
                "/path/to/difference.toml"
            )
            loader.query_one("#toml-path", Input).value = str(path)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.screen is stage  # the dialog dismissed back to it
            assert draft.document() == file_doc  # a load resets, never merges
            assert stage.form.query_one("#difference-reference", Input).value == (
                "first"
            )
            assert stage.gate_open  # the loaded file was self-contained

    asyncio.run(scenario())


def test_load_toml_restores_a_written_pipeline_config_into_the_run_level(tmp_path):
    """The config screen's load door with the pipeline's own file: the
    run level written via the save dialog, one field blanked, *Load
    TOML…* refills the run-level form from the file (the draft resets to
    the file's content)."""
    path = tmp_path / "pipeline.toml"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.form.query_one("#pipeline-raw-dir", Input).value = str(
                tmp_path / "raw"
            )
            screen.form.query_one("#pipeline-out-root", Input).value = str(
                tmp_path / "out"
            )
            await pilot.pause()
            await _click(pilot, app, "#write-toml")
            dialog = app.screen
            assert isinstance(dialog, SaveConfigDialog)
            dialog.query_one("#toml-path", Input).value = str(path)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert path.exists()
            assert app.screen is screen
            # blank a field (the gate closes), then load the file back
            screen.form.query_one("#pipeline-raw-dir", Input).value = ""
            await pilot.pause()
            assert not screen.gate_open
            await _click(pilot, app, "#load-toml")  # loading is never gated
            loader = app.screen
            assert isinstance(loader, LoadConfigDialog)
            assert loader.query_one("#toml-path", Input).placeholder == (
                "/path/to/pipeline.toml"
            )
            loader.query_one("#toml-path", Input).value = str(path)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.screen is screen
            assert screen.form.query_one(
                "#pipeline-raw-dir", Input
            ).value == str(tmp_path / "raw")
            assert screen.gate_open  # the file's run level is complete again
            draft = app.drafts["pipeline"]
            assert draft.get("pipeline", "raw_dir") == str(tmp_path / "raw")
            with open(path, "rb") as file_handle:
                file_doc = tomllib.load(file_handle)
            assert draft.get("pipeline", "out_root") == file_doc["pipeline"]["out_root"]

    asyncio.run(scenario())
