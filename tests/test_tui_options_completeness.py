"""Tests for ticket 06: the stage screens' run-level mirrors and the full
config editor with per-field reset (spec addendum decisions 2 and 5).

The mirror split: a stage screen's pipeline-owned fields come in two
kinds — the run-level mirrors (``[difference].raw_dir``/``reference``,
``[analysis].raw_dir``, plus ``out_root`` on every stage screen) are
editable inputs that write the ``[pipeline]`` key they mirror, while the
chain-wired fields (``[segmentation].input``/``diff_dir``) stay read-only
derived — so a difference run is startable entirely from its stage
screen. The config screen shows every option of every section (the run
level plus each stage's non-owned fields; the owned fields per the mirror
split), every field with an individual reset-to-default button — enabled
only while its key is set — and the live effective-TOML preview, all
registry-driven.

Async ``Pilot`` scenarios run through ``asyncio.run`` (no async test
plugin in the dev group; the suite stays dependency-free). Assertions run
inside the headless session — the app's screen stack is gone once it
ends. Reset buttons may sit below the test viewport's fold (the config
screen's sections scroll), so the clicks go through a helper that scrolls
the button into view first.
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_origin

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from colony_tracking import DifferenceSummary
from colony_tracking._options import Options, _union_members
from colony_tracking.pipeline import _OWNED, read_config
from colony_tracking.tui import runner
from colony_tracking.tui.app import ColonyTuiApp
from colony_tracking.tui.draft import RUN_LEVEL
from colony_tracking.tui.forms import kebab, run_level_specs, specs_of
from colony_tracking.tui.screens._save import SaveConfigDialog
from colony_tracking.tui.screens.config import ConfigScreen
from colony_tracking.tui.screens.run import RunScreen
from colony_tracking.tui.screens.stage import (
    STAGES,
    StageSpec,
    StageScreen,
    editable_specs,
    mirror_keys,
    mirror_specs,
    owned_entries,
)

#: How long a test waits for a (fake) run to finish.
_TIMEOUT = 10.0


@dataclass(kw_only=True)
class FutureStageOptions(Options):
    """A future stage's options (the shape the registry must accommodate):
    a Literal select and a tuple among the usual kinds."""

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


# --- the mirror split (registry-driven) ---------------------------------------------


def test_mirror_keys_follow_the_owned_run_level_split():
    """The mirrors are a stage's owned keys that name a run-level field,
    plus out_root on every stage; the chain-wired fields are the rest."""
    assert mirror_keys("difference") == ("raw_dir", "reference", "out_root")
    assert mirror_keys("segmentation") == ("out_root",)
    assert mirror_keys("analysis") == ("raw_dir", "out_root")


def test_mirror_specs_write_the_run_level_and_say_so():
    """Each mirror spec keeps the run-level field's hint, help, and
    default (one source with the config screen's run-level form), labels
    itself ``run-level: ...``, and names its ``[pipeline]`` write target."""
    specs = mirror_specs("difference")
    assert [spec.key for spec in specs] == ["raw_dir", "reference", "out_root"]
    assert [spec.label for spec in specs] == [
        "run-level: raw dir",
        "run-level: reference",
        "run-level: out root",
    ]
    assert [spec.writes for spec in specs] == [
        (RUN_LEVEL, "raw_dir"),
        (RUN_LEVEL, "reference"),
        (RUN_LEVEL, "out_root"),
    ]
    run_specs = {spec.key: spec for spec in run_level_specs()}
    for spec in specs:
        assert spec.hint == run_specs[spec.key].hint, spec.key
        assert spec.help == run_specs[spec.key].help, spec.key


def test_editable_specs_exclude_the_owned_fields():
    """A stage's editable specs are exactly its non-owned fields — for
    every registry entry (the chain-wired owned fields are not among
    them)."""
    for stage, entry in STAGES.items():
        keys = [spec.key for spec in editable_specs(stage)]
        assert not set(keys) & set(_OWNED[stage]), stage
        expected = [
            spec.key
            for spec in specs_of(entry.options)
            if spec.key not in _OWNED[stage]
        ]
        assert keys == expected, stage


# --- the stage screens' mirror group -------------------------------------------------


@pytest.mark.parametrize("stage", ["difference", "segmentation", "analysis"])
def test_every_stage_screen_shows_its_mirror_group(stage):
    """The mirror group sits in the owned/derived section, visually its
    own: one editable input per mirror key, the group's heading present,
    and the chain-wired owned fields stay read-only in the derived line
    (never widgets)."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen(stage)
            app.push_screen(screen)
            await pilot.pause()
            assert "run level" in str(
                app.screen.query_one("#mirror-title", Static).render()
            )
            for key in mirror_keys(stage):
                widget = screen.mirrors.query_one(f"#{stage}-run-{kebab(key)}")
                assert isinstance(widget, Input), key
                assert widget.value == ""  # the empty draft mirrors nothing
            labels = [str(label.render()) for label in screen.mirrors.query(Label)]
            assert any(label.startswith("run-level: ") for label in labels)
            mirrors = set(mirror_keys(stage))
            owned_text = str(app.screen.query_one("#owned-values", Static).render())
            shown = [
                key
                for key, _label, _value in owned_entries(app.draft, stage)
                if key not in mirrors
            ]
            for key in shown:
                assert kebab(key) in owned_text, key
                with pytest.raises(NoMatches):
                    app.screen.query_one(f"#{stage}-{kebab(key)}")
            if not shown:
                assert "none" in owned_text

    asyncio.run(scenario())


def test_mirror_edits_write_the_run_level_only():
    """Typing into the difference screen's mirrors updates the
    ``[pipeline]`` keys — and nothing of the stage section; the reference
    mirror takes the CLI's LO<nnn> form; invalid mirror text closes the
    gate and keeps the last valid value."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen("difference")
            app.push_screen(screen)
            await pilot.pause()
            raw = screen.mirrors.query_one("#difference-run-raw-dir", Input)
            out = screen.mirrors.query_one("#difference-run-out-root", Input)
            reference = screen.mirrors.query_one("#difference-run-reference", Input)
            raw.focus()
            await pilot.press(*"/data/raw")
            out.focus()
            await pilot.press(*"/data/out")
            await pilot.pause()
            assert app.draft.document() == {
                "pipeline": {"raw_dir": "/data/raw", "out_root": "/data/out"},
            }
            assert app.draft.stages["difference"] == {}
            assert screen.gate_open
            reference.focus()
            await pilot.press(*"LO003")
            await pilot.pause()
            assert app.draft.get(RUN_LEVEL, "reference") == 3
            reference.value = "abc"
            await pilot.pause()
            assert not screen.gate_open
            assert (
                str(app.screen.query_one("#resolve-status", Static).render())
                == "Fix the marked fields to enable the actions."
            )
            assert app.draft.get(RUN_LEVEL, "reference") == 3  # last valid kept
            reference.value = "2"
            await pilot.pause()
            assert screen.gate_open
            assert app.draft.get(RUN_LEVEL, "reference") == 2

    asyncio.run(scenario())


def test_gate_opens_with_only_mirrors_and_the_derived_tracks_them(tmp_path):
    """The gate opens with only the mirrors filled (they are the
    pipeline's required run level), and the derived values track mirror
    edits: the out_dir placeholder here, the segment screen's diff-dir
    line there."""
    out = tmp_path / "out"

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            screen = StageScreen("difference")
            app.push_screen(screen)
            await pilot.pause()
            assert not screen.gate_open
            mirrors = screen.mirrors
            mirrors.query_one("#difference-run-raw-dir", Input).value = str(
                tmp_path / "raw"
            )
            await pilot.pause()
            assert not screen.gate_open  # out_root still missing
            mirrors.query_one("#difference-run-out-root", Input).value = str(out)
            await pilot.pause()
            assert screen.gate_open
            assert screen.form.query_one(
                "#difference-out-dir", Input
            ).placeholder == str(out / "difference")
            mirrors.query_one("#difference-run-reference", Input).value = "2"
            await pilot.pause()
            assert screen.form.query_one(
                "#difference-out-dir", Input
            ).placeholder == str(out / "difference_LO002")
            # the segment screen's derived line tracks the same run level,
            # and its own out-root mirror moves it
            segment = StageScreen("segmentation")
            app.push_screen(segment)
            await pilot.pause()
            owned = str(app.screen.query_one("#owned-values", Static).render())
            assert f"diff-dir = {out / 'difference_LO002'}" in owned
            segment.mirrors.query_one("#segmentation-run-out-root", Input).value = str(
                tmp_path / "elsewhere"
            )
            await pilot.pause()
            owned = str(app.screen.query_one("#owned-values", Static).render())
            assert f"diff-dir = {tmp_path / 'elsewhere' / 'difference_LO002'}" in owned

    asyncio.run(scenario())


def test_a_difference_run_starts_from_its_stage_screen_alone(monkeypatch, tmp_path):
    """The headline: a fresh app, the difference screen only — the mirrors
    fill the run level, *Run this stage* writes the TOML and runs the
    stage with the chain's wiring. No other screen is visited."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
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
            screen.mirrors.query_one("#difference-run-raw-dir", Input).value = str(raw)
            screen.mirrors.query_one("#difference-run-out-root", Input).value = str(out)
            await pilot.pause()
            assert screen.gate_open
            await pilot.click("#run-stage")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfigDialog)
            await pilot.press("enter")  # the suggested <out_root>/pipeline.toml
            run_screen = await _await_run_screen(pilot, app)
            handle = await _await_done(pilot, run_screen)
            (options,) = calls
            assert options.raw_dir == raw
            assert options.out_dir == out / "difference"
            assert options.reference == "first"
            assert "finished — ok" in str(
                run_screen.query_one("#run-status", Static).render()
            )
            assert read_config(out / "pipeline.toml") == handle.config

    asyncio.run(scenario())


def test_mirrors_prefill_from_the_run_level_and_track_its_edits(tmp_path):
    """A mirror input shows the run level's current value on entry, and a
    run-level edit made on the config screen shows up when the stage
    screen is re-entered."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 44)) as pilot:
            app.draft.set(RUN_LEVEL, "raw_dir", "/data/raw")
            app.draft.set(RUN_LEVEL, "out_root", "/data/out")
            analysis = StageScreen("analysis")
            app.push_screen(analysis)
            await pilot.pause()
            assert (
                analysis.mirrors.query_one("#analysis-run-raw-dir", Input).value
                == "/data/raw"
            )
            assert (
                analysis.mirrors.query_one("#analysis-run-out-root", Input).value
                == "/data/out"
            )
            # the run level edited on the config screen shows on re-entry
            app.push_screen(ConfigScreen())
            await pilot.pause()
            app.screen.query_one("#pipeline-raw-dir", Input).value = "/elsewhere/raw"
            await pilot.pause()
            await pilot.click("#back")
            await pilot.pause()
            assert app.screen is analysis
            assert (
                analysis.mirrors.query_one("#analysis-run-raw-dir", Input).value
                == "/elsewhere/raw"
            )

    asyncio.run(scenario())


# --- the config screen: every field of every section ---------------------------------


def test_the_config_screen_renders_every_field_of_every_section():
    """Every option is on screen: the run level's four inputs; per stage
    section its editable fields as widgets (each with its reset button)
    and its owned fields in the derived line — mirrors marked run-level,
    chain-wired marked derived."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            for key in ("raw_dir", "out_root", "reference", "wells"):
                assert isinstance(
                    screen.form.query_one(f"#pipeline-{kebab(key)}"), Input
                )
                assert app.screen.query_one(f"#reset-pipeline-{kebab(key)}", Button)
            for stage, entry in STAGES.items():
                assert app.screen.query_one(f"#{stage}-section-title", Static)
                owned_text = str(
                    app.screen.query_one(f"#owned-{stage}", Static).render()
                )
                form = screen.stage_forms[stage]
                mirrors = set(mirror_keys(stage))
                for spec in specs_of(entry.options):
                    reset_id = f"#reset-cfg-{stage}-{kebab(spec.key)}"
                    if spec.key in _OWNED[stage]:
                        marker = "run-level" if spec.key in mirrors else "derived"
                        assert f"{kebab(spec.key)} ({marker})" in owned_text, (
                            stage,
                            spec.key,
                        )
                    else:
                        widget_id = f"#cfg-{stage}-{kebab(spec.key)}"
                        widget = form.query_one(widget_id)
                        assert type(widget) is _expected_widget(spec.hint), spec.key
                        assert app.screen.query_one(reset_id, Button)
                label_texts = [str(label.render()) for label in form.query(Label)]
                for spec in editable_specs(stage):
                    if spec.hint is not bool:  # a checkbox labels itself
                        assert kebab(spec.key) in label_texts, (stage, spec.key)

    asyncio.run(scenario())


def test_reset_clears_exactly_its_key_and_updates_the_preview():
    """A reset press drops exactly its own key back to the default: the
    draft document loses that key alone, the field re-shows the default,
    the TOML preview drops the key, and the other sections' keys
    survive."""

    async def scenario() -> None:
        app = ColonyTuiApp()
        async with app.run_test(size=(100, 60)) as pilot:
            app.draft.set(RUN_LEVEL, "raw_dir", "/data/raw")
            app.draft.set(RUN_LEVEL, "reference", 2)
            app.draft.set("difference", "display_percentile", 97.5)
            app.draft.set("analysis", "rim_width_px", 90)
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            preview = str(app.screen.query_one("#toml-preview", Static).render())
            assert "reference = 2" in preview
            assert "display_percentile = 97.5" in preview
            assert "rim_width_px = 90" in preview
            await _click(pilot, app, "#reset-cfg-analysis-rim-width-px")
            assert app.draft.document() == {
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
            assert app.draft.document() == {
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
            assert app.draft.get(RUN_LEVEL, "raw_dir") is None
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
            app.draft.stages["train"] = {}
            screen = ConfigScreen()
            app.push_screen(screen)
            await pilot.pause()
            assert app.screen.query_one("#train-section-title", Static)
            form = screen.stage_forms["train"]
            assert isinstance(form.query_one("#cfg-train-mode"), Select)
            assert isinstance(form.query_one("#cfg-train-shape"), Input)
            assert isinstance(form.query_one("#cfg-train-label-dir"), Input)
            assert app.screen.query_one("#reset-cfg-train-mode", Button)
            assert "none" in str(app.screen.query_one("#owned-train", Static).render())
            form.query_one("#cfg-train-mode", Select).value = "b"
            await pilot.pause()
            assert app.draft.get("train", "mode") == "b"
            await _click(pilot, app, "#reset-cfg-train-mode")
            assert app.draft.get("train", "mode") is None

    asyncio.run(scenario())
