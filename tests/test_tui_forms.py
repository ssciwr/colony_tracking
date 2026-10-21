"""Tests for the dataclass-driven option forms (ticket 02, spec story 36).

The form machinery under test: an options dataclass (its type hints +
``metadata["help"]`` — the CLI's single source) maps to widgets; typed
text parses back into the plain draft values (paths as strings, tuples as
lists, numbers as numbers); blank input means "unset" (the key clears, the
default applies at resolve time); text that does not parse marks the field
invalid inline and the draft keeps the key's last valid value. The
Select/tuple kinds the three pipeline stages don't exercise (their only
Literal field is pipeline-owned) are covered with a stand-in dataclass
shaped like a future stage's.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Label, Select

from colony_tracking._options import Options
from colony_tracking.tui.draft import ConfigDraft
from colony_tracking.tui.forms import (
    UNSET,
    FieldError,
    OptionForm,
    format_field,
    kebab,
    parse_field,
    run_level_specs,
    specs_of,
)


@dataclass(kw_only=True)
class DemoOptions(Options):
    """A stand-in for a future stage's options: the hint kinds the three
    pipeline stages don't exercise (a Literal select, fixed and open
    tuples, an optional int, a default-True checkbox)."""

    mode: Literal["raw", "diff"] = field(
        default="raw", metadata={"help": "Input mode."}
    )
    patch: tuple[int, int] = field(
        default=(512, 512), metadata={"help": "Tile size in px."}
    )
    series: tuple[int, ...] = field(
        default=(1, 2, 3), metadata={"help": "Frame numbers."}
    )
    limit: int | None = field(default=None, metadata={"help": "Optional cap."})
    flag: bool = field(default=True, metadata={"help": "A default-True flag."})


class _FormHost(App[None]):
    """The minimal host for a bare OptionForm (mounted alone, with the
    draft handed to it at construction)."""

    def __init__(self, form: OptionForm) -> None:
        super().__init__()
        self._form = form

    def compose(self) -> ComposeResult:
        yield self._form


def _run_form(form: OptionForm, observe) -> None:
    """Mount ``form`` headless and hand app + form + pilot to the async
    ``observe`` inside the still-running session (the widget tree is gone
    once it ends). ``observe`` awaits ``pilot.pause()`` between setting a
    widget and asserting on the draft — change events are posted, not
    synchronous."""

    async def scenario() -> None:
        app = _FormHost(form)
        async with app.run_test() as pilot:
            await pilot.pause()
            await observe(app, form, pilot)

    asyncio.run(scenario())


def _demo_draft() -> ConfigDraft:
    """A draft with a ``demo`` section — the shape a future stage gets
    (``draft.stages`` gains the section; ``set`` accepts it)."""
    draft = ConfigDraft()
    draft.stages["demo"] = {}
    return draft


# --- the parsing machinery (pure) -----------------------------------------------


def test_kebab_matches_the_cli_flags():
    """Labels are the CLI flag spellings (kebab-case, with the CLI's
    rename for seg_dirs)."""
    assert kebab("min_size") == "min-size"
    assert kebab("rim_width_px") == "rim-width-px"
    assert kebab("seg_dirs") == "seg-dir"


def test_specs_carry_hint_help_and_default():
    """specs_of reads the dataclass: declaration order, the help metadata,
    the defaults; the run level is PipelineConfig's own four fields."""
    specs = specs_of(DemoOptions)
    assert [spec.key for spec in specs] == ["mode", "patch", "series", "limit", "flag"]
    assert {spec.key: spec.help for spec in specs}["mode"] == "Input mode."
    defaults = {spec.key: spec.default for spec in specs}
    assert defaults["patch"] == (512, 512)
    assert defaults["limit"] is None
    assert [spec.key for spec in run_level_specs()] == [
        "raw_dir",
        "out_root",
        "reference",
        "wells",
    ]


def test_parse_field_scalars():
    """Numbers, paths/strings, booleans, and literals parse to the plain
    values; blank means unset; nonsense raises FieldError with the
    pipeline's hint wording."""
    assert parse_field("8", int) == 8
    assert parse_field(" 8 ", int) == 8
    assert parse_field("99.5", float) == 99.5
    assert parse_field("99", float) == 99.0
    assert parse_field("/data/raw", str) == "/data/raw"
    assert parse_field("/data/raw", Path) == "/data/raw"
    assert parse_field("true", bool) is True
    assert parse_field("false", bool) is False
    assert parse_field("raw", Literal["raw", "diff"]) == "raw"
    assert parse_field("", int) is UNSET
    assert parse_field("   ", float) is UNSET
    with pytest.raises(FieldError, match="must be an integer"):
        parse_field("8.0", int)
    with pytest.raises(FieldError, match="must be a number"):
        parse_field("abc", float)
    with pytest.raises(FieldError, match="must be true or false"):
        parse_field("no", bool)
    with pytest.raises(FieldError, match=r"must be 'raw' or 'diff'"):
        parse_field("both", Literal["raw", "diff"])


def test_parse_field_reference():
    """Reference parses to "first" or a 1-based frame number, and rejects
    anything else with the union's own wording."""
    from colony_tracking._options import Reference

    assert parse_field("first", Reference) == "first"
    assert parse_field("7", Reference) == 7
    with pytest.raises(FieldError, match=r"must be 'first' or an integer"):
        parse_field("second", Reference)


def test_parse_field_sequences():
    """Lists are comma-separated, tuples comma- or space-separated (a
    fixed tuple wants exactly its arity); empty lists mean unset."""
    list_hint = list[str] | None
    assert parse_field("-A001, -B002", list_hint) == ["-A001", "-B002"]
    assert parse_field("-A001,-B002", list_hint) == ["-A001", "-B002"]
    assert parse_field(",", list_hint) is UNSET
    assert parse_field("  ", list_hint) is UNSET
    assert parse_field("/a, /b", list[Path]) == ["/a", "/b"]
    assert parse_field("512, 512", tuple[int, int]) == (512, 512)
    assert parse_field("512 512", tuple[int, int]) == (512, 512)
    assert parse_field("1 2, 3", tuple[int, ...]) == (1, 2, 3)
    with pytest.raises(FieldError, match="must be 2 values"):
        parse_field("512", tuple[int, int])


def test_parse_field_optional_members():
    """X | None parses like X; blank means the key is unset (None at
    resolve)."""
    assert parse_field("5", int | None) == 5
    assert parse_field("", int | None) is UNSET


def test_format_field_inverts_parse():
    """format_field renders every plain draft value back to text that
    parses to the same value."""
    for value, text in [
        (None, ""),
        (True, "true"),
        (False, "false"),
        (8, "8"),
        (99.5, "99.5"),
        ("first", "first"),
        (3, "3"),
        (["-A001", "-B002"], "-A001, -B002"),
        ([512, 512], "512, 512"),
        ("/data/raw", "/data/raw"),
    ]:
        assert format_field(value) == text


# --- the mounted form -----------------------------------------------------------


def test_form_renders_and_parses_every_kind():
    """Every hint kind renders as its widget and parses back into the
    draft: a Select for the Literal, Inputs for the tuples and the
    optional int; blanking a field unsets its key."""
    draft = _demo_draft()
    form = OptionForm("demo", specs_of(DemoOptions), draft)

    async def observe(app, form, pilot):
        assert isinstance(form.query_one("#demo-mode"), Select)
        assert isinstance(form.query_one("#demo-patch"), Input)
        assert isinstance(form.query_one("#demo-series"), Input)
        assert isinstance(form.query_one("#demo-limit"), Input)
        assert isinstance(form.query_one("#demo-flag"), Checkbox)
        form.query_one("#demo-mode", Select).value = "diff"
        form.query_one("#demo-patch", Input).value = "640, 480"
        form.query_one("#demo-series", Input).value = "1 2 3"
        form.query_one("#demo-limit", Input).value = "5"
        await pilot.pause()
        assert draft.get("demo", "mode") == "diff"
        assert draft.get("demo", "patch") == [640, 480]
        assert draft.get("demo", "series") == [1, 2, 3]
        assert draft.get("demo", "limit") == 5
        form.query_one("#demo-mode", Select).clear()
        form.query_one("#demo-patch", Input).value = ""
        await pilot.pause()
        assert draft.get("demo", "mode") is None  # unset again
        assert draft.get("demo", "patch") is None

    _run_form(form, observe)


def test_form_shows_labels_and_help():
    """Each field shows its kebab label and its help text (the CLI's
    single source); the checkbox carries the label itself."""
    form = OptionForm("demo", specs_of(DemoOptions), _demo_draft())

    async def observe(app, form, pilot):
        await pilot.pause()
        texts = [str(label.render()) for label in form.query(Label)]
        for key in ("mode", "patch", "series", "limit"):
            assert kebab(key) in texts
        for help_text in ("Input mode.", "Tile size in px.", "Optional cap."):
            assert help_text in texts
        assert form.query_one("#demo-flag", Checkbox).label == "flag"

    _run_form(form, observe)


def test_invalid_text_keeps_last_value_and_recovers():
    """Unparseable text: inline error, form invalid, draft keeps the last
    valid value; fixing or emptying the field recovers (empty clears)."""
    draft = _demo_draft()
    form = OptionForm("demo", specs_of(DemoOptions), draft)

    async def observe(app, form, pilot):
        limit = form.query_one("#demo-limit", Input)
        limit.value = "5"
        await pilot.pause()
        assert form.error("limit") == ""
        limit.value = "abc"
        await pilot.pause()
        assert form.error("limit") == "must be an integer (got 'abc')"
        assert not form.valid
        assert limit.has_class("invalid")
        assert draft.get("demo", "limit") == 5  # last valid value kept
        limit.value = ""
        await pilot.pause()
        assert form.error("limit") == ""
        assert form.valid
        assert not limit.has_class("invalid")
        assert draft.get("demo", "limit") is None  # unset

    _run_form(form, observe)


def test_checkbox_stores_only_non_defaults():
    """A checkbox stores only non-default values: mounted at its default
    it writes nothing; toggled away it stores; back at the default the key
    clears (both default polarities)."""
    draft = _demo_draft()
    form = OptionForm("demo", specs_of(DemoOptions), draft)

    async def observe(app, form, pilot):
        flag = form.query_one("#demo-flag", Checkbox)  # default True
        await pilot.pause()
        assert flag.value is True
        assert draft.get("demo", "flag") is None  # default stores nothing
        flag.value = False
        await pilot.pause()
        assert draft.get("demo", "flag") is False
        flag.value = True
        await pilot.pause()
        assert draft.get("demo", "flag") is None  # back at default: cleared

    _run_form(form, observe)


def test_difference_fields_parse_into_the_draft():
    """The difference stage's real fields: typed text becomes the plain
    draft values (paths as strings, numbers as numbers, the comma list as
    a list, the checkbox as an explicit non-default)."""
    from colony_tracking._options import DifferenceOptions

    draft = ConfigDraft()
    form = OptionForm("difference", specs_of(DifferenceOptions), draft)

    async def observe(app, form, pilot):
        form.query_one("#difference-out-dir", Input).value = "/data/diff"
        form.query_one("#difference-wells", Input).value = "-A001, -B002"
        form.query_one("#difference-display-percentile", Input).value = "97.5"
        form.query_one("#difference-subsample", Input).value = "16"
        form.query_one("#difference-dry-run", Checkbox).value = True
        await pilot.pause()
        assert draft.get("difference", "out_dir") == "/data/diff"
        assert draft.get("difference", "wells") == ["-A001", "-B002"]
        assert draft.get("difference", "display_percentile") == 97.5
        assert draft.get("difference", "subsample") == 16
        assert draft.get("difference", "dry_run") is True
        form.query_one("#difference-dry-run", Checkbox).value = False
        await pilot.pause()
        assert draft.get("difference", "dry_run") is None  # back at default

    _run_form(form, observe)
