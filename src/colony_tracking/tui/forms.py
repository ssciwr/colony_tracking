"""Dataclass-driven option forms: widgets from field type hints.

An options dataclass — every field carrying ``metadata={"help": ...}``, the
same single source the CLI's flags come from — becomes a form: one widget
per field, chosen from the field's type hint alone:

- ``Path`` — an ``Input`` (the path string);
- ``bool`` — a ``Checkbox``;
- ``int`` / ``float`` — an ``Input`` with numeric validation;
- ``Literal[...]`` — a ``Select`` of the literal values;
- ``Reference`` (``"first" | int``) — an ``Input`` parsed by the shared
  :func:`~colony_tracking._options.parse_reference` (the CLI's
  spellings);
- ``list[str] | None`` — an ``Input``, comma-separated;
- ``tuple[int, ...]`` (and fixed ``tuple[int, int]``) — an ``Input``,
  comma- or space-separated;
- ``X | None`` — like ``X``, with empty input meaning ``None``.

``str`` falls to ``Input``. Field labels are the CLI flag names
(kebab-case); the help metadata is the description line under the widget.

Placeholders
------------
A blank field shows in gray exactly what a blank resolves to, laid from
the field's declared option state by one shared rule
(:func:`placeholder_text`, applied to every :class:`FieldSpec` at spec
time):

- a declared default that is not ``None`` — the bare default value
  (``reference`` → ``first``, ``display_percentile`` → ``99.5``);
- default ``None`` — ``(default: unset)`` (``wells``, ``checkpoint``);
- required (no default) — ``required: <example>``
  (:data:`REQUIRED_EXAMPLES`).

An ``Input`` shows it as ``placeholder``, a blank ``Select`` as its
prompt (never a pre-picked default value), a ``Checkbox`` not at all —
its default shows as the check state. The config screen overlays the
within-chain derived tree names onto the stage forms' static grays
(:meth:`OptionForm.refresh_placeholders`) and falls back to them again
whenever the ``out_root`` they derive from is cleared.

Value flow — on change
----------------------
Values flow into the screen's :class:`~colony_tracking.tui.draft.ConfigDraft`
on every change, not on submit: each edit parses immediately and writes
through ``ConfigDraft.set``, and an emptied field clears its key, so the
draft always matches the form and a screen's actions gate on exactly what
is shown. Mid-edit states:

- text that does not parse marks the field invalid (red border plus an
  inline message under it) and the draft keeps the key's last valid
  value; the owning screen keeps its actions disabled until every field
  parses again and the whole draft resolves;
- a checkbox fires when it is initialized too; one whose value equals the
  unset key's default stores nothing — ``resolve`` applies the default
  anyway, so the document stays free of no-op keys.

Reset
-----
Every field also carries a small *reset* button next to its widget: it
clears exactly that field's draft key (``ConfigDraft.clear``) and stays
enabled only while the key is set, so the set-vs-at-default state is
visible at a glance.

Browse
------
A single-``Path`` field also carries a *browse* button in its field row:
pressing it opens the directory-picker modal, textualize's
``textual-fspicker`` package (:class:`~textual_fspicker.SelectDirectory`).
The picker starts at the field's current value's nearest existing
directory (:func:`picker_location`), and the picked directory lands in
the input as if typed — the normal on-change flow parses and stores it;
cancelling closes the picker and changes nothing. A list-of-path field
(``seg_dirs``) keeps manual entry.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, get_args, get_origin, get_type_hints

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)
from textual_fspicker import SelectDirectory

from .._options import (
    Options,
    Reference,
    _union_members,
    parse_reference,
)
from ..pipeline import ConfigError, PipelineConfig, _RUN_LEVEL_FIELDS, _human_hint
from .draft import ConfigDraft
from .screens._save import SaveConfigDialog

#: Field-name → label spelling, mirroring the CLI's ``_RENAMES``, so a
#: field's label matches its CLI flag.
RENAMES: Final[dict[str, str]] = {"seg_dirs": "seg-dir"}


def kebab(name: str) -> str:
    """The CLI flag spelling of a field name: ``min_size`` → ``min-size``."""
    return RENAMES.get(name, name.replace("_", "-"))


def focus_field(widgets: Sequence[Widget], forward: bool) -> None:
    """Move the focus one field along ``widgets`` (DOM order), wrapping at
    the ends; when the focused widget is not among ``widgets``, enter at
    the near end. A no-op when none of ``widgets`` is focusable."""
    fields = [widget for widget in widgets if widget.focusable]
    if not fields:
        return
    focused = fields[0].app.focused
    try:
        index = fields.index(focused)
    except ValueError:
        index = 0 if forward else len(fields) - 1
        fields[index].focus()
        return
    step = 1 if forward else -1
    fields[(index + step) % len(fields)].focus()


class FieldError(ValueError):
    """A field's typed text does not parse as its declared type; the
    message is shown inline under the field."""


class _Unset:
    """The sentinel :func:`parse_field` returns for blank input."""

    def __repr__(self) -> str:
        return "UNSET"


#: Returned by :func:`parse_field` for blank text: the key is unset and
#: ``ConfigDraft.clear`` applies.
UNSET: Final[_Unset] = _Unset()


def parse_field(text: str, hint: Any) -> Any:
    """Parse one field's typed text into its plain draft value.

    Blank text returns :data:`UNSET`; a parse failure raises
    :class:`FieldError` with a short human message (the pipeline's
    ``_human_hint`` wording, except the reference field's, which is the
    shared :func:`~colony_tracking._options.parse_reference` message).
    """
    text = text.strip()
    if not text:
        return UNSET
    if hint is Reference:  # one shared parser — the CLI's spellings
        try:
            return parse_reference(text)
        except ValueError as err:
            raise FieldError(str(err)) from None
    members = _union_members(hint)
    if members is not None:
        if len(members) == 1:  # X | None: parse as X (blank already handled)
            return parse_field(text, members[0])
        for member in members:  # e.g. Reference: "first" | int
            try:
                return parse_field(text, member)
            except FieldError:
                pass
        raise FieldError(f"must be {_human_hint(hint)} (got {text!r})")
    if hint is bool:
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        raise FieldError(f"must be true or false (got {text!r})")
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        if text in args:
            return text
        raise FieldError(f"must be {_human_hint(hint)} (got {text!r})")
    if hint is int:
        try:
            return int(text)
        except ValueError:
            raise FieldError(f"must be an integer (got {text!r})") from None
    if hint is float:
        try:
            return float(text)
        except ValueError:
            raise FieldError(f"must be a number (got {text!r})") from None
    if origin in (list, tuple):
        item_hint = args[0] if args else str
        if origin is list:
            items = [piece.strip() for piece in text.split(",")]
            items = [item for item in items if item]
            if not items:
                return UNSET
            return [parse_field(item, item_hint) for item in items]
        items = [piece for piece in re.split(r"[,\s]+", text) if piece]
        if not items:
            return UNSET
        if Ellipsis not in args and len(items) != len(args):
            raise FieldError(
                f"must be {len(args)} values separated by commas or spaces "
                f"(got {text!r})"
            )
        return tuple(parse_field(item, item_hint) for item in items)
    # str, Path, and anything unhinted: the text as-is (the pipeline's own
    # config check is the final word at resolve time).
    return text


def format_field(value: Any) -> str:
    """The text an ``Input`` shows for a plain draft ``value`` — the
    inverse of :func:`parse_field` (``None`` becomes empty text;
    sequences render comma-separated)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


class _NoDefault:
    """The marker for "this field has no dataclass default" in a
    :class:`FieldSpec` — ``dataclasses.MISSING`` cannot play the role: a
    dataclass attribute defaulting to it reads as required."""

    def __repr__(self) -> str:
        return "NO_DEFAULT"


#: FieldSpec.default's marker for a required field.
NO_DEFAULT: Final[_NoDefault] = _NoDefault()

#: Required fields' example texts — the ``required:`` gray copy, keyed by
#: field name (these are all the required keys of the pipeline's own
#: options dataclasses; an unmapped required key falls back to its name).
REQUIRED_EXAMPLES: Final[dict[str, str]] = {
    "raw_dir": "/path/to/raw-plate-frames",
    "out_root": "/path/to/output-root",
    "out_dir": "/path/to/output-dir",
    "seg_dirs": "/path/to/segmentation-trees",
}

#: The gray text of a field whose declared default is ``None``.
UNSET_DEFAULT_TEXT: Final[str] = "(default: unset)"


def placeholder_text(key: str, default: Any) -> str:
    """The gray text a blank field shows — what a blank resolves to, from
    the declared option state in priority order: a default that is not
    ``None`` as :func:`format_field` renders it, ``None`` as
    :data:`UNSET_DEFAULT_TEXT`, and required (no default) as
    ``required: <example>`` (:data:`REQUIRED_EXAMPLES`, one text per key
    name, shared across screens)."""
    if default is NO_DEFAULT:
        return f"required: {REQUIRED_EXAMPLES.get(key, key)}"
    if default is None:
        return UNSET_DEFAULT_TEXT
    return format_field(default)


@dataclass(frozen=True)
class FieldSpec:
    """One option field of a form: the draft key plus the declared type,
    the gray text of a blank field, and the ``help`` metadata."""

    #: The dataclass field name — the draft key.
    key: str
    #: The declared type hint (drives the widget and the parsing).
    hint: Any
    #: The field's help text (the description line under the widget).
    help: str = ""
    #: The dataclass default (:data:`NO_DEFAULT` when required); a
    #: checkbox shows it while the key is unset.
    default: Any = NO_DEFAULT
    #: The gray text a blank field shows (:func:`placeholder_text` at
    #: spec time — an ``Input``'s placeholder, a blank ``Select``'s
    #: prompt); the config screen overlays derived texts over it.
    placeholder: str = ""
    #: The widget's label (``kebab(key)`` when empty).
    label: str = ""


def specs_of(cls: type, keys: Collection[str] | None = None) -> tuple[FieldSpec, ...]:
    """The form specs of a dataclass: one :class:`FieldSpec` per field in
    declaration order (hint, help, default, and the blank field's gray
    text — :func:`placeholder_text`), restricted to ``keys`` when given."""
    hints = get_type_hints(cls)
    out = []
    for f in fields(cls):
        if keys is not None and f.name not in keys:
            continue
        default = f.default
        if default is MISSING and f.default_factory is not MISSING:
            default = f.default_factory()
        if default is MISSING:
            default = NO_DEFAULT
        out.append(
            FieldSpec(
                f.name,
                hints[f.name],
                f.metadata.get("help", ""),
                default,
                placeholder_text(f.name, default),
            )
        )
    return tuple(out)


def run_level_specs() -> tuple[FieldSpec, ...]:
    """The run-level form specs: the ``[pipeline]`` fields of
    :class:`~colony_tracking.pipeline.PipelineConfig` (``raw_dir``,
    ``out_root``, ``reference``, ``wells``)."""
    return specs_of(PipelineConfig, keys=_RUN_LEVEL_FIELDS)


def _optional_member(hint: Any) -> Any:
    """The hint with a one-member optional wrapper removed
    (``X | None`` → ``X``)."""
    members = _union_members(hint)
    if members is not None and len(members) == 1:
        return members[0]
    return hint


def _literal_values(hint: Any) -> tuple[Any, ...]:
    """The values of a ``Literal`` hint (empty when it is not one)."""
    hint = _optional_member(hint)
    if get_origin(hint) is Literal:
        return get_args(hint)
    return ()


def _widget_kind(spec: FieldSpec) -> str:
    """The widget kind for ``spec``: ``checkbox``, ``select``, or
    ``input`` (see the module docstring for the mapping)."""
    hint = _optional_member(spec.hint)
    if hint is bool:
        return "checkbox"
    if get_origin(hint) is Literal:
        return "select"
    return "input"


def _is_path_field(spec: FieldSpec) -> bool:
    """Whether ``spec`` is a single-path field (a ``Path`` or
    ``Path | None`` hint — the ones that get a browse button)."""
    return _optional_member(spec.hint) is Path


#: A browse button's label (the field-row counterpart of ``reset``).
BROWSE_LABEL: Final[str] = "browse"


class BrowseButton(Button):
    """A path field's *browse* button: it opens the directory-picker
    modal for its field (the hosting screen wires the press; see
    :func:`open_directory_picker`). Click-only — ``can_focus`` is off,
    like the reset buttons — so tab and the arrow-key field walk stay
    field-to-field; styled through its container's ``.field-browse``
    rules, because a ``Button``'s own type rules outrank a bare type
    selector."""

    can_focus = False


def picker_location(value: str) -> Path:
    """The directory a picker opens at for a path field showing ``value``:
    the value's nearest existing directory (the path itself, else the
    first existing ancestor), or the working directory when nothing
    exists yet."""
    stripped = value.strip()
    if not stripped:
        return Path.cwd()
    path = Path(stripped)
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    return Path.cwd()


def open_directory_picker(
    app: App[Any],
    location: Path,
    accept: Callable[[Path], None],
) -> None:
    """Open the directory-picker modal — textual-fspicker's
    :class:`~textual_fspicker.SelectDirectory`, starting at ``location``.
    Picking a directory hands it to ``accept``; cancelling (escape or the
    dialog's *Cancel*) calls nothing.

    The hand-off waits for the dismissed screen's reveal to settle
    (``call_after_refresh``): the pop resumes the screen behind the
    picker, whose entry refresh re-shows the draft's values — a write
    racing that refresh would be re-set from the draft before its change
    event landed, losing the pick."""

    def picked(path: Path | None) -> None:
        if path is not None:
            app.call_after_refresh(accept, path)

    app.push_screen(SelectDirectory(location), picked)


class OptionForm(VerticalScroll):
    """The option fields of one draft section (see the module docstring).

    The form owns the draft-writing behavior: every change parses and
    writes immediately (``ConfigDraft.set`` / ``clear``), invalid text
    marks the field inline and writes nothing, and :attr:`valid` tells
    the owning screen whether every field currently parses. ``draft``
    must be set before the first change — screens pass it at
    construction. Every field carries its reset button, every
    single-``Path`` field a browse button; every field reads and writes
    the form's own section and key.
    """

    DEFAULT_CSS = """
    OptionForm {
        height: 1fr;
    }
    OptionForm .option-field {
        height: auto;
        margin-bottom: 1;
    }
    OptionForm .field-label {
        text-style: bold;
    }
    OptionForm .field-help {
        color: $text 60%;
    }
    OptionForm .field-error {
        color: $error;
    }
    OptionForm .field-row {
        height: auto;
        align-vertical: middle;
    }
    OptionForm .field-row Input {
        width: 1fr;  # not 100%: the row shares it with the other buttons
    }
    OptionForm .field-reset {
        min-width: 0;
        width: auto;
        height: 1;
        border: none;
        background: $panel;
        color: $text 60%;
        margin-left: 1;
    }
    OptionForm .field-reset:hover {
        background: $surface-darken-1;
        color: $text;
    }
    OptionForm .field-browse {
        min-width: 0;
        width: auto;
        height: 1;
        border: none;
        background: $panel;
        color: $text 60%;
        margin-left: 1;
    }
    OptionForm .field-browse:hover {
        background: $surface-darken-1;
        color: $text;
    }
    OptionForm Input.invalid,
    OptionForm Input.invalid:focus {
        border: tall $error !important;
    }
    """

    #: Up/down step the field focus instead of the scroll-container
    #: scrolling the base class binds them to. A focused widget's own
    #: bindings keep precedence (an expanded Select's overlay navigates
    #: with the arrows), and left/right stay inside the focused input.
    BINDINGS = [
        Binding("up", "screen.previous_field", "Previous field", show=False),
        Binding("down", "screen.next_field", "Next field", show=False),
    ]

    def __init__(
        self,
        section: str,
        specs: Iterable[FieldSpec],
        draft: ConfigDraft,
        *,
        id_prefix: str | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        #: The draft section the form edits (``pipeline`` for the run level).
        self.section = section
        #: The draft the form writes into.
        self.draft = draft
        self._specs = tuple(specs)
        self._on_change = on_change
        self._errors: dict[str, str] = {}
        #: The widget-id prefix (the section, unless overridden — a screen
        #: hosting a section's form a second time keeps ids unique, e.g.
        #: the config screen's per-stage copies).
        self._id_prefix = section if id_prefix is None else id_prefix
        self._ids = {self._field_id(spec.key): spec for spec in self._specs}
        self._kinds = {spec.key: _widget_kind(spec) for spec in self._specs}
        self._literals = {spec.key: _literal_values(spec.hint) for spec in self._specs}
        self._resets = {
            f"reset-{self._field_id(spec.key)}": spec for spec in self._specs
        }
        self._browses = {
            f"browse-{self._field_id(spec.key)}": spec
            for spec in self._specs
            if _is_path_field(spec)
        }

    @property
    def valid(self) -> bool:
        """Whether every field's current text parses (the whole draft may
        still fail to resolve — the owning screen shows that separately)."""
        return not self._errors

    def error(self, key: str) -> str:
        """The inline error message of one field (``""`` when it parses)."""
        return self._errors.get(key, "")

    def _field_id(self, key: str) -> str:
        """The widget id of a field (prefix-prefixed, so ids stay unique
        across the coexisting screens of the stack)."""
        return f"{self._id_prefix}-{kebab(key)}"

    def _is_set(self, spec: FieldSpec) -> bool:
        """Whether the field's draft key is set (its reset affordance is
        enabled only then — at-default state stays visible)."""
        return self.draft.get(self.section, spec.key) is not None

    def compose(self) -> ComposeResult:
        for spec in self._specs:
            kind = self._kinds[spec.key]
            label = spec.label or kebab(spec.key)
            widget_id = self._field_id(spec.key)
            if kind == "checkbox":
                widget = Checkbox(label, value=self._checked(spec), id=widget_id)
            elif kind == "select":
                values = self._literals[spec.key]
                current = self.draft.get(self.section, spec.key)
                # the blank select shows its gray default in the prompt —
                # never a pre-picked value (the draft stays unset then)
                widget = Select(
                    [(str(value), value) for value in values],
                    prompt=spec.placeholder,
                    allow_blank=True,
                    value=current if current in values else Select.NULL,
                    id=widget_id,
                )
            else:
                widget = Input(
                    value=format_field(self.draft.get(self.section, spec.key)),
                    placeholder=spec.placeholder,
                    id=widget_id,
                )
            reset = Button("reset", id=f"reset-{widget_id}", classes="field-reset")
            reset.can_focus = False  # click-only: not a tab stop between fields
            reset.disabled = not self._is_set(spec)
            with Vertical(classes="option-field"):
                if kind != "checkbox":  # the checkbox shows its own label
                    yield Label(label, classes="field-label")
                with Horizontal(classes="field-row"):
                    yield widget
                    if _is_path_field(spec):  # one directory to pick: browse
                        yield BrowseButton(
                            BROWSE_LABEL,
                            id=f"browse-{widget_id}",
                            classes="field-browse",
                        )
                    yield reset
                yield Label(Text(spec.help), classes="field-help")
                yield Label(Text(""), classes="field-error", id=f"error-{widget_id}")

    def _checked(self, spec: FieldSpec) -> bool:
        """A checkbox's initial state: the draft value, else the field's
        default (a required bool defaults to unchecked)."""
        value = self.draft.get(self.section, spec.key)
        if value is None:
            value = spec.default if spec.default is not NO_DEFAULT else False
        return bool(value)

    def set_placeholder(self, key: str, text: str) -> None:
        """Set the greyed hint of one field's input (the derived default a
        blank input falls back to)."""
        self.query_one(f"#{self._field_id(key)}", Input).placeholder = text

    def refresh_placeholders(self, derived: Mapping[str, str]) -> None:
        """Re-lay every field's gray after a draft change — the config
        screen's overlay: ``derived``'s within-chain tree names wherever
        the chain derives them (while ``out_root`` is set), every other
        input its static gray (:attr:`FieldSpec.placeholder`); an empty
        mapping (a blank ``out_root``) re-lays the statics, so no stale
        derived name lingers after a clear. Selects keep their prompt and
        checkboxes their check state — the overlay never targets them."""
        for spec in self._specs:
            if self._kinds[spec.key] != "input":
                continue
            self.set_placeholder(spec.key, derived.get(spec.key, spec.placeholder))

    def refresh_from_draft(self) -> None:
        """Set every widget's value from the draft again.

        A screen is mounted once but entered many times, and another
        screen may have edited the same draft meanwhile. Idempotent:
        widgets already showing the draft's value emit no change events.
        """
        for spec in self._specs:
            self._refresh_one(spec)
        self._refresh_resets()

    def _refresh_one(self, spec: FieldSpec) -> None:
        """Set one field's widget from the draft again (the per-field half
        of :meth:`refresh_from_draft`, also the reset's re-show)."""
        kind = self._kinds[spec.key]
        widget = self.query_one(f"#{self._field_id(spec.key)}")
        if kind == "input":
            widget.value = format_field(self.draft.get(self.section, spec.key))
        elif kind == "checkbox":
            widget.value = self._checked(spec)
        else:
            value = self.draft.get(self.section, spec.key)
            widget.value = value if value in self._literals[spec.key] else Select.NULL

    # --- change handling ------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """An input changed: parse and write, or mark the field invalid."""
        spec = self._ids.get(event.input.id)
        if spec is not None:
            self._apply_text(spec, event.value)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """A checkbox changed: store the explicit value, unless it equals
        the field's default — which ``resolve`` applies to an unset key
        anyway. (A checkbox fires on initialization too; this keeps
        mount-time events from writing defaults into the draft.)"""
        spec = self._ids.get(event.checkbox.id)
        if spec is None:
            return
        default = bool(spec.default) if spec.default is not NO_DEFAULT else False
        if event.value == default:
            self.draft.clear(self.section, spec.key)
        else:
            self.draft.set(self.section, spec.key, event.value)
        self._refresh_reset(spec)
        self._changed()

    def on_select_changed(self, event: Select.Changed) -> None:
        """A select changed: store the chosen literal; blank clears."""
        spec = self._ids.get(event.select.id)
        if spec is None:
            return
        if event.select.is_blank():
            self.draft.clear(self.section, spec.key)
        else:
            self.draft.set(self.section, spec.key, event.value)
        self._refresh_reset(spec)
        self._changed()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """A field's browse button opens the directory picker for that
        field; a field's reset button clears exactly that field's draft
        key and re-shows the field."""
        browse = self._browses.get(event.button.id)
        if browse is not None:
            self._open_picker(browse)
            return
        spec = self._resets.get(event.button.id)
        if spec is None:
            return
        self.draft.clear(self.section, spec.key)
        self._refresh_one(spec)
        self._refresh_reset(spec)
        self._changed()

    def _open_picker(self, spec: FieldSpec) -> None:
        """Open the directory picker for ``spec``, starting at the input's
        current value's nearest existing directory; the picked directory
        lands in the input, and the normal change flow does the rest."""
        widget = self.query_one(f"#{self._field_id(spec.key)}", Input)

        def accept(path: Path) -> None:
            widget.value = str(path)

        open_directory_picker(self.app, picker_location(widget.value), accept)

    def _apply_text(self, spec: FieldSpec, text: str) -> None:
        """Parse ``text`` for ``spec`` and write it — or mark the field
        invalid and keep the draft's last valid value."""
        try:
            value = parse_field(text, spec.hint)
        except FieldError as err:
            self._errors[spec.key] = str(err)
            self._set_error(spec.key, str(err))
            self._changed()
            return
        if spec.key in self._errors:
            del self._errors[spec.key]
            self._set_error(spec.key, "")
        if value is UNSET:
            self.draft.clear(self.section, spec.key)
        else:
            self.draft.set(self.section, spec.key, value)
        self._refresh_reset(spec)
        self._changed()

    def _refresh_reset(self, spec: FieldSpec) -> None:
        """Sync one field's reset button with its key's set-vs-default
        state (enabled only while the key is set)."""
        button = self.query_one(f"#reset-{self._field_id(spec.key)}", Button)
        button.disabled = not self._is_set(spec)

    def _refresh_resets(self) -> None:
        """Sync every reset button."""
        for spec in self._specs:
            self._refresh_reset(spec)

    def _set_error(self, key: str, message: str) -> None:
        """Show or clear one field's inline error (message + red border).
        The message goes in as plain ``Text`` — error text must never be
        parsed as markup."""
        self.query_one(f"#error-{self._field_id(key)}", Label).update(Text(message))
        widget = self.query_one(f"#{self._field_id(key)}")
        if message:
            widget.add_class("invalid")
        else:
            widget.remove_class("invalid")

    def _changed(self) -> None:
        """Tell the owning screen the form (or its validity) changed."""
        if self._on_change is not None:
            self._on_change()


class FormScreen(Screen[None]):
    """Base for the screens that edit one run type's draft through an
    :class:`OptionForm`.

    The screen resolves its draft (the constructor's explicit one, else
    the running app's ``ColonyTuiApp.drafts`` entry for the screen's
    :attr:`run_type`), refreshes the form from it on every entry, and
    owns the action gate: the buttons named in :attr:`gated_actions`
    stay disabled until every field parses and the whole draft resolves;
    otherwise the status line shows the draft's exact ``ConfigError``
    message. Subclasses provide :meth:`compose_body` (which must set
    ``self._form``), their action buttons, and :meth:`refresh_extras`. A
    screen may edit several draft sections: :attr:`forms` lists them
    all, and the entry refresh and the gate cover every one.
    """

    #: The button ids the action gate enables and disables.
    gated_actions: ClassVar[tuple[str, ...]] = ()

    #: The form (set by :meth:`compose_body`).
    _form: OptionForm

    #: The first form input takes the focus on entry (the form's scroll
    #: container is focusable too and would win a plain ``*``); on
    #: re-entry textual restores the previous focus.
    AUTO_FOCUS = "OptionForm Input"

    #: Screen-level arrow-key field navigation: down/up land the focus in
    #: the form's fields when it is elsewhere (e.g. on the action
    #: buttons); inside the form the OptionForm's own bindings route the
    #: same keys here. Left/right stay inside the focused input.
    BINDINGS = [
        Binding("down", "next_field", "Next field", show=False),
        Binding("up", "previous_field", "Previous field", show=False),
    ]

    CSS = """
    #resolve-status {
        height: auto;
        color: $error;
    }
    .actions {
        height: auto;
    }
    .actions Button {
        margin: 0 1 1 0;
    }
    """

    def __init__(self, draft: ConfigDraft | None = None) -> None:
        super().__init__()
        self._draft = draft
        #: The draft this screen edits (resolved on entry).
        self.draft: ConfigDraft | None = None
        self._enabled = False

    @property
    def run_type(self) -> str:
        """The run type this screen edits (a key of
        :data:`~colony_tracking.tui.draft.RUN_TYPES`): the pipeline chain
        by default; a stage screen's is its own stage."""
        return "pipeline"

    def _resolve_draft(self) -> ConfigDraft:
        """The draft to edit: the explicit one, else the app's entry for
        :attr:`run_type` (an app without a ``drafts`` dict is a wiring
        mistake — fail loudly)."""
        if self._draft is not None:
            return self._draft
        drafts = getattr(self.app, "drafts", None)
        if drafts is None:
            raise RuntimeError(
                "no draft to edit — pass one to the screen, or run it on an "
                "app with a .drafts dict (ColonyTuiApp)"
            )
        return drafts[self.run_type]

    def compose_body(self) -> ComposeResult:
        """The screen's content between the header and the status line:
        the form (which must set ``self._form``) plus the screen's extras."""
        raise NotImplementedError

    def compose_actions(self) -> ComposeResult:
        """The action buttons (the gated ones, then *Back*)."""
        raise NotImplementedError

    def refresh_extras(self) -> None:
        """Refresh the screen's extras after a draft change (the config
        screen's TOML preview)."""
        raise NotImplementedError

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self.compose_body()
        yield Static(Text(""), id="resolve-status")
        with Vertical(classes="actions"):
            yield from self.compose_actions()
        yield Footer()

    def on_mount(self) -> None:
        self.draft = self._resolve_draft()
        self.refresh_screen()

    def on_screen_resume(self, event: events.ScreenResume) -> None:
        """Every entry refreshes from the draft (screens are cached across
        pushes; another screen may have edited the same draft meanwhile)."""
        self.refresh_screen()

    def refresh_screen(self) -> None:
        """Re-read the draft and refresh the forms, the extras, and the
        action gate (called on every entry; also right after editing the
        draft from outside the forms)."""
        self.draft = self._resolve_draft()
        for form in self.forms:
            form.refresh_from_draft()
        self.refresh_extras()
        self.refresh_gate()

    def refresh_gate(self) -> None:
        """Re-check the draft and the forms: enable the gated actions when
        everything resolves; otherwise disable them and show why — the
        draft's exact ``ConfigError`` message, verbatim."""
        message = ""
        try:
            self.draft.resolve()
        except ConfigError as err:
            self._enabled, message = False, str(err)
        else:
            self._enabled = all(form.valid for form in self.forms)
            if not self._enabled:
                message = "Fix the marked fields to enable the actions."
        self.query_one("#resolve-status", Static).update(Text(message))
        for action_id in self.gated_actions:
            self.query_one(f"#{action_id}", Button).disabled = not self._enabled

    @property
    def gate_open(self) -> bool:
        """Whether the actions are currently enabled (every field parses
        and the draft resolves)."""
        return self._enabled

    @property
    def forms(self) -> list[OptionForm]:
        """The screen's option forms (the primary :attr:`form` first) —
        the entry refresh and the action gate cover every one. Screens
        that edit a single section need not override this."""
        return [self._form]

    @property
    def form(self) -> OptionForm:
        """The screen's option form."""
        return self._form

    def action_write_toml(self) -> None:
        """*Write TOML…*: open the shared save-config dialog — it asks for
        the path, resolves the draft, and writes; errors stay inside it."""
        if not self._enabled:
            return
        self.app.push_screen(SaveConfigDialog(self.draft))

    def action_back(self) -> None:
        """*Back*: pop to the previous screen (the same escape-to-go-back
        the app provides; a no-op on the stack's bottom)."""
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def _field_widgets(self) -> list[Widget]:
        """The screen's field widgets in DOM order (one Input, Select, or
        Checkbox per field)."""
        return list(self.query("Input, Select, Checkbox"))

    def action_next_field(self) -> None:
        """Down arrow: move the focus to the next field of the form
        (wrapping; :func:`focus_field` rules)."""
        focus_field(self._field_widgets(), True)

    def action_previous_field(self) -> None:
        """Up arrow: move the focus to the previous field of the form
        (wrapping; :func:`focus_field` rules)."""
        focus_field(self._field_widgets(), False)

    def _resolved_or_report(self) -> PipelineConfig | Options | None:
        """The draft's resolved config, or ``None`` after showing why not
        (a defensive re-check at press time)."""
        try:
            return self.draft.resolve()
        except ConfigError as err:
            self.query_one("#resolve-status", Static).update(Text(str(err)))
            return None

    def _on_form_change(self) -> None:
        """The form changed the draft: refresh the extras and the gate,
        never the form's own widgets (mid-typing text must not be
        clobbered). Skipped while the screen is still mounting."""
        if not self.is_mounted:
            return
        self.refresh_extras()
        self.refresh_gate()
