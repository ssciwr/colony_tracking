"""The quick-start screen: the full-auto flow.

Four run-level fields — the raw directory and the out root (both
required, matching the pipeline's contract), plus the optional reference
frame and wells filter — and two gated actions:

- *Write TOML…* writes the draft through the shared
  :class:`~colony_tracking.tui.screens._save.SaveConfigDialog` — a valid
  :class:`~colony_tracking.pipeline.PipelineConfig` document the CLI's
  ``pipeline --config`` accepts unchanged (config-only mode);
- *Run now* is the full-auto chain: the same shared dialog writes the
  TOML first (reproducibility — the suggested path is
  ``<out_root>/pipeline.toml``), then the whole chain runs on the
  resolved config with every stage option at its default, on the shared
  :class:`~colony_tracking.tui.screens.run.RunScreen`.

Each empty field shows in gray what a blank resolves to — the forms'
shared placeholder rule (:attr:`~colony_tracking.tui.forms.FieldSpec.placeholder`),
the same run-level grays the config screen shows.

The quick start edits the pipeline run type's draft only — the chain's
own :class:`~colony_tracking.tui.draft.ConfigDraft`
(:attr:`draft`, the app's ``drafts["pipeline"]`` entry), the document
whose ``pipeline.toml`` it writes: the stage run types' drafts are
independent of it and untouched here. Like the stage screens' forms, the
fields live-sync into the draft on every change — an emptied field
clears its key, back to the default — and both actions stay disabled
until the draft resolves (the stage screens' gate): a field the shared
parser rejects keeps its last valid value and shows its error on the
status line, everything else shows the pipeline's exact
:class:`~colony_tracking.pipeline.ConfigError` message verbatim (the
same text the CLI reports). Nothing is written or run while the gate is
closed; the dialog reports deeper config errors (exact problem, nothing
written).

The two path fields (raw directory, out root) carry the forms' browse
button (see :mod:`colony_tracking.tui.forms`): it opens the directory
picker, and the picked directory lands in the input — the change flow
above does the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Final

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from ..._options import parse_reference
from ...pipeline import ConfigError
from ..draft import RUN_LEVEL, ConfigDraft
from ..forms import (
    BROWSE_LABEL,
    BrowseButton,
    FieldSpec,
    focus_field,
    open_directory_picker,
    picker_location,
    run_level_specs,
)
from ._save import SaveConfigDialog
from .run import open_pipeline_run

#: The run-level fields the quick start shows. The labels pull their
#: help text from the config dataclass's metadata — the single source,
#: like the CLI's flag help.
_FIELDS: tuple[str, ...] = ("raw_dir", "out_root", "reference", "wells")

#: The fields that get a browse button: the run level's single-Path
#: fields (reference and wells are not paths — see
#: :mod:`colony_tracking.tui.forms`, the Browse section).
_BROWSE_FIELDS: Final[tuple[str, ...]] = ("raw_dir", "out_root")

#: The run-level form specs, by field name — a field's help and its
#: gray text (what a blank resolves to, :attr:`FieldSpec.placeholder`)
#: come from here: the same specs the config screen's run-level form
#: renders from, so the two screens show the same grays.
_SPECS: Final[dict[str, FieldSpec]] = {
    spec.key: spec for spec in run_level_specs()
}


def _field_id(name: str) -> str:
    """The widget id of one field (kebab-case, like the CLI's flags)."""
    return f"qs-{name.replace('_', '-')}"


#: The browse buttons' ids → their field (the press handler's routing).
_BROWSE_BUTTONS: Final[dict[str, str]] = {
    f"browse-{_field_id(name)}": name for name in _BROWSE_FIELDS
}


def _help(name: str) -> str:
    """The field's help text (the spec's — the config dataclass's own
    metadata)."""
    return _SPECS[name].help


def _prefill(draft: ConfigDraft, name: str) -> str:
    """The input's initial value from the draft (an unset key starts
    empty; lists render comma-separated)."""
    value = draft.get(RUN_LEVEL, name)
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


class QuickStartScreen(Screen[None]):
    """The quick-start screen: the full-auto flow (see the module
    docstring). Edits the pipeline chain's draft — the app's
    ``drafts["pipeline"]`` entry (:attr:`draft`) — and gates its actions
    on it like the stage screens do.
    """

    @property
    def draft(self) -> ConfigDraft:
        """The draft this screen edits: the pipeline chain's — the app's
        ``drafts["pipeline"]`` entry. The quick start writes the chain's
        ``pipeline.toml``; every stage run type's draft (and its
        ``difference.toml`` / ``segmentation.toml`` / ``analysis.toml``)
        is independent of it."""
        return self.app.drafts["pipeline"]

    CSS = """
    #qs-body { height: 1fr; }
    #qs-intro { padding: 0 1; margin-bottom: 1; }
    .qs-label { margin-top: 1; padding: 0 1; }
    .qs-row { height: auto; align-vertical: middle; }
    .qs-row Input { width: 1fr; }  # not 100%: the row shares it with the browse button
    .qs-row .field-browse {
        min-width: 0;
        width: auto;
        height: 1;
        border: none;
        background: $panel;
        color: $text 60%;
        margin-left: 1;
    }
    .qs-row .field-browse:hover {
        background: $surface-darken-1;
        color: $text;
    }
    #qs-error { color: red; padding: 0 1; margin-top: 1; display: none; }
    #qs-actions { height: auto; margin-top: 1; padding: 0 1; }
    #qs-actions Button { margin-right: 1; }
    """

    #: The action buttons the gate enables and disables (the stage
    #: screens' gate: disabled until the draft resolves).
    GATED_ACTIONS: ClassVar[tuple[str, ...]] = ("qs-write", "qs-run-now")

    #: Screen-level arrow-key field navigation (the form screens'
    #: bindings): down/up step between the run-level fields, wrapping;
    #: the inputs' own cursor bindings keep left/right in-field.
    BINDINGS = [
        Binding("down", "next_field", "Next field", show=False),
        Binding("up", "previous_field", "Previous field", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._enabled = False

    def compose(self) -> ComposeResult:
        draft = self.draft
        yield Header()
        # One scrollable body: the actions below stay reachable on a small
        # terminal whatever the fields' combined height (the stage
        # screens' pattern).
        with VerticalScroll(id="qs-body"):
            yield Static(
                "Full-auto: raw frames to final analysis in one action — every "
                "stage option takes its default. The config TOML is written "
                "first, so the run is reproducible.",
                id="qs-intro",
                markup=False,
            )
            for name in _FIELDS:
                yield Label(
                    f"{name.replace('_', '-')} — {_help(name)}", classes="qs-label"
                )
                widget = Input(
                    value=_prefill(draft, name),
                    placeholder=_SPECS[name].placeholder,
                    id=_field_id(name),
                )
                if name in _BROWSE_FIELDS:  # a single-Path field: browse joins it
                    with Horizontal(classes="qs-row"):
                        yield widget
                        yield BrowseButton(
                            BROWSE_LABEL,
                            id=f"browse-{_field_id(name)}",
                            classes="field-browse",
                        )
                else:
                    yield widget
        yield Static("", id="qs-error", markup=False)
        with Horizontal(id="qs-actions"):
            yield Button("Write TOML…", id="qs-write")
            yield Button("Run now", id="qs-run-now", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """The gate's first pass: whatever the draft holds resolves (or
        not) and the actions are set accordingly."""
        self.refresh_gate()

    def on_input_changed(self, event: Input.Changed) -> None:
        """A field changed: the gate re-syncs the fields into the draft
        and re-checks it, so the actions gate on exactly what is on
        screen (the stage screens' forms behave the same way)."""
        self.refresh_gate()

    def action_next_field(self) -> None:
        """Down arrow: move the focus to the next run-level field
        (wrapping; :func:`~colony_tracking.tui.forms.focus_field` rules)."""
        focus_field(list(self.query(Input)), True)

    def action_previous_field(self) -> None:
        """Up arrow: move the focus to the previous run-level field
        (wrapping; :func:`~colony_tracking.tui.forms.focus_field` rules)."""
        focus_field(list(self.query(Input)), False)

    def refresh_gate(self) -> None:
        """Re-check the draft and enable or disable the actions: the
        fields sync into the draft first, then the draft resolves —
        enabled when it does; otherwise disabled with the reason on the
        status line: the field's parse error, or the pipeline's exact
        ``ConfigError`` message, verbatim (the stage screens' gate, see
        :meth:`~colony_tracking.tui.forms.FormScreen.refresh_gate`)."""
        message = self._sync_and_resolve()
        self._enabled = message is None
        error = self.query_one("#qs-error", Static)
        error.update(message or "")
        error.display = message is not None
        for action_id in self.GATED_ACTIONS:
            self.query_one(f"#{action_id}", Button).disabled = not self._enabled

    @property
    def gate_open(self) -> bool:
        """Whether the actions are currently enabled (the fields sync and
        the draft resolves)."""
        return self._enabled

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """A browse button opens the directory picker for its path field
        (the picked directory lands in the input — the change flow syncs
        the draft and re-gates the actions from there); the two actions
        (see the module docstring) keep the gate: disabled until the draft
        resolves, and a press re-syncs and re-resolves defensively (the
        gate normally guarantees this) and reports the problem instead of
        writing or running."""
        field = _BROWSE_BUTTONS.get(event.button.id or "")
        if field is not None:
            self._open_picker(field)
            return
        if event.button.id not in self.GATED_ACTIONS or not self._enabled:
            return
        problem = self._sync_and_resolve()
        label = self.query_one("#qs-error", Static)
        if problem is not None:
            label.update(problem)
            label.display = True
            return
        label.display = False
        if event.button.id == "qs-write":
            self.app.push_screen(SaveConfigDialog(self.draft))
            return
        run_problem = open_pipeline_run(self.app, self.draft)
        if run_problem is not None:  # reported as a toast too; keep a line here
            label.update(run_problem)
            label.display = True

    def _open_picker(self, name: str) -> None:
        """Open the directory picker for the path field ``name``, starting
        at the field's current value's nearest existing directory
        (:func:`~colony_tracking.tui.forms.picker_location`): the picked
        directory lands in the field's input, which fires the normal
        change flow (the draft sync, the gate)."""
        widget = self.query_one(f"#{_field_id(name)}", Input)

        def accept(path: Path) -> None:
            widget.value = str(path)

        open_directory_picker(self.app, picker_location(widget.value), accept)

    def _sync_and_resolve(self) -> str | None:
        """The fields → the draft, then the draft resolved: the problem
        message when either step fails (the field's parse error, or the
        pipeline's exact ``ConfigError``), ``None`` when the config is
        good to go."""
        problem = self._sync_draft()
        if problem is not None:
            return problem
        try:
            self.draft.resolve()
        except ConfigError as err:
            return str(err)
        return None

    def _sync_draft(self) -> str | None:
        """The inputs → the app's draft (set or clear — an emptied field
        is back to its default). Each field syncs on its own, like the
        stage screens' forms: a reference the shared parser
        (:func:`~colony_tracking._options.parse_reference`) rejects leaves
        its key at the last valid value and comes back as the problem
        message, while the other fields still sync."""
        draft = self.draft
        wells_text = self.query_one("#qs-wells", Input).value.strip()
        wells = (
            None
            if not wells_text
            else [part.strip() for part in wells_text.split(",") if part.strip()]
        )
        values: dict[str, Any] = {
            "raw_dir": self.query_one("#qs-raw-dir", Input).value.strip(),
            "out_root": self.query_one("#qs-out-root", Input).value.strip(),
            "wells": wells,
        }
        for name, value in values.items():
            if value is None or value == "" or value == []:
                draft.clear(RUN_LEVEL, name)
            else:
                draft.set(RUN_LEVEL, name, value)
        text = self.query_one("#qs-reference", Input).value.strip()
        if not text:
            draft.clear(RUN_LEVEL, "reference")
            return None
        try:
            draft.set(RUN_LEVEL, "reference", parse_reference(text))
        except ValueError as err:
            return str(err)
        return None
