"""The quick-start screen: the full-auto flow (spec requirement 1).

Four run-level fields — the raw directory and the out root (both
required, matching the pipeline's contract), plus the optional reference
frame and wells filter — and two actions:

- *Write TOML…* writes the draft through the shared
  :class:`~colony_tracking.tui.screens._save.SaveConfigDialog` — a valid
  :class:`~colony_tracking.pipeline.PipelineConfig` document the CLI's
  ``pipeline --config`` accepts unchanged (config-only mode, spec
  requirement 4);
- *Run now* is the full-auto chain: the same shared dialog writes the
  TOML first (reproducibility, spec requirement 5 — the suggested path
  is ``<out_root>/pipeline.toml``), then the whole chain runs on the
  resolved config with every stage option at its default, on the shared
  :class:`~colony_tracking.tui.screens.run.RunScreen`.

The field values sync into the app's one
:class:`~colony_tracking.tui.draft.ConfigDraft` on every action — an
emptied field clears its key, back to the default — so the other screens
see exactly what the quick start shows. A value the config cannot accept
(a reference that is neither ``first`` nor a frame number) is reported
inline and nothing is written or run; the dialog reports deeper config
errors the same way the spec asks for (exact problem, nothing written).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from ...pipeline import PipelineConfig
from ..draft import RUN_LEVEL, ConfigDraft
from ._save import SaveConfigDialog
from .run import open_pipeline_run

#: The run-level fields the quick start shows, in the spec's order. The
#: labels pull their help text from the config dataclass's metadata —
#: the single source, like the CLI's flag help.
_FIELDS: tuple[str, ...] = ("raw_dir", "out_root", "reference", "wells")

#: Placeholder text per field (the hint an empty input shows).
_PLACEHOLDERS: dict[str, str] = {
    "raw_dir": "/path/to/raw-plate-frames",
    "out_root": "/path/to/output-root",
    "reference": "first",
    "wells": "-A001, -A002",
}


def _field_id(name: str) -> str:
    """The widget id of one field (kebab-case, like the CLI's flags)."""
    return f"qs-{name.replace('_', '-')}"


def _help(name: str) -> str:
    """The field's help text from :class:`PipelineConfig`'s metadata."""
    (field,) = [f for f in fields(PipelineConfig) if f.name == name]
    return field.metadata.get("help", "")


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
    docstring). Edits the app's shared draft — :attr:`App.draft` — the
    same document every other screen sees."""

    CSS = """
    #qs-intro { padding: 0 1; margin-bottom: 1; }
    .qs-label { margin-top: 1; padding: 0 1; }
    #qs-error { color: red; padding: 0 1; margin-top: 1; display: none; }
    #qs-actions { height: auto; margin-top: 1; padding: 0 1; }
    #qs-actions Button { margin-right: 1; }
    """

    def compose(self) -> ComposeResult:
        draft = self.app.draft
        yield Header()
        yield Static(
            "Full-auto: raw images to final analysis in one action — every "
            "stage option takes its default. The config TOML is written "
            "first, so the run is reproducible.",
            id="qs-intro",
            markup=False,
        )
        for name in _FIELDS:
            yield Label(f"{name.replace('_', '-')} — {_help(name)}", classes="qs-label")
            yield Input(
                value=_prefill(draft, name),
                placeholder=_PLACEHOLDERS[name],
                id=_field_id(name),
            )
        yield Static("", id="qs-error", markup=False)
        with Horizontal(id="qs-actions"):
            yield Button("Write TOML…", id="qs-write")
            yield Button("Run now", id="qs-run-now", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """The two actions (see the module docstring). Both sync the
        fields into the draft first; a field the config cannot accept is
        reported inline and nothing is written or run."""
        if event.button.id not in ("qs-write", "qs-run-now"):
            return
        error = self._sync_draft()
        label = self.query_one("#qs-error", Static)
        if error is not None:
            label.update(error)
            label.display = True
            return
        label.display = False
        if event.button.id == "qs-write":
            self.app.push_screen(SaveConfigDialog(self.app.draft))
            return
        problem = open_pipeline_run(self.app, self.app.draft)
        if problem is not None:  # reported as a toast too; keep a line here
            label.update(problem)
            label.display = True

    def _sync_draft(self) -> str | None:
        """The inputs → the app's draft (set or clear — an emptied field
        is back to its default). Returns the problem message when a
        field's value is not usable; nothing is synced then."""
        draft = self.app.draft
        text = self.query_one("#qs-reference", Input).value.strip()
        if text and text != "first" and not text.isdigit():
            return f'reference must be "first" or a 1-based frame number (got {text!r})'
        reference: Any
        if not text:
            reference = None
        elif text == "first":
            reference = "first"
        else:
            reference = int(text)
        wells_text = self.query_one("#qs-wells", Input).value.strip()
        wells = (
            None
            if not wells_text
            else [part.strip() for part in wells_text.split(",") if part.strip()]
        )
        values: dict[str, Any] = {
            "raw_dir": self.query_one("#qs-raw-dir", Input).value.strip(),
            "out_root": self.query_one("#qs-out-root", Input).value.strip(),
            "reference": reference,
            "wells": wells,
        }
        for name, value in values.items():
            if value is None or value == "" or value == []:
                draft.clear(RUN_LEVEL, name)
            else:
                draft.set(RUN_LEVEL, name, value)
        return None
