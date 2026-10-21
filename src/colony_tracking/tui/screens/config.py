"""The pipeline-config screen: the run level and the effective TOML.

The run-level form (``raw_dir``, ``out_root``, ``reference``, ``wells`` —
the ``[pipeline]`` section, generated from :class:`~colony_tracking.
pipeline.PipelineConfig`'s own fields through
:mod:`colony_tracking.tui.forms`) plus the draft's effective TOML preview:
the whole document as it stands right now, rendered from
``ConfigDraft.document`` — exactly the keys the TUI has set, with the
defaults left to apply at resolve time. *Write TOML…* opens the shared
save-config dialog, which writes the resolved config (the written file
carries the full effective document and
``colony-tracking pipeline --config`` accepts it unchanged).

The stage sections are edited on their own stage screens; this screen is
the run level's home — it shares the draft with every other screen, so a
change here re-gates the stage screens' actions on their next entry.
"""

from __future__ import annotations

from typing import ClassVar

import tomli_w
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from ..draft import RUN_LEVEL
from ..forms import FormScreen, OptionForm, run_level_specs


class ConfigScreen(FormScreen):
    """The run-level screen: the ``[pipeline]`` form, the draft's effective
    TOML preview, and *Write TOML…* (plus *Back*). The write action gates
    on the draft exactly like the stage screens' actions do."""

    gated_actions: ClassVar[tuple[str, ...]] = ("write-toml",)

    BINDINGS = [Binding("w", "write_toml", "Write TOML…")]

    CSS = """
    #config-title {
        text-style: bold;
        color: $accent;
    }
    #toml-pane {
        height: 1fr;
        border: round $border;
    }
    #toml-preview {
        padding: 0 1;
    }
    """

    def compose_body(self) -> ComposeResult:
        self._form = OptionForm(
            RUN_LEVEL,
            run_level_specs(),
            self._resolve_draft(),
            on_change=self._on_form_change,
        )
        yield Static("Pipeline config — run level", id="config-title")
        yield self._form
        with VerticalScroll(id="toml-pane"):
            yield Static(Text(""), id="toml-preview")

    def refresh_extras(self) -> None:
        """The effective TOML preview: the whole draft document as it
        stands, rendered as TOML (markup-free — section names like
        ``[pipeline]`` must show as typed)."""
        self.query_one("#toml-preview", Static).update(
            Text(tomli_w.dumps(self.draft.document()))
        )

    def compose_actions(self) -> ComposeResult:
        yield Button("Write TOML…", id="write-toml")
        yield Button("Back", id="back")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "write-toml":
            self.action_write_toml()
        elif event.button.id == "back":
            self.action_back()
