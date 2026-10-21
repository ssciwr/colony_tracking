"""The save-config dialog: one shared path prompt for every "write the
TOML" action.

Every screen that can write the draft's config file — the stage screens'
*Write TOML…*, the quick start's *Write TOML…* and *Run now* (which writes
first so the run is reproducible), the pipeline-config screen — opens
:class:`SaveConfigDialog` instead of owning its own path prompt. The dialog
does the whole write itself: it asks for the path, resolves the draft,
writes the file through
:meth:`~colony_tracking.tui.draft.ConfigDraft.write`, and dismisses with
``(path, config)`` on success or ``None`` on cancel. An invalid draft
(:class:`~colony_tracking.pipeline.ConfigError`) or an unwritable path is
reported inside the dialog — the message stays on screen, nothing is
written, and the dialog stays open for correction.

The suggested default is ``<out_root>/pipeline.toml`` (see
:func:`default_toml_path`), so a run started right after saving keeps its
config next to its outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label

from ...pipeline import ConfigError, PipelineConfig
from ..draft import RUN_LEVEL, ConfigDraft


def default_toml_path(draft: ConfigDraft) -> Path | None:
    """The suggested config path for ``draft``: ``<out_root>/pipeline.toml``
    when the run level has an out root, else ``None`` (the input starts
    empty and the user types a path)."""
    out_root = draft.get(RUN_LEVEL, "out_root")
    if not out_root:
        return None
    return Path(str(out_root)) / "pipeline.toml"


class SaveConfigDialog(ModalScreen[tuple[Path, PipelineConfig] | None]):
    """Write the draft to a TOML file chosen here (see the module
    docstring).

    Dismisses with ``(path, resolved config)`` after a successful write and
    with ``None`` when cancelled — the caller's push-screen callback
    receives exactly that.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, draft: ConfigDraft) -> None:
        super().__init__()
        #: The draft written on submit (the caller's shared source of truth).
        self.draft = draft

    def compose(self) -> ComposeResult:
        default = default_toml_path(self.draft)
        yield Label(
            "Write the config TOML to (runs later with: colony-tracking "
            "pipeline --config):"
        )
        yield Input(
            value=str(default) if default is not None else "",
            placeholder="/path/to/pipeline.toml",
            id="toml-path",
        )
        yield Label("", id="toml-error")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the path field writes the config."""
        if event.input.id == "toml-path":
            self._write(event.value)

    def action_cancel(self) -> None:
        """Escape: close without writing."""
        self.dismiss(None)

    def _write(self, raw: str) -> None:
        """Write the draft to ``raw`` (typed path), or report why not."""
        path_text = raw.strip()
        error_label = self.query_one("#toml-error", Label)
        if not path_text:
            error_label.update("Enter a path for the config TOML.")
            return
        path = Path(path_text)
        try:
            config = self.draft.write(path)
        except ConfigError as err:
            error_label.update(str(err))
            return
        except OSError as err:
            error_label.update(f"could not write {path}: {err}")
            return
        self.notify(f"Config written to {path}")
        self.dismiss((path, config))
