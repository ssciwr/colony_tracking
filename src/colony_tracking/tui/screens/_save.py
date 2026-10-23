"""The save- and load-config dialogs: the TUI's config-file doors.

Every screen that can write its draft's config file opens
:class:`SaveConfigDialog` instead of owning its own path prompt. The
dialog does the whole write itself: it asks for the path, resolves the
draft, writes the file through
:meth:`~colony_tracking.tui.draft.ConfigDraft.write`, and dismisses with
``(path, config)`` on success or ``None`` on cancel. An invalid draft
(:class:`~colony_tracking.pipeline.ConfigError`) or an unwritable path
is reported inside the dialog — the message stays on screen, nothing is
written, and the dialog stays open for correction. The suggested path
(:func:`default_toml_path`) is the run type's own file name
(:data:`~colony_tracking.tui.draft.TOML_NAMES`) next to its own output
setting, so a run started right after saving keeps its config beside its
outputs.

:class:`LoadConfigDialog` is the read door: it asks for a path and loads
the file into the caller's draft through
:meth:`~colony_tracking.tui.draft.ConfigDraft.load`. An unreadable or
structurally wrong file is reported inside the dialog, which stays open;
a success notifies, dismisses with the loaded path, and the caller
screen refreshes its forms from the (reset) draft on resume.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label

from ..._options import Options
from ...pipeline import ConfigError, PipelineConfig
from ..draft import TOML_NAMES, TITLES, ConfigDraft


def default_toml_path(draft: ConfigDraft) -> Path | None:
    """The suggested config path for ``draft``: each run type its own file
    name, next to its own output setting and only its own —

    - the pipeline draft: ``<out_root>/pipeline.toml`` from the run
      level, ``None`` without an out root (unchanged);
    - ``difference`` / ``segmentation``: the ``out_dir``'s parent with
      the run type's :data:`~colony_tracking.tui.draft.TOML_NAMES` name
      (the trees are written into the ``out_dir``, the config sits beside
      them), ``None`` without an out_dir;
    - ``analysis``: the stage's own ``out_root`` with ``analysis.toml``,
      ``None`` without one.

    ``None`` starts the input empty (the user types a path).
    """
    run_type = draft.run_type
    if run_type in ("difference", "segmentation"):
        out_dir = draft.get(run_type, "out_dir")
        if not out_dir:
            return None
        return Path(str(out_dir)).parent / TOML_NAMES[run_type]
    # The pipeline (out_root is the run level's own) and the analysis
    # stage (out_root is its own section's) read out_root alike.
    out_root = draft.get(run_type, "out_root")
    if not out_root:
        return None
    return Path(str(out_root)) / TOML_NAMES[run_type]


class SaveConfigDialog(ModalScreen[tuple[Path, PipelineConfig | Options] | None]):
    """Write the draft to a TOML file chosen here (see the module
    docstring).

    Dismisses with ``(path, resolved config)`` after a successful write
    and with ``None`` when cancelled — the caller's push-screen callback
    receives exactly that. The resolved config is the draft's run type's
    own: a :class:`~colony_tracking.pipeline.PipelineConfig` for the
    chain, the stage's own :class:`~colony_tracking._options.Options`
    otherwise.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, draft: ConfigDraft) -> None:
        super().__init__()
        #: The draft written on submit (the caller's source of truth for
        #: its run type).
        self.draft = draft

    def compose(self) -> ComposeResult:
        default = default_toml_path(self.draft)
        if self.draft.is_pipeline:
            caption = (
                "Write the config TOML to (runs later with: colony-tracking "
                "pipeline --config):"
            )
        else:
            caption = (
                f"Write the {TITLES[self.draft.run_type].lower()} run's "
                "config TOML to:"
            )
        yield Label(caption)
        yield Input(
            value=str(default) if default is not None else "",
            placeholder=f"/path/to/{TOML_NAMES[self.draft.run_type]}",
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
        """Write the draft to ``raw`` (typed path), or report why not. The
        on-screen messages go in as plain ``Text`` — they name config
        sections (``[pipeline]``) and paths, which markup would eat — and
        the success toast goes out as a plain markup-free string:
        ``App.notify`` re-processes the content itself, so a rich ``Text``
        crashes it and markup would eat bracket-bearing paths."""
        path_text = raw.strip()
        error_label = self.query_one("#toml-error", Label)
        if not path_text:
            error_label.update("Enter a path for the config TOML.")
            return
        path = Path(path_text)
        try:
            config = self.draft.write(path)
        except ConfigError as err:
            error_label.update(Text(str(err)))
            return
        except OSError as err:
            error_label.update(Text(f"could not write {path}: {err}"))
            return
        self.notify(f"Config written to {path}", markup=False)
        self.dismiss((path, config))


class LoadConfigDialog(ModalScreen[Path | None]):
    """Load a config TOML file into the caller's draft (see the module
    docstring).

    Dismisses with the loaded path after a successful
    :meth:`~colony_tracking.tui.draft.ConfigDraft.load` and with ``None``
    when cancelled — the caller's push-screen callback receives exactly
    that. An unreadable or structurally wrong file is reported inside the
    dialog and keeps it open for correction.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, draft: ConfigDraft) -> None:
        super().__init__()
        #: The draft the loaded document replaces (a load resets, it never
        #: merges; the caller refreshes its forms from it on resume).
        self.draft = draft

    def compose(self) -> ComposeResult:
        yield Label(
            f"Load the {TITLES[self.draft.run_type].lower()} config TOML from:"
        )
        yield Input(
            placeholder=f"/path/to/{TOML_NAMES[self.draft.run_type]}",
            id="toml-path",
        )
        yield Label("", id="toml-error")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the path field loads the config."""
        if event.input.id == "toml-path":
            self._load(event.value)

    def action_cancel(self) -> None:
        """Escape: close without loading."""
        self.dismiss(None)

    def _load(self, raw: str) -> None:
        """Load ``raw`` (typed path) into the draft, or report why not.
        The on-screen messages go in as plain ``Text`` — they name config
        sections and paths, which markup would eat — and the success
        toast goes out as a plain markup-free string: ``App.notify``
        re-processes the content itself, so a rich ``Text`` crashes it
        and markup would eat bracket-bearing paths."""
        path_text = raw.strip()
        error_label = self.query_one("#toml-error", Label)
        if not path_text:
            error_label.update("Enter a path for the config TOML.")
            return
        path = Path(path_text)
        try:
            self.draft.load(path)
        except ConfigError as err:
            error_label.update(Text(str(err)))
            return
        except OSError as err:
            error_label.update(Text(f"could not read {path}: {err}"))
            return
        self.notify(f"Config loaded from {path}", markup=False)
        self.dismiss(path)
