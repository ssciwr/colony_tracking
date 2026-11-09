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
:meth:`~colony_tracking.tui.draft.ConfigDraft.load`. A stage screen's
draft accepts both its own flat TOML and a sectioned ``pipeline.toml``
(the stage's wired options are derived from it). An unreadable or
structurally wrong file is reported inside the dialog, which stays open;
a success notifies, dismisses with the loaded path, and the caller
screen refreshes its forms from the (reset) draft on resume. A loaded
config's relative path values are rebased onto the directory they were
written from (:func:`~colony_tracking.tui.draft.resolve_loaded_paths`),
so a ``pipeline.toml`` saved from a project directory still points at
its raw tree and outputs when the TUI is started somewhere else.

Both dialogs' path row carries the forms' *browse* button: it opens
textual-fspicker's file picker (``FileSave`` / ``FileOpen``) preferring
``*.toml`` (with an all-files fallback), starting at the typed path's
nearest existing directory with the typed file name pre-filled. The picked
file lands in the input and is written or loaded at once — choosing it in
the picker is the confirmation — and cancelling changes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Final, Generic, TypeVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Input, Label
from textual_fspicker import FileOpen, FileSave, Filters

from ..._options import Options
from ...pipeline import ConfigError, PipelineConfig
from ..draft import TOML_NAMES, TITLES, ConfigDraft, resolve_loaded_paths
from ..pickers import (
    BROWSE_BUTTON_CSS,
    BROWSE_LABEL,
    BrowseButton,
    open_picker,
    picker_location,
)


def _is_toml(path: Path) -> bool:
    """Whether a path looks like a config TOML file."""
    return path.suffix.lower() == ".toml"


def _any_file(_path: Path) -> bool:
    """The file picker's fallback filter: every file stays reachable."""
    return True


#: The file pickers' filters: config TOMLs first, every file still
#: reachable (a config without the conventional suffix).
TOML_FILTERS: Final[Filters] = Filters(
    ("TOML files (*.toml)", _is_toml),
    ("All files", _any_file),
)


def _value_name(raw: str) -> str:
    """The file name the path input's text names, or ``""`` when it names
    a directory (or nothing) — the picker's pre-filled file name."""
    text = raw.strip()
    path = Path(text)
    if not text or path.is_dir() or text.endswith(("/", "\\")):
        return ""
    return path.name


#: Shared layout for the two dialogs' path rows: the input takes the row,
#: the browse button sits beside it like a form field's button.
_PATH_ROW_CSS = (
    """
#toml-path-row {
    height: auto;
    align-vertical: middle;
}
#toml-path-row Input {
    width: 1fr;
}
"""
    + BROWSE_BUTTON_CSS
)


def default_toml_path(draft: ConfigDraft) -> Path | None:
    """The suggested config path for ``draft``: each run type its own file
    name, next to its own output setting and only its own —

    - the pipeline draft: ``<out_root>/pipeline.toml`` from the run
      level, ``None`` without an out root (unchanged);
    - ``difference`` / ``segmentation`` / ``prepare``: the ``out_dir``'s
      parent with the run type's
      :data:`~colony_tracking.tui.draft.TOML_NAMES` name (the trees and
      the h5 set are written into the ``out_dir``, the config sits beside
      them), ``None`` without an out_dir;
    - ``analysis``: the stage's own ``out_root`` with ``analysis.toml``,
      ``None`` without one.

    ``None`` starts the input empty (the user types a path).
    """
    run_type = draft.run_type
    if run_type in ("difference", "segmentation", "prepare"):
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


#: One config-file dialog's dismissed result.
ResultT = TypeVar("ResultT")


class _ConfigFileDialog(ModalScreen[ResultT], Generic[ResultT]):
    """The shared body of the two config-file dialogs: one path row with
    the forms' browse button, Enter and the browse pick committing through
    the same ``_commit``, and escape cancelling.

    Subclasses build their picker (:meth:`_make_picker`) and commit the
    chosen path (:meth:`_commit`); everything else — the row's widgets,
    the click-only button, the picker hand-off — stays here so the two
    doors cannot drift.
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    CSS = _PATH_ROW_CSS

    def __init__(self, draft: ConfigDraft) -> None:
        super().__init__()
        #: The draft the dialog reads or replaces (the caller's source of
        #: truth for its run type).
        self.draft = draft

    def _path_row(self, value: str = "") -> ComposeResult:
        """The caption row's widgets: the path input and its browse
        button, in the shared layout."""
        with Horizontal(id="toml-path-row"):
            yield Input(
                value=value,
                placeholder=f"/path/to/{TOML_NAMES[self.draft.run_type]}",
                id="toml-path",
            )
            yield BrowseButton(
                BROWSE_LABEL,
                id="browse-toml-path",
                classes="field-browse",
            )

    def _input(self) -> Input:
        """The dialog's path input."""
        return self.query_one("#toml-path", Input)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the path field commits the typed path."""
        if event.input.id == "toml-path":
            self._commit(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """The browse button opens the file picker for the typed path; the
        picked file lands in the input and is committed at once."""
        if event.button.id == "browse-toml-path":
            self._browse()

    def action_cancel(self) -> None:
        """Escape: close without writing or loading."""
        self.dismiss(None)

    def _browse(self) -> None:
        """Open the file picker at the typed path's nearest existing
        directory; the picked file is committed at once (choosing it is
        the confirmation)."""
        raw = self._input().value
        picker = self._make_picker(raw)

        def accept(path: Path) -> None:
            self._input().value = str(path)
            self._commit(str(path))

        open_picker(self.app, picker, accept)

    def _make_picker(self, raw: str) -> Screen[Any]:
        """The file picker for the typed path (subclass hook)."""
        raise NotImplementedError

    def _commit(self, path: str) -> None:
        """Write or load the chosen path (subclass hook)."""
        raise NotImplementedError


class SaveConfigDialog(_ConfigFileDialog[tuple[Path, PipelineConfig | Options] | None]):
    """Write the draft to a TOML file chosen here (see the module
    docstring).

    Dismisses with ``(path, resolved config)`` after a successful write
    and with ``None`` when cancelled — the caller's push-screen callback
    receives exactly that. The resolved config is the draft's run type's
    own: a :class:`~colony_tracking.pipeline.PipelineConfig` for the
    chain, the stage's own :class:`~colony_tracking._options.Options`
    otherwise.
    """

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
        yield from self._path_row(
            str(default) if default is not None else ""
        )
        yield Label("", id="toml-error")
        yield Footer()

    def _make_picker(self, raw: str) -> Screen[Any]:
        return FileSave(
            picker_location(raw),
            title="Write config TOML",
            filters=TOML_FILTERS,
            default_file=(
                _value_name(raw) or TOML_NAMES[self.draft.run_type]
            ),
        )

    def _commit(self, path: str) -> None:
        self._write(path)

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


class LoadConfigDialog(_ConfigFileDialog[Path | None]):
    """Load a config TOML file into the caller's draft (see the module
    docstring).

    Dismisses with the loaded path after a successful
    :meth:`~colony_tracking.tui.draft.ConfigDraft.load` and with ``None``
    when cancelled — the caller's push-screen callback receives exactly
    that. An unreadable or structurally wrong file is reported inside the
    dialog and keeps it open for correction.
    """

    def compose(self) -> ComposeResult:
        yield Label(
            f"Load the {TITLES[self.draft.run_type].lower()} config TOML from:"
        )
        yield from self._path_row()
        yield Label("", id="toml-error")
        yield Footer()

    def _make_picker(self, raw: str) -> Screen[Any]:
        return FileOpen(
            picker_location(raw),
            title="Load config TOML",
            filters=TOML_FILTERS,
            default_file=_value_name(raw) or None,
        )

    def _commit(self, path: str) -> None:
        self._load(path)

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
        base = resolve_loaded_paths(self.draft, path)
        message = f"Config loaded from {path}"
        if base is not None and base != Path.cwd():
            message += f" (relative paths resolved against {base})"
        self.notify(message, markup=False)
        self.dismiss(path)
