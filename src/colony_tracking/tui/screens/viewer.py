"""The viewer screen: pick wells and open them in napari.

The screen offers exactly the wells the chain draft's configured sources
actually hold — the same discovery the ``view`` command uses
(:func:`colony_tracking.viewer.available_wells` over the
:class:`~colony_tracking._options.ViewOptions` the pipeline draft
resolves to), so an empty well is never offered. The list is
multi-select; *Open in napari* starts :mod:`colony_tracking.tui.launch`,
which runs ``colony-tracking view`` as a separate process with the
selected wells and the draft's resolved paths. The TUI never imports
napari and stays usable while the viewer is open.

Everything about a launch — the command, the child's lines, and a
failure (napari missing, no display) — lands in the screen's log pane
through the package logger
(:data:`colony_tracking.log`, forwarded by
:func:`~colony_tracking.tui.records.forwarding`). A failed launch
never takes the interface down.

A draft that does not resolve (the chain's required run level missing)
shows the exact :class:`~colony_tracking.pipeline.ConfigError` on the
status line and in the log; the list stays empty and the screen stays
usable.
"""

from __future__ import annotations

import logging

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, RichLog, SelectionList, Static
from textual.widgets.selection_list import Selection

from ... import log
from ..._options import ViewOptions
from ...pipeline import ConfigError
from ...viewer import available_wells
from .. import launch
from ..draft import ConfigDraft
from .run import _record_text

#: The status line's normal hint (replaced by an error when the draft
#: does not resolve).
HINT = "Wells in the pipeline draft's sources — select one or more."


class ViewerScreen(Screen[None]):
    """Pick wells and open them in napari (see the module docstring).

    The screen edits nothing: it reads the app's ``drafts["pipeline"]``
    entry on every entry (the draft a quick start or the config screen
    fills) and offers its sources' wells.
    """

    BINDINGS = [Binding("o", "open_viewer", "Open in napari")]

    CSS = """
    #view-hint { padding: 0 1; color: $text 60%; }
    #view-status { padding: 0 1; color: $error; height: auto; }
    #view-wells { height: 1fr; }
    #view-actions { height: auto; }
    #view-actions Button { margin: 0 1 1 0; }
    #view-log { height: 1fr; padding: 0 1; }
    """

    def __init__(self, draft: ConfigDraft | None = None) -> None:
        super().__init__()
        self._draft = draft
        #: The draft the screen reads (the app's pipeline entry by default).
        self.draft: ConfigDraft | None = None
        #: The resolved viewer session (None when the draft does not resolve).
        self.options: ViewOptions | None = None
        #: The wells the configured sources hold, sorted — the selection's
        #: values, and nothing else.
        self.wells: list[str] = []
        #: Every launch this screen started, in order (the last one is the
        #: most recent).
        self.handles: list[launch.ViewHandle] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(Text(HINT), id="view-hint")
        yield Static(Text(""), id="view-status")
        yield SelectionList(id="view-wells")
        with Horizontal(id="view-actions"):
            yield Button("Open in napari", id="open-viewer", disabled=True)
            yield Button("Back", id="view-back")
        yield RichLog(id="view-log", wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.draft = (
            self._draft if self._draft is not None else self.app.drafts["pipeline"]
        )
        self.refresh_wells()

    def on_screen_resume(self) -> None:
        """Every entry re-reads the draft: another screen may have filled
        or changed its paths meanwhile."""
        self.refresh_wells()

    def refresh_wells(self) -> None:
        """List the wells the draft's configured sources hold, or report
        why the draft does not resolve (the list stays empty then)."""
        selection = self.query_one("#view-wells", SelectionList)
        selection.clear_options()
        try:
            config = self.draft.resolve()
        except ConfigError as err:
            self.options, self.wells = None, []
            self._report(str(err))
            self._sync_actions()
            return
        self.options = launch.view_options(config)
        self.wells = available_wells(self.options)
        selection.add_options(
            [Selection(well, well) for well in self.wells]
        )
        self._set_status("")
        self._sync_actions()

    @property
    def selected_wells(self) -> list[str]:
        """The wells currently selected, in selection order."""
        return list(self.query_one("#view-wells", SelectionList).selected)

    def _sync_actions(self) -> None:
        """*Open in napari* is enabled only with a resolved session and at
        least one selected well."""
        can_open = self.options is not None and bool(self.selected_wells)
        self.query_one("#open-viewer", Button).disabled = not can_open

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        if event.selection_list.id == "view-wells":
            self._sync_actions()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-viewer":
            self.action_open_viewer()
        elif event.button.id == "view-back":
            self.action_back()

    def action_back(self) -> None:
        """*Back*: pop to the previous screen (the app's escape)."""
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def action_open_viewer(self) -> None:
        """*Open in napari*: launch the viewer for the selected wells in a
        separate process — the UI thread keeps running, and a failure is a
        log-pane line, never a crash."""
        wells = self.selected_wells
        if not wells or self.options is None:
            return
        handle = launch.start(
            wells,
            self.options,
            on_done=self._on_done,
            on_record=self._on_record,
            app=self.app,
        )
        self.handles.append(handle)
        self._set_status(
            f"opening {', '.join(wells)} in a separate process — "
            "the TUI stays usable"
        )

    def _on_record(self, record: logging.LogRecord) -> None:
        """One record of a launch: into the log pane (on the UI thread)."""
        if not self.is_mounted:
            return
        self.query_one("#view-log", RichLog).write(_record_text(record))

    def _on_done(self, handle: launch.ViewHandle) -> None:
        """The launch finished: the outcome on the status line (the error
        itself is in the log pane; the app stays alive)."""
        if not self.is_mounted:
            return
        if handle.error is not None:
            self._set_status(
                f"the viewer could not be started: {handle.error} — "
                "the TUI keeps running"
            )
        elif handle.failed:
            self._set_status(
                f"the viewer exited with status {handle.returncode} — "
                "the TUI keeps running"
            )
        else:
            self._set_status("the viewer closed")

    def _set_status(self, message: str) -> None:
        self.query_one("#view-status", Static).update(Text(message))

    def _write_line(self, text: str, style: str = "") -> None:
        self.query_one("#view-log", RichLog).write(Text(text, style=style))

    def _report(self, message: str) -> None:
        """Report a pre-flight problem: the exact message on the status
        line and in the log pane (and through the package logger)."""
        log.error("viewer: %s", message)
        self._set_status(message)
        self._write_line(f"   error {message}", "bold red")
