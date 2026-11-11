"""Shared body of the one-run-type screens: stages and finetune steps.

A stage screen and a finetune step screen are the same decision point on
different registries: one run type's options as one form, *Run this
stage/step*, *Write TOML…*, *Load TOML…*, and *Back*, with the run/write
actions gated on the draft's own resolve (see
:class:`~colony_tracking.tui.forms.FormScreen`). This base owns that body;
a subclass supplies the registry entry (title and options dataclass), the
run-type name, the labels/ids, and the run-flow hand-off. The registries
themselves live with their screens (:mod:`colony_tracking.tui.screens.stage`
and :mod:`colony_tracking.tui.screens.step`).
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from ..._options import Options
from ..forms import FormScreen, OptionForm, specs_of
from ._save import LoadConfigDialog


class StandaloneRunScreen(FormScreen):
    """One standalone run type's option form and its four actions.

    Subclasses set :attr:`RUN_ACTION` / :attr:`RUN_LABEL` (and the body and
    title ids), implement :meth:`_entry` (the registry entry's title and
    options dataclass), :attr:`run_type`, and :meth:`start_run` (the run
    flow hand-off).
    """

    #: The run button's id (also the gated action id).
    RUN_ACTION: ClassVar[str] = "run-stage"
    #: The run button's label.
    RUN_LABEL: ClassVar[str] = "Run this stage"
    #: The scrollable body's id.
    BODY_ID: ClassVar[str] = "stage-body"
    #: The options title's id.
    TITLE_ID: ClassVar[str] = "stage-title"

    def _entry(self) -> Any:
        """The registry entry: ``.title`` and ``.options`` (subclass hook)."""
        raise NotImplementedError

    def screen_note(self) -> str | None:
        """A behavioural note shown above the form, or ``None`` (subclass
        hook; stage screens have none)."""
        return None

    def compose_body(self) -> ComposeResult:
        entry = self._entry()
        self._form = OptionForm(
            self.run_type,
            specs_of(entry.options),
            self._resolve_draft(),
            on_change=self._on_form_change,
        )
        # One scrollable body: the actions below stay reachable on a small
        # terminal whatever the form's height.
        with VerticalScroll(id=self.BODY_ID):
            yield Static(f"{entry.title} — options", id=self.TITLE_ID)
            note = self.screen_note()
            if note is not None:
                yield Static(note, id=f"{self.run_type}-note", markup=False)
            yield self._form

    def compose_actions(self) -> ComposeResult:
        yield Button(self.RUN_LABEL, id=self.RUN_ACTION)
        yield Button("Write TOML…", id="write-toml")
        yield Button("Load TOML…", id="load-toml")
        yield Button("Back", id="back")

    def refresh_extras(self) -> None:
        """Nothing beyond the form itself: every field is ordinary and
        lives on it (the gate re-checks the draft on every change)."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == self.RUN_ACTION:
            self.action_run()
        elif event.button.id == "write-toml":
            self.action_write_toml()
        elif event.button.id == "load-toml":
            self.action_load_toml()
        elif event.button.id == "back":
            self.action_back()

    def action_load_toml(self) -> None:
        """*Load TOML…*: open the shared load-config dialog (see
        :class:`~colony_tracking.tui.screens._save.LoadConfigDialog`);
        loading is never gated."""
        self.app.push_screen(LoadConfigDialog(self.draft))

    def action_run(self) -> None:
        """*Run this stage/step*: resolve the draft and hand the run to the
        app's run flow (the config is the run type's own resolved options)."""
        if not self._enabled:
            return
        config = self._resolved_or_report()
        if config is not None:
            self.start_run(self.run_type, config)

    def start_run(self, name: str, config: Options) -> None:
        """Hand a resolved run request to the app's run flow (subclass
        hook)."""
        raise NotImplementedError
