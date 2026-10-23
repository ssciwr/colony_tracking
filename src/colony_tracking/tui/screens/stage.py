"""The per-stage screens: one :class:`StageScreen` per stage, generated
from the stage registry.

The registry (:data:`STAGES`) maps a draft section name to its options
dataclass and user-visible title (the titles are
:data:`~colony_tracking.tui.draft.TITLES` — the same names the config
file's stage sections and the home menu use). There is exactly one
reusable screen class, instantiated per stage.

Every field of the stage's options dataclass renders as a form field,
generated from the type hints and help metadata alone through
:mod:`colony_tracking.tui.forms` — a stage run type's config is fully
explicit, so every field is ordinary and editable: the segmentation
screen takes ``input`` as a select (raw frames or difference frames) and
``diff_dir`` as a plain path input, and nothing on a stage screen is
derived from a run level, from another stage's settings, or from another
run type's outputs. Each instance edits its own
:class:`~colony_tracking.tui.draft.ConfigDraft` — the one run type it
stands for (:attr:`run_type`): the app's
``drafts`` entry for that run type — and writes and reads that run
type's own independent config TOML.

The actions are the stage-level decision points: *Run this stage*,
*Write TOML…* and *Load TOML…* (the shared save/load dialogs in
:mod:`~colony_tracking.tui.screens._save`), and *Back*. The run/write
actions stay disabled until the screen's draft resolves with the values
on screen; a failing resolve shows the draft's exact
:class:`~colony_tracking.pipeline.ConfigError` message instead. The load
action is never gated — loading works on an incomplete draft, and the
loaded document is reported at the next resolve either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from ..._options import (
    AnalysisOptions,
    DifferenceOptions,
    Options,
    SegmentationOptions,
)
from ..draft import TITLES, ConfigDraft
from ..forms import FormScreen, OptionForm, specs_of
from ._save import LoadConfigDialog
from .run import open_stage_run


@dataclass(frozen=True)
class StageSpec:
    """One entry of the stage registry: the draft section (the run type
    the stage screen edits), the user-visible title, and the options
    dataclass the form is generated from."""

    #: The draft section name (the config's stage name).
    name: str
    #: The user-visible title (the home menu's wording).
    title: str
    #: The options dataclass the form is generated from.
    options: type[Options]


#: The stage registry: draft section name → spec. The titles come from
#: :data:`~colony_tracking.tui.draft.TITLES` — the same wording the save
#: and load dialogs show.
STAGES: dict[str, StageSpec] = {
    "difference": StageSpec(
        "difference", TITLES["difference"], DifferenceOptions
    ),
    "segmentation": StageSpec(
        "segmentation", TITLES["segmentation"], SegmentationOptions
    ),
    "analysis": StageSpec("analysis", TITLES["analysis"], AnalysisOptions),
}


class StageScreen(FormScreen):
    """The screen for one stage: all of the stage's options as one form,
    and the four actions.

    Instantiated per stage from the registry: ``StageScreen("difference")``
    renders the difference stage's options. The run/write actions gate on
    the draft (see :class:`~colony_tracking.tui.forms.FormScreen`); a
    passing press hands the resolved options to :meth:`start_run`.
    """

    gated_actions: ClassVar[tuple[str, ...]] = ("run-stage", "write-toml")

    BINDINGS = [
        Binding("r", "run_stage", "Run this stage"),
        Binding("w", "write_toml", "Write TOML…"),
        Binding("l", "load_toml", "Load TOML…"),
    ]

    CSS = """
    #stage-body {
        height: 1fr;
    }
    #stage-body OptionForm {
        height: auto;
    }
    """

    def __init__(
        self,
        stage: str,
        draft: ConfigDraft | None = None,
        *,
        start_run: Callable[[str, Options], None] | None = None,
    ) -> None:
        super().__init__(draft)
        self._stage = stage
        self._start_run = start_run

    @property
    def stage(self) -> str:
        """The screen's stage (the draft section name, and the run type
        whose draft the screen edits)."""
        return self._stage

    @property
    def run_type(self) -> str:
        """The run type this screen edits (a key of
        :data:`~colony_tracking.tui.draft.RUN_TYPES`): its own stage —
        the base class resolves the draft as the app's ``drafts`` entry
        for it."""
        return self._stage

    def compose_body(self) -> ComposeResult:
        entry = STAGES[self._stage]
        self._form = OptionForm(
            self._stage,
            specs_of(entry.options),
            self._resolve_draft(),
            on_change=self._on_form_change,
        )
        # One scrollable body: the actions below stay reachable on a small
        # terminal whatever the form's height.
        with VerticalScroll(id="stage-body"):
            yield Static(f"{entry.title} — options", id="stage-title")
            yield self._form

    def compose_actions(self) -> ComposeResult:
        yield Button("Run this stage", id="run-stage")
        yield Button("Write TOML…", id="write-toml")
        yield Button("Load TOML…", id="load-toml")
        yield Button("Back", id="back")

    def refresh_extras(self) -> None:
        """Nothing beyond the form itself: every field is ordinary and
        lives on it (the gate re-checks the draft on every change)."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-stage":
            self.action_run_stage()
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

    def action_run_stage(self) -> None:
        """*Run this stage*: resolve the draft and hand the run to the
        app's run flow (the config is the stage's own resolved options —
        everything the draft holds, nothing derived)."""
        if not self._enabled:
            return
        config = self._resolved_or_report()
        if config is not None:
            self.start_run(self._stage, config)

    def start_run(self, stage: str, config: Options) -> None:
        """Hand a resolved run request to the app's run flow.

        ``stage`` is the screen's stage name; ``config`` is the draft's
        resolved :class:`~colony_tracking._options.Options` for it. The
        shipped default dispatches to the run screens' entry point
        :func:`~colony_tracking.tui.screens.run.open_stage_run`, which
        writes the stage's own config TOML first (reproducibility) and
        pushes the run screen; the entry point resolves the draft
        itself, so the config handed over here is the gate's
        confirmation. Tests can pass a ``start_run`` callable to the
        constructor to observe the hand-off without the dialog and the
        runner, leaving the form machinery unchanged.
        """
        if self._start_run is not None:
            self._start_run(stage, config)
            return
        open_stage_run(self.app, self.draft, self._stage)
