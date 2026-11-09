"""The finetune step screens: one :class:`StepScreen` per registered step.

The registry (:data:`STEPS`) maps a step name to its options dataclass and
user-visible title. It follows the stage registry's pattern: one reusable
screen class, instantiated per step, rendering every field of the step's
options dataclass through :mod:`colony_tracking.tui.forms`. A step run
type's config is flat and fully explicit, so every field is ordinary and
editable — nothing is derived from another step, another run type, or a
pipeline run level.

Each instance edits its own :class:`~colony_tracking.tui.draft.ConfigDraft`
— the app's ``drafts`` entry for :attr:`run_type` — and writes and reads
that step's own config TOML (``prepare.toml``, ...). The draft resolves
through :mod:`colony_tracking.finetune.config`, the same flat-document
machinery the CLI's ``train <step> --config`` reads.

The actions are the step-level decision points: *Run this step*, *Write
TOML…* and *Load TOML…* (the shared save/load dialogs in
:mod:`~colony_tracking.tui.screens._save`), and *Back* — the body the
stage and finetune-step screens share
(:class:`~colony_tracking.tui.screens._standalone.StandaloneRunScreen`).
The run/write actions stay disabled until the screen's draft resolves with
the values on screen; a failing resolve shows the finetune machinery's
exact :class:`~colony_tracking.pipeline.ConfigError` message instead. The
load action is never gated — loading works on an incomplete draft, and the
loaded document is reported at the next resolve either way.

Adding a step — ticket 04's ``evaluate`` — means registering its
:class:`StepSpec` here and its name in the draft's live finetune run
types; the app registers the screen and the finetune submenu makes the
row live from that registry. The step's draft, TOML name, title, and run
kind are registered alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

from textual.binding import Binding

from ..._options import Options, PrepareOptions, TrainOptions
from ..draft import ConfigDraft
from ._standalone import StandaloneRunScreen
from .run import open_step_run


@dataclass(frozen=True)
class StepSpec:
    """One entry of the finetune step registry: the step's run type (its
    draft section and config TOML name), the user-visible title, the
    options dataclass the form is generated from, and an optional note
    shown above the form (a behaviour the screen must state)."""

    #: The step name (the run type, draft section, and config TOML name).
    name: str
    #: The user-visible title (the finetune submenu's wording).
    title: str
    #: The options dataclass the form is generated from.
    options: type[Options]
    #: A short note above the form, or ``None`` for no note.
    note: str | None = None


#: The finetune step registry: step name → spec. Ticket 04 adds evaluate
#: here (with the step's live run type and run kind); the screen and the
#: finetune submenu row follow from this registry.
STEPS: dict[str, StepSpec] = {
    "prepare": StepSpec("prepare", "Prepare data", PrepareOptions),
    "train": StepSpec(
        "train",
        "Train decoder",
        TrainOptions,
        note=(
            "Re-running with the same output directory resumes training; a "
            "run that already reached n_epochs returns cleanly. There is no "
            "confirmation prompt and no start-over control."
        ),
    ),
}


class StepScreen(StandaloneRunScreen):
    """The screen for one finetune step: all of the step's options as one
    form, and the four actions.

    Instantiated per step from the registry: ``StepScreen("prepare")``
    renders the prepare step's options. The run/write actions gate on the
    draft (see :class:`~colony_tracking.tui.forms.FormScreen`); a passing
    press hands the resolved options to :meth:`start_run`.
    """

    RUN_ACTION: ClassVar[str] = "run-step"
    RUN_LABEL: ClassVar[str] = "Run this step"
    BODY_ID: ClassVar[str] = "step-body"
    TITLE_ID: ClassVar[str] = "step-title"
    gated_actions: ClassVar[tuple[str, ...]] = ("run-step", "write-toml")

    BINDINGS = [
        Binding("r", "run_step", "Run this step"),
        Binding("w", "write_toml", "Write TOML…"),
        Binding("l", "load_toml", "Load TOML…"),
    ]

    CSS = """
    #step-body {
        height: 1fr;
    }
    #step-body OptionForm {
        height: auto;
    }
    """

    def __init__(
        self,
        step: str,
        draft: ConfigDraft | None = None,
        *,
        start_run: Callable[[str, Options], None] | None = None,
    ) -> None:
        super().__init__(draft)
        self._step = step
        self._start_run = start_run

    @property
    def step(self) -> str:
        """The screen's step (the draft section name, and the run type
        whose draft the screen edits)."""
        return self._step

    @property
    def run_type(self) -> str:
        """The run type this screen edits (a key of
        :data:`~colony_tracking.tui.draft.RUN_TYPES`): its own step — the
        base class resolves the draft as the app's ``drafts`` entry for
        it."""
        return self._step

    def _entry(self) -> StepSpec:
        """The step's registry entry (its title, options dataclass, and
        note)."""
        return STEPS[self._step]

    def screen_note(self) -> str | None:
        """The step's behavioural note above the form (``None`` when the
        registry entry carries none)."""
        return self._entry().note

    def action_run_step(self) -> None:
        """*Run this step*: delegate to ``StandaloneRunScreen.action_run``."""
        self.action_run()

    def start_run(self, step: str, config: Options) -> None:
        """Hand a resolved run request to the app's run flow.

        The shipped default dispatches to the run screens' entry point
        :func:`~colony_tracking.tui.screens.run.open_step_run`, which
        writes the step's own config TOML first (reproducibility) and
        pushes the run screen. Tests can pass a ``start_run`` callable to
        the constructor to observe the hand-off without the dialog and the
        runner, leaving the form machinery unchanged.
        """
        if self._start_run is not None:
            self._start_run(step, config)
            return
        open_step_run(self.app, self.draft, self._step)
