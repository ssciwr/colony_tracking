"""The textual application: navigation, screen registry, and the four
run-type drafts.

:class:`ColonyTuiApp` is the whole TUI's root: it owns the four
:class:`~colony_tracking.tui.draft.ConfigDraft` instances the screens
edit — one per run type, keyed by
:data:`~colony_tracking.tui.draft.RUN_TYPES` in :attr:`drafts`; each is
its run type's config source of truth and each writes and reads its own
independent config TOML (:data:`~colony_tracking.tui.draft.TOML_NAMES`)
— the screen registry (:attr:`SCREENS`) every screen is opened through,
and the escape-to-go-back navigation:

- ``quick_start`` — the full-auto flow
  (:class:`~colony_tracking.tui.screens.quick_start.QuickStartScreen`),
  which edits the chain's ``drafts["pipeline"]``;
- ``difference`` / ``segmentation`` / ``analysis`` — one
  :class:`~colony_tracking.tui.screens.stage.StageScreen` per registered
  stage, generated from the stage's options dataclass via the stage
  registry (:data:`colony_tracking.tui.screens.stage.STAGES`) — a future
  stage (train) slots in there, with no per-stage class anywhere; each
  stage screen edits its own run type's draft (its ``drafts`` entry), a
  flat and fully-explicit document;
- ``config`` — the chain's run level plus the effective TOML
  (:class:`~colony_tracking.tui.screens.config.ConfigScreen`), the
  pipeline draft again.

No screen edits another run type's draft — each run type holds an
independent config and derives no option from any other run.

There is no ``run`` entry: the live run screen
(:class:`~colony_tracking.tui.screens.run.RunScreen`) is pushed by the
flow that starts a run, never opened from a menu. The stage screens' run
actions reach it through
:func:`~colony_tracking.tui.screens.run.open_stage_run` (the shipped
default of
:meth:`StageScreen.start_run <colony_tracking.tui.screens.stage.StageScreen.start_run>`),
the quick start's *Run now* through
:func:`~colony_tracking.tui.screens.run.open_pipeline_run` — both entry
points write their run type's own config TOML first (reproducibility)
and then push the run screen, which starts the run through the
:mod:`~colony_tracking.tui.runner` in-process.

Everything user-visible goes into widgets; runtime diagnostics flow through
the package logger (:data:`colony_tracking.log`) — nothing ever prints.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from textual.app import App
from textual.binding import Binding
from textual.screen import Screen

from .draft import RUN_TYPES, ConfigDraft
from .screens.config import ConfigScreen
from .screens.home import HomeScreen
from .screens.quick_start import QuickStartScreen
from .screens.stage import STAGES, StageScreen


def _stage_screen(name: str) -> Callable[[], Screen]:
    """The screen factory for one registered stage: a
    :class:`~colony_tracking.tui.screens.stage.StageScreen` for ``name``
    (the draft comes from the running app's ``drafts`` entry for the
    stage's own run type; the run action uses the shipped default run
    flow — see
    :meth:`~colony_tracking.tui.screens.stage.StageScreen.start_run`)."""

    def factory() -> Screen:
        return StageScreen(name)

    return factory


class ColonyTuiApp(App[None]):
    """The colony-tracking TUI (see the module docstring).

    The screen registry (:attr:`SCREENS`) names every screen the home
    menu opens; the home screen is the bottom of the stack, and
    :attr:`drafts` holds the four run types' independent config
    documents — a screen edits its own run type's entry and nothing
    else's.
    """

    TITLE = "colony-tracking"

    BINDINGS = [Binding("escape", "back", "Back")]

    #: The screen registry: registered name -> screen factory. The names
    #: are the contract between the home menu and the screens; the run
    #: screen is not registered — runs are pushed by the flow that starts
    #: them (see the module docstring). The stage entries come straight
    #: from the stage registry, so a future stage's screen is its
    #: ``STAGES`` entry (plus its home-menu line and draft section), not a
    #: line here.
    SCREENS: ClassVar[dict[str, Callable[[], Screen]]] = {
        "quick_start": QuickStartScreen,
        "config": ConfigScreen,
        **{spec.name: _stage_screen(spec.name) for spec in STAGES.values()},
    }

    def __init__(self) -> None:
        super().__init__()
        #: The four run types' drafts — one :class:`ConfigDraft` per run
        #: type, keyed by :data:`~colony_tracking.tui.draft.RUN_TYPES`
        #: (the config sources of truth): the quick start and the config
        #: screen edit the chain's pipeline entry, every stage screen its
        #: own stage's flat entry, and each run type writes and reads
        #: only its own config TOML.
        self.drafts: dict[str, ConfigDraft] = {
            run_type: ConfigDraft(run_type) for run_type in RUN_TYPES
        }

    def get_default_screen(self) -> Screen:
        """The home screen: the first screen on the stack."""
        return HomeScreen()

    def action_back(self) -> None:
        """Pop to the previous screen (escape); the home screen is the
        bottom of the stack, so escape does nothing there."""
        if len(self.screen_stack) > 1:
            self.pop_screen()
