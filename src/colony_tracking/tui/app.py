"""The textual application: navigation, screen registry, and the shared
draft.

:class:`ColonyTuiApp` is the whole TUI's root: it owns the one
:class:`~colony_tracking.tui.draft.ConfigDraft` every screen edits (the
config source of truth), the screen registry (:attr:`SCREENS`) every
screen is opened through, and the escape-to-go-back navigation:

- ``quick_start`` — the full-auto flow
  (:class:`~colony_tracking.tui.screens.quick_start.QuickStartScreen`);
- ``difference`` / ``segmentation`` / ``analysis`` — one
  :class:`~colony_tracking.tui.screens.stage.StageScreen` per registered
  stage, generated from the stage's options dataclass via the stage
  registry (:data:`colony_tracking.tui.screens.stage.STAGES`) — a future
  stage (train) slots in there, with no per-stage class anywhere;
- ``config`` — the run level plus the effective TOML
  (:class:`~colony_tracking.tui.screens.config.ConfigScreen`).

There is no ``run`` entry: the live run screen
(:class:`~colony_tracking.tui.screens.run.RunScreen`) is pushed by the
flow that starts a run, never opened from a menu. The stage screens'
run actions reach it through
:func:`~colony_tracking.tui.screens.run.open_stage_run` /
:func:`~colony_tracking.tui.screens.run.open_pipeline_run` (the shipped
default of :meth:`~colony_tracking.tui.screens.stage.StageScreen.
start_run`), the quick start's *Run now* through
:func:`~colony_tracking.tui.screens.run.open_pipeline_run` — both entry
points write the config TOML first (reproducibility) and then push the
run screen, which starts the run through the
:class:`~colony_tracking.tui.runner` in-process.

Everything user-visible goes into widgets; runtime diagnostics flow through
the package logger (:data:`colony_tracking.log`) — nothing ever prints.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from textual.app import App
from textual.binding import Binding
from textual.screen import Screen

from .draft import ConfigDraft
from .screens.config import ConfigScreen
from .screens.home import HomeScreen
from .screens.quick_start import QuickStartScreen
from .screens.stage import STAGES, StageScreen


def _stage_screen(name: str) -> Callable[[], Screen]:
    """The screen factory for one registered stage: a
    :class:`~colony_tracking.tui.screens.stage.StageScreen` for ``name``
    (the draft comes from the running app; the run actions use the
    shipped default run flow — see
    :meth:`~colony_tracking.tui.screens.stage.StageScreen.start_run`)."""

    def factory() -> Screen:
        return StageScreen(name)

    return factory


class ColonyTuiApp(App[None]):
    """The colony-tracking TUI (see the module docstring).

    The screen registry (:attr:`SCREENS`) names every screen the home
    menu opens; the home screen is the bottom of the stack and
    :attr:`draft` is the one config document every screen edits.
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
        #: The one draft every screen edits (the config source of truth).
        self.draft = ConfigDraft()

    def get_default_screen(self) -> Screen:
        """The home screen: the first screen on the stack."""
        return HomeScreen()

    def action_back(self) -> None:
        """Pop to the previous screen (escape); the home screen is the
        bottom of the stack, so escape does nothing there."""
        if len(self.screen_stack) > 1:
            self.pop_screen()
