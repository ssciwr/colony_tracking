"""The textual application: navigation, screen registry, and the
run-type drafts.

:class:`ColonyTuiApp` owns the four :class:`~colony_tracking.tui.draft.ConfigDraft`
instances the screens edit — one per run type, keyed by
:data:`~colony_tracking.tui.draft.RUN_TYPES` in :attr:`drafts`, each its
run type's config source of truth writing and reading its own config TOML
(:data:`~colony_tracking.tui.draft.TOML_NAMES`) — and the screen registry
(:attr:`SCREENS`) every screen is opened through, plus escape-to-go-back:

- ``quick_start`` — the full-auto flow, which edits the chain's
  ``drafts["pipeline"]``;
- ``difference`` / ``segmentation`` / ``analysis`` — one
  :class:`~colony_tracking.tui.screens.stage.StageScreen` per registered
  stage, generated from the stage registry
  (:data:`colony_tracking.tui.screens.stage.STAGES`); each edits its own
  run type's draft;
- ``config`` — the chain's run level plus the effective TOML
  (:class:`~colony_tracking.tui.screens.config.ConfigScreen`), the
  pipeline draft again.

No screen edits another run type's draft. There is no ``run`` entry: the
run screen (:class:`~colony_tracking.tui.screens.run.RunScreen`) is
pushed by the flow that starts a run — the stage screens' run actions
through :func:`~colony_tracking.tui.screens.run.open_stage_run`, the
quick start's *Run now* through
:func:`~colony_tracking.tui.screens.run.open_pipeline_run` — both write
the run type's config TOML first, then push the run screen, which runs
the run in-process through :mod:`~colony_tracking.tui.runner`.

Diagnostics flow through the package logger (:data:`colony_tracking.log`);
nothing ever prints.
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
    (the draft comes from the running app's ``drafts`` entry; the run
    action uses the shipped default — see
    :meth:`~colony_tracking.tui.screens.stage.StageScreen.start_run`)."""

    def factory() -> Screen:
        return StageScreen(name)

    return factory


class ColonyTuiApp(App[None]):
    """The colony-tracking TUI (see the module docstring)."""

    TITLE = "colony-tracking"

    BINDINGS = [Binding("escape", "back", "Back")]

    #: The screen registry: registered name -> screen factory; the names
    #: are the contract between the home menu and the screens. The run
    #: screen is not registered — runs are pushed by the flow that starts
    #: them (see the module docstring).
    SCREENS: ClassVar[dict[str, Callable[[], Screen]]] = {
        "quick_start": QuickStartScreen,
        "config": ConfigScreen,
        **{spec.name: _stage_screen(spec.name) for spec in STAGES.values()},
    }

    def __init__(self) -> None:
        super().__init__()
        #: One :class:`ConfigDraft` per run type, keyed by
        #: :data:`~colony_tracking.tui.draft.RUN_TYPES`: the quick start
        #: and config screen edit the pipeline entry, every stage screen
        #: its own.
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
