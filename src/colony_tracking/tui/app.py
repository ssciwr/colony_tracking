"""The textual application: navigation, screen registry, and the shared
draft.

:class:`ColonyTuiApp` is the whole TUI's root: it owns the one
:class:`~colony_tracking.tui.draft.ConfigDraft` every screen edits (the
config source of truth), the screen registry every screen is opened
through, and the escape-to-go-back navigation. The home screen is real
from ticket 01 on; the other screens are placeholders until their tickets
mount the real ones — a later ticket replaces its registry entry with the
real screen factory and leaves the app alone otherwise:

- ``quick_start`` — the full-auto flow (ticket 03);
- ``difference`` / ``segmentation`` / ``analysis`` — one
  :class:`~colony_tracking.tui.screens.stage.StageScreen` each, generated
  from the stage's options dataclass (ticket 02);
- ``config`` — run-level settings plus the effective TOML (ticket 02);
- ``run`` — the live run screen (ticket 03), pushed by the screens that
  start runs.

Everything user-visible goes into widgets; runtime diagnostics flow through
the package logger (:data:`colony_tracking.log`) — nothing ever prints.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from .draft import ConfigDraft
from .screens.home import HomeScreen


class _PlaceholderScreen(Screen[None]):
    """A stand-in for a screen a later ticket mounts (see the module
    docstring): shows its title, waits for escape."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"[b]{self._title}[/b] — planned; a later ticket mounts the "
            "real screen here. Escape goes back.",
        )
        yield Footer()


def _placeholder(title: str) -> Callable[[], Screen]:
    """The screen factory for one of the later tickets' screens."""

    def factory() -> Screen:
        return _PlaceholderScreen(title)

    return factory


class ColonyTuiApp(App[None]):
    """The colony-tracking TUI (see the module docstring).

    The screen registry (:attr:`SCREENS`) names every screen; the home
    screen is the bottom of the stack and :attr:`draft` is the one config
    document every screen edits.
    """

    TITLE = "colony-tracking"

    BINDINGS = [Binding("escape", "back", "Back")]

    #: The screen registry: registered name -> screen factory. The names
    #: are the contract between the home menu, the screens, and the later
    #: tickets (which replace the placeholder factories).
    SCREENS: ClassVar[dict[str, Callable[[], Screen]]] = {
        "quick_start": _placeholder("Quick start"),
        "difference": _placeholder("Difference"),
        "segmentation": _placeholder("Segment"),
        "analysis": _placeholder("Analyze"),
        "config": _placeholder("Pipeline config"),
        "run": _placeholder("Run"),
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
