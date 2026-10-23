"""The home screen: the TUI's top-level menu.

One entry per top-level action: Quick start (full-auto) — raw frames to
final analysis with zero configuration beyond the raw directory; the
three stages (Difference, Segment, Analyze), each startable on its own
and re-runnable (per-well done markers make a re-run a resume), each on
its own independent config (its run type's own draft and TOML — nothing
derived from another run); Pipeline config (the chain's run-level
settings, stage sections, and TOML); and the disabled Train entry.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, OptionList
from textual.widgets.option_list import Option

#: The home menu: ``(option id, label, target)`` where ``target`` is the
#: registered screen name to open, or ``None`` for a disabled entry. The
#: ids are the app's screen-registry names.
ENTRIES: tuple[tuple[str, str, str | None], ...] = (
    ("quick_start", "Quick start (full-auto)", "quick_start"),
    ("difference", "Difference", "difference"),
    ("segmentation", "Segment", "segmentation"),
    ("analysis", "Analyze", "analysis"),
    ("config", "Pipeline config", "config"),
    ("train", "Train — planned", None),
)


class HomeScreen(Screen[None]):
    """The home screen: the app's first screen and navigation hub."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield OptionList(
            *[
                Option(label, id=option_id, disabled=target is None)
                for option_id, label, target in ENTRIES
            ],
            id="home-menu",
        )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the selected entry's screen (disabled entries never
        select)."""
        for option_id, _label, target in ENTRIES:
            if event.option.id == option_id and target is not None:
                self.app.push_screen(target)
                return
