"""The finetune submenu: the home menu's one ``Finetune`` row opens this
screen, whose rows name the work, not the subcommand.

The rows are the spec's three names — ``Prepare data``, ``Train decoder``,
``Evaluate checkpoint`` — while the run-type keys and TOML names stay
``prepare`` / ``train`` / ``evaluate``. A row is live exactly when its
step has a registered :class:`~colony_tracking.tui.screens.step.StepSpec`
(and therefore a registered screen); the rest stay disabled planned rows.
Tickets 03/04 make their rows live by registering the step's spec and its
draft/runner kind — this submenu needs no edit.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, OptionList
from textual.widgets.option_list import Option

from .step import STEPS

#: The submenu's rows: ``(step name, user-visible label)`` in the spec's
#: order. The step name is the run type and the screen-registry name.
ENTRIES: tuple[tuple[str, str], ...] = (
    ("prepare", "Prepare data"),
    ("train", "Train decoder"),
    ("evaluate", "Evaluate checkpoint"),
)


class FinetuneScreen(Screen[None]):
    """The finetune submenu: one row per step (see the module docstring)."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield OptionList(
            *[
                Option(
                    label,
                    id=step,
                    disabled=step not in STEPS,
                )
                for step, label in ENTRIES
            ],
            id="finetune-menu",
        )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the selected step's registered screen (disabled rows never
        select)."""
        for step, _label in ENTRIES:
            if event.option.id == step and step in STEPS:
                self.app.push_screen(step)
                return
