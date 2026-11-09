"""Shared path-picker pieces: location, browse button, and hand-off.

The option forms, the quick start, and the config-file dialogs all put a
click-only *browse* button beside a path input. This module owns the
pieces they share — the button widget, its label and styles, the rule for
where a picker opens (:func:`picker_location`), and the hand-off that
waits for the dismissed screen's reveal to settle (:func:`open_picker`) —
so the call sites cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from textual.app import App
from textual.screen import Screen
from textual.widgets import Button
from textual_fspicker import SelectDirectory

#: A browse button's label (the field-row counterpart of ``reset``).
BROWSE_LABEL: Final[str] = "browse"

#: The browse button's styles, shared by every container that hosts one
#: (the option forms and the config-file dialogs): a button's own type
#: rules outrank a bare type selector, so the look is keyed on the class.
BROWSE_BUTTON_CSS = """
.field-browse {
    min-width: 0;
    width: auto;
    height: 1;
    border: none;
    background: $panel;
    color: $text 60%;
    margin-left: 1;
}
.field-browse:hover {
    background: $surface-darken-1;
    color: $text;
}
"""


class BrowseButton(Button):
    """A path field's *browse* button: it opens the picker modal for its
    field (the hosting screen wires the press; see :func:`open_picker`).
    Click-only — ``can_focus`` is off, like the reset buttons — so tab and
    the arrow-key field walk stay field-to-field; styled through its
    container's ``.field-browse`` rules, because a ``Button``'s own type
    rules outrank a bare type selector."""

    can_focus = False


def picker_location(value: str) -> Path:
    """The directory a picker opens at for a path field showing ``value``:
    the value's nearest existing directory (the path itself, else the
    first existing ancestor), or the working directory when nothing
    exists yet."""
    stripped = value.strip()
    if not stripped:
        return Path.cwd()
    path = Path(stripped)
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    return Path.cwd()


def open_picker(
    app: App[Any],
    picker: Screen[Any],
    accept: Callable[[Path], None],
) -> None:
    """Open a textual-fspicker modal, starting at its own location.

    Picking a path hands it to ``accept``; cancelling (escape or the
    dialog's *Cancel*) calls nothing. The hand-off waits for the dismissed
    screen's reveal to settle (``call_after_refresh``): the pop resumes the
    screen behind the picker, whose entry refresh re-shows the draft's
    values — a write racing that refresh would be re-set from the draft
    before its change event landed, losing the pick.
    """

    def picked(path: Path | None) -> None:
        if path is not None:
            app.call_after_refresh(accept, path)

    app.push_screen(picker, picked)


def open_directory_picker(
    app: App[Any],
    location: Path,
    accept: Callable[[Path], None],
) -> None:
    """Open the directory-picker modal — textual-fspicker's
    :class:`~textual_fspicker.SelectDirectory`, starting at ``location``.
    Picking a directory hands it to ``accept``; cancelling calls
    nothing."""
    open_picker(app, SelectDirectory(location), accept)
