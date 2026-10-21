"""The colony-tracking terminal user interface.

The public entry point is :func:`run_tui`, launched by the ``tui``
subcommand of the CLI. It starts the textual app (:mod:`.app`), whose
screens edit one :class:`~colony_tracking.tui.draft.ConfigDraft` and start
runs through :mod:`colony_tracking.tui.runner`. Errors of runs and forms
are handled in the UI; the app exits cleanly.
"""

from __future__ import annotations

import logging
import sys

from .. import log
from .app import ColonyTuiApp


def _configure_logging() -> None:
    """The TUI's logging setup, mirroring the CLI's ``_configure_logging``
    (that module is the precedent; it is mirrored, not imported): the
    package logger goes to stderr at INFO. Idempotent — launching through
    the CLI has already attached the stderr handler, so a handler is added
    only when none exists (nothing here ever double-adds one)."""
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def run_tui(argv: list[str] | None = None) -> int:
    """Launch the TUI; returns the exit code (0 after a clean quit).

    The ``tui`` subcommand takes no flags in v1, so passing any argument is
    a usage error (exit code 2, the CLI's convention). Run failures and
    invalid drafts are handled in the UI; the app itself exits cleanly.
    """
    if argv:
        log.error(
            "usage error: the tui command takes no arguments (got %r)",
            " ".join(argv),
        )
        return 2
    _configure_logging()
    ColonyTuiApp().run()
    return 0
