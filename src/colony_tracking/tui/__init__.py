"""The colony-tracking terminal user interface.

:func:`run_tui` (the CLI's ``tui`` subcommand) starts the textual app
(:mod:`.app`), whose screens edit one config draft per run type
(:class:`~colony_tracking.tui.draft.ConfigDraft`) and start runs through
:mod:`colony_tracking.tui.runner`. Run and form errors are handled in the
UI; the app exits cleanly.
"""

from __future__ import annotations

import logging

from textual.logging import TextualHandler

from .. import log
from .app import ColonyTuiApp


def _configure_logging() -> None:
    """Route the package logger for on-screen TUI operation (idempotent).

    While the textual app runs it owns the terminal, so any handler that
    writes raw text to it would garble the on-screen frame. Every
    terminal-bound ``StreamHandler`` on the package logger — the CLI
    installs a stderr one before dispatching here — is therefore removed,
    and a :class:`textual.logging.TextualHandler` takes over: UI-thread
    records go to the app's log, worker-thread records are dropped (the
    runner's per-run forwarder already delivers those to the run screen's
    log pane). :func:`run_tui` logs usage errors before this
    reconfiguration, so they still reach stderr.
    """
    for handler in list(log.handlers):
        if isinstance(handler, logging.StreamHandler):
            log.removeHandler(handler)
    if not any(isinstance(handler, TextualHandler) for handler in log.handlers):
        log.addHandler(TextualHandler(stderr=False))
    log.setLevel(logging.INFO)


def run_tui(argv: list[str] | None = None) -> int:
    """Launch the TUI; returns the exit code (0 after a clean quit).

    The ``tui`` subcommand takes no flags, so passing any argument is a
    usage error (exit code 2, the CLI's convention). Run failures and
    invalid drafts are handled in the UI.
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
