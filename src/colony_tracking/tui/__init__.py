"""The colony-tracking terminal user interface.

The public entry point is :func:`run_tui`, launched by the ``tui``
subcommand of the CLI. It starts the textual app (:mod:`.app`), whose
screens edit one :class:`~colony_tracking.tui.draft.ConfigDraft` and start
runs through :mod:`colony_tracking.tui.runner`. Errors of runs and forms
are handled in the UI; the app exits cleanly.
"""

from __future__ import annotations

import logging

from textual.logging import TextualHandler

from .. import log
from .app import ColonyTuiApp


def _configure_logging() -> None:
    """Route the package logger for on-screen TUI operation (idempotent).

    While the textual app runs it parks the real terminal cursor inside the
    focused input after every frame, so any handler that writes raw text to
    the terminal stream would materialize log lines right there — the
    reported bug of the draft's ``ConfigError`` appearing inside the
    last-selected text input (the gate's ``refresh_gate`` logs one
    ``pipeline config:`` line per failed resolve, i.e. on every field
    change). Every terminal-bound ``StreamHandler`` on the package logger —
    the CLI installs a stderr one before dispatching here — is therefore
    removed, and a :class:`textual.logging.TextualHandler` takes over:
    records logged on the UI thread go to the app's own log, records from
    worker threads are dropped (the runner's per-run forwarder already
    delivers those to the run screen's log pane), and nothing ever reaches
    the terminal. The usage-error path in :func:`run_tui` logs before this
    reconfiguration, so pre-app diagnostics still reach stderr."""
    for handler in list(log.handlers):
        if isinstance(handler, logging.StreamHandler):
            log.removeHandler(handler)
    if not any(isinstance(handler, TextualHandler) for handler in log.handlers):
        log.addHandler(TextualHandler(stderr=False))
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
