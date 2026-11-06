"""Log routing from the package logger to the TUI.

While a block of work runs — a stage run (:mod:`colony_tracking.tui.runner`)
or a viewer launch (:mod:`colony_tracking.tui.launch`) — a
:class:`logging.Handler` attached to the package logger
:data:`colony_tracking.log` forwards every record to the UI.
:func:`forwarding` owns that lifetime: it attaches one handler on entry and
detaches exactly that one on exit, so repeated blocks can never stack
handlers.

With a running textual ``App`` (the normal TUI case) records are marshalled
onto the UI thread through ``App.call_from_thread`` (:func:`deliverer`);
without one (tests, scripts) they run directly on the worker thread.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from textual.app import App

from .. import log


class _ForwardHandler(logging.Handler):
    """The per-block log forwarder: ``emit`` hands the record itself to the
    deliver callback — not formatted text, so the pane chooses its own
    rendering. A delivery failure is never fatal: ``handleError``
    reports it and the block goes on."""

    def __init__(
        self,
        deliver: Callable[..., None],
        on_record: Callable[[logging.LogRecord], None] | None,
    ) -> None:
        super().__init__()
        self._deliver = deliver
        self._on_record = on_record

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._on_record is not None:
                self._deliver(self._on_record, record)
        except Exception:  # noqa: BLE001 — logging must never kill a run
            self.handleError(record)


def deliverer(app: App[Any] | None) -> Callable[..., None]:
    """The post-to-UI-thread wrapper for one block: ``app.call_from_thread``
    when a textual App is given, a direct call otherwise."""
    if app is None:
        return lambda callback, *args: callback(*args)
    return lambda callback, *args: app.call_from_thread(callback, *args)


@contextmanager
def forwarding(
    deliver: Callable[..., None],
    on_record: Callable[[logging.LogRecord], None] | None,
) -> Iterator[None]:
    """Route the package logger's records to ``on_record`` (posted through
    ``deliver``, the :func:`deliverer` wrapper of the running app) for the
    block's duration: one handler attached on entry, detached on exit."""
    handler = _ForwardHandler(deliver, on_record)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)
