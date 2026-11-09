"""Log routing from the package logger to the TUI.

While a block of work runs — a stage run (:mod:`colony_tracking.tui.runner`)
or a viewer launch (:mod:`colony_tracking.tui.launch`) — a
:class:`logging.Handler` attached to the package logger
:data:`colony_tracking.log` forwards every record to the UI.
:func:`forwarding` owns that lifetime: it attaches one handler on entry and
detaches exactly that one on exit, so repeated blocks can never stack
handlers.

Finetune steps also print: micro-sam/torch_em write progress to
stdout/stderr and draw a tqdm bar there, not through the package logger.
:func:`output_capture` is that separate door: it redirects the two streams
into plain unprefixed lines, collapsing a write's carriage-return repaints
to their last segment so a bar does not flood the pane. The runner uses it
for finetune kinds only; the stages' logging is left untouched.

With a running textual ``App`` (the normal TUI case) records and output
lines are marshalled onto the UI thread through ``App.call_from_thread``
(:func:`deliverer`); without one (tests, scripts) they run directly on the
worker thread.
"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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


class _LineWriter(io.TextIOBase):
    """A text stream that emits plain lines to ``emit``.

    Newlines end a line. Carriage returns are tqdm repaints: everything
    since the previous ``\\r`` is the same line's next frame, so it is
    discarded until the line ends. That makes a bar one line per finished
    line (its last segment), however many frames it drew, instead of a
    line per refresh flooding the pane. An unterminated line is emitted by
    :meth:`flush` (the run's end), and empty lines are dropped.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        data = self._buffer + text
        self._buffer = ""
        start = 0
        index = 0
        while index < len(data):
            char = data[index]
            if char == "\r":
                if index + 1 < len(data) and data[index + 1] == "\n":
                    self._emit_segment(data, start, index)  # CRLF line end
                    index += 2
                else:  # a repaint: drop the segment drawn before it
                    index += 1
                start = index
            elif char == "\n":
                self._emit_segment(data, start, index)
                index += 1
                start = index
            else:
                index += 1
        self._buffer = data[start:]
        return len(text)

    def _emit_segment(self, data: str, start: int, end: int) -> None:
        """Emit ``data[start:end]`` when it is not empty (a line's last
        repaint segment)."""
        segment = data[start:end]
        if segment:
            self._emit(segment)

    def flush(self) -> None:
        last = self._buffer
        self._buffer = ""
        if last:
            self._emit(last)

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


@contextmanager
def output_capture(
    deliver: Callable[..., None],
    on_output: Callable[[str], None],
) -> Iterator[None]:
    """Redirect stdout/stderr into ``on_output`` for the block's duration.

    Each completed line is delivered through ``deliver`` (the
    :func:`deliverer` wrapper of the running app), exactly like a log
    record, so the pane sees it on the UI thread; a carriage-return
    repaint is kept as the same line's last segment. A delivery failure
    never kills the run (the UI may be closing as the step writes), and
    the original streams are restored on exit, flushing any final
    unterminated line.
    """

    def emit(line: str) -> None:
        try:
            deliver(on_output, line)
        except Exception:  # noqa: BLE001 — a lost line must not kill a run
            log.debug("tui: dropping an output line", exc_info=True)

    stdout = _LineWriter(emit)
    stderr = _LineWriter(emit)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
    finally:
        stdout.flush()
        stderr.flush()
