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
into plain unprefixed lines, turning carriage-return repaints (even across
the per-write flushes tqdm makes) into periodic snapshots — immediate, then
at most one per :data:`SNAPSHOT_INTERVAL` — so a long bar shows it is
alive without flooding the pane. The final frame is always emitted when
the stream ends. The runner uses it for finetune kinds only; the stages'
logging is left untouched.

With a running textual ``App`` (the normal TUI case) records and output
lines are marshalled onto the UI thread through ``App.call_from_thread``
(:func:`deliverer`); without one (tests, scripts) they run directly on the
worker thread.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Iterator

from textual.app import App

from .. import log

#: Minimum seconds between progress snapshots from one carriage-return
#: repaint stream. A bar redraws many times per second; one snapshot per
#: interval shows it is alive without flooding the log pane. The first
#: snapshot is immediate, and the final one is always emitted at the
#: run's end, so even a short run shows both ends of the bar.
SNAPSHOT_INTERVAL = 10.0


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

    Newlines end a line, and so does CRLF (even when the two characters
    arrive in separate writes). A lone carriage return is a tqdm repaint:
    everything since the previous ``\\r`` is the same line's next frame.
    Frames are snapshots of one line, so they are emitted at most once
    per :data:`SNAPSHOT_INTERVAL` (the first immediately) instead of one
    line per refresh — a long training run shows the bar while it runs,
    and a fast one cannot flood the pane. An unterminated line's last
    frame is emitted once by :meth:`finish` (the run's end), and empty
    lines are dropped.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        min_interval: float = SNAPSHOT_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._emit = emit
        self._buffer = ""
        self._min_interval = min_interval
        self._clock = clock
        self._last_snapshot: float | None = None

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
                    start = index
                elif index + 1 == len(data):
                    # A trailing CR may be the first half of a CRLF split
                    # across writes: defer it (with its segment) until the
                    # next write, which re-reads it in context.
                    break
                else:  # a repaint: snapshot the frame drawn before it
                    self._emit_snapshot(data, start, index)
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

    def _emit_snapshot(self, data: str, start: int, end: int) -> None:
        """Emit one repaint frame as a line, throttled to one per
        ``min_interval`` — the first frame is immediate so a slow bar
        appears as soon as it draws."""
        segment = data[start:end]
        if not segment:
            return
        now = self._clock()
        if (
            self._last_snapshot is None
            or now - self._last_snapshot >= self._min_interval
        ):
            self._emit(segment)
            self._last_snapshot = now

    def flush(self) -> None:
        """No-op: a flush is not a line boundary.

        tqdm's printer calls ``file.flush()`` after every repaint write
        (torch_em uses it), so emitting the buffer here would turn each
        frame into its own line and defeat the repaint collapse. The
        run's end calls :meth:`finish` instead.
        """

    def finish(self) -> None:
        """Emit the final unterminated segment, if any (the run's end).

        A trailing ``\\r`` is the last repaint's boundary, not part of the
        segment. Idempotent: a second call finds an empty buffer.
        """
        last = self._buffer.removesuffix("\r")
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
    record, so the pane sees it on the UI thread. A carriage-return
    repaint stream is delivered as periodic snapshots (see
    :class:`_LineWriter`), and :meth:`_LineWriter.finish` emits the final
    unterminated frame once. A stream flush is not a line boundary (tqdm
    flushes after every repaint). A delivery failure never kills the run
    (the UI may be closing as the step writes), and the original streams
    are restored on exit.
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
        stdout.finish()
        stderr.finish()
