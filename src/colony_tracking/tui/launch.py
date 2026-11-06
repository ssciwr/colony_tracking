"""Launch the viewer as a separate process from the TUI.

The TUI never imports napari: Qt must own the process's main thread and
Textual already holds it. :func:`start` therefore runs the CLI's own
``view`` command — ``[sys.executable, "-m", "colony_tracking", "view",
...]`` (:func:`command`) — so the launched session shows exactly the
layers running the command by hand for the same wells and paths would.

The paths are the chain draft's resolved :class:`PipelineConfig`
(:func:`view_options`): ``raw_dir`` and each stage's output
(``difference.out_dir``, ``segmentation.out_dir``,
``analysis.out_root``). Nothing re-derives a tree name here — that is
what the config's resolution is for, and it keeps
:mod:`colony_tracking.viewer` free of the pipeline.

The child runs on a worker thread (``app.run_worker(..., thread=True)``
when a running textual ``App`` is given, a plain daemon thread
otherwise) that spawns it and watches it to completion, so the UI thread
never blocks. Its output is captured and logged back through the package
logger :data:`colony_tracking.log`; :func:`start` attaches the shared
record forwarder (:func:`~colony_tracking.tui.runner.record_forwarder`)
around the launch's own messages, so the screen's log pane shows the
child's lines and — when it exits non-zero (napari missing, no display) —
the failure, instead of a silent orphan. A failed spawn is stored on the
handle too: nothing about a launch is fatal to the TUI.

Tests replace :data:`SPAWN` with a stub process, so no real viewer is
ever started and no display is needed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from textual.app import App

from .. import log
from .._options import ViewOptions
from ..pipeline import PipelineConfig
from .runner import _deliverer, record_forwarder

#: The package's module entry point (``python -m colony_tracking``) the
#: child runs; see ``src/colony_tracking/__main__.py``.
VIEW_MODULE = "colony_tracking"

#: The process spawner. Tests replace it with a stub (the signature is
#: :class:`subprocess.Popen`'s: the argument vector plus the capture
#: keyword arguments), so a test never starts a real viewer.
SPAWN: Callable[..., Any] = subprocess.Popen

#: The viewer's path flags: the CLI flag and the :class:`ViewOptions` field
#: it carries. Every set path is passed explicitly, so the child resolves
#: the same session the draft did (`--config` is not used: the draft is
#: not necessarily on disk).
_PATH_FLAGS: tuple[tuple[str, str], ...] = (
    ("--raw-dir", "raw_dir"),
    ("--diff-dir", "diff_dir"),
    ("--seg-dir", "seg_dir"),
    ("--analysis-dir", "analysis_dir"),
)


def view_options(config: PipelineConfig) -> ViewOptions:
    """The viewer session the chain config describes.

    The config's resolved paths, in the viewer's own
    :class:`~colony_tracking._options.ViewOptions`: the run level's raw
    directory and each stage's output tree — the derived names follow the
    chain's reference frame, exactly as the CLI's ``--config`` resolution
    would read them.
    """
    return ViewOptions(
        raw_dir=config.raw_dir,
        diff_dir=config.difference.out_dir,
        seg_dir=config.segmentation.out_dir,
        analysis_dir=config.analysis.out_root,
    )


def command(wells: Sequence[str], options: ViewOptions) -> list[str]:
    """The viewer subprocess's exact argument vector.

    The wells go as one comma-separated ``--wells=`` value (well names
    start with a dash, so the value must stay inside the flag), every set
    path as ``--<flag>=<path>``.
    """
    argv = [
        sys.executable,
        "-m",
        VIEW_MODULE,
        "view",
        "--wells=" + ",".join(wells),
    ]
    for flag, name in _PATH_FLAGS:
        value = getattr(options, name)
        if value is not None:
            argv.append(f"{flag}={value}")
    return argv


@dataclass
class ViewHandle:
    """One launched viewer process: the command, the child, and its outcome.

    ``done`` is set last, after the record forwarder is detached; UI code
    learns completion through ``on_done`` — never block the UI thread on
    ``done`` (``call_from_thread`` would deadlock); ``done`` is for
    non-UI waiters such as tests and scripts.
    """

    #: The wells the session shows.
    wells: list[str]
    #: The argument vector handed to :data:`SPAWN`.
    argv: list[str]
    #: The child process once spawned (None when the spawn failed).
    process: Any = None
    #: The child's exit status once it returned (None before that, and
    #: after a failed spawn).
    returncode: int | None = None
    #: The exception when spawning failed (never fatal).
    error: BaseException | None = None
    #: Set when the launch is over (the child exited, or the spawn failed).
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def failed(self) -> bool:
        """Whether the launch failed: the spawn raised, or the child exited
        non-zero (napari missing, no display)."""
        return self.error is not None or (self.returncode or 0) != 0


def _log_child_output(stdout: str | None, stderr: str | None) -> None:
    """Log the child's captured output line by line (the viewer's own
    INFO lines and napari's diagnostics), through the package logger."""
    for stream in (stdout, stderr):
        for line in (stream or "").splitlines():
            if line.strip():
                log.info("viewer: %s", line)


@contextmanager
def _forwarding(
    deliver: Callable[..., None],
    on_record: Callable[[logging.LogRecord], None] | None,
) -> Iterator[None]:
    """Route the package logger's records to the screen for the block's
    duration, through the shared per-run forwarder seam.

    Attached only around the launch's own messages — the opening lines and
    the outcome — never while the viewer runs: a napari session is open
    for as long as the user keeps it, and forwarding every other screen's
    records to it meanwhile would be noise (and an ever-growing pane).
    """
    handler = record_forwarder(deliver, on_record)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)


def _run(
    handle: ViewHandle,
    deliver: Callable[..., None],
    on_record: Callable[[logging.LogRecord], None] | None,
    on_done: Callable[[ViewHandle], None],
) -> None:
    """The thread body of one launch: report and spawn the child, wait for
    it, report its outcome, and report completion.

    The child outlives the launch handshake — the viewer stays open until
    the user closes it — so the record forwarder is attached only around
    the messages this launch logs, not for the child's lifetime.
    """
    try:
        with _forwarding(deliver, on_record):
            log.info(
                "viewer: opening %s in a separate process",
                ", ".join(handle.wells),
            )
            log.info("viewer: %s", " ".join(handle.argv))
            process = SPAWN(
                handle.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        handle.process = process
        stdout, stderr = process.communicate()
        handle.returncode = process.returncode
    except Exception as exc:  # noqa: BLE001 — a launch failure is not fatal
        handle.error = exc
        with _forwarding(deliver, on_record):
            log.error("viewer: could not start the viewer process: %s", exc)
    else:
        with _forwarding(deliver, on_record):
            _log_child_output(stdout, stderr)
            if handle.failed:
                log.error(
                    "viewer: the viewer process exited with status %s",
                    handle.returncode,
                )
            else:
                log.info("viewer: the viewer process closed")
    finally:
        handle.done.set()
        try:
            deliver(on_done, handle)
        except Exception:  # noqa: BLE001 — reporting must never be fatal
            log.error("viewer: reporting the finished launch failed")


def start(
    wells: Sequence[str],
    options: ViewOptions,
    *,
    on_done: Callable[[ViewHandle], None],
    on_record: Callable[[logging.LogRecord], None] | None = None,
    app: App[Any] | None = None,
) -> ViewHandle:
    """Start the viewer for ``wells`` on ``options`` in a separate process.

    Returns as soon as the launch is under way; ``on_done(handle)`` fires
    when the child has exited (or the spawn failed), ``on_record``
    optionally receives every log record of the launch live. With ``app``
    set both are marshalled onto the UI thread via
    ``app.call_from_thread``; the TUI stays responsive throughout.

    No exception escapes: a failed spawn and a non-zero child are stored
    on the handle and logged (napari missing, no display), and the app
    keeps running.
    """
    handle = ViewHandle(wells=list(wells), argv=command(wells, options))
    deliver = _deliverer(app)

    def body() -> None:
        _run(handle, deliver, on_record, on_done)

    if app is None:
        threading.Thread(
            target=body,
            name="colony-tracking-view",
            daemon=True,
        ).start()
    else:
        app.run_worker(body, name="view wells", thread=True)
    return handle
