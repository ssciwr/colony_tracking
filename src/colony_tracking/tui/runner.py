"""StageRunner: run the stages in-process from the TUI.

:func:`start` runs one of ``run_difference`` / ``run_segmentation`` /
``run_analysis`` / ``run_pipeline`` on its kind's resolved config in a
thread — the stage functions are called unchanged, so a TUI run behaves
exactly like a CLI run of the same config (resume via done markers
included). One run at a time: starting a second run while one is active
is an error the UI reports, and there is no cancellation — stopping
means quitting the TUI with Ctrl-Q, and the stages' done markers make
the next start a resume.

Each run kind carries its run type's own resolved config: a stage kind
(:data:`KINDS` minus ``pipeline``) receives that stage's own
:class:`~colony_tracking._options.Options`, the ``pipeline`` kind the
chain's :class:`~colony_tracking.pipeline.PipelineConfig`.

Log routing: while a run is active, a :class:`logging.Handler` attached
to the package logger :data:`colony_tracking.log` forwards every record
to the UI. With a running textual ``App`` (the normal TUI case) records
and the completion callback are marshalled onto the UI thread via
``App.call_from_thread``; without one (tests, scripts) they run directly
on the worker thread. The handler is attached on start and detached on
completion — exactly what was attached, so repeated runs cannot stack
handlers.

Failures are never fatal: an exception escaping a stage is caught,
stored on the :class:`RunHandle`, logged, and reported through
``on_done``. A summary that reports failed work (failed wells, the
analyze stage's failures, a failed pipeline status) is reported through
the same lines the CLI logs for an exit-1 run
(:func:`failed_summary_lines` is the single source). Run handles also
carry the run's ``config`` so the UI can show what is running.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from textual.app import App

from .. import log
from .._options import Options
from ..analysis import AnalysisSummary, run_analysis
from ..difference import DifferenceSummary, run_difference
from ..pipeline import PipelineConfig, PipelineSummary, run_pipeline
from ..segmentation import SegmentationSummary, run_segmentation

#: The kinds :func:`start` can run: one per stage plus the whole chain.
KINDS: tuple[str, ...] = ("difference", "segmentation", "analysis", "pipeline")

#: The runner's table: kind -> the run to make on the kind's resolved
#: config — a stage kind takes that stage's own
#: :class:`~colony_tracking._options.Options`, the ``pipeline`` kind the
#: chain's :class:`~colony_tracking.pipeline.PipelineConfig`. The
#: lambdas resolve the ``run_*`` globals at call time, so tests can
#: monkeypatch them on this module.
_RUNNERS: dict[str, Callable[[PipelineConfig | Options], Any]] = {
    "difference": lambda options: run_difference(options),
    "segmentation": lambda options: run_segmentation(options),
    "analysis": lambda options: run_analysis(options),
    "pipeline": lambda config: run_pipeline(config),
}

#: The run in progress, if any (one run at a time; guarded by ``_LOCK``).
_ACTIVE: RunHandle | None = None
_LOCK = threading.Lock()

#: The one-run-at-a-time message, formatted with the active run's kind:
#: :func:`start` raises it as its :class:`RuntimeError` and the run
#: screens report the same text.
ACTIVE_MESSAGE = (
    "a {kind} run is already active — the runner runs one at a time and "
    "cannot cancel: wait for it to finish, or quit the TUI (the next "
    "start resumes via done markers)"
)


def summary_failed(summary: object) -> bool:
    """Whether a stage/pipeline summary reports failed work — the CLI's
    exit-code-1 semantics, which the TUI surfaces as an error panel."""
    if isinstance(summary, PipelineSummary):
        return summary.status == "failed"
    if isinstance(summary, (DifferenceSummary, SegmentationSummary)):
        return bool(summary.failed)
    if isinstance(summary, AnalysisSummary):
        return bool(summary.failures or summary.failed_wells)
    return False


def failed_summary_lines(kind: str, summary: Any) -> list[str]:
    """The failed-work lines of one finished run of ``kind``, in the
    CLI's ERROR-line wording — the single source the runner logs and the
    run screen's error panel shows. Empty when the summary reports no
    failed work of that kind."""
    if kind == "difference":
        failed = [
            (result.well, result.error)
            for result in summary.wells.values()
            if result.status == "failed"
        ]
        if not failed:
            return []
        return [
            f"difference: {len(failed)} well(s) failed: "
            + ", ".join(f"{well} ({error})" for well, error in failed)
        ]
    if kind == "segmentation":
        if not summary.failed_wells:
            return []
        return [
            f"segment: {summary.failed} well(s) failed: "
            + ", ".join(summary.failed_wells)
        ]
    if kind == "analysis":
        lines = []
        if summary.failures:
            lines.append(
                f"analyze: {len(summary.failures)} root(s) failed: "
                + "; ".join(summary.failures)
            )
        if summary.failed_wells:
            lines.append(
                f"analyze: {len(summary.failed_well_names)} well(s) failed: "
                + ", ".join(summary.failed_well_names)
            )
        return lines
    if kind == "pipeline":
        return [f"pipeline: the {summary.failed_stage} stage failed"]
    return []


def _log_failed_summary(kind: str, summary: Any) -> None:
    """Log a failed summary with the CLI's ERROR-line wording
    (:func:`failed_summary_lines`)."""
    for line in failed_summary_lines(kind, summary):
        log.error("%s", line)


@dataclass
class RunHandle:
    """One started run: its kind, config, outcome, and completion signal.

    ``summary`` is set when the run returned one; ``error`` is set when
    the stage raised. ``done`` is set last, after the handler is detached
    and the runner is free for the next run. UI code learns completion
    through ``on_done`` — never block the UI thread on ``done``
    (``call_from_thread`` would deadlock); ``done`` is for non-UI
    waiters such as tests and scripts.
    """

    kind: str
    #: The resolved config the run runs on: the stage's own Options for a
    #: stage kind, the chain's PipelineConfig for ``pipeline``.
    config: PipelineConfig | Options
    #: The stage/pipeline summary when the run returned one.
    summary: object | None = None
    #: The exception when the stage raised one (never fatal).
    error: BaseException | None = None
    #: Set when the run is finished (success, failed summary, or error).
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def failed(self) -> bool:
        """Whether the run failed: an exception, or a summary reporting
        failed work (the CLI's exit-1 semantics)."""
        if self.error is not None:
            return True
        return self.summary is not None and summary_failed(self.summary)


def active() -> RunHandle | None:
    """The run in progress, or ``None`` when the runner is free."""
    with _LOCK:
        handle = _ACTIVE
    return None if handle is None or handle.done.is_set() else handle


class _ForwardHandler(logging.Handler):
    """The per-run log forwarder: ``emit`` hands the record itself to the
    deliver callback — not formatted text, so the pane chooses its own
    rendering. A delivery failure is never fatal: ``handleError``
    reports it and the run goes on."""

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


def _deliverer(
    app: App[Any] | None,
) -> Callable[..., None]:
    """The post-to-UI-thread wrapper for one run: ``app.call_from_thread``
    when a textual App is given, a direct call otherwise."""
    if app is None:
        return lambda callback, *args: callback(*args)
    return lambda callback, *args: app.call_from_thread(callback, *args)


def _run(
    handle: RunHandle,
    deliver: Callable[..., None],
    on_record: Callable[[logging.LogRecord], None] | None,
    on_done: Callable[[RunHandle], None],
) -> None:
    """The thread body of one run: attach the forwarder, run the stage,
    capture the outcome, detach, and report.

    Ordering matters at the end: the handler is detached and the runner
    freed *before* ``done`` is set and ``on_done`` fires, so a completion
    callback that immediately starts the next run can never stack
    handlers or trip the one-run-at-a-time check.
    """
    handler = _ForwardHandler(deliver, on_record)
    global _ACTIVE  # noqa: PLW0603 — module-level, one run at a time
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        summary = _RUNNERS[handle.kind](handle.config)
    except Exception as exc:  # noqa: BLE001 — a stage failure is not fatal
        handle.error = exc
        log.error("tui: %s run failed\n%s", handle.kind, traceback.format_exc())
    else:
        handle.summary = summary
        if summary_failed(summary):
            _log_failed_summary(handle.kind, summary)
    finally:
        log.removeHandler(handler)
        with _LOCK:
            if _ACTIVE is handle:
                _ACTIVE = None
        handle.done.set()
        try:
            deliver(on_done, handle)
        except Exception:  # noqa: BLE001 — reporting must never be fatal
            log.error(
                "tui: reporting the finished %s run failed\n%s",
                handle.kind,
                traceback.format_exc(),
            )


def start(
    kind: str,
    config: PipelineConfig | Options,
    *,
    on_done: Callable[[RunHandle], None],
    on_record: Callable[[logging.LogRecord], None] | None = None,
    app: App[Any] | None = None,
) -> RunHandle:
    """Start one run of ``kind`` on the resolved ``config``.

    ``kind`` is one of :data:`KINDS`; the run happens in a thread — via
    ``app.run_worker(..., thread=True)`` when the running textual ``app``
    is given, on a plain daemon thread otherwise — calling the kind's run
    function unchanged with the resolved config: a stage kind receives
    that stage's own :class:`~colony_tracking._options.Options`, the
    ``pipeline`` kind the chain's
    :class:`~colony_tracking.pipeline.PipelineConfig`.

    ``on_done(handle)`` fires when the run is finished; ``on_record``
    optionally receives every log record of the run live. With ``app``
    set, both are marshalled onto the UI thread via
    ``app.call_from_thread``.

    Starting a second run while one is active raises :class:`RuntimeError`
    (the UI reports it; there is no cancellation — quitting and
    re-starting resumes via done markers).
    """
    global _ACTIVE  # noqa: PLW0603 — module-level, one run at a time
    if kind not in _RUNNERS:
        raise ValueError(f"unknown run kind {kind!r} — expected one of {list(KINDS)}")
    with _LOCK:
        if _ACTIVE is not None and not _ACTIVE.done.is_set():
            raise RuntimeError(ACTIVE_MESSAGE.format(kind=_ACTIVE.kind))
        handle = RunHandle(kind=kind, config=config)
        _ACTIVE = handle
    _spawn(handle, on_done=on_done, on_record=on_record, app=app)
    return handle


def _spawn(
    handle: RunHandle,
    *,
    on_done: Callable[[RunHandle], None],
    on_record: Callable[[logging.LogRecord], None] | None,
    app: App[Any] | None,
) -> None:
    """Start :func:`_run` for ``handle`` in the right kind of thread."""
    deliver = _deliverer(app)

    def body() -> None:
        _run(handle, deliver, on_record, on_done)

    if app is None:
        threading.Thread(
            target=body,
            name=f"colony-tracking-{handle.kind}",
            daemon=True,
        ).start()
    else:
        app.run_worker(body, name=f"run {handle.kind}", thread=True)
