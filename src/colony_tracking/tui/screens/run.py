"""The run screen: one run's live log, summary-driven status, and errors.

:class:`RunScreen` shows one run from start to finish: the streamed log
(a :class:`~textual.widgets.RichLog` fed by the ticket-01 log routing), a
live status line, the final summary block, and an error panel when the
run fails — an exception, or a summary that reports failed work (the
CLI's exit-1 semantics). Every outcome stays in the UI: the app survives
any run failure.

The screen starts its own run on mount — ``runner.start`` with the
screen's app — so the log handler attaches before the worker thread
spawns and no record is lost. Status comes from the summary objects
alone (spec story 40: log text is never parsed for status); the single
exception is the current-stage hint during a pipeline run, which
recognizes the exact structured marker ``run_pipeline`` emits as it
starts a chain stage (a match on the record's template and arguments,
never on formatted text) — the log pane itself tells the full story
meanwhile. While a run is active the status line also records the
no-cancellation decision: quitting the TUI (Ctrl-Q) mid-run is safe, the
stages are crash-safe via their per-well done markers, and the next
start resumes where the run left off.

Direct runs are written first (spec requirement 5): the entry points
below resolve the caller's draft, then write the TOML through the shared
:class:`~colony_tracking.tui.screens._save.SaveConfigDialog` (each run
type's own suggested path), then push the run screen with the dialog's
resolved config:

- :func:`open_pipeline_run` — the quick start's *Run now* (and any
  whole-chain start): the full chain on the pipeline draft as it stands,
  its within-run wiring intact.
- :func:`open_stage_run` — the stage screens' *Run this stage*: one
  stage on its own draft — the stage's own independent, fully-explicit
  config, and the run takes its config's inputs as given. There is no
  cross-run pre-flight: whether another run type has produced what this
  run's paths point at is the config's own content, not the runner's
  business. Re-running a stage re-enters it — the per-well done markers
  make that a resume, not a wipe.

Both entry points return the problem message when a run cannot start
(the runner busy, an invalid draft) after reporting it in the UI, and
``None`` once the flow is underway — the dialog takes over from there
(cancelling it writes nothing and runs nothing).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static

from ..._options import Options
from ...pipeline import (
    AnalysisSummary,
    ConfigError,
    DifferenceSummary,
    PipelineConfig,
    PipelineSummary,
    SegmentationSummary,
)
from .. import runner
from ..draft import STAGES, ConfigDraft
from ..runner import ACTIVE_MESSAGE, RunHandle, failed_summary_lines
from ._save import SaveConfigDialog

#: The exact template of the pipeline's stage-start marker:
#: ``run_pipeline`` logs ``"pipeline: <label> → <target>"`` as it starts
#: each chain stage. Matching the record's *template* — the structured
#: marker, never its formatted text — drives the status line's
#: current-stage hint without parsing log text (spec story 40).
_STAGE_START_TEMPLATE = "pipeline: %s → %s"

#: Rich style per log level for the pane's records (INFO stays plain).
_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG: "dim",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
}


def _record_text(record: logging.LogRecord) -> Text:
    """One log record as styled text for the pane: a right-aligned level
    tag, then the message — plain text, no markup, so paths with brackets
    render exactly as logged."""
    text = Text()
    text.append(
        f"{record.levelname.lower():>7} ",
        style=_LEVEL_STYLES.get(record.levelno, ""),
    )
    text.append(record.getMessage())
    return text


def _stage_start_marker(record: logging.LogRecord) -> tuple[str, str] | None:
    """The ``(stage label, target)`` of a pipeline stage-start marker
    record, or ``None`` (see :data:`_STAGE_START_TEMPLATE`)."""
    if record.msg != _STAGE_START_TEMPLATE or len(record.args or ()) != 2:
        return None
    label, target = record.args
    return str(label), str(target)


def _failed_names(summary: Any) -> list[str]:
    """The names a stage summary reports as failed (empty when none)."""
    if isinstance(summary, DifferenceSummary):
        return [
            result.well
            for result in summary.wells.values()
            if result.status == "failed"
        ]
    if isinstance(summary, SegmentationSummary):
        return list(summary.failed_wells)
    if isinstance(summary, AnalysisSummary):
        return [*summary.failures, *summary.failed_well_names]
    return []


def _stage_detail(summary: Any) -> str:
    """One stage summary as a status-line detail: the totals, and the
    failed work named when the stage failed — from the summary object
    alone (spec story 40)."""
    if isinstance(summary, AnalysisSummary):
        detail = (
            f"roots {len(summary.roots)} · colonies "
            f"{sum(root.colonies for root in summary.roots.values())} · "
            f"trajectory rows "
            f"{sum(root.rows for root in summary.roots.values())}"
        )
    else:
        detail = f"processed {summary.processed} · skipped {summary.skipped}"
    if runner.summary_failed(summary):
        names = _failed_names(summary)
        detail += " — failed" + (f": {', '.join(names)}" if names else "")
    return detail


def _pipeline_summary_lines(summary: PipelineSummary) -> list[str]:
    """The per-stage lines of a pipeline summary, in chain order: the
    stages that completed carry their summaries; the stage that failed
    with an exception has none (it says so), and the later ones were not
    reached."""
    lines = []
    failed_at = (
        STAGES.index(summary.failed_stage) if summary.failed_stage is not None else None
    )
    for position, name in enumerate(STAGES):
        stage_summary = getattr(summary, name)
        if stage_summary is not None:
            lines.append(f"{name:<13}{_stage_detail(stage_summary)}")
        elif summary.failed_stage == name:
            lines.append(f"{name:<13}failed (exception — see the log)")
        elif failed_at is not None and position > failed_at:
            lines.append(f"{name:<13}not reached")
    return lines


def _summary_lines(kind: str, summary: object) -> list[str]:
    """The summary block's lines for a finished run — per-stage status
    lines driven by the summary objects (spec story 40)."""
    if isinstance(summary, PipelineSummary):
        return _pipeline_summary_lines(summary)
    return [f"{kind:<13}{_stage_detail(summary)}"]


def _outcome_text(handle: RunHandle) -> str:
    """The final one-line status of a finished run."""
    finished = time.strftime("%H:%M:%S")
    if handle.error is not None:
        return (
            f"{handle.kind} failed: {type(handle.error).__name__}: "
            f"{handle.error} (finished {finished})"
        )
    if isinstance(handle.summary, PipelineSummary):
        if handle.summary.status == "ok":
            return f"pipeline finished — all three stages ok ({finished})"
        return (
            f"pipeline finished — stopped at the "
            f"{handle.summary.failed_stage} stage ({finished})"
        )
    if handle.failed:
        return f"{handle.kind} finished with failures ({finished})"
    return f"{handle.kind} finished — ok ({finished})"


def _problem_lines(handle: RunHandle) -> list[str]:
    """The error panel's lines for a finished run (empty when it did not
    fail): the exception, or the failed work the summary reports — never
    log text, and worded exactly like the runner's logged ERROR lines
    (the shared
    :func:`~colony_tracking.tui.runner.failed_summary_lines`)."""
    if handle.error is not None:
        raised = (
            f"the {handle.kind} run raised "
            f"{type(handle.error).__name__}: {handle.error}"
        )
        return [raised, "the full traceback is in the log below."]
    summary = handle.summary
    if summary is None or not runner.summary_failed(summary):
        return []
    if isinstance(summary, PipelineSummary):
        lines = [f"the pipeline stopped at the {summary.failed_stage} stage."]
        if summary.failed_stage is not None:
            embedded = getattr(summary, summary.failed_stage)
            if embedded is not None:
                lines.extend(failed_summary_lines(summary.failed_stage, embedded))
        return lines
    return failed_summary_lines(handle.kind, summary)


def _report(app: App[Any], message: str) -> str:
    """Report a pre-flight problem in the UI and return it (callers may
    also keep it on screen — the toast is transient). The toast goes out
    as a plain markup-free string: the message names config sections
    (``[difference]``) and paths, which markup would eat."""
    app.notify(message, severity="error", markup=False)
    return message


class RunScreen(Screen[None]):
    """One run from start to finish (see the module docstring).

    The run starts on mount; :attr:`handle` holds the runner's handle as
    soon as it exists — ``None`` when a pre-flight problem replaced the
    run (shown in the error panel instead; the app stays alive).
    """

    CSS = """
    #run-status { padding: 0 1; }
    #run-error {
        border: round red; padding: 0 1; margin: 1 1 0 1; display: none;
    }
    #run-summary {
        border: round green; padding: 0 1; margin: 1 1 0 1; display: none;
    }
    #run-log { height: 1fr; padding: 0 1; }
    """

    def __init__(
        self,
        kind: str,
        config: PipelineConfig | Options,
        *,
        toml_path: Path | None = None,
    ) -> None:
        super().__init__()
        if kind not in runner.KINDS:
            raise ValueError(
                f"unknown run kind {kind!r} — expected one of {list(runner.KINDS)}"
            )
        self.kind = kind
        #: The resolved config the run goes out with — for a direct start
        #: the dialog's write result, so the run is reproducible from
        #: it: a stage kind's own
        #: :class:`~colony_tracking._options.Options`, the chain's
        #: :class:`~colony_tracking.pipeline.PipelineConfig`.
        self.config = config
        #: Where the run's config TOML was written (None: not recorded).
        self.toml_path = toml_path
        #: The runner's handle once the run started (None before that).
        self.handle: RunHandle | None = None
        self._started_at: str | None = None
        self._current_stage: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="run-status", markup=False)
        yield Static("", id="run-error", markup=False)
        yield Static("", id="run-summary", markup=False)
        yield RichLog(id="run-log", wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start the run, or report why not (never fatal)."""
        problem = self._preflight_problem()
        if problem is None:
            try:
                self.handle = runner.start(
                    self.kind,
                    self.config,
                    on_done=self._on_done,
                    on_record=self._on_record,
                    app=self.app,
                )
            except RuntimeError as err:  # a run slipped in meanwhile
                problem = str(err)
        if problem is not None:
            self._show_error([problem])
            return
        self._started_at = time.strftime("%H:%M:%S")
        self._refresh_running_status()

    def _preflight_problem(self) -> str | None:
        """Why the run must not start (None: good to go) — the runner's
        one-at-a-time rule."""
        active = runner.active()
        if active is None:
            return None
        return ACTIVE_MESSAGE.format(kind=active.kind)

    def _on_record(self, record: logging.LogRecord) -> None:
        """One log record of the active run: into the pane, plus the
        current-stage hint when it is one of the pipeline's structured
        stage-start markers."""
        self.query_one("#run-log", RichLog).write(_record_text(record))
        if self.kind != "pipeline":
            return
        marker = _stage_start_marker(record)
        if marker is not None:
            self._current_stage = f"{marker[0]} → {marker[1]}"
            self._refresh_running_status()

    def _refresh_running_status(self) -> None:
        """The live status line: what is running, since when, the
        no-cancellation note, and — during a pipeline run — the chain
        stage the markers say is running now."""
        running = (
            f"running {self.kind} (started {self._started_at}) — no "
            "cancellation in v1: quitting (Ctrl-Q) is safe, the next "
            "start resumes via done markers"
        )
        lines = [running]
        if self._current_stage is not None:
            lines.append(f"now: {self._current_stage}")
        self.query_one("#run-status", Static).update("\n".join(lines))

    def _on_done(self, handle: RunHandle) -> None:
        """The run finished: the final status and summary block, and — on
        a failure — the error panel (the app stays alive either way)."""
        self.handle = handle
        self.query_one("#run-status", Static).update(_outcome_text(handle))
        if handle.summary is not None:
            summary = self.query_one("#run-summary", Static)
            summary.update("\n".join(_summary_lines(handle.kind, handle.summary)))
            summary.display = True
        problem = _problem_lines(handle)
        if problem:
            self._show_error(problem)

    def _show_error(self, lines: list[str]) -> None:
        """Show the error panel with ``lines`` (the screen stays alive
        and interactive)."""
        error = self.query_one("#run-error", Static)
        error.update("\n".join(lines))
        error.display = True


def _open_run(app: App[Any], draft: ConfigDraft, kind: str) -> str | None:
    """The shared direct-run flow (see the module docstring): the
    active-run check, then the draft resolved, then the shared save
    dialog — the reproducibility write — and, on a successful write, the
    run screen with the dialog's resolved config. Returns the problem
    message when a check failed (also reported in the UI), None once the
    dialog is up. There is no cross-run pre-flight: the draft's own
    inputs are taken as given (:func:`resolve` is where anything wrong
    with them reports)."""
    active = runner.active()
    if active is not None:
        return _report(app, ACTIVE_MESSAGE.format(kind=active.kind))
    try:
        config = draft.resolve()
    except ConfigError as err:
        return _report(app, str(err))

    def after_write(result: tuple[Path, PipelineConfig | Options] | None) -> None:
        if result is None:  # cancelled: nothing runs
            return
        path, written = result
        app.push_screen(RunScreen(kind, written, toml_path=path))

    app.push_screen(SaveConfigDialog(draft), after_write)
    return None


def open_pipeline_run(app: App[Any], draft: ConfigDraft) -> str | None:
    """Start the whole chain from the pipeline draft — also the quick
    start's *Run now*: the full chain on ``draft`` as it stands, its own
    TOML written first (see the module docstring).

    Returns the problem message when a check failed (also reported in
    the UI), None once the flow is underway.
    """
    return _open_run(app, draft, "pipeline")


def open_stage_run(app: App[Any], draft: ConfigDraft, stage: str) -> str | None:
    """Start one stage on its own — the stage screens' action (the
    ticket-04 wiring calls this): a standalone run of ``stage`` on ``draft``,
    that stage type's own independent, fully-explicit config, its own
    TOML written first (see the module docstring). The run takes its
    config's inputs as given — no other run type's state is consulted
    (a stage's ``resolve`` is the only gate on the config itself).
    Re-running a stage re-enters it (a resume via the done markers).

    Returns the problem message when a check failed (also reported in
    the UI), None once the flow is underway.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} — expected one of {list(STAGES)}")
    return _open_run(app, draft, stage)
