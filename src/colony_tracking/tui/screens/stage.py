"""The per-stage screens: one :class:`StageScreen` per stage, generated
from the stage registry (spec story 36, requirement 2).

The registry (:data:`STAGES`) maps a draft section name to its options
dataclass and user-visible title — the same names the config file's stage
sections and the home menu use. There is exactly one reusable screen class,
instantiated per stage: a future train stage slots in by adding a registry
entry (and a ``draft.stages`` section), with no per-stage class anywhere.
The editable fields come from the dataclass through
:mod:`colony_tracking.tui.forms` (type hints + help metadata only); the
pipeline-owned fields (:data:`colony_tracking.pipeline._OWNED` — they come
from the run level and the chain wiring, never from a stage section) are
not form fields: the screen shows them read-only with their derived values
(:func:`owned_values`), and the free fields that default to a derived name
grey that default in their empty input (:func:`placeholders`).

The actions are the spec's per-stage decision point (requirement 5):
*Run this stage*, *Write TOML…* (the shared
:class:`~colony_tracking.tui.screens._save.SaveConfigDialog`), *Run whole
pipeline from draft*, and *Back*. The run/write actions stay disabled
until the draft resolves with the values on screen; a failing resolve
shows the pipeline's exact :class:`~colony_tracking.pipeline.ConfigError`
message instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Static

from ..._options import (
    AnalysisOptions,
    DifferenceOptions,
    Options,
    SegmentationOptions,
)
from ...pipeline import (
    PipelineConfig,
    _OWNED,
    _derived_difference_tree,
    _derived_segment_root,
    _reference_number,
)
from ..draft import RUN_LEVEL, ConfigDraft
from ..forms import FormScreen, OptionForm, kebab, specs_of


@dataclass(frozen=True)
class StageSpec:
    """One entry of the stage registry: the draft section, the user-visible
    title, and the options dataclass the form is generated from."""

    #: The draft section name (the config's stage name).
    name: str
    #: The user-visible title (the home menu's wording).
    title: str
    #: The options dataclass the form is generated from.
    options: type[Options]


#: The stage registry: draft section name → spec. A future stage (train)
#: slots in by adding an entry — the screens are one class, generated from
#: the entry's dataclass.
STAGES: dict[str, StageSpec] = {
    "difference": StageSpec("difference", "Difference", DifferenceOptions),
    "segmentation": StageSpec("segmentation", "Segment", SegmentationOptions),
    "analysis": StageSpec("analysis", "Analyze", AnalysisOptions),
}


def _shown(value: object) -> str:
    """A derived value for display: ``—`` while it cannot be derived (the
    run level is still incomplete)."""
    return "—" if value is None else str(value)


def owned_values(draft: ConfigDraft, stage: str) -> tuple[tuple[str, str], ...]:
    """The owned fields of ``stage`` with their derived values, for the
    stage screen's read-only display: ``(kebab label, value text)`` pairs,
    in the options dataclass's declaration order.

    The values are the derivation ``PipelineConfig.resolve_stages``
    performs: the raw dir and the reference from ``[pipeline]``, the
    segment input forced to ``diff``, and the difference tree behind the
    segment stage's ``diff_dir`` (the difference section's own ``out_dir``
    when set, else the derived tree name). A value that cannot be derived
    yet shows as ``—``. The owned keys come from the registry entry's
    dataclass and :data:`~colony_tracking.pipeline._OWNED` — a future
    stage without owned fields displays nothing.
    """
    owned = _OWNED.get(stage, frozenset())
    keys = [spec.key for spec in specs_of(STAGES[stage].options) if spec.key in owned]
    raw = draft.get(RUN_LEVEL, "raw_dir")
    out_root = draft.get(RUN_LEVEL, "out_root")
    reference = draft.get(RUN_LEVEL, "reference")
    if reference is None:
        reference = "first"
    number = _reference_number(reference)
    diff_dir = draft.get("difference", "out_dir")
    if diff_dir is None and out_root is not None:
        diff_dir = str(_derived_difference_tree(Path(str(out_root)), reference))
    shown: dict[tuple[str, str], str] = {
        ("difference", "raw_dir"): _shown(raw),
        ("difference", "reference"): "first" if number is None else str(number),
        ("segmentation", "input"): "diff",
        ("segmentation", "diff_dir"): _shown(diff_dir),
        ("analysis", "raw_dir"): _shown(raw),
    }
    return tuple((kebab(key), shown.get((stage, key), "—")) for key in keys)


def placeholders(draft: ConfigDraft, stage: str) -> dict[str, str]:
    """The derived default each free field of ``stage`` falls back to when
    left blank — greyed in the empty input so the user sees what the chain
    would pick. Stages without such fields (a future train stage) get an
    empty mapping."""
    out_root = draft.get(RUN_LEVEL, "out_root")
    if out_root is None:
        return {}
    root = Path(str(out_root))
    reference = draft.get(RUN_LEVEL, "reference")
    if reference is None:
        reference = "first"
    tree = str(_derived_difference_tree(root, reference))
    segment_root = str(_derived_segment_root(root, reference))
    per_stage: dict[str, dict[str, str]] = {
        "difference": {"out_dir": tree},
        "segmentation": {"out_dir": segment_root},
        "analysis": {"seg_dirs": segment_root, "out_root": str(root / "analysis")},
    }
    return per_stage.get(stage, {})


class StageScreen(FormScreen):
    """The screen for one stage: its options as a form, the pipeline-owned
    fields read-only with their derived values, and the spec's four
    actions.

    Instantiated per stage from the registry: ``StageScreen("difference")``
    renders the difference stage's options — a future stage needs only a
    :data:`STAGES` entry. The run/write actions gate on the draft (see
    :class:`~colony_tracking.tui.forms.FormScreen`); a passing press hands
    the resolved config to :meth:`start_run`.
    """

    gated_actions: ClassVar[tuple[str, ...]] = (
        "run-stage",
        "write-toml",
        "run-pipeline",
    )

    BINDINGS = [
        Binding("r", "run_stage", "Run this stage"),
        Binding("w", "write_toml", "Write TOML…"),
        Binding("p", "run_pipeline", "Run pipeline"),
    ]

    def __init__(
        self,
        stage: str,
        draft: ConfigDraft | None = None,
        *,
        start_run: Callable[[str, PipelineConfig], None] | None = None,
    ) -> None:
        super().__init__(draft)
        self._stage = stage
        self._start_run = start_run

    @property
    def stage(self) -> str:
        """The screen's stage (the draft section name)."""
        return self._stage

    def compose_body(self) -> ComposeResult:
        entry = STAGES[self._stage]
        owned = _OWNED.get(self._stage, frozenset())
        specs = specs_of(entry.options)
        self._form = OptionForm(
            self._stage,
            [spec for spec in specs if spec.key not in owned],
            self._resolve_draft(),
            on_change=self._on_form_change,
        )
        yield Static(f"{entry.title} — stage options", id="stage-title")
        yield Static(Text(""), id="owned-values")
        yield self._form

    def compose_actions(self) -> ComposeResult:
        yield Button("Run this stage", id="run-stage")
        yield Button("Write TOML…", id="write-toml")
        yield Button("Run whole pipeline from draft", id="run-pipeline")
        yield Button("Back", id="back")

    def refresh_extras(self) -> None:
        """The derived owned values and the greyed derived defaults — both
        change when the run level changes (edited on the config screen)."""
        values = owned_values(self.draft, self._stage)
        joined = "  ".join(f"{label} = {value}" for label, value in values)
        self.query_one("#owned-values", Static).update(
            Text("owned by the pipeline (derived): " + (joined if joined else "none"))
        )
        for key, text in placeholders(self.draft, self._stage).items():
            self._form.set_placeholder(key, text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-stage":
            self.action_run_stage()
        elif event.button.id == "write-toml":
            self.action_write_toml()
        elif event.button.id == "run-pipeline":
            self.action_run_pipeline()
        elif event.button.id == "back":
            self.action_back()

    def action_run_stage(self) -> None:
        """*Run this stage*: resolve the draft and hand the run to the
        app's run flow (the config is the one ``run_pipeline`` would give
        this stage — owned fields forced)."""
        if not self._enabled:
            return
        config = self._resolved_or_report()
        if config is not None:
            self.start_run(self._stage, config)

    def action_run_pipeline(self) -> None:
        """*Run whole pipeline from draft*: resolve the draft and hand the
        whole-chain run to the app's run flow."""
        if not self._enabled:
            return
        config = self._resolved_or_report()
        if config is not None:
            self.start_run("pipeline", config)

    def start_run(self, kind: str, config: PipelineConfig) -> None:
        """Hand a resolved run request to the app's run flow.

        ``kind`` is the stage's name or ``"pipeline"``; ``config`` is the
        resolved :class:`~colony_tracking.pipeline.PipelineConfig`. The
        default reports that runs are not wired yet — the run screens'
        ticket replaces this hand-off (write the TOML first, start the run
        through the runner, show it on the run screen) by passing a
        ``start_run`` callable to the constructor, leaving the form
        machinery unchanged.
        """
        if self._start_run is not None:
            self._start_run(kind, config)
            return
        self.app.notify(
            "Running is not wired yet — write the TOML and start the run "
            "with: colony-tracking pipeline --config"
        )
