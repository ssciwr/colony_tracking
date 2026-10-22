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
from the run level and the chain wiring, never from a stage section) split
per the spec addendum's decision 2:

- the *run-level mirrors* (:func:`mirror_keys` — an owned key that names a
  run-level field, plus ``out_root`` on every stage, since every derived
  value hangs off it) are editable inputs in their own group under the
  derived line (:func:`mirror_specs`, labeled ``run-level: ...``): every
  edit writes the ``[pipeline]`` key the mirror names, so a stage run — a
  difference run, say — is startable entirely from its stage screen;
- the *chain-wired* fields (``[segmentation].input``/``diff_dir``) stay
  read-only: the derived line shows them with their derived values
  (:func:`owned_values`, filtered to them), and the free fields that
  default to a derived name grey that default in their empty input
  (:func:`placeholders`).

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
from textual.containers import VerticalScroll
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
    _RUN_LEVEL_FIELDS,
    _derived_difference_tree,
    _derived_segment_root,
    _reference_number,
)
from ..draft import RUN_LEVEL, ConfigDraft
from ..forms import (
    FieldSpec,
    FormScreen,
    OptionForm,
    kebab,
    run_level_mirror,
    specs_of,
)
from .run import open_pipeline_run, open_stage_run


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


def owned_entries(draft: ConfigDraft, stage: str) -> tuple[tuple[str, str, str], ...]:
    """The owned fields of ``stage`` with their derived values, as
    ``(key, kebab label, value text)`` triples in the options dataclass's
    declaration order (see :func:`owned_values` for the derivation)."""
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
    return tuple((key, kebab(key), shown.get((stage, key), "—")) for key in keys)


def owned_values(draft: ConfigDraft, stage: str) -> tuple[tuple[str, str], ...]:
    """Every owned field of ``stage`` with its derived value, as
    ``(kebab label, value text)`` pairs in the options dataclass's
    declaration order.

    The values are the derivation ``PipelineConfig.resolve_stages``
    performs: the raw dir and the reference from ``[pipeline]``, the
    segment input forced to ``diff``, and the difference tree behind the
    segment stage's ``diff_dir`` (the difference section's own ``out_dir``
    when set, else the derived tree name). A value that cannot be derived
    yet shows as ``—``. The owned keys come from the registry entry's
    dataclass and :data:`~colony_tracking.pipeline._OWNED` — a future
    stage without owned fields contributes nothing.

    The stage screen shows only the chain-wired half of these (the
    run-level mirrors are editable there — :func:`mirror_keys`); the
    config screen shows every entry read-only.
    """
    return tuple((label, value) for _key, label, value in owned_entries(draft, stage))


def mirror_keys(stage: str) -> tuple[str, ...]:
    """The run-level keys ``stage``'s screen mirrors, in display order:
    the stage's owned fields that name a run-level field —
    ``[difference].raw_dir``/``reference`` and ``[analysis].raw_dir``
    (spec addendum decision 2: editable, writing the run level) — then
    ``out_root``, editable on every stage screen because every derived
    value hangs off it. The chain-wired owned fields
    (``[segmentation].input``/``diff_dir``) are not mirrors: they stay
    read-only derived. A future stage without owned fields mirrors only
    ``out_root``."""
    owned = [
        spec.key
        for spec in specs_of(STAGES[stage].options)
        if spec.key in _OWNED.get(stage, frozenset())
    ]
    mirrors = [key for key in owned if key in _RUN_LEVEL_FIELDS]
    if "out_root" not in mirrors:
        mirrors.append("out_root")
    return tuple(mirrors)


def mirror_specs(stage: str) -> tuple[FieldSpec, ...]:
    """The mirror fields' form specs for ``stage``'s screen (see
    :func:`mirror_keys`): :func:`colony_tracking.tui.forms.
    run_level_mirror` per key."""
    return tuple(run_level_mirror(key) for key in mirror_keys(stage))


def editable_specs(stage: str) -> tuple[FieldSpec, ...]:
    """The stage's own editable option specs: every field of the registry
    entry's dataclass except the pipeline-owned ones (a stage section
    carries only those — the owned fields are derived, shown read-only)."""
    owned = _OWNED.get(stage, frozenset())
    return tuple(
        spec for spec in specs_of(STAGES[stage].options) if spec.key not in owned
    )


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


#: The mirror group's heading: what the fields under it are (they write
#: the ``[pipeline]`` section every stage shares).
MIRROR_TITLE = "run level (writes [pipeline]; shared by every stage)"


class StageScreen(FormScreen):
    """The screen for one stage: the run-level mirror group, its options
    as a form, the chain-wired owned fields read-only with their derived
    values, and the spec's four actions.

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

    CSS = """
    #stage-body {
        height: 1fr;
    }
    #stage-body OptionForm {
        height: auto;
    }
    #mirror-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    OptionForm.mirror-form {
        border-left: tall $accent 40%;
        padding-left: 1;
    }
    """

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

    @property
    def mirrors(self) -> OptionForm:
        """The screen's run-level mirror form (the owned fields the mirror
        split makes editable, writing ``[pipeline]``)."""
        return self._mirrors

    @property
    def forms(self) -> list[OptionForm]:
        """The screen's option forms: the stage's own, then the run-level
        mirrors (the entry refresh and the action gate cover both)."""
        return [self._form, self._mirrors]

    def compose_body(self) -> ComposeResult:
        entry = STAGES[self._stage]
        self._mirrors = OptionForm(
            RUN_LEVEL,
            mirror_specs(self._stage),
            self._resolve_draft(),
            id_prefix=f"{self._stage}-run",
            on_change=self._on_form_change,
        )
        self._mirrors.add_class("mirror-form")
        self._form = OptionForm(
            self._stage,
            editable_specs(self._stage),
            self._resolve_draft(),
            on_change=self._on_form_change,
        )
        # One scrollable body: the actions below stay reachable on a small
        # terminal whatever the two forms' combined height.
        with VerticalScroll(id="stage-body"):
            yield Static(f"{entry.title} — stage options", id="stage-title")
            yield Static(Text(""), id="owned-values")
            yield Static(Text(MIRROR_TITLE), id="mirror-title")
            yield self._mirrors
            yield self._form

    def compose_actions(self) -> ComposeResult:
        yield Button("Run this stage", id="run-stage")
        yield Button("Write TOML…", id="write-toml")
        yield Button("Run whole pipeline from draft", id="run-pipeline")
        yield Button("Back", id="back")

    def refresh_extras(self) -> None:
        """The chain-wired owned values and the greyed derived defaults —
        both change when the run level changes (the mirrors edit it right
        here; the config screen edits it too)."""
        mirrors = set(mirror_keys(self._stage))
        derived = [
            (label, value)
            for key, label, value in owned_entries(self.draft, self._stage)
            if key not in mirrors
        ]
        joined = "  ".join(f"{label} = {value}" for label, value in derived)
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
        shipped default dispatches to the run screens' entry points —
        :func:`~colony_tracking.tui.screens.run.open_stage_run` for a
        single stage, :func:`~colony_tracking.tui.screens.run.
        open_pipeline_run` for the whole chain — which write the TOML
        first (reproducibility, spec requirement 5) and push the run
        screen; the entry points resolve the draft themselves, so the
        config handed over here is the gate's confirmation. Tests can
        pass a ``start_run`` callable to the constructor to observe the
        hand-off without the dialog and the runner, leaving the form
        machinery unchanged.
        """
        if self._start_run is not None:
            self._start_run(kind, config)
            return
        if kind == "pipeline":
            open_pipeline_run(self.app, self.draft)
        else:
            open_stage_run(self.app, self.draft, kind)
