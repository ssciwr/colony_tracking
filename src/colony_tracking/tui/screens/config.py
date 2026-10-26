"""The pipeline-config screen: the whole chain's config in one place,
plus the effective TOML.

The full editor of the pipeline run type's draft: the run-level form
(``raw_dir``, ``out_root``, ``reference``, ``wells`` — the
``[pipeline]`` section, generated from
:class:`~colony_tracking.pipeline.PipelineConfig`'s own fields through
:mod:`colony_tracking.tui.forms`), then one form per registered stage
with that stage's non-owned fields. The sections come from the stage
registry (:data:`~colony_tracking.tui.screens.stage.STAGES`).

This screen edits the *chain's* draft (the app's ``drafts["pipeline"]``
entry), and a pipeline run is the chain: the within-run wiring stays.
The pipeline-owned fields (:data:`colony_tracking.pipeline._OWNED`) are
refused on the pipeline draft at ``set`` — within one chain run they are
derived from the run level and the chain wiring, never stored in a stage
section — so their stage sections show them in a read-only line instead:
the values the chain derives for them
(:func:`owned_entries` — the same derivation
``PipelineConfig.resolve_stages`` performs), every entry marked
``(derived)``. The run level is *this screen's form*, not a line among
them. The free fields that default to a derived name grey that default
in their empty input while ``out_root`` is set (:func:`placeholders`,
laid over the forms' static grays through
:meth:`~colony_tracking.tui.forms.OptionForm.refresh_placeholders`) —
a blank or cleared ``out_root`` re-lays the static grays, no stale
derived name lingers; everywhere else the greys are the forms' own
static ones (the field's declared default, ``(default: unset)``, or its
``required:`` text). This wiring is the chain's
own — nothing here is derived from another run type's configuration or
outputs.

Every field of every form carries an individual *reset* button (the form
machinery's; see :class:`~colony_tracking.tui.forms.OptionForm`): it
clears exactly that field's draft key — back to the default — and is
enabled only while the key is set, so the set-vs-at-default state stays
visible.

The draft's effective TOML preview sits under the forms: the whole
document as it stands right now, rendered from ``ConfigDraft.document``
— exactly the keys the TUI has set, with the defaults left to apply at
resolve time — and live, updated on every edit. *Write TOML…* opens the
shared save-config dialog, which writes the resolved config (the written
file carries the full effective document and ``colony-tracking pipeline
--config`` accepts it unchanged); *Load TOML…* opens the shared
load-config dialog, which replaces the draft with a ``pipeline.toml``,
one of four independent per-run-type config files.

The draft is the screen's own run type's; nothing is shared with the
stage screens' drafts — each run type holds an independent config.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import tomli_w
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from ...pipeline import (
    _OWNED,
    _derived_difference_tree,
    _derived_segment_root,
    _reference_number,
)
from ..draft import RUN_LEVEL, ConfigDraft
from ..forms import FieldSpec, FormScreen, OptionForm, kebab, run_level_specs, specs_of
from ._save import LoadConfigDialog
from .stage import STAGES


def _shown(value: object) -> str:
    """A derived value for display: ``—`` while it cannot be derived (the
    run level is still incomplete)."""
    return "—" if value is None else str(value)


def owned_entries(draft: ConfigDraft, stage: str) -> tuple[tuple[str, str, str], ...]:
    """The owned fields of ``stage`` with the values the chain derives
    for them, as ``(key, kebab label, value text)`` triples in the
    options dataclass's declaration order — this screen shows every
    entry read-only; the stage screens need none of it.

    The values are the derivation ``PipelineConfig.resolve_stages``
    performs for one chain run: the raw dir and the reference from
    ``[pipeline]``, the segment input forced to ``diff``, and the
    difference tree behind the segment stage's ``diff_dir`` (the
    difference section's own ``out_dir`` when set, else the derived
    tree name). A value that cannot be derived yet shows as ``—``. The
    owned keys come from the registry entry's dataclass and
    :data:`~colony_tracking.pipeline._OWNED`.
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
    return tuple((key, kebab(key), shown.get((stage, key), "—")) for key in keys)


def editable_specs(stage: str) -> tuple[FieldSpec, ...]:
    """The stage's own editable option specs: every field of the registry
    entry's dataclass except the pipeline-owned ones (the chain refuses
    owned keys on the pipeline draft — they are derived within the run,
    shown read-only, never stored in a stage section)."""
    owned = _OWNED.get(stage, frozenset())
    return tuple(
        spec for spec in specs_of(STAGES[stage].options) if spec.key not in owned
    )


def placeholders(draft: ConfigDraft, stage: str) -> dict[str, str]:
    """The derived default each free field of ``stage`` falls back to when
    left blank — greyed in the empty input so the user sees what the
    chain run would pick; stages without such fields get an empty
    mapping, and an unset ``out_root`` gets one too — the overlay
    (:meth:`~colony_tracking.tui.forms.OptionForm.refresh_placeholders`)
    re-lays every field's static gray then. These greys are the chain's
    own within-run wiring — tree names built from the run level on the
    same draft, never from another run type's configuration."""
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


class ConfigScreen(FormScreen):
    """The full pipeline-config editor (see the module docstring): the
    run level, one form per stage section with every field's reset
    button, the within-chain owned lines, and the effective TOML
    preview. The write action gates on the draft exactly like the stage
    screens' actions do; the load action is never gated."""

    gated_actions: ClassVar[tuple[str, ...]] = ("write-toml",)

    BINDINGS = [
        Binding("w", "write_toml", "Write TOML…"),
        Binding("l", "load_toml", "Load TOML…"),
    ]

    CSS = """
    #config-title {
        text-style: bold;
        color: $accent;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    .owned-line {
        color: $text 60%;
    }
    #config-body {
        height: 1fr;
    }
    #config-body OptionForm {
        height: auto;
    }
    #toml-pane {
        height: 10;
        border: round $border;
    }
    #toml-preview {
        padding: 0 1;
    }
    """

    def compose_body(self) -> ComposeResult:
        self._form = OptionForm(
            RUN_LEVEL,
            run_level_specs(),
            self._resolve_draft(),
            on_change=self._on_form_change,
        )
        self._stage_forms: dict[str, OptionForm] = {}
        with VerticalScroll(id="config-body"):
            yield Static("Pipeline config — run level", id="config-title")
            yield self._form
            for stage in STAGES:
                entry = STAGES[stage]
                self._stage_forms[stage] = OptionForm(
                    stage,
                    editable_specs(stage),
                    self._resolve_draft(),
                    id_prefix=f"cfg-{stage}",
                    on_change=self._on_form_change,
                )
                yield Static(
                    f"[{stage}] — {entry.title}",
                    id=f"{stage}-section-title",
                    classes="section-title",
                )
                yield Static(Text(""), id=f"owned-{stage}", classes="owned-line")
                yield self._stage_forms[stage]
        with VerticalScroll(id="toml-pane"):
            yield Static(Text(""), id="toml-preview")

    @property
    def stage_forms(self) -> dict[str, OptionForm]:
        """The stage sections' forms, by section name (the run level's
        form is :attr:`form`)."""
        return self._stage_forms

    @property
    def forms(self) -> list[OptionForm]:
        """Every section's form: the run level, then one per registered
        stage (the entry refresh and the action gate cover them all)."""
        return [self._form, *self._stage_forms.values()]

    def refresh_extras(self) -> None:
        """The effective TOML preview — the whole draft document as it
        stands, rendered as TOML (markup-free — section names like
        ``[pipeline]`` must show as typed) — plus every stage section's
        within-chain owned line (every entry marked ``derived``; the run
        level is this screen's own form) and every stage form's grays
        re-laid over their static defaults: the derived tree names where
        the chain derives them, the forms' own grays everywhere else (a
        cleared ``out_root`` re-lays the statics, no stale derived name)."""
        self.query_one("#toml-preview", Static).update(
            Text(tomli_w.dumps(self.draft.document()))
        )
        for stage in STAGES:
            parts = [
                f"{label} (derived) = {value}"
                for _key, label, value in owned_entries(self.draft, stage)
            ]
            joined = "  ".join(parts)
            self.query_one(f"#owned-{stage}", Static).update(
                Text("wired within the chain: " + (joined if joined else "none"))
            )
            form = self._stage_forms[stage]
            form.refresh_placeholders(placeholders(self.draft, stage))

    def compose_actions(self) -> ComposeResult:
        yield Button("Write TOML…", id="write-toml")
        yield Button("Load TOML…", id="load-toml")
        yield Button("Back", id="back")

    def action_load_toml(self) -> None:
        """*Load TOML…*: open the shared load-config dialog (see
        :class:`~colony_tracking.tui.screens._save.LoadConfigDialog`);
        loading is never gated."""
        self.app.push_screen(LoadConfigDialog(self.draft))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "write-toml":
            self.action_write_toml()
        elif event.button.id == "load-toml":
            self.action_load_toml()
        elif event.button.id == "back":
            self.action_back()
