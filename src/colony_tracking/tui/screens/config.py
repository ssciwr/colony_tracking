"""The pipeline-config screen: every option in one place, plus the
effective TOML.

The full editor (spec addendum decision 5): the run-level form
(``raw_dir``, ``out_root``, ``reference``, ``wells`` — the ``[pipeline]``
section, generated from :class:`~colony_tracking.pipeline.PipelineConfig`'s
own fields through :mod:`colony_tracking.tui.forms`), then one form per
registered stage with that stage's non-owned fields. The sections come
from the stage registry
(:data:`~colony_tracking.tui.screens.stage.STAGES`), so a future stage
slots in with its registry entry alone.

The pipeline-owned fields follow the mirror split (spec addendum decision
2): the run-level mirrors (``[difference].raw_dir``/``reference``,
``[analysis].raw_dir``) are the ``[pipeline]`` inputs themselves —
editable, writing the run level, shown once here rather than duplicated
per stage section — while the chain-wired fields (``[segmentation].input``
/ ``diff_dir``) stay read-only. Every stage section shows its owned
fields' derived values in a read-only line, mirrors marked as the
run-level key they mirror.

Every field of every form carries an individual *reset* button (the form
machinery's; see :class:`~colony_tracking.tui.forms.OptionForm`): it
clears exactly that field's draft key — back to the default — and is
enabled only while the key is set, so the set-vs-at-default state stays
visible.

The draft's effective TOML preview sits under the forms: the whole
document as it stands right now, rendered from ``ConfigDraft.document`` —
exactly the keys the TUI has set, with the defaults left to apply at
resolve time — and live, updated on every edit. *Write TOML…* opens the
shared save-config dialog, which writes the resolved config (the written
file carries the full effective document and ``colony-tracking pipeline
--config`` accepts it unchanged).

The stage sections are also edited on their own stage screens; this
screen shares the draft with every other screen, so a change here
re-gates the stage screens' actions on their next entry.
"""

from __future__ import annotations

from typing import ClassVar

import tomli_w
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from ..draft import RUN_LEVEL
from ..forms import FormScreen, OptionForm, run_level_specs
from .stage import STAGES, editable_specs, mirror_keys, owned_entries, placeholders


class ConfigScreen(FormScreen):
    """The full pipeline-config editor (see the module docstring): the
    run level, one form per stage section with every field's reset
    button, the derived owned lines, and the effective TOML preview. The
    write action gates on the draft exactly like the stage screens'
    actions do."""

    gated_actions: ClassVar[tuple[str, ...]] = ("write-toml",)

    BINDINGS = [Binding("w", "write_toml", "Write TOML…")]

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
        derived owned line (mirrors marked as the run level they mirror)
        and the stage forms' greyed derived defaults."""
        self.query_one("#toml-preview", Static).update(
            Text(tomli_w.dumps(self.draft.document()))
        )
        for stage in STAGES:
            mirrors = set(mirror_keys(stage))
            parts = [
                f"{label} ({'run-level' if key in mirrors else 'derived'}) = {value}"
                for key, label, value in owned_entries(self.draft, stage)
            ]
            joined = "  ".join(parts)
            self.query_one(f"#owned-{stage}", Static).update(
                Text("owned by the pipeline: " + (joined if joined else "none"))
            )
            form = self._stage_forms[stage]
            for key, text in placeholders(self.draft, stage).items():
                form.set_placeholder(key, text)

    def compose_actions(self) -> ComposeResult:
        yield Button("Write TOML…", id="write-toml")
        yield Button("Back", id="back")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "write-toml":
            self.action_write_toml()
        elif event.button.id == "back":
            self.action_back()
