"""ConfigDraft: the TUI's one config document per run type.

The TUI never edits a partially-resolved config object; each run type
edits a plain, TOML-shaped document of its own. The four run types
(:data:`RUN_TYPES`) — the full pipeline chain, and each of the three
stages on its own — carry one :class:`ConfigDraft` and write and read one
independent config TOML (:data:`TOML_NAMES`): nothing a draft stores,
resolves, or suggests comes from any other run type's configuration or
outputs.

Independence boundary
---------------------
The pipeline draft is the chain. Its document keeps the sectioned shape a
``pipeline.toml`` always had — the ``[pipeline]`` run level plus one
optional section per stage, plain values only (paths as strings, tuples
as lists) so a TOML round-trip is exact — resolves through
``PipelineConfig.from_dict``, and keeps the within-chain wiring: the
pipeline-owned fields (:data:`colony_tracking.pipeline._OWNED`) are still
refused at ``set`` with the pipeline's own reason, because *within one
chain run* they are derived from the run level and the chain wiring, not
from another run. See the module docstring of
:mod:`colony_tracking.pipeline` for what the chain derives and why.

A stage run type's draft (``difference``, ``segmentation``, ``analysis``)
is flat and fully explicit: its document is one table — a detached copy
of its own section — where every field of the stage's options dataclass
(:mod:`colony_tracking._options`) is an ordinary key. ``input`` /
``diff_dir`` (segmentation), ``raw_dir`` / ``reference`` (difference) and
``raw_dir`` (analysis) are stored like any other key, nothing falls back
to a run level or to a sibling stage's settings, and there is no
cross-run pre-flight: a standalone run's config is self-contained.

Validation and loading
----------------------
:meth:`ConfigDraft.resolve` is where a draft's values are fully
validated — unknown keys, missing required keys, and mistyped values
raise :class:`~colony_tracking.pipeline.ConfigError` with the machinery's
exact wordings, logged through :data:`colony_tracking.log` (nothing ever
prints) and shown verbatim by a screen's action gate.
:meth:`ConfigDraft.load` is the inverse: reading a config TOML of its run
type *replaces* the draft (a load resets, it never merges) and validates
structure only — unknown sections/keys are refused so a typo cannot sit
in the draft unnoticed — while owned keys, missing requireds, and
mistyped values pass and surface verbatim the next time the gate
resolves.
"""

from __future__ import annotations

import tomllib
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_type_hints

import tomli_w

from .. import log
from .._options import (
    AnalysisOptions,
    DifferenceOptions,
    Options,
    SegmentationOptions,
    _from_plain,
    _to_plain,
)
from ..pipeline import (
    ConfigError,
    PipelineConfig,
    _KNOWN_SECTIONS,
    _OWNED,
    _OWNED_REASON,
    _RUN_LEVEL_FIELDS,
    _STAGES,
    _compatible,
    _config_error,
    _human_hint,
    write_config,
)

#: The run types the TUI edits — one :class:`ConfigDraft` (and one config
#: TOML) per type: the full chain, and every stage run on its own.
RUN_TYPES: tuple[str, ...] = ("pipeline", "difference", "segmentation", "analysis")

#: The stage sections of a pipeline document (the config's stage names,
#: also the standalone stage run types).
STAGES: tuple[str, ...] = _STAGES

#: Each stage run type's options dataclass — the dataclass a stage draft's
#: flat document is fully explicit about.
_STAGE_OPTIONS: dict[str, type[Options]] = {
    "difference": DifferenceOptions,
    "segmentation": SegmentationOptions,
    "analysis": AnalysisOptions,
}

#: User-visible titles per run type (screens and dialogs reuse them).
TITLES: dict[str, str] = {
    "pipeline": "Pipeline",
    "difference": "Difference",
    "segmentation": "Segment",
    "analysis": "Analyze",
}

#: Each run type's own config file name — the four independent config TOMLs.
TOML_NAMES: dict[str, str] = {
    "pipeline": "pipeline.toml",
    "difference": "difference.toml",
    "segmentation": "segmentation.toml",
    "analysis": "analysis.toml",
}

#: The run-level section of a pipeline draft document (``[pipeline]``).
RUN_LEVEL = "pipeline"


def _copy_value(value: Any) -> Any:
    """A copy of one plain draft value (a list value must not alias the
    draft's own, or editing a document copy could corrupt the draft)."""
    return list(value) if isinstance(value, list) else value


def _unknown_run_type(run_type: str) -> ConfigError:
    """The config error for a run type no registry knows (logged; see
    :data:`RUN_TYPES`)."""
    message = f"unknown run type {run_type!r} — expected one of {list(RUN_TYPES)}"
    log.error("tui draft: %s", message)
    return ConfigError(message)


def _stage_config_error(run_type: str, message: str) -> ConfigError:
    """Log one stage config error with the run type's own prefix, then
    return it for raising (the per-run-type counterpart of
    :func:`colony_tracking.pipeline._config_error`)."""
    log.error("%s config: %s", run_type, message)
    return ConfigError(message)


def _check_pipeline_sections(doc: dict[str, Any]) -> None:
    """Structural validation of a pipeline document (a load's only check):
    known sections, known keys inside each one — with the exact wordings
    ``PipelineConfig.from_dict`` raises for them. Owned keys, missing
    required keys, and mistyped values pass: they sit in the draft and
    the gate reports them verbatim at ``resolve``."""
    unknown_sections = sorted(set(doc) - set(_KNOWN_SECTIONS))
    if unknown_sections:
        raise _config_error(
            f"unknown section(s) {unknown_sections} — expected one section "
            f"per name: {list(_KNOWN_SECTIONS)}"
        )
    section = doc.get(RUN_LEVEL, {})
    if not isinstance(section, dict):
        raise _config_error(f"[pipeline] must be a table, got {section!r}")
    unknown_keys = sorted(set(section) - set(_RUN_LEVEL_FIELDS))
    if unknown_keys:
        raise _config_error(
            f"unknown key(s) {unknown_keys} in [pipeline] — expected one "
            f"key per name: {list(_RUN_LEVEL_FIELDS)}"
        )
    for stage in STAGES:
        section = doc.get(stage)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise _config_error(f"[{stage}] must be a table, got {section!r}")
        stage_fields = {f.name for f in fields(_STAGE_OPTIONS[stage])}
        unknown_keys = sorted(set(section) - stage_fields)
        if unknown_keys:
            raise _config_error(
                f"unknown key(s) {unknown_keys} in [{stage}] — expected "
                f"one key per name: {sorted(stage_fields)}"
            )


def resolve_stage_document(run_type: str, doc: dict[str, Any]) -> Options:
    """Validate and build one stage's options from its flat document.

    The checkpoints mirror ``PipelineConfig.from_dict``'s, restated for a
    stage's own table — in this order, each raising (logged with the
    run-type prefix) :class:`ConfigError`: the document must be a mapping;
    its keys must be exactly the dataclass's field names; every field
    without a default must be present (in the dataclass's declaration
    order); every present value must be compatible with its declared type.
    Values coerce through :func:`~colony_tracking._options._from_plain`
    exactly as a read-back TOML value would; omitted keys take their
    dataclass defaults. Every field is ordinary here — there is no
    owned-key refusal and nothing is derived.
    """
    options = _STAGE_OPTIONS.get(run_type)
    if options is None:
        raise _unknown_run_type(run_type)
    hints = get_type_hints(options)
    declared = fields(options)
    names = [f.name for f in declared]
    if not isinstance(doc, dict):
        raise _stage_config_error(
            run_type, f"{doc!r} is not a table of {run_type} options"
        )
    unknown = sorted(set(doc) - set(names))
    if unknown:
        raise _stage_config_error(
            run_type,
            f"unknown key(s) {unknown} — expected one key per name: {sorted(names)}",
        )
    missing = [
        f.name
        for f in declared
        if f.name not in doc and f.default is MISSING and f.default_factory is MISSING
    ]
    if missing:
        raise _stage_config_error(run_type, f"missing required key(s): {missing}")
    for key, value in doc.items():
        hint = hints[key]
        if not _compatible(value, hint):
            raise _stage_config_error(
                run_type,
                f"{run_type}: key {key!r} must be {_human_hint(hint)} "
                f"(got {value!r})",
            )
    kwargs = {key: _from_plain(value, hints[key]) for key, value in doc.items()}
    return options(**kwargs)


def write_stage_config(run_type: str, options: Options, path: Path | str) -> Path:
    """Write ``options`` as ``run_type``'s flat TOML config file.

    The effective document — the options'
    :meth:`~colony_tracking._options.Options.to_dict` minus the ``None``
    values — is written whole, defaults included, so the file is
    self-contained and round-trips through the dataclass's ``from_dict``
    unchanged. Parent directories are created. Returns the path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {key: value for key, value in options.to_dict().items() if value is not None}
    with open(path, "wb") as handle:
        tomli_w.dump(doc, handle)
    log.info("%s config: written to %s", run_type, path)
    return path


class ConfigDraft:
    """One run type's in-progress config document (see the module docstring).

    The run type is fixed at construction: the draft holds exactly one
    run's document, and the app keys its four drafts by
    :data:`RUN_TYPES`. A draft is its run type's config state: the screens
    edit it, and every action (write the TOML, run) goes through
    :meth:`resolve` / :meth:`write`. A key that is not set takes its
    dataclass default at ``resolve`` time, exactly like an omitted key in
    the TOML config file.
    """

    def __init__(self, run_type: str = "pipeline") -> None:
        if run_type not in RUN_TYPES:
            raise _unknown_run_type(run_type)
        #: This draft's run type: which of :data:`RUN_TYPES` it is — and
        #: which config TOML (:data:`TOML_NAMES`) it writes and reads.
        self.run_type: str = run_type
        #: The run-level (``[pipeline]``) section: plain values only. Only
        #: the pipeline draft carries values here — a stage run derives
        #: nothing from a run level.
        self.run_level: dict[str, Any] = {}
        #: The stage sections held: stage name -> plain values (may be
        #: empty). The pipeline draft carries one per stage (an empty one
        #: stays out of the document); a stage draft carries only its own.
        self.stages: dict[str, dict[str, Any]] = (
            {stage: {} for stage in STAGES}
            if self.is_pipeline
            else {self.run_type: {}}
        )

    @property
    def is_pipeline(self) -> bool:
        """Whether this is the pipeline chain's draft (its document is
        sectioned; a stage draft's is flat and fully explicit)."""
        return self.run_type == "pipeline"

    def _section(self, stage: str) -> dict[str, Any]:
        """The named section: the run level for the pipeline draft's
        ``pipeline``, else the named stage's — for a stage draft, its own
        single section only. Anything else raises :class:`ConfigError`."""
        if stage == RUN_LEVEL and self.is_pipeline:
            return self.run_level
        if stage in self.stages:
            return self.stages[stage]
        if self.is_pipeline:
            raise _config_error(
                f"unknown draft section {stage!r} — expected [{RUN_LEVEL}] or "
                f"one of the stage sections {list(STAGES)}"
            )
        raise _stage_config_error(
            self.run_type,
            f"unknown draft section {stage!r} — the {self.run_type} draft "
            f"edits its [{self.run_type}] section only (each run type holds "
            f"an independent config)",
        )

    def set(self, stage: str, key: str, value: Any) -> None:
        """Set ``key`` in section ``stage`` (``pipeline`` for a run-level
        setting on the pipeline draft) to the plain value ``value``.

        ``Path`` values are stored as strings and tuples as lists, so the
        document stays TOML-shaped and round-trips exactly. On the
        pipeline draft an owned key
        (:data:`colony_tracking.pipeline._OWNED`) is refused with the
        pipeline's own reason — within one chain run those fields are
        derived, never stored. A stage draft stores every key of its
        dataclass — a standalone run's config is fully explicit,
        ``input``/``diff_dir``/``raw_dir``/``reference`` included. An
        unknown section raises :class:`ConfigError`.
        """
        section = self._section(stage)
        # A section added after construction (a future stage's) owns
        # nothing yet; _OWNED only knows the pipeline's own stages.
        if (
            self.is_pipeline
            and stage != RUN_LEVEL
            and key in _OWNED.get(stage, frozenset())
        ):
            raise _config_error(
                f"[{stage}] key {key!r} is owned by the pipeline — "
                f"{_OWNED_REASON[stage, key]}"
            )
        section[key] = _to_plain(value)

    def get(self, stage: str, key: str, default: Any = None) -> Any:
        """The value of ``key`` in section ``stage``, or ``default`` when
        the key is unset (the stage default then applies at ``resolve``)."""
        return self._section(stage).get(key, default)

    def clear(self, stage: str, key: str) -> None:
        """Drop ``key`` from section ``stage`` — back to the stage default
        (dropping an unset key is a no-op)."""
        self._section(stage).pop(key, None)

    def document(self) -> dict[str, Any]:
        """A TOML-shaped copy of the draft as it stands right now.

        The pipeline draft's document is sectioned: the ``[pipeline]``
        section plus the non-empty stage sections — the shape a
        ``pipeline.toml`` has. A stage draft's document is flat: a
        detached copy of its own section's plain values, every key an
        ordinary field of the stage's dataclass. The copy is detached
        from the draft (editing it cannot corrupt the draft), which makes
        it safe to hand to config resolution, TOML preview, and tests
        alike.
        """
        if not self.is_pipeline:
            section = self.stages[self.run_type]
            return {key: _copy_value(value) for key, value in section.items()}
        doc: dict[str, dict[str, Any]] = {
            RUN_LEVEL: {
                key: _copy_value(value) for key, value in self.run_level.items()
            }
        }
        for stage, section in self.stages.items():
            if section:
                doc[stage] = {key: _copy_value(value) for key, value in section.items()}
        return doc

    def resolve(self) -> PipelineConfig | Options:
        """The resolved config for the draft as it stands right now.

        The pipeline draft goes through ``PipelineConfig.from_dict``:
        owned fields are derived from the run level and the chain wiring,
        defaults fill every unset key, and every invalid draft raises
        :class:`ConfigError` with the pipeline's own message. A stage
        draft goes through :func:`resolve_stage_document` — the stage's
        own options, fully explicit, that machinery's missing/mistyped
        wordings included (a screen's gate reports them verbatim).
        """
        if self.is_pipeline:
            return PipelineConfig.from_dict(self.document())
        return resolve_stage_document(self.run_type, self.document())

    def write(self, path: Path | str) -> PipelineConfig | Options:
        """Resolve the draft and write it as its run type's TOML config
        file (a run started from the draft writes the same file first, so
        the run is reproducible). Returns the resolved config — a
        :class:`~colony_tracking.pipeline.PipelineConfig` for the chain,
        the stage's own :class:`~colony_tracking._options.Options`
        otherwise."""
        config = self.resolve()
        if self.is_pipeline:
            write_config(config, path)
        else:
            write_stage_config(self.run_type, config, path)
        return config

    def load(self, path: Path | str) -> Path:
        """Read a config TOML of this run type into the draft, replacing
        the draft's content with the file's (a load resets, it never
        merges). Structural validation only: unknown sections/keys are
        refused with the resolvers' wordings, so a typo cannot sit in the
        draft unnoticed — while owned keys, missing requireds, and
        mistyped values pass and surface verbatim the next time
        :meth:`resolve` runs. Returns the path.
        """
        path = Path(path)
        try:
            with open(path, "rb") as handle:
                doc = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            message = f"{path}: not valid TOML: {exc}"
            if self.is_pipeline:
                raise _config_error(message) from exc
            raise _stage_config_error(self.run_type, message) from exc
        if self.is_pipeline:
            _check_pipeline_sections(doc)
            self.run_level = dict(doc.get(RUN_LEVEL, {}))
            self.stages = {stage: dict(doc.get(stage, {})) for stage in STAGES}
        else:
            names = {f.name for f in fields(_STAGE_OPTIONS[self.run_type])}
            unknown = sorted(set(doc) - names)
            if unknown:
                raise _stage_config_error(
                    self.run_type,
                    f"unknown key(s) {unknown} — expected one key per name: "
                    f"{sorted(names)}",
                )
            self.run_level = {}
            self.stages = {self.run_type: dict(doc)}
        log.info("%s config: loaded %s", self.run_type, path)
        return path
