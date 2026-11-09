"""ConfigDraft: the TUI's config document per run type.

Each run type (:data:`RUN_TYPES` — the pipeline chain, one per stage, and
the live finetune steps) edits a plain, TOML-shaped document of its own
and writes and reads its own config TOML (:data:`TOML_NAMES`); nothing a
draft stores, resolves, or suggests comes from another run type's
configuration or outputs.

The pipeline draft keeps the sectioned ``pipeline.toml`` shape — the
``[pipeline]`` run level plus one optional section per stage, plain
values only (paths as strings, tuples as lists) so a TOML round-trip is
exact — resolves through ``PipelineConfig.from_dict``, and keeps the
within-chain wiring: the pipeline-owned fields
(:data:`colony_tracking.pipeline._OWNED`) are refused at ``set``
(:meth:`~colony_tracking.pipeline.PipelineConfig`'s module docstring for
what the chain derives and why).

A stage draft is flat and fully explicit: its document is one table — a
detached copy of its own section — where every field of the stage's
options dataclass (:mod:`colony_tracking._options`) is an ordinary key;
a standalone run's config derives nothing. The flat-document machinery
(:func:`~colony_tracking.pipeline.resolve_stage_document` /
:func:`~colony_tracking.pipeline.write_stage_config`) lives in
:mod:`colony_tracking.pipeline`, so the stage CLIs read the same files
without importing the TUI.

A finetune step draft (:data:`FINETUNE_RUN_TYPES`) is the same flat,
fully-explicit shape, but resolves, writes, and loads through
:mod:`colony_tracking.finetune.config` — the machinery the CLI's
``train <step> --config`` uses — so a prepare draft and a
``train prepare --config`` read one document format.

:meth:`ConfigDraft.resolve` is where a draft's values are fully
validated, raising :class:`~colony_tracking.pipeline.ConfigError` with
the machinery's exact wordings (logged through :data:`colony_tracking.log`,
shown verbatim by a screen's action gate). :meth:`ConfigDraft.load`
replaces the draft with a config TOML of its run type. A flat stage
document is validated structurally only — owned keys, missing requireds,
and mistyped values pass and surface verbatim at the next resolve — while
a sectioned pipeline document loaded into a stage draft resolves to that
stage's wired options.
:func:`resolve_loaded_paths` is the load dialogs' follow-up: it rebases a
loaded draft's relative path values onto the directory they were written
from, so a file saved from one working directory keeps working when the
TUI runs in another.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from .. import log
from .._options import Options, _to_plain, _union_members
from ..config import FlatConfig
from ..finetune.config import (
    STEP_OPTIONS as FINETUNE_STEP_OPTIONS,
    resolve_step_document,
    step_config,
    write_step_config,
)
from ..pipeline import (
    ConfigError,
    PipelineConfig,
    _KNOWN_SECTIONS,
    _OWNED,
    _OWNED_REASON,
    _RUN_LEVEL_FIELDS,
    _STAGE_OPTIONS,
    _STAGES,
    _config_error,
    _stage_config_error,
    resolve_stage_document,
    stage_config_document,
    write_config,
    write_stage_config,
)

#: The finetune steps the TUI edits — one flat document each, resolved
#: through :mod:`colony_tracking.finetune.config`.
FINETUNE_RUN_TYPES: tuple[str, ...] = ("prepare", "train", "evaluate")

#: The run types the TUI edits — one :class:`ConfigDraft` (and one config
#: TOML) per type.
RUN_TYPES: tuple[str, ...] = (
    "pipeline",
    "difference",
    "segmentation",
    "analysis",
    *FINETUNE_RUN_TYPES,
)

#: The stage sections of a pipeline document (the config's stage names,
#: also the standalone stage run types).
STAGES: tuple[str, ...] = _STAGES

#: User-visible titles per run type (screens and dialogs reuse them).
TITLES: dict[str, str] = {
    "pipeline": "Pipeline",
    "difference": "Difference",
    "segmentation": "Segment",
    "analysis": "Analyze",
    "prepare": "Prepare",
    "train": "Train",
    "evaluate": "Evaluate",
}

#: Each run type's own config file name.
TOML_NAMES: dict[str, str] = {
    "pipeline": "pipeline.toml",
    "difference": "difference.toml",
    "segmentation": "segmentation.toml",
    "analysis": "analysis.toml",
    "prepare": "prepare.toml",
    "train": "train.toml",
    "evaluate": "evaluate.toml",
}

#: The run-level section of a pipeline draft document (``[pipeline]``).
RUN_LEVEL = "pipeline"


def _copy_value(value: Any) -> Any:
    """A copy of one plain draft value (a list value must not alias the
    draft's own, or editing a document copy could corrupt the draft)."""
    return list(value) if isinstance(value, list) else value


def _is_path_hint(hint: Any) -> bool:
    """Whether a declared field type is ``Path`` or a list of ``Path``."""
    members = _union_members(hint)
    if members is not None and len(members) == 1:
        return _is_path_hint(members[0])
    origin, args = get_origin(hint), get_args(hint)
    if origin is list and args:
        return _is_path_hint(args[0])
    return hint is Path


def _options_of(run_type: str) -> type[Options]:
    """The options dataclass a non-pipeline run type's draft describes: a
    pipeline stage's or a live finetune step's."""
    if run_type in FINETUNE_RUN_TYPES:
        return FINETUNE_STEP_OPTIONS[run_type]
    return _STAGE_OPTIONS[run_type]


def _is_finetune(run_type: str) -> bool:
    """Whether ``run_type`` is a finetune step registered with the TUI."""
    return run_type in FINETUNE_RUN_TYPES


def _path_hints(run_type: str) -> dict[str, dict[str, Any]]:
    """The declared types of the draft's fields, by section: the run level
    and every stage section for the pipeline draft, the own section for a
    standalone run type's draft (which has no run level)."""
    if run_type != "pipeline":
        return {run_type: get_type_hints(_options_of(run_type))}
    hints = {"pipeline": get_type_hints(PipelineConfig)}
    for stage, options in _STAGE_OPTIONS.items():
        hints[stage] = get_type_hints(options)
    return hints


def _path_slots(
    draft: ConfigDraft,
) -> list[tuple[dict[str, Any], str, int | None, str]]:
    """Every relative path value in ``draft``, as ``(section, key, index,
    text)`` slots (``index`` is set for one entry of a path list)."""
    hints = _path_hints(draft.run_type)
    if draft.is_pipeline:
        sections = [(RUN_LEVEL, draft.run_level)]
        sections.extend((stage, draft.stages[stage]) for stage in STAGES)
    else:
        sections = [(draft.run_type, draft.stages[draft.run_type])]
    slots: list[tuple[dict[str, Any], str, int | None, str]] = []
    for name, section in sections:
        for key, value in section.items():
            hint = hints.get(name, {}).get(key)
            if hint is None or not _is_path_hint(hint):
                continue
            if isinstance(value, str):
                values: list[tuple[int | None, Any]] = [(None, value)]
            elif isinstance(value, list):
                values = list(enumerate(value))
            else:
                continue
            for index, text in values:
                if isinstance(text, str) and not Path(text).is_absolute():
                    slots.append((section, key, index, text))
    return slots


def _candidate_bases(config_path: Path) -> list[Path]:
    """The directories a loaded config's relative paths may have been
    written from, nearest first: the config's own directory and its
    ancestors, then the process's working directory."""
    bases: list[Path] = []
    path = Path(config_path)
    for base in (path.parent, *path.parent.parents, Path.cwd()):
        resolved = base.resolve()
        if resolved not in bases:
            bases.append(resolved)
    return bases


def _choose_base(
    slots: list[tuple[dict[str, Any], str, int | None, str]],
    bases: list[Path],
) -> Path | None:
    """The directory the draft's relative paths were written from.

    A relative ``raw_dir`` names the run's input tree, so the first base
    (nearest first) where it exists wins. Without one (or when it exists
    nowhere), the base resolving the most slots wins; when no base
    resolves any, the config is left untouched.
    """
    raw = next(
        (
            value
            for _section, key, index, value in slots
            if key == "raw_dir" and index is None
        ),
        None,
    )
    if raw is not None:
        for base in bases:
            if (base / raw).is_dir():
                return base

    def score(base: Path) -> int:
        return sum((base / value).exists() for _s, _k, _i, value in slots)

    best = max(bases, key=score)
    return best if score(best) > 0 else None


def resolve_loaded_paths(draft: ConfigDraft, config_path: Path | str) -> Path | None:
    """Resolve the relative paths of a just-loaded draft to absolute ones.

    A config written by the TUI keeps paths as they were typed — often
    relative to the directory the TUI was started in — while the file
    itself sits next to the run's outputs. Loading that file later from a
    different directory would otherwise point every source at the wrong
    place (the viewer then finds no wells). Each relative path value is
    rebased onto :func:`_choose_base`; absolute values, non-path values,
    and a config whose paths resolve nowhere are left as written. Returns
    the base used, or ``None`` when nothing was rebased.
    """
    slots = _path_slots(draft)
    if not slots:
        return None
    base = _choose_base(slots, _candidate_bases(Path(config_path)))
    if base is None:
        log.info(
            "%s config: relative paths resolve nowhere — left as written",
            draft.run_type,
        )
        return None
    for section, key, index, value in slots:
        resolved = str((base / value).resolve())
        if index is None:
            section[key] = resolved
        else:
            section[key][index] = resolved
    log.info("%s config: relative paths resolved against %s", draft.run_type, base)
    return base


def _unknown_run_type(run_type: str) -> ConfigError:
    """The config error for a run type no registry knows."""
    message = f"unknown run type {run_type!r} — expected one of {list(RUN_TYPES)}"
    log.error("tui draft: %s", message)
    return ConfigError(message)


def _check_pipeline_sections(doc: dict[str, Any]) -> None:
    """Structural validation of a pipeline document (a load's only
    check): known sections, known keys inside each one — ``PipelineConfig.from_dict``'s
    exact wordings. Owned keys, missing requireds, and mistyped values
    pass; the gate reports them verbatim at ``resolve``."""
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


class ConfigDraft:
    """One run type's in-progress config document (see the module
    docstring).

    The run type is fixed at construction. A key that is not set takes
    its dataclass default at ``resolve`` time, exactly like an omitted
    key in the TOML config file.
    """

    def __init__(self, run_type: str = "pipeline") -> None:
        if run_type not in RUN_TYPES:
            raise _unknown_run_type(run_type)
        #: This draft's run type: which of :data:`RUN_TYPES` it is, and
        #: which config TOML (:data:`TOML_NAMES`) it writes and reads.
        self.run_type: str = run_type
        #: The run-level (``[pipeline]``) section: plain values only. Only
        #: the pipeline draft carries values here.
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
        document stays TOML-shaped. On the pipeline draft an owned key
        (:data:`colony_tracking.pipeline._OWNED`) is refused with the
        pipeline's own reason — within one chain run those fields are
        derived. A stage draft stores every key of its dataclass. An
        unknown section raises :class:`ConfigError`.
        """
        section = self._section(stage)
        # _OWNED only covers the pipeline's registered stages.
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
        the key is unset."""
        return self._section(stage).get(key, default)

    def clear(self, stage: str, key: str) -> None:
        """Drop ``key`` from section ``stage`` — back to the stage
        default (an unset key is a no-op)."""
        self._section(stage).pop(key, None)

    def document(self) -> dict[str, Any]:
        """A TOML-shaped copy of the draft as it stands right now.

        The pipeline draft's document is sectioned: the ``[pipeline]``
        section plus the non-empty stage sections. A stage draft's
        document is flat: a detached copy of its own section's plain
        values. Editing the copy cannot corrupt the draft.
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
        """The resolved config for the draft as it stands right now: a
        :class:`~colony_tracking.pipeline.PipelineConfig` from
        ``PipelineConfig.from_dict`` for the chain, the stage's own
        :class:`~colony_tracking._options.Options` from
        :func:`resolve_stage_document` otherwise; invalid drafts raise
        :class:`ConfigError` with the machinery's own message.
        """
        if self.is_pipeline:
            return PipelineConfig.from_dict(self.document())
        if _is_finetune(self.run_type):
            return resolve_step_document(self.run_type, self.document())
        return resolve_stage_document(self.run_type, self.document())

    def write(self, path: Path | str) -> PipelineConfig | Options:
        """Resolve the draft and write it as its run type's TOML config
        file (a run started from the draft writes the same file first, so
        the run is reproducible). Returns the resolved config."""
        config = self.resolve()
        if self.is_pipeline:
            write_config(config, path)
        elif _is_finetune(self.run_type):
            write_step_config(self.run_type, config, path)
        else:
            write_stage_config(self.run_type, config, path)
        return config

    def load(self, path: Path | str) -> Path:
        """Read a config TOML of this run type into the draft, replacing
        the draft's content with the file's (a load resets, it never
        merges). Returns the path.

        A stage draft accepts its own flat document and a sectioned
        ``pipeline.toml``: the latter resolves through
        ``PipelineConfig.from_dict`` to the stage's wired options (run
        level and derived values included), so the draft becomes fully
        explicit. A flat document is validated structurally only —
        unknown keys are refused at load, while owned keys, missing
        requireds, and mistyped values surface verbatim at the next
        :meth:`resolve`.
        """
        path = Path(path)
        if self.is_pipeline:
            try:
                with open(path, "rb") as handle:
                    doc = tomllib.load(handle)
            except tomllib.TOMLDecodeError as exc:
                raise _config_error(
                    f"{path}: not valid TOML: {exc}"
                ) from exc
            _check_pipeline_sections(doc)
            self.run_level = dict(doc.get(RUN_LEVEL, {}))
            self.stages = {stage: dict(doc.get(stage, {})) for stage in STAGES}
        elif _is_finetune(self.run_type):
            flat = step_config(self.run_type)
            doc = flat.read_document(path)
            flat.validate_structure(doc)
            self.run_level = {}
            self.stages = {self.run_type: dict(doc)}
        else:
            flat = FlatConfig(self.run_type, _STAGE_OPTIONS[self.run_type])
            doc = stage_config_document(
                self.run_type, flat.read_document(path)
            )
            flat.validate_structure(doc)
            self.run_level = {}
            self.stages = {self.run_type: dict(doc)}
        log.info("%s config: loaded %s", self.run_type, path)
        return path
