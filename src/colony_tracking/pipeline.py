"""Pipeline: the TOML config and the orchestration of the standard chain
(difference → segment on the difference tree → analyze).

The standard chain
------------------
:func:`run_pipeline` runs the three stages in order, wiring each stage to the
previous stage's output:

1. **difference** — the raw plate (``[pipeline].raw_dir``) into the
   difference tree;
2. **segment** — the difference tree just produced (``input = "diff"``,
   ``diff_dir`` = the difference tree);
3. **analyze** — the segment root (``seg_dirs`` = [the segment root]) and the
   run-level raw dir.

Each stage's own resume logic applies (per-well done markers), so an
interrupted run re-enters at the first incomplete stage's incomplete wells;
the pipeline keeps no extra bookkeeping and always calls all three stages. A
stage failure — an exception, or a failed stage summary (``failed`` wells,
or the analyze stage's ``failures``/``failed_wells``) — aborts the chain:
later stages are not called, the error is logged, and
:class:`PipelineSummary` reports ``status = "failed"`` with the failing
stage's name. Raw-input segmentation stays available through the ``segment``
subcommand directly.

Derived names
-------------
All intermediate tree names derive from ``[pipeline].out_root`` and the
reference, and every one is overridable in the stage sections:

- difference tree: ``<out_root>/difference_LO<nnn>`` for a numeric reference
  (zero-padded 3 digits), ``<out_root>/difference`` for ``reference =
  "first"`` — overridden by ``[difference].out_dir``;
- segment root: ``<out_root>/seg_growth_diff_LO<nnn>`` /
  ``<out_root>/seg_growth_diff`` — overridden by ``[segmentation].out_dir``;
- the segment stage's ``exclude_before`` defaults to the reference frame
  number when the reference is numeric (``"first"`` leaves it unset) —
  overridden by ``[segmentation].exclude_before``;
- analysis out root ``<out_root>/analysis`` — overridden by
  ``[analysis].out_root``; ``seg_dirs`` defaults to ``[the segment root]`` —
  overridden by ``[analysis].seg_dirs``. ``raw_seg_dir`` is never invented:
  set it in ``[analysis]`` to probe a raw-input segmentation root.

Config contract
---------------
A config file is a TOML document with a ``[pipeline]`` section (run-level:
``raw_dir``, ``out_root``, ``reference``, ``wells`` — the first two required)
plus one optional section per stage holding any of that stage's options;
omitted keys take their defaults. :func:`read_config` parses one,
:func:`write_config` writes one (for the future TUI), and
``PipelineConfig.from_dict`` / ``to_dict`` round-trip the plain document.

Some fields are owned by the pipeline and cannot be set in a stage section:
``[difference].raw_dir`` / ``reference`` and ``[analysis].raw_dir`` come from
``[pipeline]``; ``[segmentation].input`` / ``diff_dir`` are wired to the
difference tree the chain just produced. Unknown keys in a known section,
owned keys, unknown sections, missing required keys, and mistyped values are
usage errors: logged with a clear message and raised as :class:`ConfigError`
(the CLI maps it to exit code 2).

Run-level settings are shared defaults: a stage section's own ``wells``
overrides ``[pipeline].wells`` for that stage. A stage dataclass set directly
on the config (programmatic/TUI use) resolves like the equivalent section:
owned fields are forced, unset ``wells`` / ``exclude_before`` fall back to
the run-level values, and everything else is kept as set.
"""

from __future__ import annotations

import dataclasses
import tomllib
import traceback
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

import tomli_w

from . import log
from ._options import (
    AnalysisOptions,
    DifferenceOptions,
    Options,
    Reference,
    SegmentationOptions,
    _from_plain,
    _to_plain,
    _union_members,
)
from .analysis import AnalysisSummary, run_analysis
from .difference import DifferenceSummary, run_difference
from .segmentation import SegmentationSummary, run_segmentation

#: The config's top-level sections: the run-level one plus one per stage.
_RUN_LEVEL_FIELDS = ("raw_dir", "out_root", "reference", "wells")
_STAGES = ("difference", "segmentation", "analysis")
_KNOWN_SECTIONS = ("pipeline", *_STAGES)

#: Fields the pipeline owns — taken from the run-level settings and the chain
#: wiring, never from a stage section (a config that sets them is a usage
#: error; the written config omits them).
_OWNED: dict[str, frozenset[str]] = {
    "difference": frozenset({"raw_dir", "reference"}),
    "segmentation": frozenset({"input", "diff_dir"}),
    "analysis": frozenset({"raw_dir"}),
}

#: Why each owned field belongs to the pipeline (for the error message).
_OWNED_REASON: dict[tuple[str, str], str] = {
    ("difference", "raw_dir"): "it comes from [pipeline].raw_dir",
    ("difference", "reference"): "it comes from [pipeline].reference",
    ("segmentation", "input"): 'the pipeline always runs input = "diff"; '
    "use the segment subcommand for raw frames",
    ("segmentation", "diff_dir"): "the chain wires it to the difference tree "
    "the pipeline just produced",
    ("analysis", "raw_dir"): "it comes from [pipeline].raw_dir",
}


class ConfigError(ValueError):
    """An invalid pipeline config: unknown key/section, owned key, missing
    required key, mistyped value, or malformed TOML.

    Logged with a clear message and raised; the CLI maps it to exit code 2.
    """


def _reference_number(reference: Reference) -> int | None:
    """The numeric frame number of a reference (``"first"`` → None)."""
    if isinstance(reference, bool) or not isinstance(reference, int):
        return None
    return reference


def _derived_difference_tree(out_root: Path, reference: Reference) -> Path:
    """The difference tree name for an out root and reference (LO-prefixed,
    zero-padded 3 digits, for a numeric reference)."""
    return Path(out_root) / (
        "difference"
        if _reference_number(reference) is None
        else f"difference_LO{reference:03d}"
    )


def _derived_segment_root(out_root: Path, reference: Reference) -> Path:
    """The segment root name for an out root and reference."""
    return Path(out_root) / (
        "seg_growth_diff"
        if _reference_number(reference) is None
        else f"seg_growth_diff_LO{reference:03d}"
    )


# --- config validation ----------------------------------------------------------


#: Scalar hint → (plain-value predicate, phrase, plural phrase): one table
#: shared by ``_human_hint`` and ``_compatible`` so the wording cannot drift.
_SCALARS: dict[type, tuple[Callable[[Any], bool], str, str]] = {
    Path: (lambda v: isinstance(v, str), "a path string", "path strings"),
    bool: (lambda v: isinstance(v, bool), "true or false", "booleans"),
    int: (lambda v: isinstance(v, int) and not isinstance(v, bool),
          "an integer", "integers"),
    float: (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "a number", "numbers"),
    str: (lambda v: isinstance(v, str), "a string", "strings"),
}
#: phrase → plural, derived from the same table.
_PLURAL = {phrase: plural for _, (_, phrase, plural) in _SCALARS.items()}


def _human_hint(hint: Any) -> str:
    """A human-readable description of a type hint (for error messages)."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        return " or ".join(repr(a) for a in args)
    if origin in (list, tuple):
        item = _human_hint(args[0])
        return f"a list of {_PLURAL.get(item, item)}"
    if (members := _union_members(hint)) is not None:  # a Union
        return " or ".join(_human_hint(a) for a in members)
    if args:  # another parameterized hint: list its members
        return " or ".join(_human_hint(a) for a in args if a is not type(None))
    if hint in _SCALARS:
        return _SCALARS[hint][1]
    return getattr(hint, "__name__", str(hint))


def _compatible(value: Any, hint: Any) -> bool:
    """Whether a plain (TOML) value can be read as the declared field type."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        return value in args
    if origin in (list, tuple):
        return isinstance(value, list) and all(
            _compatible(v, args[0]) for v in value)
    if (members := _union_members(hint)) is not None:  # a Union
        return (value is None and type(None) in args) or any(
            _compatible(value, a) for a in members)
    scalar = _SCALARS.get(hint)
    return scalar[0](value) if scalar else True  # unhinted/unknown: accept


def _check_value(where: str, name: str, value: Any, hint: Any) -> None:
    """Log and raise a usage error when a config value has the wrong type."""
    if _compatible(value, hint):
        return
    message = f"{where}: key {name!r} must be {_human_hint(hint)} (got {value!r})"
    log.error("pipeline config: %s", message)
    raise ConfigError(message)


def _config_error(message: str) -> ConfigError:
    """Log a config error clearly, then return it for raising."""
    log.error("pipeline config: %s", message)
    return ConfigError(message)


@dataclass(kw_only=True)
class PipelineConfig(Options):
    """The run-level settings plus the three stage option dataclasses.

    Run-level fields (the ``[pipeline]`` config section): ``raw_dir``,
    ``out_root`` (both required), ``reference`` (``"first"`` or a 1-based
    frame number) and ``wells``. The three stage fields hold each stage's
    options; ``None`` means "derive from the run-level settings and the
    derived names" (see the module docstring) and is resolved immediately in
    ``__post_init__``, so a config always carries fully resolved stages and
    round-trips through TOML unchanged.

    Run-level settings are shared defaults: a stage's own ``wells`` overrides
    ``[pipeline].wells`` for that stage. The pipeline-owned fields (see
    ``_OWNED``) are forced onto provided stage instances. Every field carries
    per-field help text for the CLI and the future TUI.
    """

    raw_dir: Path = field(
        metadata={"help": "Directory of raw plate frames (flat or per-well layout)."}
    )
    out_root: Path = field(
        metadata={"help": "Root directory every tree of the run is written into."}
    )
    reference: Reference = field(
        default="first",
        metadata={"help": 'Reference frame: "first" or a 1-based frame number; '
        "drives the derived tree names."},
    )
    wells: list[str] | None = field(
        default=None,
        metadata={
            "help": "Run only these wells (repeatable, comma-separated). "
            "Default: all wells."
        },
    )
    difference: DifferenceOptions | None = field(
        default=None,
        metadata={"help": "Difference stage options; None derives them."},
    )
    segmentation: SegmentationOptions | None = field(
        default=None,
        metadata={"help": "Segment stage options; None derives them."},
    )
    analysis: AnalysisOptions | None = field(
        default=None,
        metadata={"help": "Analyze stage options; None derives them."},
    )

    def __post_init__(self) -> None:
        # Resolve unset stages right away so a config always carries fully
        # resolved ones (and round-trips through TOML unchanged).
        self.raw_dir = Path(self.raw_dir)
        self.out_root = Path(self.out_root)
        diff, seg, ana = self.resolve_stages()
        self.difference, self.segmentation, self.analysis = diff, seg, ana

    def resolve_stages(
        self,
    ) -> tuple[DifferenceOptions, SegmentationOptions, AnalysisOptions]:
        """The three stage options for this config.

        Derives every unset stage from the run-level settings and the derived
        names, forces the pipeline-owned fields, and falls unset
        ``wells`` / ``exclude_before`` back to the run-level values / the
        reference frame number. Idempotent: a resolved config resolves to
        itself.
        """
        diff = self.difference
        if diff is None:
            diff = DifferenceOptions(
                raw_dir=self.raw_dir,
                out_dir=_derived_difference_tree(self.out_root, self.reference),
                reference=self.reference,
                wells=self.wells,
            )
        else:
            diff = dataclasses.replace(
                diff,
                raw_dir=self.raw_dir,
                reference=self.reference,
                wells=diff.wells if diff.wells is not None else self.wells,
            )
        exclude_before = _reference_number(diff.reference)
        seg = self.segmentation
        if seg is None:
            seg = SegmentationOptions(
                out_dir=_derived_segment_root(self.out_root, self.reference)
            )
        seg = dataclasses.replace(
            seg,
            input="diff",
            diff_dir=diff.out_dir,
            wells=seg.wells if seg.wells is not None else self.wells,
            exclude_before=(
                seg.exclude_before
                if seg.exclude_before is not None
                else exclude_before
            ),
        )
        ana = self.analysis
        if ana is None:
            ana = AnalysisOptions(
                raw_dir=self.raw_dir,
                out_root=Path(self.out_root) / "analysis",
                seg_dirs=[seg.out_dir],
            )
        ana = dataclasses.replace(
            ana,
            raw_dir=self.raw_dir,
            wells=ana.wells if ana.wells is not None else self.wells,
        )
        return diff, seg, ana

    def to_dict(self) -> dict[str, Any]:
        """The full config as a TOML-ready document: a ``pipeline`` section
        plus one section per stage (paths as strings, unset values omitted,
        pipeline-owned keys omitted from the stage sections)."""
        doc: dict[str, Any] = {
            "pipeline": {
                name: _to_plain(getattr(self, name))
                for name in _RUN_LEVEL_FIELDS
                if getattr(self, name) is not None
            }
        }
        for stage in _STAGES:
            opts = getattr(self, stage)
            doc[stage] = {
                key: value
                for key, value in opts.to_dict().items()
                if value is not None and key not in _OWNED[stage]
            }
        return doc

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> PipelineConfig:
        """Build a config from a plain, sectioned document (the TOML shape).

        Known sections only: ``pipeline`` plus one per stage. Omitted keys
        take their defaults (stage sections over the derived names, which in
        turn come from the run-level settings). Unknown keys or sections,
        owned keys, missing required keys, and mistyped values are logged and
        raised as :class:`ConfigError`.
        """
        if not isinstance(doc, dict):
            raise _config_error(f"expected a table of sections, got {doc!r}")
        unknown = sorted(set(doc) - set(_KNOWN_SECTIONS))
        if unknown:
            raise _config_error(
                f"unknown section(s) {unknown} — expected one section per "
                f"name: {list(_KNOWN_SECTIONS)}"
            )
        hints = get_type_hints(cls)
        section = doc.get("pipeline", {})
        if not isinstance(section, dict):
            raise _config_error(f"[pipeline] must be a table, got {section!r}")
        unknown_keys = sorted(set(section) - set(_RUN_LEVEL_FIELDS))
        if unknown_keys:
            raise _config_error(
                f"unknown key(s) {unknown_keys} in [pipeline] — expected one "
                f"key per name: {list(_RUN_LEVEL_FIELDS)}"
            )
        missing = [name for name in ("raw_dir", "out_root") if name not in section]
        if missing:
            raise _config_error(f"missing required [pipeline] key(s): {missing}")
        run_kwargs: dict[str, Any] = {}
        for name in _RUN_LEVEL_FIELDS:
            if name in section:
                value = section[name]
                _check_value("[pipeline]", name, value, hints[name])
                run_kwargs[name] = _from_plain(value, hints[name])
        number = _reference_number(run_kwargs.get("reference", "first"))
        if number is not None and number < 1:
            raise _config_error(
                f"[pipeline]: key 'reference' must be \"first\" or a 1-based "
                f"frame number (got {run_kwargs['reference']!r})"
            )
        config = cls(**run_kwargs)
        # Apply the stage sections over the resolved stages, in chain order,
        # then re-resolve so the cross-stage wiring tracks the overrides.
        for stage in _STAGES:
            stage_section = doc.get(stage)
            if stage_section is None:
                continue
            if not isinstance(stage_section, dict):
                raise _config_error(
                    f"[{stage}] must be a table, got {stage_section!r}"
                )
            opts = getattr(config, stage)
            stage_fields = {f.name for f in fields(type(opts))}
            unknown_keys = sorted(set(stage_section) - stage_fields)
            if unknown_keys:
                raise _config_error(
                    f"unknown key(s) {unknown_keys} in [{stage}] — expected "
                    f"one key per name: {sorted(stage_fields)}"
                )
            owned = sorted(set(stage_section) & _OWNED[stage])
            if owned:
                reasons = "; ".join(
                    f"{key!r}: {_OWNED_REASON[stage, key]}" for key in owned
                )
                raise _config_error(f"[{stage}] key(s) are owned by the "
                                    f"pipeline — {reasons}")
            stage_hints = get_type_hints(type(opts))
            overrides = {}
            for key, value in stage_section.items():
                _check_value(f"[{stage}]", key, value, stage_hints[key])
                overrides[key] = _from_plain(value, stage_hints[key])
            setattr(config, stage, dataclasses.replace(opts, **overrides))
            if stage == "segmentation" and not _has_key(doc, "analysis", "seg_dirs"):
                config.analysis = dataclasses.replace(
                    config.analysis, seg_dirs=[config.segmentation.out_dir]
                )
        diff, seg, ana = config.resolve_stages()
        config.difference, config.segmentation, config.analysis = diff, seg, ana
        return config


def _has_key(doc: dict[str, Any], section: str, key: str) -> bool:
    """Whether a config document's section carries a key."""
    return isinstance(doc.get(section), dict) and key in doc[section]


# --- TOML config file ---------------------------------------------------------


def read_config(path: Path | str) -> PipelineConfig:
    """Read a TOML config file (see the module docstring for the contract)."""
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            doc = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise _config_error(f"{path}: not valid TOML: {exc}") from exc
    log.info("pipeline config: reading %s", path)
    return PipelineConfig.from_dict(doc)


def write_config(config: PipelineConfig, path: Path | str) -> Path:
    """Write ``config`` as a TOML config file (the future TUI's save path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        tomli_w.dump(config.to_dict(), handle)
    log.info("pipeline config: written to %s", path)
    return path


# --- orchestration --------------------------------------------------------------


#: The stage a failed run stopped at (the config section names).
Stage = Literal["difference", "segmentation", "analysis"]


@dataclass(frozen=True)
class PipelineSummary:
    """Outcome of one pipeline run (for the caller's status line).

    ``status`` is ``ok`` only when all three stages succeeded;
    ``failed_stage`` names the stage the run stopped at (None when ok). The
    three embedded stage summaries are set as the stages complete — a stage
    that was never reached (or raised) is None.
    """

    status: Literal["ok", "failed"]
    failed_stage: Stage | None = None
    difference: DifferenceSummary | None = None
    segmentation: SegmentationSummary | None = None
    analysis: AnalysisSummary | None = None


def _failed_wells(summary) -> str | None:
    """Failure detail for the well-counting stages (None when the stage ok)."""
    return f"for {summary.failed} well(s)" if summary.failed else None


def _failed_analysis(summary) -> str | None:
    """Failure detail for the analyze stage (None when the stage ok).

    Covers both failure channels: roots that raised (``failures``) and
    wells that failed inside an otherwise-completed root
    (``failed_wells``).
    """
    parts = []
    if summary.failures:
        parts.append(f"root(s) {summary.failures}")
    if summary.failed_wells:
        parts.append(f"well(s) {summary.failed_well_names}")
    return f"for {' and '.join(parts)}" if parts else None


def _run_stage(
    stage: Stage,
    label: str,
    target: Any,
    run: Callable[[], Any],
    done: dict[str, Any],
    fail_detail: Callable[[Any], str | None],
) -> tuple[Any, PipelineSummary | None]:
    """Run one pipeline stage with the shared try/abort discipline.

    Logs the start line, runs the stage, and on a failure — an exception,
    or ``fail_detail(summary)`` returning a non-None description — logs the
    error and builds the failed :class:`PipelineSummary` that aborts the
    chain. The stage's own summary is embedded in the abort summary only
    when it completed but reported a failure; ``done`` holds the summaries
    of the stages already completed.

    Returns ``(summary, abort)``: the stage's summary (None when it raised)
    and the abort summary (None when the stage succeeded).
    """
    log.info("pipeline: %s → %s", label, target)
    try:
        summary = run()
    except Exception:
        log.error("pipeline: %s stage failed\n%s", label, traceback.format_exc())
        return None, PipelineSummary(status="failed", failed_stage=stage, **done)
    detail = fail_detail(summary)
    if detail is not None:
        log.error("pipeline: %s stage failed %s — aborting", label, detail)
        return summary, PipelineSummary(
            status="failed", failed_stage=stage, **done, **{stage: summary},
        )
    return summary, None


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    """Run the standard chain (see the module docstring).

    Stage failures — an exception, or a failed stage summary (``failed``
    wells, or the analyze stage's ``failures``/``failed_wells``) — abort
    the chain: later stages are not called, the error is logged, and the
    returned summary reports ``status = "failed"`` with the failing
    stage's name. An invalid config raises :class:`ConfigError` instead
    (the CLI maps it to exit 2).
    """
    diff_opts, seg_opts, ana_opts = config.resolve_stages()
    log.info(
        "pipeline: reference %s, raw %s → out %s",
        config.reference, config.raw_dir, config.out_root,
    )
    diff_summary, abort = _run_stage(
        "difference", "[1/3] difference", diff_opts.out_dir,
        lambda: run_difference(diff_opts), {}, _failed_wells,
    )
    if abort:
        return abort
    seg_summary, abort = _run_stage(
        "segmentation", "[2/3] segment (input=diff)", seg_opts.out_dir,
        lambda: run_segmentation(seg_opts), {"difference": diff_summary},
        _failed_wells,
    )
    if abort:
        return abort
    ana_summary, abort = _run_stage(
        "analysis", "[3/3] analyze", ana_opts.out_root,
        lambda: run_analysis(ana_opts),
        {"difference": diff_summary, "segmentation": seg_summary},
        _failed_analysis,
    )
    if abort:
        return abort

    log.info("pipeline: done — all three stages ok")
    return PipelineSummary(
        status="ok",
        difference=diff_summary,
        segmentation=seg_summary,
        analysis=ana_summary,
    )