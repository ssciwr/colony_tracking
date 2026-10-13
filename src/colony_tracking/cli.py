"""The ``colony-tracking`` command line interface.

One entry point with subcommands ``difference | segment | analyze | pipeline
| train`` (``train`` has sub-subcommands ``prepare | train | evaluate``).
Every option of a stage's options dataclass maps 1:1 to a kebab-case flag
derived programmatically from the dataclass fields, with the per-field help
text as the flag's help (the single source the TUI reuses too). Well names
start with a dash, so ``--wells``-style values must stay inside the flag
value — ``--wells=-A001,-A002`` and ``--wells -A001`` both parse.

The exit-code contract: 0 success, 1 runtime error (a stage exception, or a
summary that reports failed work — logged at ERROR with context), 2 usage
error (argparse errors, invalid option combinations, config errors).
``main`` returns the exit code; the installed console script wraps it.

This module is the only place in the package that configures logging: the
package logger goes to stderr at INFO. Nothing ever prints.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from dataclasses import MISSING, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from . import log
from ._options import (
    AnalysisOptions,
    DifferenceOptions,
    Reference,
    SegmentationOptions,
    _union_members,
)
from .analysis import AnalysisSummary, run_analysis
from .difference import run_difference
from .finetune import (
    EvaluateOptions,
    PrepareOptions,
    TrainOptions,
    run_evaluate,
    run_prepare,
    run_train,
)
from .pipeline import ConfigError, PipelineConfig, run_pipeline
from .segmentation import run_segmentation

#: Flag names that are not the kebab-cased field name (repeatable path lists
#: read as singular directories on the command line).
_RENAMES: dict[str, str] = {"seg_dirs": "seg-dir"}


class _UsageError(Exception):
    """An invalid command line (raised where argparse would exit)."""


class _Parser(argparse.ArgumentParser):
    """An argument parser that raises :class:`_UsageError` instead of
    exiting, so ``main`` can return the exit code instead of raising."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Widened value matcher so dash-leading flag values (wells) parse.
        self._negative_number_matcher = _DashValueMatcher(self)  # noqa: SLF001

    def error(self, message: str) -> None:  # noqa: D401 (argparse API)
        raise _UsageError(message)


class _DashValueMatcher:
    """The ``_negative_number_matcher`` hook of :mod:`argparse`, widened for
    this CLI: an unknown single-dash token (e.g. the well ``-A001``) is a
    flag *value*, not an option. Well names start with a dash, so
    ``--wells -A001`` must parse; known option strings (``-h``) and
    double-dash tokens are left alone."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self._parser = parser

    def match(self, arg_string: str) -> object | None:
        if arg_string.startswith("--"):
            return None
        if arg_string in self._parser._option_string_actions:  # noqa: SLF001
            return None
        return True


class _AppendComma(argparse.Action):
    """Accumulate the (already comma-split) values of repeatable flags:
    every occurrence appends its values to the destination, flattened."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None) or []
        setattr(namespace, self.dest, [*current, *values])


class _Tuple(argparse.Action):
    """Collect an n-value flag (``--patch-shape 512 512``) as a tuple."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


def _comma_list(item_type):
    """A ``type=`` callable splitting one flag value on commas into items
    of ``item_type`` (``--wells=-A001,-A002`` → ``["-A001", "-A002"]``)."""

    def split(text: str) -> list:
        return [item_type(part.strip()) for part in text.split(",")
                if part.strip()]

    return split


def _parse_reference(text: str) -> Reference:
    """``first``, a bare 1-based frame number (``5``), or ``LO005``."""
    lowered = text.strip().lower()
    if lowered == "first":
        return "first"
    if lowered.startswith("lo"):
        lowered = lowered[2:]
    try:
        return int(lowered)
    except ValueError:
        raise ValueError(
            f"invalid reference {text!r}: expected 'first', a 1-based frame "
            "number, or LO<nnn>"
        ) from None


def _add_flags(parser: argparse.ArgumentParser, cls: type, *,
               renames: dict[str, str] | None = None) -> None:
    """One kebab-case flag per field of the options dataclass ``cls``,
    with the field's help text (the dataclass is the single source)."""
    renames = _RENAMES if renames is None else renames
    hints = get_type_hints(cls)
    for f in fields(cls):
        flag = "--" + renames.get(f.name, f.name.replace("_", "-"))
        required = f.default is MISSING and f.default_factory is MISSING
        default = None
        if not required:
            default = (f.default if f.default is not MISSING
                       else f.default_factory())
        hint = hints[f.name]
        origin = get_origin(hint)
        members = _union_members(hint)
        if members is not None and len(members) == 1:  # X | None
            hint, origin = members[0], get_origin(members[0])
        kwargs: dict = {"help": f.metadata["help"], "default": default,
                        "required": required, "dest": f.name}
        if hint is bool:
            if default:
                parser.add_argument("--no-" + flag[2:], dest=f.name,
                                    action="store_false", help=kwargs["help"])
            else:
                parser.add_argument(flag, dest=f.name, action="store_true",
                                    help=kwargs["help"])
            continue
        if hint is Reference:
            kwargs["type"] = _parse_reference
        elif origin is list:
            kwargs["action"] = _AppendComma
            kwargs["type"] = _comma_list(get_args(hint)[0])
            kwargs["metavar"] = renames.get(f.name, f.name).upper()
        elif origin is tuple:  # e.g. patch_shape: two values
            item_type = get_args(hint)[0]
            kwargs["action"] = _Tuple
            kwargs["nargs"] = len(get_args(hint))
            kwargs["type"] = item_type
            kwargs["metavar"] = " ".join(
                [renames.get(f.name, f.name).upper()] * len(get_args(hint)))
        else:
            kwargs["type"] = hint
            if origin is not None:  # Literal: fixed choices
                kwargs["choices"] = list(get_args(hint))
                kwargs.pop("type")
        parser.add_argument(flag, **kwargs)


def _stage_kwargs(args: argparse.Namespace, cls: type) -> dict:
    """The parsed flag values that are fields of ``cls`` (one namespace per
    subcommand, dests named after the fields)."""
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in vars(args).items() if key in names}


def _difference_options(args: argparse.Namespace) -> DifferenceOptions:
    return DifferenceOptions(**_stage_kwargs(args, DifferenceOptions))


def _segmentation_options(args: argparse.Namespace) -> SegmentationOptions:
    """The segment options, with the input mode's directory required exactly
    when the input mode needs it (a usage error, exit code 2)."""
    options = SegmentationOptions(**_stage_kwargs(args, SegmentationOptions))
    if options.input == "raw" and options.raw_dir is None:
        raise _UsageError("--raw-dir is required when --input raw")
    if options.input == "diff" and options.diff_dir is None:
        raise _UsageError("--diff-dir is required when --input diff")
    return options


def _analysis_options(args: argparse.Namespace) -> AnalysisOptions:
    return AnalysisOptions(**_stage_kwargs(args, AnalysisOptions))


#: The pipeline's run-level settings — the only flags the ``pipeline``
#: subcommand takes besides ``--config`` (per-stage tuning lives in the
#: config file).
_PIPELINE_OVERRIDES = ("raw_dir", "out_root", "reference", "wells")


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """The pipeline config: the TOML file (optional — omit it to drive the
    run from the command line alone) plus the run-level command-line
    overrides (which take precedence — they are merged into the plain
    document before :meth:`PipelineConfig.from_dict` resolves the stages,
    so the derived names follow the overridden values)."""
    if args.config is None:
        missing = [name for name in ("raw_dir", "out_root")
                   if getattr(args, name) is None]
        if missing:
            flags = " and ".join(f"--{name.replace('_', '-')}"
                                 for name in missing)
            raise _UsageError(f"{flags} required when --config is omitted")
        doc: dict = {}
    else:
        path = Path(args.config)
        try:
            with open(path, "rb") as handle:
                doc = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: not valid TOML: {exc}") from exc
        log.info("pipeline config: reading %s", path)
    section = doc.setdefault("pipeline", {})
    if not isinstance(section, dict):
        section = doc["pipeline"] = {}
    overrides = {name: getattr(args, name) for name in _PIPELINE_OVERRIDES}
    for name, value in overrides.items():
        if value is not None:
            section[name] = str(value) if isinstance(value, Path) else value
    return PipelineConfig.from_dict(doc)


def _add_pipeline_flags(parser: argparse.ArgumentParser) -> None:
    help_by_field = {f.name: f.metadata["help"] for f in fields(PipelineConfig)}
    parser.add_argument(
        "--config", type=Path,
        help="TOML config file: a [pipeline] section plus one optional "
        "section per stage. Omit to drive the run from the command line "
        "alone (--raw-dir and --out-root required then).",
    )
    parser.add_argument(
        "--raw-dir", dest="raw_dir", type=Path,
        help=help_by_field["raw_dir"] + " Overrides the config file.",
    )
    parser.add_argument(
        "--out-root", dest="out_root", type=Path,
        help=help_by_field["out_root"] + " Overrides the config file.",
    )
    parser.add_argument(
        "--reference", dest="reference", type=_parse_reference,
        help=help_by_field["reference"] + " Overrides the config file.",
    )
    parser.add_argument(
        "--wells", dest="wells", action=_AppendComma, type=_comma_list(str),
        metavar="WELLS",
        help=help_by_field["wells"] + " Overrides the config file.",
    )


def _train_options(args: argparse.Namespace):
    """The finetune options for the given ``train`` sub-subcommand."""
    cls = {"prepare": PrepareOptions, "train": TrainOptions,
           "evaluate": EvaluateOptions}[args.train_command]
    return cls(**_stage_kwargs(args, cls))


def _train_run(args: argparse.Namespace):
    """Run the ``train`` sub-subcommand's stage (each returns a summary
    with an ``error`` field instead of raising)."""
    options = _train_options(args)
    if args.train_command == "prepare":
        return run_prepare(options)
    if args.train_command == "train":
        return run_train(options)
    return run_evaluate(options)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="colony-tracking",
        description="Colony-size measurement pipeline (difference → segment "
        "→ analyze, plus the finetuning track).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       metavar="command")
    _add_flags(
        subparsers.add_parser(
            "difference",
            help="Per-well uint8 difference frames from a raw plate.",
            description="Difference frames from a raw plate directory "
            "(see --help for every flag).",
        ),
        DifferenceOptions,
    )
    _add_flags(
        subparsers.add_parser(
            "segment",
            help="Colony segmentation and tracking of raw or difference frames.",
            description="Colony segmentation and tracking of raw or "
            "difference frames (see --help for every flag).",
        ),
        SegmentationOptions,
    )
    _add_flags(
        subparsers.add_parser(
            "analyze",
            help="Growth measurements, aggregates, and plots for segmentation trees.",
            description="Growth measurements, aggregates, and plots for one "
            "or more segmentation trees (see --help for every flag).",
        ),
        AnalysisOptions,
        renames=_RENAMES,
    )
    _add_pipeline_flags(subparsers.add_parser(
        "pipeline",
        help="The standard chain (difference → segment → analyze) from a "
        "TOML config or the command line.",
        description="The standard chain (difference → segment → analyze) "
        "from a TOML config file, or from the command line alone; the "
        "command line carries the config path and the run-level overrides "
        "only — per-stage tuning belongs in the config file.",
    ))
    train_parser = subparsers.add_parser(
        "train",
        help="The finetuning track: prepare a well-split h5 set, train, evaluate.",
        description="The finetuning track (see --help on a subcommand for "
        "every flag).",
    )
    train_subparsers = train_parser.add_subparsers(
        dest="train_command", required=True, metavar="command")
    _add_flags(train_subparsers.add_parser(
        "prepare",
        help="Build the well-split h5 training set from raw frames and labels.",
    ), PrepareOptions)
    _add_flags(train_subparsers.add_parser(
        "train",
        help="Run micro-sam decoder training on the prepared split.",
    ), TrainOptions)
    _add_flags(train_subparsers.add_parser(
        "evaluate",
        help="Benchmark a checkpoint: per-instance IoU on the held-out split.",
    ), EvaluateOptions)
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "difference":
        summary = run_difference(_difference_options(args))
        if summary.failed:
            failed = [(result.well, result.error)
                      for result in summary.wells.values()
                      if result.status == "failed"]
            log.error(
                "difference: %d well(s) failed: %s",
                len(failed),
                ", ".join(f"{well} ({error})" for well, error in failed),
            )
            return 1
        return 0
    if args.command == "segment":
        summary = run_segmentation(_segmentation_options(args))
        if summary.failed:
            log.error(
                "segment: %d well(s) failed: %s",
                summary.failed, ", ".join(summary.failed_wells),
            )
            return 1
        return 0
    if args.command == "analyze":
        summary = run_analysis(_analysis_options(args))
        if summary.failures:
            log.error(
                "analyze: %d root(s) failed: %s",
                len(summary.failures), "; ".join(summary.failures),
            )
            return 1
        if summary.failed_wells:
            log.error("analyze: %d well(s) failed: %s",
                      len(summary.failed_well_names),
                      ", ".join(summary.failed_well_names))
            return 1
        return 0
    if args.command == "pipeline":
        summary = run_pipeline(_pipeline_config(args))
        if summary.status == "failed":
            log.error("pipeline: the %s stage failed", summary.failed_stage)
            return 1
        return 0
    if args.command == "train":
        summary = _train_run(args)
        if summary.error is not None:
            log.error("train %s: %s", args.train_command, summary.error)
            return 1
        return 0
    raise _UsageError(f"unknown command {args.command!r}")


def _configure_logging() -> None:
    """The package's only logging configuration: the package logger →
    stderr at INFO (idempotent; ``main`` may run many times in-process)."""
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``colony-tracking`` console script: returns the
    exit code (0 success, 1 runtime error, 2 usage error)."""
    _configure_logging()
    parser = _build_parser()
    try:
        args = parser.parse_args(
            sys.argv[1:] if argv is None else [str(a) for a in argv]
        )
    except _UsageError as exc:
        log.error("usage error: %s", exc)
        return 2
    try:
        return _dispatch(args)
    except _UsageError as exc:  # invalid option combinations
        log.error("usage error: %s", exc)
        return 2
    except ConfigError as exc:  # a ValueError: must map to 2, not 1
        log.error("config error: %s", exc)
        return 2
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())