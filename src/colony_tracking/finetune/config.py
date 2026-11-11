"""The finetune steps' flat config documents, shared by the CLI and the TUI.

Each step — ``prepare``, ``train``, ``evaluate`` — has one flat TOML
document: one key per option of its :class:`~colony_tracking._options.Options`
dataclass, exactly what the step's dataclass round-trips. The reader and
writer live in :class:`~colony_tracking.config.FlatConfig`; this module only
registers the three steps and exposes the step-addressed helpers, so the
CLI's ``--config`` and the TUI's drafts read and write the same files with
the same validation wordings.

A document with an unknown key, a missing required option, or a mistyped
value is a usage error: logged under the step's prefix and raised as
:class:`~colony_tracking.config.ConfigError` (the CLI maps it to exit 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._options import EvaluateOptions, Options, PrepareOptions, TrainOptions
from ..config import ConfigError, FlatConfig, config_error

#: Each step's options dataclass — what its flat document is explicit about.
STEP_OPTIONS: dict[str, type[Options]] = {
    "prepare": PrepareOptions,
    "train": TrainOptions,
    "evaluate": EvaluateOptions,
}

#: The finetune steps, in run order (each is a run type and a config file).
STEPS: tuple[str, ...] = tuple(STEP_OPTIONS)


def step_config(step: str) -> FlatConfig:
    """The flat-document description of one finetune step.

    Unknown steps raise :class:`ConfigError` naming the known ones.
    """
    options = STEP_OPTIONS.get(step)
    if options is None:
        raise config_error(
            "finetune",
            f"unknown finetune step {step!r} — expected one of "
            f"{list(STEPS)}",
        )
    return FlatConfig(step, options)


def resolve_step_document(step: str, doc: Any) -> Options:
    """Validate and build one step's options from its flat document.

    The pipeline config's checkpoints, in this order: the document must be
    a table; its keys must be exactly the dataclass's field names; every
    field without a default must be present; every value must match its
    declared type. Omitted keys take their dataclass defaults.
    """
    return step_config(step).resolve(doc)


def read_step_document(step: str, path: Path | str) -> dict[str, Any]:
    """Read one step's flat TOML document, without resolving it.

    :class:`~colony_tracking.config.ConfigError` for malformed TOML;
    structural validation happens at :func:`resolve_step_document` (so the
    CLI can merge explicit flags over the document first).
    """
    return step_config(step).read_document(path)


def read_step_options(step: str, path: Path | str) -> Options:
    """Read and resolve one step's flat TOML document."""
    return step_config(step).read(path)


def write_step_config(step: str, options: Options, path: Path | str) -> Path:
    """Write ``options`` as one step's self-contained flat TOML config.

    Defaults are written, unset (``None``) values omitted; parent
    directories are created. Returns the path.
    """
    return step_config(step).write(options, path)


__all__ = [
    "ConfigError",
    "STEPS",
    "STEP_OPTIONS",
    "step_config",
    "resolve_step_document",
    "read_step_document",
    "read_step_options",
    "write_step_config",
]
