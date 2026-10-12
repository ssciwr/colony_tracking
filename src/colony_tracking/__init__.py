"""colony_tracking — stable package for the colony-size measurement pipeline.

The public API (option dataclasses, ``run_*`` functions, ``PipelineConfig``,
and the summary types) is re-exported from here as its ticket lands;
``__all__`` is the single source of truth for that namespace.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

#: The single package logger. All runtime output goes through it — never print.
#: Defined before the re-exports below: the submodules they import do
#: ``from . import log`` and would otherwise see a partially initialized
#: package.
log = logging.getLogger("colony_tracking")

from .finetune import (  # noqa: E402
    EvaluateOptions,
    EvaluateSummary,
    PrepareOptions,
    PrepareSummary,
    TrainOptions,
    TrainSummary,
    run_evaluate,
    run_prepare,
    run_train,
)

__all__: list[str] = [
    # Finetune track (ticket 07):
    "PrepareOptions",
    "TrainOptions",
    "EvaluateOptions",
    "PrepareSummary",
    "TrainSummary",
    "EvaluateSummary",
    "run_prepare",
    "run_train",
    "run_evaluate",
    # The remaining stages re-export from here as their tickets land:
    # difference, segmentation, analysis, pipeline.
]
