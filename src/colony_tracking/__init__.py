"""colony_tracking — stable package for the colony-size measurement pipeline.

The public API (option dataclasses, ``run_*`` functions, ``PipelineConfig``,
and the summary types) is re-exported from here as its ticket lands;
``__all__`` is the single source of truth for that namespace.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

#: The single package logger. All runtime output goes through it — never print.
log = logging.getLogger("colony_tracking")

from ._options import AnalysisOptions
from .analysis import AnalysisSummary, RootSummary, run_analysis

__all__: list[str] = [
    "AnalysisOptions",
    "AnalysisSummary",
    "RootSummary",
    "run_analysis",
    # The remaining public API re-exports (option dataclasses, run_*
    # functions, PipelineConfig, summary types) land here as their tickets
    # do.
]
