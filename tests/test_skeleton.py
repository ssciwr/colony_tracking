"""Smoke tests: the package imports, the package logger exists, the public
API namespace is a list, and every module of the skeleton imports cleanly.
"""

import logging

import colony_tracking


def test_package_logger_exists():
    assert isinstance(colony_tracking.log, logging.Logger)
    assert colony_tracking.log.name == "colony_tracking"


def test_all_is_a_list():
    assert isinstance(colony_tracking.__all__, list)


def test_all_is_exactly_the_public_api():
    """``__all__`` exports exactly the public API: the option dataclasses,
    the ``run_*`` functions, ``PipelineConfig``, and the summary types."""
    assert sorted(colony_tracking.__all__) == sorted([
        "AnalysisOptions", "AnalysisSummary", "RootSummary", "run_analysis",
        "DifferenceOptions", "DifferenceSummary", "run_difference",
        "SegmentationOptions", "SegmentationSummary", "run_segmentation",
        "PrepareOptions", "TrainOptions", "EvaluateOptions",
        "PrepareSummary", "TrainSummary", "EvaluateSummary",
        "run_prepare", "run_train", "run_evaluate",
        "PipelineConfig", "PipelineSummary", "run_pipeline",
    ])
    for name in colony_tracking.__all__:
        assert hasattr(colony_tracking, name)


def test_all_modules_import():
    from colony_tracking import (
        analysis,
        cli,
        difference,
        finetune,
        io,
        pipeline,
        segmentation,
        tracking,
    )
    from colony_tracking.finetune import _convert

    for module in (
        analysis,
        cli,
        difference,
        finetune,
        io,
        _convert,
        pipeline,
        segmentation,
        tracking,
    ):
        assert module is not None
