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
