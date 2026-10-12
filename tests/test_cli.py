"""Contract tests for the CLI entry point (stub; the real CLI is ticket 09)."""

from colony_tracking.cli import main


def test_main_returns_zero():
    assert main() == 0


def test_main_prints_nothing(capsys):
    # Runtime output goes through the package logger, never stdout/stderr.
    main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
