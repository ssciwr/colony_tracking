"""Command-line interface for colony_tracking.

One entry point with subcommands ``difference | segment | analyze | pipeline
| train``. The real CLI is built in a later ticket; this stub makes the
``colony-tracking`` console script resolve.
"""

from __future__ import annotations

from colony_tracking import log


def main() -> int:
    """Entry point for the ``colony-tracking`` console script (stub)."""
    log.info("colony-tracking: stub entry point — the real CLI lands in a later ticket")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
