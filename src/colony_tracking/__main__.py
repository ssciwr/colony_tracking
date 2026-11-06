"""``python -m colony_tracking``: the console script's entry point.

The TUI launches the viewer as ``[sys.executable, "-m", "colony_tracking",
"view", ...]`` so the child runs the installed package even when the
``colony-tracking`` script is not on the child's ``PATH``.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
