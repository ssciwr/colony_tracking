"""Shared test scaffolding for the colony_tracking suite.

The suite is hermetic: no real plate data, no GPU, no network.
"""

import matplotlib

# Force the non-interactive Agg backend before any test imports plotting code.
matplotlib.use("Agg")
