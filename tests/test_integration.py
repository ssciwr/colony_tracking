"""The one-time real-plate integration check (spec: Testing Decisions).

Excluded from the default run by ``addopts = "-m 'not integration'"``; run
explicitly with ``uv run pytest -m integration``. It requires the real
plate data root (via ``COLONY_TRACKING_DATA_ROOT``) and a GPU, and skips
when either is absent so the suite stays hermetic. The full check — the
standard chain into a scratch out root, colony-count sanity, and column
equality against the prototype's trees — lands with the pipeline ticket.
"""

import os

import pytest


@pytest.mark.integration
def test_full_pipeline_on_real_plate():
    if not os.environ.get("COLONY_TRACKING_DATA_ROOT"):
        pytest.skip("COLONY_TRACKING_DATA_ROOT not set")
    pytest.skip("real-plate check lands with the pipeline ticket")
