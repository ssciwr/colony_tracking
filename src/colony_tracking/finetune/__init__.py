"""Finetuning track: ``train prepare | train | evaluate``.

``run_prepare`` builds the well-split h5 training set from raw frames and
ground-truth labels — the conversion itself lives in the private
``_convert`` module and reuses the io helpers. ``run_train`` runs
micro-sam's ``train_sam`` on that split (a one-batch smoke test runs
before training), and ``run_evaluate`` benchmarks a checkpoint on the
held-out split: every ground-truth instance is prompted with its own
bounding box and scored by per-instance IoU in the best containing tile.

micro-sam/torch are imported lazily inside the run functions — importing
this package never imports torch.
"""

from __future__ import annotations

from .._options import EvaluateOptions, PrepareOptions, TrainOptions

__all__ = [
    "PrepareOptions",
    "TrainOptions",
    "EvaluateOptions",
    # Summaries and run functions land with the implementation slices.
]
