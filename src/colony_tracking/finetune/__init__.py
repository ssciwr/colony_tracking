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

from dataclasses import dataclass, field
from pathlib import Path

from .. import log
from .._options import EvaluateOptions, PrepareOptions, TrainOptions
from ._convert import convert

__all__ = [
    "PrepareOptions",
    "TrainOptions",
    "EvaluateOptions",
    "PrepareSummary",
    "run_prepare",
    # run_train / run_evaluate and their summaries land with the next slice.
]


@dataclass
class PrepareSummary:
    """What ``run_prepare`` did: the pairs written and the well split."""

    n_pairs: int = 0  # h5 files written
    n_train: int = 0  # pairs in the train split
    n_val: int = 0  # pairs in the val split
    n_instances: int = 0  # instances across the kept pairs
    train_wells: list[str] = field(default_factory=list)
    val_wells: list[str] = field(default_factory=list)
    n_dropped: int = 0  # labels with < min_instances instances
    n_missing_raw: int = 0  # labels without a matching raw frame
    error: str | None = None


def run_prepare(options: PrepareOptions) -> PrepareSummary:
    """Build the well-split h5 set from raw frames and ground-truth labels.

    Pairs are matched by file stem, labels below ``min_instances`` are
    dropped, the raw frames are stretched to uint8 via the shared io
    helper, and whole wells (never individual files) go to validation —
    greedily the instance-richest until ``val_fraction`` (capped by
    ``max_val_wells``), or the explicit ``val_wells`` list. One h5 per
    pair (``raw`` uint8, ``label`` int32, gzip 4) lands under
    ``<out_dir>/{train,val}``.
    """
    result = convert(
        raw_dir=options.raw_dir,
        label_dir=options.label_dir,
        out_dir=options.out_dir,
        val_fraction=options.val_fraction,
        max_val_wells=options.max_val_wells,
        min_instances=options.min_instances,
        n_workers=options.n_workers,
        val_wells=options.val_wells,
    )
    if result.error is not None:
        log.error(result.error)
    return PrepareSummary(
        n_pairs=result.n_pairs,
        n_train=result.n_train,
        n_val=result.n_val,
        n_instances=result.n_instances,
        train_wells=list(result.train_wells),
        val_wells=list(result.val_wells),
        n_dropped=result.n_dropped,
        n_missing_raw=result.n_missing_raw,
        error=result.error,
    )
