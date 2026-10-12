"""Private data conversion for the finetuning track.

Builds the well-split h5 training set from raw frames and ground-truth
labels, reusing the io helpers (no duplicated io logic). Not public API —
the public surface is ``run_prepare`` in the package module.

Why the steps it takes (the prototype's rationale, kept):

* The image set is defined by the label directory: only frames with a
  matching label are converted. Pairs are matched by file stem — the same
  file name in the label tree.
* Labels with fewer than ``min_instances`` instances are dropped:
  micro-sam samples crops with a sampler that needs a real object per
  crop, so an all-zero label would abort training with "Could not sample
  a valid batch in 500 attempts".
* The raw frames are stretched to uint8 (1/99 percentile) via the shared
  io helper: SAM expects input in [0, 255], the raw images are uint16.
* The split is by well, never by file: neighbouring timepoints of one well
  correlate at r > 0.99, so a file split would leak the same colonies into
  train and validation. Whole wells move, greedily the instance-richest
  until ``val_fraction`` (capped by ``max_val_wells``), or an explicit
  val-well list.
* One h5 per pair holds ``raw`` (uint8) and ``label`` (int32), gzip 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile

from .. import io, log

#: h5 compression level for both datasets (prototype: gzip 4).
GZIP_LEVEL = 4
#: Log write progress every this many files (prototype cadence).
PROGRESS_EVERY = 25


@dataclass(frozen=True)
class PreparedSplit:
    """The conversion result (private; ``run_prepare`` wraps it)."""

    n_pairs: int = 0  # h5 files written
    n_train: int = 0  # pairs in the train split
    n_val: int = 0  # pairs in the val split
    n_instances: int = 0  # instances across the kept pairs
    train_wells: tuple[str, ...] = ()
    val_wells: tuple[str, ...] = ()
    n_dropped: int = 0  # labels with < min_instances instances
    n_missing_raw: int = 0  # labels without a matching raw frame
    error: str | None = None


def convert(
    raw_dir: Path | str,
    label_dir: Path | str,
    out_dir: Path | str,
    val_fraction: float = 0.2,
    max_val_wells: int = 3,
    min_instances: int = 1,
    n_workers: int = 8,
    val_wells: Sequence[str] | None = None,
) -> PreparedSplit:
    """Build the well-split h5 set under ``<out_dir>/{train,val}``."""
    raw_dir, label_dir, out_dir = Path(raw_dir), Path(label_dir), Path(out_dir)

    if not raw_dir.is_dir():
        return PreparedSplit(error=f"no such raw frame directory: {raw_dir}")
    if not label_dir.is_dir():
        return PreparedSplit(error=f"no such label directory: {label_dir}")
    if min_instances < 1:
        return PreparedSplit(
            error=f"min_instances must be >= 1 (empty labels cannot be "
            f"sampled), got {min_instances}"
        )

    # Pair raw frames and labels by file stem (the same file name in the
    # label tree); the label set defines the image set, as in the prototype.
    raw_by_key = {
        (f.well, f.name): f for frames in io.discover_wells(raw_dir).values()
        for f in frames
    }
    label_frames = [
        f for frames in io.discover_wells(label_dir).values() for f in frames
    ]
    if not label_frames:
        return PreparedSplit(error=f"no ground-truth labels found in {label_dir}")

    pairs = []
    n_missing_raw = 0
    for lf in label_frames:
        rf = raw_by_key.get((lf.well, lf.name))
        if rf is None:
            log.warning("no raw image for %s, skipping", lf.name)
            n_missing_raw += 1
            continue
        pairs.append((lf.well, lf.name, rf.path, lf.path))

    # Inspect the labels once to drop empty ones and count instances.
    def inspect(pair: tuple[str, str, Path, Path]) -> tuple[str, int]:
        _, _, _, label_path = pair
        label = np.asarray(tifffile.imread(label_path))
        return pair[0], int(label.max())

    log.info("inspecting %d raw/label pairs ...", len(pairs))
    with ThreadPoolExecutor(n_workers) as pool:
        inspected = list(pool.map(inspect, pairs))

    kept = []
    n_dropped = 0
    for pair, (_, n_inst) in zip(pairs, inspected):
        if n_inst < min_instances:
            n_dropped += 1
            continue
        kept.append(
            {"well": pair[0], "name": pair[1], "raw": pair[2],
             "label": pair[3], "n_inst": n_inst}
        )
    log.info(
        "  %d pairs dropped (label has < %d instances), %d kept",
        n_dropped, min_instances, len(kept),
    )
    if not kept:
        return PreparedSplit(
            n_dropped=n_dropped, n_missing_raw=n_missing_raw,
            error="no usable pairs left (all labels dropped)",
        )

    # Group by well and pick whole wells for validation.
    per_well: dict[str, list[dict]] = {}
    for rec in kept:
        per_well.setdefault(rec["well"], []).append(rec)

    # Wells present in the label tree whose labels were all dropped.
    for well in sorted({p[0] for p in pairs} - set(per_well)):
        log.warning(
            "well %s: all labels dropped (< %d instances); excluded from "
            "the split", well, min_instances,
        )

    if val_wells:
        val_set = set(val_wells)
        unknown = sorted(val_set - set(per_well))
        if unknown:
            return PreparedSplit(
                n_dropped=n_dropped, n_missing_raw=n_missing_raw,
                error=f"requested validation wells without labels: "
                f"{', '.join(unknown)}",
            )
    else:
        # Greedily take the instance-richest wells until the target is
        # reached, so the split follows the data instead of a file count
        # that would be mostly empty frames.
        total = sum(r["n_inst"] for r in kept)
        target = val_fraction * total
        val_set: set[str] = set()
        n_val_inst = 0
        order = sorted(
            per_well, key=lambda w: (-sum(r["n_inst"] for r in per_well[w]), w)
        )
        for well in order:
            if n_val_inst >= target or len(val_set) >= max_val_wells:
                break
            val_set.add(well)
            n_val_inst += sum(r["n_inst"] for r in per_well[well])

    train_recs = [r for r in kept if r["well"] not in val_set]
    val_recs = [r for r in kept if r["well"] in val_set]
    if not train_recs or not val_recs:
        return PreparedSplit(
            n_dropped=n_dropped, n_missing_raw=n_missing_raw,
            error="split produced an empty train or val set, adjust "
            "val_wells/val_fraction",
        )

    log.info("split by well (%d wells carry labels):", len(per_well))
    for tag, recs in (("train", train_recs), ("val", val_recs)):
        wells = sorted({r["well"] for r in recs})
        log.info(
            "  %s %3d images, %3d instances, wells %s",
            tag, len(recs), sum(r["n_inst"] for r in recs), ",".join(wells),
        )

    # Write one h5 per pair: raw uint8 (shared io stretch), label int32.
    n_pairs = 0
    for tag, recs in (("train", train_recs), ("val", val_recs)):
        split_dir = out_dir / tag
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(recs, 1):
            raw = np.asarray(tifffile.imread(rec["raw"]))
            label = np.asarray(tifffile.imread(rec["label"]))
            if raw.shape != label.shape:
                log.warning("shape mismatch for %s, skipping", rec["name"])
                continue
            out_path = split_dir / (Path(rec["name"]).stem + ".h5")
            with h5py.File(out_path, "w") as h5:
                h5.create_dataset(
                    "raw", data=io.stretch_uint8(raw),
                    compression="gzip", compression_opts=GZIP_LEVEL,
                )
                h5.create_dataset(
                    "label", data=label.astype(np.int32),
                    compression="gzip", compression_opts=GZIP_LEVEL,
                )
            n_pairs += 1
            if i % PROGRESS_EVERY == 0:
                log.info("  %s: %d/%d", tag, i, len(recs))

    log.info(
        "done: %d train + %d val files in %s/",
        len(train_recs), len(val_recs), out_dir,
    )
    return PreparedSplit(
        n_pairs=n_pairs,
        n_train=len(train_recs),
        n_val=len(val_recs),
        n_instances=sum(r["n_inst"] for r in kept),
        train_wells=sorted({r["well"] for r in train_recs}),
        val_wells=sorted({r["well"] for r in val_recs}),
        n_dropped=n_dropped,
        n_missing_raw=n_missing_raw,
    )
