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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import log
from .._options import EvaluateOptions, PrepareOptions, TrainOptions
from ._convert import convert

__all__ = [
    "PrepareOptions",
    "TrainOptions",
    "EvaluateOptions",
    "PrepareSummary",
    "TrainSummary",
    "EvaluateSummary",
    "run_prepare",
    "run_train",
    "run_evaluate",
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


@dataclass
class TrainSummary:
    """What ``run_train`` did: smoke-test status, dataset size, checkpoint."""

    smoke_test: bool = False  # the one-batch smoke test ran and yielded data
    smoke_raw_shape: tuple[int, ...] | None = None
    smoke_label_shape: tuple[int, ...] | None = None
    n_train_images: int = 0  # h5 files in the train split
    n_val_images: int = 0  # h5 files in the val split
    checkpoint: Path | None = None  # best checkpoint, when it exists
    error: str | None = None


def run_train(options: TrainOptions) -> TrainSummary:
    """Run micro-sam decoder training on the prepared h5 split.

    A one-batch smoke test (shapes via the logger) runs before training,
    to catch dataset problems (e.g. empty labels) up front. Checkpoints
    and TensorBoard logs land under ``out_dir``. The model stack is
    imported lazily, only once the prerequisites (the h5 split) check
    out — a missing split is a logged error, not a traceback.
    """
    train_dir = Path(options.data_dir) / "train"
    val_dir = Path(options.data_dir) / "val"
    train_paths = sorted(train_dir.glob("*.h5")) if train_dir.is_dir() else []
    val_paths = sorted(val_dir.glob("*.h5")) if val_dir.is_dir() else []
    if not train_paths:
        message = f"no training files in {train_dir} — run train prepare first"
        log.error(message)
        return TrainSummary(error=message)
    if not val_paths:
        message = f"no validation files in {val_dir} — run train prepare first"
        log.error(message)
        return TrainSummary(error=message)

    # Lazy: the heavy model stack is imported only when training runs.
    import torch
    import torch_em

    from micro_sam.training import default_sam_dataset, train_sam
    from micro_sam.training.training import CONFIGURATIONS
    from micro_sam.training.util import get_device

    torch.manual_seed(options.seed)
    np.random.seed(torch.initial_seed() % 2**32)

    device = get_device(None)
    log.info("device: %s", device)
    if str(device).startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(
            "GPU VRAM: %.1f GB (model: %s, configuration: %s)",
            total, options.model_type, options.configuration,
        )

    def build_loader(paths: list[Path], is_train: bool):
        per_image = (
            options.samples_per_image
            if is_train
            else options.val_samples_per_image
        )
        dataset = default_sam_dataset(
            raw_paths=paths,
            raw_key="raw",
            label_paths=paths,
            label_key="label",
            patch_shape=tuple(options.patch_shape),
            with_segmentation_decoder=True,
            # The data is already 8-bit, so `require_8bit` passes it
            # through unchanged.
            raw_transform=None,
            n_samples=per_image * len(paths),
            is_train=is_train,
        )
        return torch_em.get_data_loader(
            dataset, batch_size=1, shuffle=is_train, num_workers=4,
        )

    log.info("loading dataset from %s ...", options.data_dir)
    train_loader = build_loader(train_paths, is_train=True)
    val_loader = build_loader(val_paths, is_train=False)
    log.info(
        "  train: %d images / %d samples per epoch",
        len(train_paths), len(train_loader.dataset),
    )
    log.info(
        "  val:   %d images / %d samples per epoch",
        len(val_paths), len(val_loader.dataset),
    )

    # Catch dataset problems (e.g. empty labels) before the run starts.
    x, y = next(iter(train_loader))
    log.info(
        "  smoke test, sample shapes: raw %s %s in [%d, %d], label %s",
        tuple(x.shape), x.dtype, int(x.min()), int(x.max()), tuple(y.shape),
    )
    smoke_raw, smoke_label = tuple(x.shape), tuple(y.shape)

    if options.configuration not in CONFIGURATIONS:
        message = (
            f"unknown configuration '{options.configuration}', "
            f"expect one of {list(CONFIGURATIONS)}"
        )
        log.error(message)
        return TrainSummary(
            smoke_test=True,
            smoke_raw_shape=smoke_raw,
            smoke_label_shape=smoke_label,
            n_train_images=len(train_paths),
            n_val_images=len(val_paths),
            error=message,
        )
    n_sub_iteration = CONFIGURATIONS[options.configuration].get(
        "n_sub_iteration", 4
    )
    log.info(
        "  preset '%s': %d objects/batch, %d sub-iterations",
        options.configuration, options.n_objects_per_batch, n_sub_iteration,
    )

    start = time.time()
    train_sam(
        name="sam_colony",
        model_type=options.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=options.n_epochs,
        early_stopping=options.early_stopping if options.early_stopping > 0 else None,
        n_objects_per_batch=options.n_objects_per_batch,
        n_sub_iteration=n_sub_iteration,
        lr=options.lr,
        with_segmentation_decoder=True,
        device=device,
        save_root=options.out_dir,
        overwrite_training=False,
        # Labels were already validated by prepare, so skip the slow re-check.
        verify_n_labels_in_loader=None,
    )
    log.info("training finished in %.1f min", (time.time() - start) / 60)
    checkpoint = Path(options.out_dir) / "checkpoints" / "sam_colony" / "best.pt"
    log.info("best checkpoint: %s", checkpoint)
    return TrainSummary(
        smoke_test=True,
        smoke_raw_shape=smoke_raw,
        smoke_label_shape=smoke_label,
        n_train_images=len(train_paths),
        n_val_images=len(val_paths),
        checkpoint=checkpoint if checkpoint.exists() else None,
    )


@dataclass
class EvaluateSummary:
    """What ``run_evaluate`` measured on the held-out split."""

    n_images: int = 0  # h5 files of the split
    n_instances: int = 0  # ground-truth instances scored
    mean_iou: float = 0.0
    median_iou: float = 0.0
    iou_at_05: float = 0.0  # fraction of instances with IoU >= 0.5
    iou_at_075: float = 0.0  # fraction of instances with IoU >= 0.75
    iou_at_09: float = 0.0  # fraction of instances with IoU >= 0.9
    n_clipped: int = 0  # instances larger than a tile
    per_image: list[dict] = field(default_factory=list)
    split: str = "val"
    model_type: str = "vit_t"
    patch_shape: tuple[int, int] = (512, 512)
    checkpoint: Path | None = None
    error: str | None = None


def _predict_instances(predictor, raw, label, patch_shape) -> list[dict]:
    """Segment every ground-truth instance in one image; per-instance IoU.

    Each instance is scored exactly once, in the tile that best contains
    its bounding box — scoring a tile-local copy in every containing tile
    would count an instance several times against a clipped mask. Boxes
    are computed once on the full image so an instance keeps one identity.
    """
    from micro_sam.util import get_centers_and_bounding_boxes

    results = []
    ids = np.unique(label)
    ids = ids[ids > 0]
    _, boxes = get_centers_and_bounding_boxes(label, mode="p")
    h, w = raw.shape

    for iid in ids:
        y0, x0, y1, x1 = boxes[iid]
        # Choose the tile that best contains this instance.
        ty = int(
            np.clip(
                (y0 + y1) // 2 - patch_shape[0] // 2, 0, max(0, h - patch_shape[0])
            )
        )
        tx = int(
            np.clip(
                (x0 + x1) // 2 - patch_shape[1] // 2, 0, max(0, w - patch_shape[1])
            )
        )
        ty = min(ty, max(0, h - patch_shape[0]))
        tx = min(tx, max(0, w - patch_shape[1]))
        patch = raw[ty : ty + patch_shape[0], tx : tx + patch_shape[1]]
        patch_gt = label[ty : ty + patch_shape[0], tx : tx + patch_shape[1]]

        # micro-sam returns boxes as [y0, x0, y1, x1]; SAM expects
        # [x0, y0, x1, y1] — the swap is handled here.
        box = np.array([[x0 - tx, y0 - ty, x1 - tx, y1 - ty]])

        # SAM's predictor expects an HWC image with 3 channels; replicate
        # the greyscale channel (apply_image drops a trailing singleton
        # axis).
        predictor.set_image(np.repeat(patch[..., None], 3, axis=-1))
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        pred = masks[0]
        true = patch_gt == iid
        union = np.logical_or(pred, true).sum()
        results.append(
            {
                "id": int(iid),
                "iou": float(np.logical_and(pred, true).sum() / union)
                if union
                else 0.0,
                "clipped": bool(
                    (y1 - y0) > patch_shape[0] or (x1 - x0) > patch_shape[1]
                ),
            }
        )
    return results


def _report(summary: EvaluateSummary) -> dict:
    """The JSON report (the prototype's evaluate_sam.py keys)."""
    return {
        "model_type": summary.model_type,
        "checkpoint_path": (
            str(summary.checkpoint) if summary.checkpoint is not None else None
        ),
        "split": summary.split,
        "patch_shape": list(summary.patch_shape),
        "n_images": summary.n_images,
        "n_instances": summary.n_instances,
        "mean_iou": summary.mean_iou,
        "median_iou": summary.median_iou,
        "iou@0.5": summary.iou_at_05,
        "iou@0.75": summary.iou_at_075,
        "iou@0.9": summary.iou_at_09,
        "n_clipped": summary.n_clipped,
        "per_image": summary.per_image,
    }


def run_evaluate(options: EvaluateOptions) -> EvaluateSummary:
    """Benchmark a checkpoint on the held-out split.

    Every ground-truth instance is prompted with its own bounding box, so
    the score measures the mask decoder rather than a prompt generator.
    The per-instance IoU is computed in the best containing tile
    (``patch_shape``); instances larger than a tile are flagged clipped.
    Reports mean/median IoU and the fractions above 0.5 / 0.75 / 0.9; an
    optional JSON report is written to ``output``. The model stack is
    imported lazily, only once the prerequisites check out — a missing
    split or checkpoint is a logged error, not a traceback.
    """
    split_dir = Path(options.data_dir) / options.split
    files = sorted(split_dir.glob("*.h5")) if split_dir.is_dir() else []
    if not files:
        message = f"no images in {split_dir} — run train prepare first"
        log.error(message)
        return EvaluateSummary(
            split=options.split,
            model_type=options.model_type,
            patch_shape=tuple(options.patch_shape),
            checkpoint=options.checkpoint,
            error=message,
        )
    if options.checkpoint is not None and not Path(options.checkpoint).exists():
        message = f"no such checkpoint: {options.checkpoint}"
        log.error(message)
        return EvaluateSummary(
            split=options.split,
            model_type=options.model_type,
            patch_shape=tuple(options.patch_shape),
            checkpoint=options.checkpoint,
            error=message,
        )

    # Lazy: the model stack is imported only when evaluation runs.
    import h5py

    from micro_sam.util import get_sam_model

    tag = "finetuned" if options.checkpoint is not None else "base"
    log.info(
        "evaluating the %s model (%s, box prompt) on %d %s images",
        tag, options.model_type, len(files), options.split,
    )
    if options.checkpoint is not None:
        log.info("  checkpoint: %s", options.checkpoint)

    predictor = get_sam_model(
        model_type=options.model_type, device=None,
        checkpoint_path=options.checkpoint,
    )
    patch_shape = tuple(options.patch_shape)

    all_ious: list[float] = []
    per_image: list[dict] = []
    for f in files:
        with h5py.File(f) as h5:
            raw = np.asarray(h5["raw"][:])
            label = np.asarray(h5["label"][:])
        results = _predict_instances(predictor, raw, label, patch_shape)
        if results:
            ious = [r["iou"] for r in results]
            all_ious.extend(ious)
            n_clipped = sum(r["clipped"] for r in results)
            per_image.append(
                {
                    "file": f.name,
                    "n_instances": len(ious),
                    "mean_iou": float(np.mean(ious)),
                    "n_clipped": n_clipped,
                }
            )
            note = f"  ({n_clipped} larger than a tile)" if n_clipped else ""
            log.info(
                "  %s n=%3d meanIoU=%.3f%s",
                f.name[:34], len(ious), float(np.mean(ious)), note,
            )

    if not all_ious:
        message = "no instances evaluated"
        log.error(message)
        return EvaluateSummary(
            n_images=len(files),
            split=options.split,
            model_type=options.model_type,
            patch_shape=patch_shape,
            checkpoint=options.checkpoint,
            error=message,
        )

    arr = np.array(all_ious)
    summary = EvaluateSummary(
        n_images=len(files),
        n_instances=int(arr.size),
        mean_iou=float(arr.mean()),
        median_iou=float(np.median(arr)),
        iou_at_05=float((arr >= 0.5).mean()),
        iou_at_075=float((arr >= 0.75).mean()),
        iou_at_09=float((arr >= 0.9).mean()),
        n_clipped=sum(r["n_clipped"] for r in per_image),
        per_image=per_image,
        split=options.split,
        model_type=options.model_type,
        patch_shape=patch_shape,
        checkpoint=options.checkpoint,
    )
    log.info(
        "%s model over %d instances: mean IoU %.3f, median IoU %.3f, "
        "IoU>=0.5 %.3f, IoU>=0.75 %.3f, IoU>=0.9 %.3f",
        tag, arr.size, summary.mean_iou, summary.median_iou,
        summary.iou_at_05, summary.iou_at_075, summary.iou_at_09,
    )
    if options.output is not None:
        output = Path(options.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(_report(summary), f, indent=2)
        log.info("report written to %s", output)
    return summary
