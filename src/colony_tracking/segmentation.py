"""Segment stage: growth-mask-gated instance segmentation (ADR 0002).

Colonies in a well only grow, so the final frame sees every colony at its
largest extent. One well is processed like this:

1. **Growth mask from the final frame** — instance segmentation of the
   final frame (highest frame number) at decoder foreground >= ``fg_ref``
   (0.5). Junk instances are dropped (area < ``min_size`` (200 px) or
   solidity < ``min_solidity`` (0.7) — e.g. the dark well ring if it is
   ever segmented); the union of the survivors is split into connected
   components (**groups**; a group may contain several colonies that have
   grown together) and dilated by ``mask_margin`` (15 px) = the **growth
   mask**. ``ref__labels.tif`` holds the group labels of the final frame.
2. **Earlier frames** — foreground >= ``fg_early`` (0.3: the decoder
   under-fires on the faint, small early colonies, and false positives are
   caught by the mask anyway). An instance is kept only if >= ``min_overlap``
   (0.5) of its pixels lie inside the growth mask **and** inside a single
   group — colonies only grow, so a mid-series instance belongs to one
   colony's final extent, and an instance spanning several groups cannot be
   a single colony.
3. **Drift gate** — a frame is skipped when its elevated-foreground
   fraction (the part of the whole image above the frame's foreground
   threshold) exceeds ``max(max_fg_ratio * reference-frame fraction, 0.05)``:
   drift-dominated frames make the decoder hallucinate confident giant
   phantom instances. ``max_fg_ratio = 0`` disables the gate. The final
   frame itself is never gated. Skipped frames still get their (blank)
   label image, so the outputs stay 1:1 with the input frames.
4. **Tracking** — colony IDs are assigned by the shared pure module
   :mod:`colony_tracking.tracking`: touching segments of one group are
   pre-merged, then segments are tracked forward in time with the
   continue / absorb-merge / fork / new-ID rules, and the ``exclude_before``
   pre-existing filter blanks frames before N and flags colonies first
   detected at or before N as ``pre_existing=1``.

Inputs (``input``): with ``"raw"`` each raw frame is 1/99 percentile-
stretched to uint8 (the shared io helper, matching the finetuning data
prep). With ``"diff"`` the uint8 difference frames are fed as written
(128 = no change, ADR 0001 — the colonies' difference polarity flips over
time anyway, so there is no single polarity to enforce) and the reference
frame comes from the well's ``scale.txt``: **a difference reference equal
to the final frame is rejected up front** — the growth mask would be empty.

Outputs, per well, under per-well subdirectories of ``out_dir``:
``<frame-stem>__labels.tif`` per input frame (uint32 colony IDs, 1:1 with
the input frames; all-zero for gate-skipped or ``exclude_before``-blanked
frames), ``ref__labels.tif`` (the group labels of the final frame),
``colonies.csv`` (canonical columns), and the ``_done.txt`` done marker
with the seven canonical keys — a well with a complete marker is skipped
on re-run unless ``redo``.

Re-ID (``reid_only``): no model, no GPU — the segments are re-derived from
the saved label images (each label value re-split into connected
components), tracking is re-run from the first non-blank frame, and the
label images, colonies.csv, and done marker are rewritten. Useful after
changing ``exclude_before`` or the identity logic.

micro-sam (and with it torch) is imported lazily, inside the single model-
loading helper — importing this module never pulls torch, so the
difference, analyze, and train stages do not pay that cost. ``checkpoint``
(a finetuned decoder) replaces the pretrained model when given.

Progress is logged per well and per frame through the package logger;
never printed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.measure import regionprops

from . import io, log
from ._options import SegmentationOptions
from .tracking import extract_segments, track_colonies

#: Floor of the drift gate (a fraction of the image): wells with tiny
#: colonies still get protection. The prototype's ``MIN_FG_ABS``.
MIN_FG_ABS = 0.05


# --- the drift gate -------------------------------------------------------------


def drift_gate_limit(ref_frac: float, max_fg_ratio: float) -> float:
    """The elevated-foreground fraction a frame may reach before it is skipped.

    ``max(max_fg_ratio * ref_frac, MIN_FG_ABS)``: relative to the reference
    frame's fraction (its legitimate maximum), floored so that wells with
    tiny colonies still get protection.
    """
    return max(max_fg_ratio * ref_frac, MIN_FG_ABS)


def drift_gate_skips(
    elevated: float, ref_frac: float, *, max_fg_ratio: float
) -> bool:
    """The drift-gate decision for one frame.

    True when the frame's elevated-foreground fraction exceeds the gate
    limit (:func:`drift_gate_limit`) — the decoder elevates the foreground
    globally and hallucinates confident giant instances. ``max_fg_ratio``
    of 0 (or less) disables the gate. The reference frame is never gated
    (its fraction equals ``ref_frac`` and can never exceed the limit).
    """
    if max_fg_ratio <= 0:
        return False
    return elevated > drift_gate_limit(ref_frac, max_fg_ratio)


# --- growth-mask construction -----------------------------------------------------


def build_growth_mask(
    instances: np.ndarray,
    *,
    min_size: int,
    min_solidity: float,
    mask_margin: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Junk-filter the final frame's instances and build the growth mask.

    ``instances`` is the final frame's instance segmentation (a label
    image; one positive label value per decoder instance). Instances with
    area < ``min_size`` or solidity < ``min_solidity`` are dropped (well-
    ring junk); the union of the survivors is split into connected
    components — the **groups** (a group label image, uint32, 0 =
    background) — and the groups are dilated by ``mask_margin`` px = the
    **growth mask** (a boolean image).

    Returns ``(groups, mask)``; ``(None, all-False)`` when nothing survives
    the filters (the caller then writes an all-zero ``ref__labels.tif`` and
    keeps no segments).
    """
    instances = np.asarray(instances)
    keep = np.zeros(instances.shape, dtype=bool)
    for region in regionprops(instances):
        if region.area >= min_size and region.solidity >= min_solidity:
            keep |= instances == region.label
    if not keep.any():
        return None, np.zeros(instances.shape, dtype=bool)
    groups = ndi.label(keep)[0].astype(np.uint32)
    return groups, ndi.binary_dilation(groups > 0, iterations=mask_margin)


# --- the mask-overlap filter -------------------------------------------------------


def filter_to_mask(
    instances: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
    min_overlap: float,
) -> tuple[np.ndarray, int]:
    """Keep the instances that lie (mostly) inside the growth mask.

    Two growth-based sanity checks per instance: at least ``min_overlap``
    of its pixels must lie inside the (dilated) growth mask, and at least
    ``min_overlap`` inside its best single reference group — colonies only
    grow, so a mid-series instance belongs to one colony's final extent,
    and an instance spanning several groups cannot be a single colony.

    Returns a label image with the kept instances relabelled 1..k (local
    IDs, no colony identity yet — a kept instance keeps its full extent)
    and the number of discarded instances.
    """
    instances = np.asarray(instances)
    n_groups = 0 if groups is None else int(groups.max())
    out = np.zeros(instances.shape, np.uint32)
    next_id, discarded = 1, 0
    for region in regionprops(instances):
        instance = instances == region.label
        if float((instance & mask).sum()) / region.area >= min_overlap:
            if n_groups:
                hits = np.bincount(
                    groups[instance].ravel(), minlength=n_groups + 1
                )
                if float(hits[1:].max()) / region.area < min_overlap:
                    discarded += 1
                    continue
            out[instance] = next_id
            next_id += 1
        else:
            discarded += 1
    return out, discarded

# --- the stage entry point --------------------------------------------------------


@dataclass
class SegmentationSummary:
    """Outcome of one segment-stage run (for the caller's status line).

    ``wells`` is the number of wells selected for this run (after the
    ``wells`` filter); ``processed``/``skipped``/``failed`` partition it —
    skipped covers wells with a complete done marker (unless ``redo``),
    re-ID wells without completed outputs, and wells rejected up front
    (e.g. a difference reference equal to the final frame). 
    ``gate_skipped_frames`` counts the frames the drift gate skipped, over
    all processed wells; ``failed_wells`` names the failed ones.
    """

    wells: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_wells: list[str] = field(default_factory=list)
    gate_skipped_frames: int = 0


def _diff_reference(well_dir: Path) -> tuple[int | None, float | None]:
    """Reference frame number and display limit of a difference well.

    Both come from the well's ``scale.txt`` written by the difference
    stage: ``reference`` holds the reference frame's file name (the frame
    number is parsed from its ``LO`` field), ``display_limit_int16`` the
    float display limit. ``(None, None)`` when there is no usable
    ``reference`` line.
    """
    scale = io.read_scale(well_dir)
    ref_lo = io.parse_frame(scale.get("reference", ""))
    limit: float | None = None
    raw_limit = scale.get("display_limit_int16")
    if raw_limit:
        try:
            limit = float(raw_limit)
        except ValueError:
            limit = None
    return ref_lo, limit


def _load_frame(frame: io.Frame, input_mode: str) -> np.ndarray:
    """Load one frame as a uint8 image the segmenter can use.

    Raw frames are 1/99 percentile-stretched per frame (the shared io
    helper, matching the finetuning data prep). Difference frames are fed
    as written (128 = no change, ADR 0001).
    """
    image = tifffile.imread(frame.path)
    if input_mode == "raw":
        return io.stretch_uint8(image)
    return np.asarray(image).astype(np.uint8)


def _load_segmenter(model_type: str, checkpoint: Path | None):
    """Load micro-sam's decoder segmenter for one run.

    This is the only place micro-sam — and with it torch — is imported:
    lazily, inside the model path, so that importing this module (and the
    difference, analyze, and train stages) never pays the torch import
    cost. ``checkpoint`` (a finetuned decoder) replaces the pretrained
    model when given.
    """
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        get_predictor_and_decoder,
    )

    predictor, decoder = get_predictor_and_decoder(
        model_type,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
    )
    return InstanceSegmentationWithDecoder(predictor, decoder)


def _segment_frame(
    segmenter,
    image: np.ndarray,
    *,
    fg_threshold: float,
    min_size: int,
    allowed: np.ndarray | None,
    ref_frac: float | None,
    max_fg_ratio: float,
    desc: str,
) -> tuple[np.ndarray, float]:
    """One model call: decoder foreground, drift gate, masked watershed.

    Returns ``(instances, elevated-foreground fraction)`` — the fraction of
    the whole image above the foreground threshold. With ``ref_frac`` (the
    reference frame's fraction) the frame is skipped as a drift artifact
    when its own fraction exceeds the gate limit: the decoder then elevates
    the foreground globally and hallucinates confident giant instances, so
    zeros are returned instead of instances. With ``allowed`` (the growth
    mask) the decoder foreground is cleared outside it before the
    watershed, so no instance can start there at all.
    """
    segmenter.initialize(image)
    elevated = float((segmenter._foreground >= fg_threshold).mean())
    if ref_frac is not None and drift_gate_skips(
        elevated, ref_frac, max_fg_ratio=max_fg_ratio
    ):
        gate = drift_gate_limit(ref_frac, max_fg_ratio)
        log.warning(
            "%s: %.2f of the image above the foreground threshold "
            "(gate %.2f, reference %.3f; global drift artifact); "
            "frame skipped to avoid phantom instances",
            desc, elevated, gate, ref_frac,
        )
        return np.zeros(image.shape, np.uint32), elevated
    if allowed is not None:
        segmenter._foreground[~allowed] = 0.0
    return segmenter.generate(
        foreground_threshold=fg_threshold, min_size=min_size
    ), elevated


def _rows_with_frame_info(
    rows: Sequence[dict[str, object]], frames: Sequence[io.Frame], well: str
) -> list[dict[str, object]]:
    """Add the well, its T field, and the frame file name to tracker rows.

    The tracker returns rows keyed by frame *number*; ``colonies.csv`` also
    carries ``well``, ``t_seconds`` (the raw T field) and ``image`` (the
    input frame's file name), which only the stage knows.
    """
    by_number = {fr.frame: fr for fr in frames}
    return [
        {
            "well": well,
            "t_seconds": by_number[r["frame"]].t,
            "image": by_number[r["frame"]].name,
            **r,
        }
        for r in rows
    ]


def _write_outputs(
    well_dir: Path,
    frames: Sequence[io.Frame],
    labels: Sequence[np.ndarray],
    rows: Sequence[dict[str, object]],
) -> None:
    """Write one well's per-frame label images and colonies.csv."""
    for frame, label in zip(frames, labels):
        io.write_label_image(well_dir / f"{frame.stem}__labels.tif", label)
    io.write_colonies(well_dir / "colonies.csv", rows)


def _blanking_log(well: str, frames: Sequence[io.Frame], exclude_before: int) -> None:
    """Log how many frames ``exclude_before`` blanks for this well."""
    blanked = sum(1 for fr in frames if fr.frame < exclude_before)
    if blanked:
        log.info(
            "[%s] exclude_before %d: blanking %d frames below it "
            "(their label images are written empty)",
            well, exclude_before, blanked,
        )


def _process_well(
    well: str,
    frames: Sequence[io.Frame],
    segmenter,
    options: SegmentationOptions,
) -> tuple[str, int]:
    """Segment one well: final frame -> growth mask -> filtered frames -> IDs.

    The up-front checks (done marker, difference reference) have already
    run in :func:`run_segmentation`. Returns ``("ok", gate-skipped frames)``
    — the caller counts anything else as skipped.
    """
    well_dir = Path(options.out_dir) / well
    well_dir.mkdir(parents=True, exist_ok=True)
    t_well = time.monotonic()

    n = len(frames)
    last = frames[-1]
    log.info("[%s] %d frames, reference = LO%03d", well, n, last.frame)
    exclude_before = options.exclude_before
    if exclude_before is not None:
        _blanking_log(well, frames, exclude_before)

    # 1. Segment the final frame and build the growth mask.
    t0 = time.monotonic()
    raw_last, ref_frac = _segment_frame(
        segmenter,
        _load_frame(last, options.input),
        fg_threshold=options.fg_ref,
        min_size=options.min_size,
        allowed=None,
        ref_frac=None,  # the reference frame itself is never gated
        max_fg_ratio=options.max_fg_ratio,
        desc=f"[{well}]",
    )
    groups, mask = build_growth_mask(
        raw_last,
        min_size=options.min_size,
        min_solidity=options.min_solidity,
        mask_margin=options.mask_margin,
    )
    n_groups = 0 if groups is None else int(groups.max())
    log.info(
        "[%s] LO%03d (ref): %d groups, %.1fs",
        well, last.frame, n_groups, time.monotonic() - t0,
    )
    io.write_label_image(
        well_dir / "ref__labels.tif",
        groups if groups is not None else np.zeros(raw_last.shape, np.uint32),
    )

    # 2. All frames: segment, gate, keep what is inside the growth mask.
    kept: list[np.ndarray] = []
    disc_total, gate_skips = 0, 0
    for i, frame in enumerate(frames):
        t0 = time.monotonic()
        if exclude_before is not None and frame.frame < exclude_before:
            kept.append(np.zeros(raw_last.shape, np.uint32))
            log.info(
                "[%s] %d/%d LO%03d: before LO%03d (exclude_before), "
                "labels dropped",
                well, i + 1, n, frame.frame, exclude_before,
            )
            continue
        if i == n - 1:
            # already segmented above; the decoder is deterministic, so the
            # result is identical and the second model call is saved
            raw = raw_last
        else:
            image = _load_frame(frame, options.input)
            raw, elevated = _segment_frame(
                segmenter,
                image,
                fg_threshold=options.fg_early,
                min_size=options.min_size,
                allowed=mask,
                ref_frac=ref_frac,
                max_fg_ratio=options.max_fg_ratio,
                desc=f"[{well}] LO{frame.frame:03d}",
            )
            if drift_gate_skips(
                elevated, ref_frac, max_fg_ratio=options.max_fg_ratio
            ):
                gate_skips += 1
        if groups is not None:
            kept_frame, discarded = filter_to_mask(
                raw, mask, groups, options.min_overlap
            )
        else:
            kept_frame, discarded = np.zeros(raw_last.shape, np.uint32), 0
        kept.append(kept_frame)
        disc_total += discarded
        log.info(
            "[%s] %d/%d LO%03d: %d segments (%d discarded), %.1fs",
            well, i + 1, n, frame.frame, int(np.unique(kept_frame).size) - 1,
            discarded, time.monotonic() - t0,
        )

    # 3. Colony IDs: forward tracking within each group (+ pre-existing
    #    filter); the gate-skipped and blanked frames stay empty, 1:1.
    result = track_colonies(
        [(frame.frame, k) for frame, k in zip(frames, kept)],
        groups,
        exclude_before=exclude_before,
        merge_touching=options.merge_touching,
        discarded_instances=disc_total,
        desc=well,
    )
    _write_outputs(
        well_dir, frames, result.labels,
        _rows_with_frame_info(result.rows, frames, well),
    )
    io.write_done(well_dir, result.counters)
    log.info(
        "[%s] done: %d groups, %d colonies (%d pre-existing flagged), "
        "%d merge events, %d forks, %d discarded instances, %.1fs total",
        well, result.counters["groups"], result.counters["colonies"],
        result.counters["pre_existing_colonies"],
        result.counters["merge_events"], result.counters["forked_segments"],
        result.counters["discarded_instances"], time.monotonic() - t_well,
    )
    return "ok", gate_skips


def _reid_well(
    well: str, frames: Sequence[io.Frame], options: SegmentationOptions
) -> str:
    """Re-assign colony IDs from the saved outputs (no model, no GPU).

    The segments are re-derived from the saved label images (each label
    value re-split into connected components), tracking is re-run, and the
    label images, colonies.csv, and done marker are rewritten.
    """
    well_dir = Path(options.out_dir) / well
    if not io.is_done(well_dir):
        log.info("[%s] no completed outputs, skipping", well)
        return "skipped"
    t_well = time.monotonic()
    groups = io.read_label_image(well_dir / "ref__labels.tif")
    exclude_before = options.exclude_before
    if exclude_before is not None:
        _blanking_log(well, frames, exclude_before)
    if not groups.any():
        io.write_done(well_dir, {
            "frames": len(frames), "groups": 0, "colonies": 0,
            "pre_existing_colonies": 0, "merge_events": 0,
            "forked_segments": 0, "discarded_instances": 0,
        })
        log.info("[%s] empty reference, nothing to re-assign", well)
        return "skipped"
    kept = []
    for frame in frames:
        if exclude_before is not None and frame.frame < exclude_before:
            kept.append(np.zeros(groups.shape, np.uint32))
            continue
        labels = io.read_label_image(well_dir / f"{frame.stem}__labels.tif")
        local = np.zeros(labels.shape, np.uint32)
        for local_id, segment in enumerate(extract_segments(labels), start=1):
            local[segment.mask] = local_id
        kept.append(local)
    result = track_colonies(
        [(frame.frame, k) for frame, k in zip(frames, kept)],
        groups,
        exclude_before=exclude_before,
        merge_touching=options.merge_touching,
        desc=well,
    )
    _write_outputs(
        well_dir, frames, result.labels,
        _rows_with_frame_info(result.rows, frames, well),
    )
    io.write_done(well_dir, result.counters)
    log.info(
        "[%s] re-ID done: %d groups, %d colonies (%d pre-existing flagged), "
        "%d merge events, %d forks, %.1fs",
        well, result.counters["groups"], result.counters["colonies"],
        result.counters["pre_existing_colonies"],
        result.counters["merge_events"], result.counters["forked_segments"],
        time.monotonic() - t_well,
    )
    return "ok"


def run_segmentation(options: SegmentationOptions) -> SegmentationSummary:
    """Run the segment stage over the discovered wells.

    Discovers the wells in the input tree (``raw_dir`` for ``input="raw"``,
    ``diff_dir`` for ``input="diff"``; both layouts accepted), rejects
    misconfigured wells up front (missing input directory, requested wells
    that are not in the tree, a difference reference equal to the final
    frame), and processes the rest — the model-based full run, or the
    model-free re-ID when ``reid_only`` is set.

    Returns a :class:`SegmentationSummary` with the per-well outcome
    counts and the number of drift-gate-skipped frames.
    """
    if options.input not in ("raw", "diff"):
        raise ValueError(
            f"unknown input mode {options.input!r}: expected 'raw' or 'diff'"
        )
    if options.input == "raw" and options.raw_dir is None:
        raise ValueError("raw_dir is required when input = 'raw'")
    if options.input == "diff" and options.diff_dir is None:
        raise ValueError("diff_dir is required when input = 'diff'")
    input_dir = Path(options.raw_dir if options.input == "raw" else options.diff_dir)
    out_dir = Path(options.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wells = io.discover_wells(input_dir, wells=options.wells)
    if options.wells:
        missing = [w for w in options.wells if w not in wells]
        if missing:
            raise ValueError(f"wells not found in {input_dir}: {missing}")
    log.info("input = %s, outputs -> %s", input_dir, out_dir)
    log.info(
        "found %d wells, %d frames",
        len(wells), sum(len(fs) for fs in wells.values()),
    )

    summary = SegmentationSummary(wells=len(wells))
    t_run = time.monotonic()
    if options.reid_only:
        for i, (well, frames) in enumerate(wells.items()):
            log.info("== well %d/%d (re-ID) ==", i + 1, len(wells))
            try:
                if _reid_well(well, frames, options) == "ok":
                    summary.processed += 1
                else:
                    summary.skipped += 1
            except Exception:
                summary.failed += 1
                summary.failed_wells.append(well)
                log.exception("[%s] FAILED", well)
    else:
        # Up-front per-well checks — before the model is loaded, so a
        # misconfigured well costs nothing and no outputs are written.
        todo: list[tuple[str, list[io.Frame]]] = []
        for well, frames in wells.items():
            well_dir = out_dir / well
            if not options.redo and io.is_done(well_dir):
                log.info("[%s] already done, skipping (--redo to reprocess)", well)
                summary.skipped += 1
                continue
            if options.input == "diff":
                ref_lo, _ = _diff_reference(Path(options.diff_dir) / well)
                if ref_lo is None:
                    log.error(
                        "[%s] no scale.txt with a reference line under %s; "
                        "skipping well",
                        well, Path(options.diff_dir) / well,
                    )
                    summary.skipped += 1
                    continue
                if ref_lo >= frames[-1].frame:
                    log.error(
                        "[%s] difference reference is LO%03d, the final frame "
                        "itself; the growth mask would be empty. Re-run the "
                        "difference stage with an earlier --reference.",
                        well, ref_lo,
                    )
                    summary.skipped += 1
                    continue
            todo.append((well, frames))

        if todo:
            t0 = time.monotonic()
            segmenter = _load_segmenter(options.model_type, options.checkpoint)
            log.info(
                "loaded model %s%s in %.1fs", options.model_type,
                f" + {options.checkpoint}" if options.checkpoint else "",
                time.monotonic() - t0,
            )
        else:
            segmenter = None
        for i, (well, frames) in enumerate(todo):
            log.info("== well %d/%d ==", i + 1, len(todo))
            try:
                status, gate_skips = _process_well(well, frames, segmenter, options)
                if status == "ok":
                    summary.processed += 1
                    summary.gate_skipped_frames += gate_skips
                else:
                    summary.skipped += 1
            except Exception:
                summary.failed += 1
                summary.failed_wells.append(well)
                log.exception("[%s] FAILED", well)

    log.info(
        "run finished in %.1fs; %d wells, %d processed, %d skipped, "
        "%d failed%s",
        time.monotonic() - t_run, len(wells), summary.processed,
        summary.skipped, summary.failed,
        f": {summary.failed_wells}" if summary.failed_wells else "",
    )
    return summary
