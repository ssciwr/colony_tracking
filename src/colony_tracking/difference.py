"""Difference stage: per-well difference frames.

Each well of the plate is a time series of repeated acquisitions of the same
field of view, with the bacteria growing between frames. One reference frame
per well is subtracted from every frame, so the result highlights where the
colonies changed rather than re-showing the static plate illumination. The
reference is the well's first frame by default (``reference = "first"``), or
a fixed 1-based frame number.

Algorithm (the prototype's plain-subtraction path; the flat-field correction
and the int16 output variant are cut from the package):

- Every frame is loaded as float32 intensity counts, so the subtraction
  ``cur - ref`` cannot wrap around.
- One display limit per well: the ``display_percentile`` (99.5) percentile
  of |diff|, estimated on the subsampled grid (stride ``subsample``),
  floored at 1.0 so an unchanged well still gets a defined mapping. Only the
  small strided samples are kept while estimating, so a full-resolution
  difference is never held more than one pass at a time. The limit is
  recorded in the well's ``scale.txt`` as ``display_limit_int16`` so the
  uint8 previews can be converted back to counts (at the cost of one
  quantisation step of ``2 * limit / 255`` counts).
- Each frame is written as a uint8 preview ``<frame-stem>_diff_uint8.tif``:
  ``128 + diff * 127 / limit``, clamped to 0..255, so **128 = no change**
  (ADR 0001 — the 8-bit form is the internal contract between stages). The
  reference frame itself is written as an all-128 frame, so each well folder
  stays a complete, evenly numbered time series.

Polarity: the subtraction is always ``cur - ref``. Frames acquired BEFORE
the reference come out polarity-flipped: where colonies later grow, they
read as bright (> 128) in those frames instead of dark (< 128). That is
intended and unchanged from the prototype — the uint8 mapping is
sign-agnostic (128 = no change either way), so pre- and post-reference
frames stay directly comparable as "how much did this pixel change".

Runs on CPU only; wells are processed in a thread pool of ``n_workers``.
A well whose output directory holds a done marker (``_done.txt``) is skipped
on re-run, so an interrupted run resumes instead of restarting. ``dry_run``
reports the discovered wells, frame counts, and chosen reference per well
and writes nothing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile

from . import io, log
from ._options import DifferenceOptions, Reference


#: A well's outcome in a difference run.
WellStatus = Literal["processed", "skipped", "failed"]


@dataclass(frozen=True)
class WellResult:
    """The outcome of one well in a difference run.

    ``status`` is ``processed`` (written this run), ``skipped`` (already
    complete via its done marker, or a dry run), or ``failed`` (an error was
    caught and recorded in ``error``). ``limit`` is the well's display limit
    (None when no work was done).
    """

    well: str
    status: WellStatus
    n_frames: int
    limit: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class DifferenceSummary:
    """Per-well outcomes and totals for one difference run.

    Returned by :func:`run_difference` so a caller (CLI, TUI) can update
    status without parsing log text (spec user story 40).
    """

    wells: dict[str, WellResult]
    processed: int
    skipped: int
    failed: int


def _load(path: Path) -> np.ndarray:
    """Load a frame as float32 so subtraction cannot wrap around."""
    return np.asarray(tifffile.imread(path), dtype=np.float32)


def _resolve_reference(frames: list[io.Frame], reference: Reference) -> io.Frame:
    """The well's reference frame (KeyError when a numbered frame is missing)."""
    if reference == "first":
        return frames[0]
    for frame in frames:
        if frame.frame == reference:
            return frame
    raise KeyError(reference)


def _process_well(
    well: str,
    frames: list[io.Frame],
    reference_frame: io.Frame,
    options: DifferenceOptions,
) -> WellResult:
    """Difference one well's time series against its reference frame."""
    out_dir = Path(options.out_dir) / well
    if io.is_done(out_dir):
        log.info("difference: %s: complete — skipping", well)
        return WellResult(well=well, status="skipped", n_frames=len(frames))

    out_dir.mkdir(parents=True, exist_ok=True)
    ref = _load(reference_frame.path)

    # One display scale for the whole well, so frames stay comparable over
    # time. The percentile is estimated from every frame except the reference
    # itself (its difference is all zeros by construction). Only the strided
    # samples are kept, so the full-resolution differences are not held at
    # once; io.display_limit is then fed those samples with stride 1 (the
    # same sample set, no second copy of the full frames).
    samples: list[np.ndarray] = []
    for frame in frames:
        if frame.name == reference_frame.name:
            continue
        cur = _load(frame.path)
        d = cur - ref
        samples.append(np.abs(d)[:: options.subsample, :: options.subsample].copy())
        del cur, d
    limit = io.display_limit(samples, options.display_percentile, 1)
    del samples

    io.write_scale(out_dir, reference_frame.name, limit)

    written = 0
    for frame in frames:
        if frame.name == reference_frame.name:
            # The reference frame, written as an all-128 frame so each well
            # stays a complete, evenly numbered series.
            diff = np.zeros(ref.shape, dtype=np.float32)
        else:
            cur = _load(frame.path)
            diff = cur - ref
            del cur
        out_name = f"{Path(frame.name).stem}_diff_uint8.tif"
        tifffile.imwrite(
            out_dir / out_name, io.map_to_uint8(diff, limit), compression="zlib"
        )
        del diff
        log.info("difference: %s: %s", well, out_name)
        written += 1

    io.write_done(
        out_dir,
        {
            "frames": written,
            "reference": reference_frame.name,
            "display_limit": repr(float(limit)),
        },
    )
    log.info(
        "difference: %s: %d frames, |diff| p%s = %.3f",
        well, written, options.display_percentile, limit,
    )
    return WellResult(well=well, status="processed", n_frames=written, limit=limit)


def run_difference(options: DifferenceOptions) -> DifferenceSummary:
    """Run the difference stage over a raw plate (see the module docstring).

    Discovers the wells and their frames in ``options.raw_dir`` (flat or
    per-well layout), differences each well against its reference frame, and
    writes ``<out_dir>/<well>/`` trees of uint8 difference frames plus
    ``scale.txt`` and a done marker per well.

    Returns a :class:`DifferenceSummary` with one :class:`WellResult` per
    well. Raises ``FileNotFoundError`` for a missing input tree and
    ``ValueError`` for a tree with no matching frames or a reference frame
    missing from a well — both before anything is written.
    """
    if options.subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {options.subsample}")
    if options.n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {options.n_workers}")

    frames_by_well = io.discover_wells(options.raw_dir, wells=options.wells)
    if not frames_by_well:
        message = f"no frames found in {options.raw_dir}"
        if options.wells:
            message += f" for the requested wells {sorted(options.wells)}"
        log.error("difference: %s", message)
        raise ValueError(message)

    # Resolve every well's reference frame up front, so a well without the
    # requested frame aborts the run before anything is written.
    references: dict[str, io.Frame] = {}
    missing: list[str] = []
    for well, frames in frames_by_well.items():
        try:
            references[well] = _resolve_reference(frames, options.reference)
        except KeyError:
            missing.append(well)
    if missing:
        message = (
            f"reference frame {options.reference!r} not present in "
            f"{len(missing)} well(s): {' '.join(sorted(missing))}"
        )
        log.error("difference: %s", message)
        raise ValueError(message)

    mode = (
        "first frame per well"
        if options.reference == "first"
        else f"frame {options.reference} per well"
    )
    total = sum(len(frames) for frames in frames_by_well.values())
    log.info(
        "difference: found %d well(s), %d frame(s) (reference = %s)",
        len(frames_by_well), total, mode,
    )
    for well in sorted(frames_by_well):
        log.info(
            "difference: %s: %d frames, reference LO%03d",
            well, len(frames_by_well[well]), references[well].frame,
        )

    if options.dry_run:
        log.info("difference: dry run — nothing written")
        return _summarize(
            {
                well: WellResult(well=well, status="skipped", n_frames=len(frames))
                for well, frames in frames_by_well.items()
            }
        )

    log.info(
        "difference: writing to %s (display scale = p%s of |diff|)",
        options.out_dir, options.display_percentile,
    )
    results: dict[str, WellResult] = {}
    with ThreadPoolExecutor(max_workers=options.n_workers) as executor:
        futures = {
            executor.submit(_process_well, well, frames, references[well], options): well
            for well, frames in sorted(frames_by_well.items())
        }
        for future in as_completed(futures):
            well = futures[future]
            try:
                results[well] = future.result()
            except Exception as exc:  # a failed well must not kill the run
                log.error("difference: %s failed: %s", well, exc)
                results[well] = WellResult(
                    well=well,
                    status="failed",
                    n_frames=len(frames_by_well[well]),
                    error=str(exc),
                )
    summary = _summarize(results)
    log.info(
        "difference: done — %d frames across %d well(s) in %s "
        "(%d processed, %d skipped, %d failed)",
        sum(r.n_frames for r in summary.wells.values() if r.status == "processed"),
        len(summary.wells), options.out_dir,
        summary.processed, summary.skipped, summary.failed,
    )
    return summary


def _summarize(results: dict[str, WellResult]) -> DifferenceSummary:
    """Count per-well outcomes into a :class:`DifferenceSummary`."""

    def count(status: WellStatus) -> int:
        return sum(1 for r in results.values() if r.status == status)

    return DifferenceSummary(
        wells=dict(sorted(results.items())),
        processed=count("processed"),
        skipped=count("skipped"),
        failed=count("failed"),
    )
