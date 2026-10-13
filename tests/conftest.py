"""Shared test scaffolding for the colony_tracking suite.

The suite is hermetic: no real plate data, no GPU, no network.

Synthetic fixtures (plain factory functions — a test builds only what it
needs, so construction stays lazy):

- ``build_mini_plate`` builds a synthetic acquisition tree of 2048x2048
  uint16 frames (the real plate size — the circle-fit constants assume it)
  with the real acquisition file-name vocabulary and millisecond ``T``
  fields, a planted dark meniscus ring, and planted colony disks (a pair
  that grows until it touches, a static disk, one present from frame 1,
  one approaching the rim). It returns the tree plus the planted ground
  truth (positions, radii/areas per frame, ring parameters, T values).
- ``build_label_series`` writes synthetic integer label TIFFs (no model
  involved) for tracking/segmentation tests, with canonical frame names.
- ``build_labeled_plate`` builds a small plate (default 512x512) with the
  raw tree plus a ground-truth label tree of planted-disk labels mirroring
  the raw layout, for the finetune conversion tests.
- ``disk_mask`` returns the exact boolean pixel mask the builder painted
  for one disk at one frame (clipped to the well interior), so tests have
  the planted ground-truth colony area.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

# Force the non-interactive Agg backend before any test imports plotting code.
matplotlib.use("Agg")

# --- the real plate's constants ---------------------------------------------

#: The real plate size. The circle-fit constants are calibrated to it;
#: the default planting is laid out for it — do not shrink.
PLATE_SIZE = 2048
#: Background intensity of the synthetic frames (the real plate is ~5e3..4.5e4).
BASE_INTENSITY = 5000.0
#: Per-frame shot-noise sigma (counts); 0 for fully deterministic plates.
NOISE_SIGMA = 25.0
#: The real plate's frames are hourly; T fields carry milliseconds.
T_STEP_MS = 3_600_000
#: The default mini-plate: one "full" well and one "healthy" well.
DEFAULT_WELLS = ("-A001", "-A002")


def frame_name(well: str, lo: int, t_ms: int) -> str:
    """The canonical raw-plate frame file name (real acquisition vocabulary).

    ``-<A-H>ddd--PO01--LO<nnn>--CO1--SL001--PX32500--PW0050--IN0020--TM366
    --X014222--Y011210--Z220046--T<ms, 10 digits>--WE00001.tif``
    """
    return (
        f"{well}--PO01--LO{lo:03d}--CO1--SL001--PX32500--PW0050--IN0020"
        f"--TM366--X014222--Y011210--Z220046--T{t_ms:010d}--WE00001.tif"
    )


# --- reading written outputs --------------------------------------------------


def read_marker(well_dir: Path | str) -> dict[str, int | str]:
    """Read a well's done marker (tab-separated ``key<TAB>value`` lines).

    Values parse as ints when possible and stay strings otherwise; absent
    keys are simply missing. Test-side stand-in for the removed
    ``io.read_done`` — the format it asserts is the written output's.
    """
    # Deferred: the package import chain pulls plotting code, which must
    # not load before the Agg backend is forced below.
    from colony_tracking import io

    values: dict[str, int | str] = {}
    with open(io.done_marker_path(well_dir)) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, _, value = line.partition("\t")
            try:
                values[key] = int(value)
            except ValueError:
                values[key] = value
    return values


# --- the planted objects ------------------------------------------------------


@dataclass(frozen=True)
class RingSpec:
    """A planted dark meniscus ring (the well's circle).

    Painted as a Gaussian dip: ``depth`` counts below the local background
    at radius ``R``, falling off with radial sigma ``width`` (px).
    """

    cy: float = 1024.0
    cx: float = 1024.0
    R: float = 920.0
    width: float = 12.0
    depth: float = 2500.0


@dataclass(frozen=True)
class DiskSpec:
    """A planted colony disk: a dark disk that grows geometrically.

    The radius at 1-based frame ``t`` is ``r0 * growth ** (t - first_frame)``
    for ``first_frame <= t <= last_frame`` (``last_frame`` defaults to the
    plate's final frame when the builder resolves it). ``area_at`` is the
    area of the *unclipped* planted disk; where a disk reaches the well
    wall, the painted (and ground-truth) area is smaller — use
    ``disk_mask`` for the exact painted mask.
    """

    name: str
    cy: float
    cx: float
    r0: float
    growth: float = 1.0
    first_frame: int = 1
    last_frame: int | None = None
    delta: float = 2000.0  # intensity drop vs local background (counts)

    def radius_at(self, frame: int) -> float | None:
        """The planted radius at a 1-based frame, or None if absent there."""
        if frame < self.first_frame:
            return None
        if self.last_frame is not None and frame > self.last_frame:
            return None
        return self.r0 * self.growth ** (frame - self.first_frame)

    def area_at(self, frame: int) -> float | None:
        """The planted (unclipped) disk area at a 1-based frame, or None."""
        r = self.radius_at(frame)
        return None if r is None else float(np.pi * r * r)


def _default_disks(well_index: int) -> list[DiskSpec]:
    """Default planting (calibrated for 2048x2048, ring R 920 at the centre).

    Well 0 is the "full" well: a pair that grows until it touches (merge),
    a static disk (no_growth), one present from frame 1 (from_start), and
    one that grows into the rim zone (monotone rim exclusion). Every other
    well is "healthy": one clean exponential disk.
    """
    if well_index == 0:
        return [
            DiskSpec("pair_a", cy=650.0, cx=750.0, r0=30.0, growth=1.3,
                     first_frame=2),
            DiskSpec("pair_b", cy=650.0, cx=950.0, r0=30.0, growth=1.3,
                     first_frame=2),
            DiskSpec("static", cy=1450.0, cx=1450.0, r0=50.0, growth=1.0,
                     first_frame=2),
            DiskSpec("from_start", cy=600.0, cx=1500.0, r0=30.0,
                     growth=1.25, first_frame=1),
            DiskSpec("rim", cy=1024.0, cx=1824.0, r0=40.0, growth=1.35,
                     first_frame=3),
        ]
    return [
        DiskSpec("growing", cy=1024.0, cx=600.0, r0=40.0, growth=1.35,
                 first_frame=3),
    ]


@dataclass
class MiniPlate:
    """A built synthetic plate: the tree on disk plus its ground truth."""

    root: Path
    layout: str  # "flat" or "per_well"
    wells: dict[str, list[str]]  # well -> frame file names, in frame order
    ring: dict[str, tuple[float, float, float] | None]  # well -> (cy, cx, R)
    disks: dict[str, list[DiskSpec]]  # well -> planted disks (last_frame set)
    base: float
    noise_sigma: float
    gradient: tuple[float, float]
    size: int
    n_frames: int
    t0_ms: int
    t_step_ms: int

    @property
    def t_values(self) -> list[int]:
        """The T field (ms) of every frame, in frame order."""
        return [self.t0_ms + i * self.t_step_ms for i in range(self.n_frames)]

    def disk(self, well: str, name: str) -> DiskSpec:
        """The planted disk of ``well`` called ``name`` (KeyError if absent)."""
        for d in self.disks[well]:
            if d.name == name:
                return d
        raise KeyError(f"no disk named {name!r} in well {well}")


def build_mini_plate(
    out_dir: Path | str,
    layout: str = "flat",
    wells: Sequence[str] = DEFAULT_WELLS,
    n_frames: int = 8,
    size: int = PLATE_SIZE,
    t0_ms: int = 0,
    t_step_ms: int = T_STEP_MS,
    base: float = BASE_INTENSITY,
    noise_sigma: float = NOISE_SIGMA,
    seed: int = 0,
    rings: Mapping[str, RingSpec | None] | RingSpec | None = None,
    disks: Mapping[str, Sequence[DiskSpec]] | None = None,
    gradient: tuple[float, float] = (0.0, 0.0),
) -> MiniPlate:
    """Build a synthetic mini-plate acquisition tree under ``out_dir``.

    ``layout`` is ``"flat"`` (all frames in one directory) or ``"per_well"``
    (one subdirectory per well). Every parameter is overridable so later
    tickets can build targeted plates: the wells, the frame count, the
    T fields (milliseconds), the background/noise, the planted ring(s)
    (a shared ``RingSpec``, a per-well mapping, or ``None`` per well for
    no ring) and the planted disks (a per-well mapping of ``DiskSpec``;
    omitted wells get no disks). The defaults use 2 wells x 8 frames of
    2048x2048 uint16 with the default planting of ``_default_disks`` —
    about 200 MB uncompressed, so call it with only what a test needs.

    ``gradient`` is a smooth illumination slope (total counts across the
    image, per axis) identical in every frame. Returns the :class:`MiniPlate`
    ground truth.
    """
    if layout not in ("flat", "per_well"):
        raise ValueError(f"unknown layout {layout!r}: expected 'flat' or 'per_well'")
    if not wells:
        raise ValueError("at least one well is required")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Resolve the per-well ring and disk specs (defaults fill the gaps).
    if isinstance(rings, RingSpec):
        ring_by_well: dict[str, RingSpec | None] = {w: rings for w in wells}
    elif rings is not None:
        ring_by_well = dict(rings)
    else:
        ring_by_well = {}
    for w in wells:
        ring_by_well.setdefault(w, RingSpec())

    disk_by_well: dict[str, list[DiskSpec]] = {}
    for i, w in enumerate(wells):
        specs = _default_disks(i) if disks is None else list(disks.get(w, []))
        disk_by_well[w] = [
            dataclasses.replace(d, last_frame=d.last_frame if d.last_frame is not None
                                else n_frames)
            for d in specs
        ]

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    gradient_field = (
        gradient[0] / (size - 1) * yy + gradient[1] / (size - 1) * xx
    ).astype(np.float32)

    plate = MiniPlate(
        root=out_dir, layout=layout, wells={}, ring={}, disks=disk_by_well,
        base=base, noise_sigma=noise_sigma, gradient=gradient, size=size,
        n_frames=n_frames, t0_ms=t0_ms, t_step_ms=t_step_ms,
    )

    for w in wells:
        ring = ring_by_well[w]
        well_dir = out_dir if layout == "flat" else out_dir / w
        if layout == "per_well":
            well_dir.mkdir(exist_ok=True)

        # Static per-well fields (the ring and disk geometry do not move).
        ring_field = None
        interior = None
        if ring is not None:
            d_ring = np.hypot(yy - ring.cy, xx - ring.cx).astype(np.float32)
            ring_field = (
                ring.depth
                * np.exp(-0.5 * ((d_ring - ring.R) / ring.width) ** 2)
            ).astype(np.float32)
            interior = d_ring < ring.R
        disk_fields = [
            (d, np.hypot(yy - d.cy, xx - d.cx).astype(np.float32))
            for d in disk_by_well[w]
        ]

        names = []
        for t in range(1, n_frames + 1):
            img = np.full((size, size), base, dtype=np.float32)
            img += gradient_field
            if ring_field is not None:
                img -= ring_field
            for d, d_disk in disk_fields:
                radius = d.radius_at(t)
                if radius is None:
                    continue
                mask = d_disk <= radius
                if interior is not None:  # colonies stop at the well wall
                    mask &= interior
                img -= d.delta * mask
            if noise_sigma > 0:
                img += rng.normal(0.0, noise_sigma, size=(size, size)).astype(
                    np.float32)
            name = frame_name(w, t, t0_ms + (t - 1) * t_step_ms)
            tifffile.imwrite(well_dir / name,
                             np.clip(img, 0, 65535).astype(np.uint16),
                             compression="zlib")
            names.append(name)
        plate.wells[w] = names
        plate.ring[w] = (ring.cy, ring.cx, ring.R) if ring is not None else None
    return plate


def disk_mask(
    plate: MiniPlate, well: str, disk_name: str, frame: int
) -> np.ndarray | None:
    """The boolean pixel mask the builder painted for one disk at one frame.

    None when the disk is absent at the frame (before ``first_frame`` or
    after ``last_frame``). Like the painted frame, the disk is clipped to
    the well interior (inside the planted ring), so ``mask.sum()`` is the
    ground-truth colony area for that frame.
    """
    disk = plate.disk(well, disk_name)
    radius = disk.radius_at(frame)
    if radius is None:
        return None
    size = plate.size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mask = np.hypot(yy - disk.cy, xx - disk.cx) <= radius
    ring = plate.ring.get(well)
    if ring is not None:
        cy, cx, R = ring
        mask &= np.hypot(yy - cy, xx - cx) < R
    return mask


@dataclass
class LabelSeries:
    """A written synthetic label series (no model involved)."""

    root: Path
    names: list[str]  # the input frame names (canonical vocabulary)
    label_paths: list[Path]  # the written <frame-stem>__labels.tif files


@dataclass
class LabeledPlate:
    """A synthetic plate with raw frames plus planted-disk ground-truth labels.

    The raw tree (``raw_root``) is a ``build_mini_plate`` tree (noise on,
    no ring). The label tree (``label_root``) mirrors the raw layout: for
    every well in ``disks``, one label TIFF per frame carrying the raw
    frame's file name (uint32; 0 = background; 1..N = the well's disks in
    planting order, so ``int(label.max())`` is the instance count). Wells
    not in ``disks`` have raw frames but no label files — the finetune
    conversion pairs by file stem, so they stay out of the image set.
    """

    plate: MiniPlate  # the raw tree's ground truth
    raw_root: Path
    label_root: Path
    layout: str  # "flat" or "per_well"
    label_paths: dict[str, list[Path]]  # well -> label files, in frame order
    instances: dict[str, list[int]]  # well -> instance count per frame


def build_labeled_plate(
    out_dir: Path | str,
    *,
    wells: Sequence[str],
    n_frames: int,
    disks: Mapping[str, Sequence[DiskSpec]],
    size: int = 512,
    layout: str = "flat",
    noise_sigma: float = NOISE_SIGMA,
    seed: int = 0,
) -> LabeledPlate:
    """Build a small synthetic plate with ground-truth labels under ``out_dir``.

    Raw frames go to ``out_dir/raw``, the label tree to ``out_dir/labels``.
    Each well in ``disks`` gets a label file per frame: the disk masks the
    builder painted for the raw frames (``disk_mask``), painted with values
    1..N in planting order. A well absent from ``disks`` is unlabeled (raw
    frames only). ``size`` defaults to 512 so the fixture stays small; the
    disk specs are then laid out for that size by the caller.
    """
    raw_root = Path(out_dir) / "raw"
    label_root = Path(out_dir) / "labels"
    plate = build_mini_plate(
        raw_root,
        layout=layout,
        wells=wells,
        n_frames=n_frames,
        size=size,
        noise_sigma=noise_sigma,
        seed=seed,
        rings={w: None for w in wells},
        disks=disks,
    )
    label_paths: dict[str, list[Path]] = {}
    instances: dict[str, list[int]] = {}
    for w in wells:
        if w not in disks:
            continue
        specs = list(disks[w])
        well_dir = label_root if layout == "flat" else label_root / w
        well_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        counts: list[int] = []
        for t in range(1, n_frames + 1):
            label = np.zeros((size, size), dtype=np.uint32)
            n_inst = 0
            for i, d in enumerate(specs):
                mask = disk_mask(plate, w, d.name, t)
                if mask is not None:
                    n_inst += 1
                    label[mask] = i + 1
            path = well_dir / plate.wells[w][t - 1]
            tifffile.imwrite(path, label, compression="zlib")
            paths.append(path)
            counts.append(n_inst)
        label_paths[w] = paths
        instances[w] = counts
    return LabeledPlate(
        plate=plate,
        raw_root=raw_root,
        label_root=label_root,
        layout=layout,
        label_paths=label_paths,
        instances=instances,
    )


def build_label_series(
    out_dir: Path | str,
    labels: Sequence[np.ndarray],
    well: str = "-A001",
    names: Sequence[str] | None = None,
    lo_start: int = 1,
    t0_ms: int = 0,
    t_step_ms: int = T_STEP_MS,
) -> LabelSeries:
    """Write one synthetic label series (integer label TIFFs) to ``out_dir``.

    ``labels`` is one uint32 (or integer) label array per frame; each is
    written as ``<frame-stem>__labels.tif`` (0 = background). ``out_dir``
    is the well directory the series is written into. ``names`` gives the
    input frame names (defaults to canonical names for ``well`` starting at
    ``lo_start``); the in-memory ``names`` list is what tracking/segmentation
    tests pass as the frame list.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    if names is None:
        names = [frame_name(well, lo_start + i, t0_ms + i * t_step_ms)
                 for i in range(n)]
    if len(names) != n:
        raise ValueError(f"{len(names)} names for {n} label images")
    label_paths = []
    for name, labels_arr in zip(names, labels):
        stem = name[:-4] if name.endswith(".tif") else name
        path = out_dir / f"{stem}__labels.tif"
        tifffile.imwrite(
            path,
            np.ascontiguousarray(labels_arr).astype(np.uint32, copy=False),
            compression="zlib",
        )
        label_paths.append(path)
    return LabelSeries(root=out_dir, names=list(names), label_paths=label_paths)
