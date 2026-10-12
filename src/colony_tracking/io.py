"""Input-tree discovery and shared I/O for every stage.

This module is the shared input/output layer the stages build on. It knows
the two accepted input-tree layouts (flat and per-well), parses the
well/frame/time metadata from the acquisition file names, and provides the
shared image math (the 1/99 percentile stretch and the 128-neutral
difference mapping with its display-limit estimator) plus the small on-disk
formats every stage reads and writes: label images, colonies.csv (including
the prototype's legacy column names), the per-well done marker, the
per-well ``scale.txt`` of a difference tree, and the shared
``well_geometry.csv`` cache.

It holds no stage logic — discovery, parsing, image math, and file formats
only. Every path is an explicit argument; nothing here defaults to a data
location.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from . import log

#: Suffixes of the frame files accepted during discovery.
FRAME_SUFFIXES = (".tif", ".tiff")

#: The well field leading a flat-layout file name, e.g. ``-A001``.
WELL_RE = re.compile(r"^(-[A-H]\d{3})")
#: The 1-based frame number, e.g. ``--LO005--``.
FRAME_RE = re.compile(r"--LO(\d+)--")
#: The acquisition-time field (raw value; milliseconds in this dataset).
TIME_RE = re.compile(r"--T(\d+)--")


def parse_well(name: str) -> str | None:
    """Return the well field (e.g. ``-A001``) leading a file name, or None.

    Only a well field at the very start of the name counts (a per-well
    directory name or the leading field of a flat-layout file name).
    """
    m = WELL_RE.match(name)
    return m.group(1) if m else None


def parse_frame(name: str) -> int | None:
    """Return the 1-based frame number from the ``LO<nnn>`` field, or None."""
    m = FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def parse_time(name: str) -> int | None:
    """Return the raw ``T`` field (acquisition time) as an int, or None."""
    m = TIME_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class Frame:
    """One frame of one well, located in an input tree."""

    well: str  # well field, e.g. "-A001"
    frame: int  # 1-based frame number (the LO field)
    t: int  # raw value of the T field (milliseconds in this dataset)
    name: str  # the frame's file name
    path: Path  # full path to the frame file


def discover_wells(
    root: Path | str,
    suffixes: Sequence[str] = FRAME_SUFFIXES,
    wells: Sequence[str] | None = None,
) -> dict[str, list[Frame]]:
    """Discover the wells and their frames under an input tree.

    Accepts both input-tree layouts: *per-well* (one subdirectory per well;
    the well is the directory name) and *flat* (all wells' frames in one
    directory; the well is parsed from the leading ``-<A-H>\\d{3}`` of the
    file name). A tree with well-named subdirectories is read as per-well;
    otherwise the files directly under ``root`` are read as flat. Files
    without a parseable ``LO<nnn>`` field (e.g. ``ref__labels.tif``) and
    resource forks (``._*``) are not frames and are skipped.

    Returns ``{well: [frames]}`` with each well's frames sorted by frame
    number (ties by the T field).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no such input tree: {root}")
    suffixes = tuple(s.lower() for s in suffixes)

    well_dirs = [
        entry
        for entry in sorted(root.iterdir())
        if entry.is_dir() and parse_well(entry.name) is not None
    ]
    sources = (
        [(well_dir.name, well_dir.iterdir()) for well_dir in well_dirs]
        if well_dirs
        else [(None, root.iterdir())]
    )

    frames: dict[str, list[Frame]] = {}
    for well_name, entries in sources:
        for entry in sorted(entries):
            if not entry.is_file() or entry.name.startswith("._"):
                continue
            if entry.suffix.lower() not in suffixes:
                continue
            well = well_name if well_name is not None else parse_well(entry.name)
            frame = parse_frame(entry.name)
            if well is None or frame is None:
                log.debug("skipping %s: not a frame file", entry.name)
                continue
            t = parse_time(entry.name) or 0
            frames.setdefault(well, []).append(
                Frame(well=well, frame=frame, t=t, name=entry.name, path=entry)
            )
    if wells is not None:
        requested = set(wells)
        frames = {w: fs for w, fs in frames.items() if w in requested}
    return {
        well: sorted(fs, key=lambda f: (f.frame, f.t))
        for well, fs in sorted(frames.items())
    }


def stretch_uint8(image: np.ndarray | Sequence[float]) -> np.ndarray:
    """Percentile-stretch an image (uint16 or any float) to uint8.

    Clips at the 1st and 99th percentiles of the image and maps the range to
    0..255 — the shared stretch that feeds uint8-based tools (the segment
    stage and the finetune data conversion). A flat image comes out
    all-zero.
    """
    im = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(im, (1, 99))
    if hi <= lo:  # flat image; avoid division by zero
        return np.zeros(im.shape, dtype=np.uint8)
    return np.clip((im - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def map_to_uint8(diff: np.ndarray, limit: float) -> np.ndarray:
    """Map a signed difference to uint8 with 128 = no change.

    ``128 + diff * 127 / limit``, clamped to 0..255: with the well's display
    limit, +/-limit counts map to the edges of the 8-bit range and a zero
    difference maps to exactly 128 (ADR 0001).
    """
    if limit <= 0:
        raise ValueError(f"display limit must be positive, got {limit!r}")
    scaled = 128.0 + np.asarray(diff) * (127.0 / limit)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def display_limit(
    diffs: np.ndarray | Sequence[np.ndarray],
    percentile: float = 99.5,
    subsample: int = 8,
) -> float:
    """Estimate a difference run's display limit.

    The ``percentile``-th percentile of |diff|, estimated on the subsampled
    grid (stride ``subsample``) of one difference frame or a sequence of
    them (one per frame). The result is floored at 1.0 so a well with no
    change still gets a defined mapping (and the reference frame, whose
    difference is all zero, can be included or skipped at will).
    """
    if isinstance(diffs, np.ndarray):
        diffs = [diffs]
    samples = []
    for d in diffs:
        arr = np.abs(np.asarray(d, dtype=np.float32))
        if arr.size:
            samples.append(arr[::subsample, ::subsample].ravel())
    if not samples:
        return 1.0
    return max(float(np.percentile(np.concatenate(samples), percentile)), 1.0)


#: The per-well done-marker file name. Presence marks the well complete.
DONE_MARKER = "_done.txt"

#: The canonical segment-stage marker keys (spec: I/O contract).
SEGMENT_DONE_KEYS = (
    "frames",
    "groups",
    "colonies",
    "pre_existing_colonies",
    "merge_events",
    "forked_segments",
    "discarded_instances",
)


def write_scale(well_dir: Path | str, reference: str, display_limit: float) -> Path:
    """Write a difference well's ``scale.txt`` (tab-separated key/value).

    Records ``reference`` (the reference frame's file name) and
    ``display_limit_int16`` (the float display limit), so the uint8
    previews can be converted back to counts.
    """
    path = Path(well_dir) / "scale.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"reference\t{reference}\n")
        f.write(f"display_limit_int16\t{float(display_limit)!r}\n")
    return path


def read_scale(well_dir: Path | str) -> dict[str, str]:
    """Read a difference well's ``scale.txt`` (empty dict when absent).

    Tolerates missing and extra lines (the prototype also wrote
    ``flatfield``/``sigma``).
    """
    path = Path(well_dir) / "scale.txt"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                key, _, value = line.partition("\t")
                values[key] = value
    return values


def done_marker_path(well_dir: Path | str) -> Path:
    """Path of the well's done marker."""
    return Path(well_dir) / DONE_MARKER


def is_done(well_dir: Path | str) -> bool:
    """True when the well's done marker is present (the well is complete)."""
    return done_marker_path(well_dir).exists()


def write_done(well_dir: Path | str, counters: Mapping[str, int | str]) -> Path:
    """Write the per-well done marker (tab-separated ``key<TAB>value`` lines)."""
    path = done_marker_path(well_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for key, value in counters.items():
            f.write(f"{key}\t{value}\n")
    return path


def read_done(well_dir: Path | str) -> dict[str, int | str]:
    """Read the per-well done marker; absent keys are simply missing.

    Values are parsed as ints when possible, kept as strings otherwise.
    """
    path = done_marker_path(well_dir)
    values: dict[str, int | str] = {}
    with open(path) as f:
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


#: The canonical colonies.csv columns (spec: I/O contract).
COLONIES_COLUMNS = (
    "well", "frame", "t_seconds", "image", "colony_id", "group_id",
    "merged_with", "pre_existing", "area_px", "centroid_y", "centroid_x",
)

#: The prototype's legacy column names, mapped to their canonical names.
#: New writes always use the canonical names; readers accept both.
LEGACY_COLUMNS = {
    "lo": "frame",
    "n_merged": "ancestors",
    "max_n_merged": "max_ancestors",
    "first_lo": "first_frame",
    "last_lo": "last_frame",
    "rim_excluded_lo": "rim_excluded_frame",
}


def write_csv_rows(
    path: Path | str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> Path:
    """Write rows (mappings) under the given header as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(fields))
        for row in rows:
            values = [row.get(k) for k in fields]
            writer.writerow(["" if v is None else v for v in values])
    return path


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """Read a CSV, mapping the prototype's legacy column names to canonical.

    Values are returned as strings; callers coerce them.
    """
    with open(path, newline="") as f:
        return [
            {LEGACY_COLUMNS.get(key, key): value for key, value in row.items()
             if key is not None}
            for row in csv.DictReader(f)
        ]


def write_colonies(path: Path | str, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write one well's colonies.csv (canonical columns)."""
    return write_csv_rows(path, rows, COLONIES_COLUMNS)


def read_colonies(path: Path | str) -> list[dict[str, str]]:
    """Read one well's colonies.csv (or a prototype-written one) canonically."""
    return read_csv_rows(path)


#: The shared well-geometry cache columns (spec: I/O contract).
GEOMETRY_COLUMNS = ("well", "cy", "cx", "R")


def read_geometry(path: Path | str) -> dict[str, tuple[float, float, float]]:
    """Read the shared ``well_geometry.csv`` cache (empty dict when absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {
            row["well"]: (float(row["cy"]), float(row["cx"]), float(row["R"]))
            for row in csv.DictReader(f)
        }


def write_geometry(
    path: Path | str, geometry: Mapping[str, tuple[float, float, float]]
) -> Path:
    """Write the shared ``well_geometry.csv`` cache (cy, cx, R at one decimal)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GEOMETRY_COLUMNS)
        for well in sorted(geometry):
            cy, cx, radius = geometry[well]
            writer.writerow([well, f"{cy:.1f}", f"{cx:.1f}", f"{radius:.1f}"])
    return path


def update_geometry(
    path: Path | str, well: str, cy: float, cx: float, radius: float
) -> None:
    """Read-modify-write one well's fitted circle in the geometry cache."""
    geometry = read_geometry(path)
    geometry[well] = (float(cy), float(cx), float(radius))
    write_geometry(path, geometry)


def write_label_image(path: Path | str, labels: np.ndarray) -> Path:
    """Write a colony label image (uint32; 0 = background) as a TIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path, np.ascontiguousarray(labels).astype(np.uint32, copy=False),
        compression="zlib",
    )
    return path


def read_label_image(path: Path | str) -> np.ndarray:
    """Read a colony label image as a uint32 array."""
    return np.asarray(tifffile.imread(path), dtype=np.uint32)
