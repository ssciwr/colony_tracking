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

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
