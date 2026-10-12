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
