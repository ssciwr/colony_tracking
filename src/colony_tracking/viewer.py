"""The viewer: open a run's wells in napari as image layers.

:func:`view_wells` loads every requested well from every *configured
source* — one layer per (well, source), named ``"<well> <kind>"`` (``-A001
raw``), all visible — and adds it to a napari session; it returns the
viewer and never calls ``napari.run()`` (the caller owns the event loop;
the CLI runs it).

The module is deliberately thin. Beyond the package logger, it imports
only :mod:`colony_tracking.io` (discovery and the frame records) and
:mod:`colony_tracking._options` (the resolved
:class:`~colony_tracking._options.ViewOptions`): the CLI has
already resolved every path, so the viewer holds no knowledge of the
pipeline's tree naming and derives nothing. Raw is the only source today;
the derived kinds (difference, labels, QA overlays) join
:func:`configured_sources`, each optional and skipped when its path is
unset.

napari is an implicit dependency (it ships with micro-sam): it is imported
lazily inside :func:`_new_viewer`, so importing the package — or this
module — never pulls napari, PyQt, or a display. A missing napari is
reported as a clear error, not a traceback from inside napari.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from . import log
from ._options import ViewOptions
from .io import Frame, discover_wells

#: Loader of one well's source: its frames (in time order) → the stack the
#: layer shows. No pixel conversion happens here — the stack keeps the
#: files' dtype exactly.
Loader = Callable[[list[Frame]], np.ndarray]


@dataclass(frozen=True)
class _Source:
    """One configured source: the layer kind, the tree it lives in, and how
    one well's frames become the layer's data."""

    kind: str
    #: The resolved directory of this source (never a derived tree name).
    root: Path
    load: Loader


def _load_raw(frames: list[Frame]) -> np.ndarray:
    """One well's raw acquisition as a ``(T, H, W)`` stack (frames are
    time, not depth, so napari's 2D display gives the frame slider)."""
    return np.stack([tifffile.imread(frame.path) for frame in frames])


def configured_sources(options: ViewOptions) -> list[_Source]:
    """The sources ``options`` configures, in layer order.

    A source whose path the options do not carry is not configured (and is
    skipped). Raw is always the first source; the derived image kinds are
    added here as their tickets land.
    """
    sources: list[_Source] = []
    if options.raw_dir is not None:
        sources.append(_Source("raw", options.raw_dir, _load_raw))
    return sources


def available_wells(options: ViewOptions) -> list[str]:
    """The wells present in at least one of the configured sources, sorted.

    The one notion of "a well that exists", shared by the viewer's own
    check and the CLI's usage error.
    """
    wells: set[str] = set()
    for source in configured_sources(options):
        wells.update(discover_wells(source.root))
    return sorted(wells)


def _requested_wells(options: ViewOptions, available: list[str]) -> list[str]:
    """The wells to show, or a ValueError naming the wells available."""
    names = ", ".join(available) or "none"
    if not options.wells:
        raise ValueError(f"no wells requested: available wells: {names}")
    unknown = [well for well in options.wells if well not in available]
    if unknown:
        raise ValueError(
            f"unknown well(s) {', '.join(unknown)}: available wells: {names}"
        )
    return list(options.wells)


def _layers(options: ViewOptions) -> list[tuple[str, np.ndarray]]:
    """``(layer name, data)`` for every requested well and configured
    source that holds it: ``"<well> <kind>"``, raw before the derived
    kinds."""
    available = available_wells(options)
    layers: list[tuple[str, np.ndarray]] = []
    for well in _requested_wells(options, available):
        for source in configured_sources(options):
            frames = discover_wells(source.root, wells=[well]).get(well)
            if not frames:
                continue
            data = source.load(frames)
            log.info(
                "viewer: %s %s: %d frame(s), %dx%d",
                well, source.kind, data.shape[0], data.shape[1], data.shape[2],
            )
            layers.append((f"{well} {source.kind}", data))
    if not layers:
        raise ValueError(
            "nothing to load: available wells: "
            + (", ".join(available) or "none")
        )
    return layers


def _new_viewer():
    """A napari ``Viewer``; the import is lazy (see the module docstring)."""
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed: the view command needs it (it ships "
            "with micro-sam); install micro-sam in this environment"
        ) from exc
    return napari.Viewer()


def view_wells(options: ViewOptions, viewer=None):
    """Open ``options``' wells in a napari session and return the viewer.

    Builds one greyscale layer per well and configured source, named
    ``"<well> <kind>"`` (``-A001 raw``), and adds it visible. Never calls
    ``napari.run()``: the caller runs the event loop (the CLI does), which
    also lets a test pass a recording stub as ``viewer`` and keep napari,
    PyQt, and any display out of the test. An unknown or missing well, or a
    session with nothing to load, raises a ValueError listing the wells
    available in the configured sources.
    """
    layers = _layers(options)
    if viewer is None:
        viewer = _new_viewer()
    for name, data in layers:
        viewer.add_image(data, name=name, colormap="gray")
    return viewer
