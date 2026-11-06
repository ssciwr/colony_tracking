"""The viewer: open a run's wells in napari as layers.

:func:`view_wells` loads every requested well from every *configured
source* — one layer per (well, source), named ``"<well> <kind>"`` (``-A001
raw``), all visible — and adds it to a napari session; it returns the
viewer and never calls ``napari.run()`` (the caller owns the event loop;
the CLI runs it).

Four sources exist, each optional and skipped when its path is unset:

``raw``
    the acquisition frames, one greyscale image stack (``add_image``);
``diff``
    the stored difference previews, an ``uint8`` image stack (``128`` = no
    change, ADR 0001) shown as written;
``labels``
    the segmentation label images, an ``uint32`` napari ``Labels`` layer
    (a rendering mode, not a data conversion);
``qa``
    the analysis stage's ``qa/<well>_rim.png`` overlay, loaded as a bitmap
    and placed on the raw frame's footprint by a coordinate ``scale``
    (``H/512``, ``W/512``) when a raw stack is loaded, unscaled otherwise.

Missing artifacts (an absent tree, a well without its files) never fail the
command: that one layer is dropped with an INFO line and the rest opens.

The module is deliberately thin. Beyond the package logger, it imports
numpy and tifffile (the frame stacks), PIL (the QA bitmap),
:mod:`colony_tracking.io` (discovery and the frame records), and
:mod:`colony_tracking._options` (the resolved
:class:`~colony_tracking._options.ViewOptions`). Every path reaches the
viewer already resolved — the CLI and the TUI read them from the config —
so it derives no pipeline tree name. The one layout it does know is the
analysis stage's QA output, where the overlay bitmap lives:
``<analysis dir>/<segmentation root name>/qa/<well>_rim.png`` — the names
:data:`QA_DIR` and :func:`_qa_paths` build.

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
from PIL import Image

from . import io, log
from ._options import ViewOptions
from .io import discover_wells, read_label_image

#: Discovery of one source's wells, sorted (a missing tree holds none).
Discover = Callable[[], list[str]]
#: Loader of one well's layer data: the stack (or bitmap) the layer shows,
#: or None when the well's artifact is missing. No pixel conversion happens
#: here — the data keeps the files' dtype exactly.
Load = Callable[[str], np.ndarray | None]

#: The QA overlays' directory name, inside one segmentation root's analysis
#: output (``<analysis root>/<segmentation root>/qa/<well>_rim.png``).
QA_DIR = "qa"


@dataclass(frozen=True)
class _Source:
    """One configured source: the layer kind, the tree it lives in, how its
    wells are discovered, and how one well becomes the layer's data."""

    kind: str
    #: The resolved directory of this source (never a derived tree name).
    root: Path
    discover: Discover
    load: Load
    #: Whether the layer renders as a napari ``Labels`` layer (label images).
    as_labels: bool = False


@dataclass(frozen=True)
class _Layer:
    """One layer to add: its name, its data, its layer type, and its
    coordinate scale (None = napari's default)."""

    name: str
    data: np.ndarray
    as_labels: bool = False
    scale: tuple[float, ...] | None = None


# --- the per-source loaders --------------------------------------------------


def _tree_wells(root: Path) -> list[str]:
    """The wells of a per-well input tree, sorted; a missing tree has none."""
    if not root.is_dir():
        return []
    return sorted(discover_wells(root))


def _tree_stack(root: Path, well: str, read):
    """One well's frames of a tree as a ``(T, H, W)`` stack read by ``read``,
    in the files' frame order; None when the tree or the well is absent."""
    if not root.is_dir():
        return None
    frames = discover_wells(root, wells=[well]).get(well)
    if not frames:
        return None
    return np.stack([read(frame.path) for frame in frames])


def _stack_source(kind: str, root: Path, read, *, as_labels=False) -> _Source:
    """A per-well frame-series source (raw, diff, labels)."""
    return _Source(
        kind=kind,
        root=root,
        discover=lambda: _tree_wells(root),
        load=lambda well: _tree_stack(root, well, read),
        as_labels=as_labels,
    )


def _qa_roots(analysis_dir: Path, seg_dir: Path | None) -> list[Path]:
    """The analysis-output directories that hold this session's QA overlays.

    With a segmentation root, the precise path — the analysis output of
    that root alone, so an analysis root holding several segmentation trees
    never shows the wrong one. Without one (a QA-only session) every
    immediate subdirectory of the analysis root is a candidate;
    :func:`_qa_image` shows a well's overlay only when exactly one candidate
    holds it.
    """
    if seg_dir is not None:
        return [analysis_dir / Path(seg_dir).name]
    if not analysis_dir.is_dir():
        return []
    return [entry for entry in sorted(analysis_dir.iterdir())
            if entry.is_dir()]


def _qa_paths(analysis_dir: Path, seg_dir: Path | None,
              well: str) -> list[Path]:
    """The existing ``qa/<well>_rim.png`` overlays, in root order."""
    return [
        path
        for root in _qa_roots(analysis_dir, seg_dir)
        if (path := root / QA_DIR / f"{well}_rim.png").is_file()
    ]


def _qa_wells(analysis_dir: Path, seg_dir: Path | None) -> list[str]:
    """The wells with a QA overlay, sorted."""
    wells = set()
    for root in _qa_roots(analysis_dir, seg_dir):
        qa_dir = root / QA_DIR
        if not qa_dir.is_dir():
            continue
        for entry in sorted(qa_dir.iterdir()):
            well = io.parse_well(entry.name)
            if well is not None and entry.name == f"{well}_rim.png":
                wells.add(well)
    return sorted(wells)


def _qa_image(analysis_dir: Path, seg_dir: Path | None,
              well: str) -> np.ndarray | None:
    """One well's QA overlay as the PNG's pixels, or None when absent."""
    paths = _qa_paths(analysis_dir, seg_dir, well)
    if len(paths) > 1:
        roots = ", ".join(str(path.parent.parent) for path in paths)
        log.info(
            "viewer: %s qa: ambiguous, %d analysis roots hold it (%s); "
            "skipping (name the segmentation root with --seg-dir)",
            well, len(paths), roots,
        )
        return None
    if not paths:
        return None
    with Image.open(paths[0]) as image:
        return np.asarray(image)


def configured_sources(options: ViewOptions) -> list[_Source]:
    """The sources ``options`` configures, in layer order.

    A source whose path the options do not carry is not configured (and is
    skipped), so any non-empty subset of the four kinds is a session. Raw is
    always the first source, then the derived kinds.
    """
    sources: list[_Source] = []
    if options.raw_dir is not None:
        sources.append(_stack_source("raw", options.raw_dir, tifffile.imread))
    if options.diff_dir is not None:
        sources.append(_stack_source("diff", options.diff_dir,
                                     tifffile.imread))
    if options.seg_dir is not None:
        sources.append(_stack_source("labels", options.seg_dir,
                                     read_label_image, as_labels=True))
    if options.analysis_dir is not None:
        analysis_dir, seg_dir = options.analysis_dir, options.seg_dir
        sources.append(_Source(
            kind="qa",
            root=analysis_dir,
            discover=lambda: _qa_wells(analysis_dir, seg_dir),
            load=lambda well: _qa_image(analysis_dir, seg_dir, well),
        ))
    return sources


def available_wells(options: ViewOptions) -> list[str]:
    """The wells present in at least one of the configured sources, sorted.

    The one notion of "a well that exists", shared by the viewer's own
    check and the CLI's usage error.
    """
    wells: set[str] = set()
    for source in configured_sources(options):
        wells.update(source.discover())
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


def _log_layer(well: str, kind: str, data: np.ndarray) -> None:
    """The INFO line of one loaded layer (a frame stack or a bitmap)."""
    if data.ndim == 3:
        log.info("viewer: %s %s: %d frame(s), %dx%d",
                 well, kind, *data.shape)
    else:
        log.info("viewer: %s %s: %dx%d image", well, kind, *data.shape)


def _qa_scale(qa_shape: tuple[int, int],
              raw_shape: tuple[int, int] | None) -> tuple[float, float]:
    """The QA bitmap's coordinate scale: the raw frame's footprint over the
    bitmap (``H/512``, ``W/512`` for the 512 px overlays), or napari's
    default when no raw stack is loaded to size it against."""
    if raw_shape is None:
        return (1.0, 1.0)
    return (raw_shape[0] / qa_shape[0], raw_shape[1] / qa_shape[1])


def _layers(options: ViewOptions) -> list[_Layer]:
    """The layers for every requested well and configured source that holds
    it: ``"<well> <kind>"``, raw before the derived kinds."""
    available = available_wells(options)
    sources = configured_sources(options)
    layers: list[_Layer] = []
    for well in _requested_wells(options, available):
        loaded: list[tuple[_Source, np.ndarray]] = []
        for source in sources:
            data = source.load(well)
            if data is None:
                log.info("viewer: %s %s: no artifact under %s, skipping",
                         well, source.kind, source.root)
                continue
            _log_layer(well, source.kind, data)
            loaded.append((source, data))
        raw_shape = next(
            (data.shape[-2:] for source, data in loaded
             if source.kind == "raw"),
            None,
        )
        for source, data in loaded:
            scale = None
            if source.kind == "qa":
                scale = _qa_scale(data.shape[:2], raw_shape)
            layers.append(_Layer(f"{well} {source.kind}", data,
                                 source.as_labels, scale))
    if not layers:
        raise ValueError(
            "nothing to load: available wells: "
            + (", ".join(available) or "none")
        )
    return layers


def _new_viewer():
    """A napari ``Viewer``; the import and the viewer construction are lazy
    (see the module docstring). A failed import — napari itself, or its Qt
    binding — is one clear RuntimeError, never a traceback."""
    try:
        import napari

        return napari.Viewer()
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed: the view command needs it (it ships "
            "with micro-sam); install micro-sam in this environment"
        ) from exc


def view_wells(options: ViewOptions, viewer=None):
    """Open ``options``' wells in a napari session and return the viewer.

    Builds one layer per well and configured source, named ``"<well>
    <kind>"`` (``-A001 raw``): greyscale image stacks for raw and diff, a
    ``Labels`` layer for the segmentation, and the QA overlay bitmap scaled
    onto the raw frame. Adds every layer visible. Never calls
    ``napari.run()``: the caller runs the event loop (the CLI does), which
    also lets a test pass a recording stub as ``viewer`` and keep napari,
    PyQt, and any display out of the test. An unknown or missing well, or a
    session with nothing to load, raises a ValueError listing the wells
    available in the configured sources.
    """
    layers = _layers(options)
    if viewer is None:
        viewer = _new_viewer()
    for layer in layers:
        if layer.as_labels:
            viewer.add_labels(layer.data, name=layer.name)
        elif layer.scale is not None:
            viewer.add_image(layer.data, name=layer.name, scale=layer.scale)
        else:
            viewer.add_image(layer.data, name=layer.name, colormap="gray")
    return viewer
