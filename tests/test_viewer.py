"""Contract tests for the viewer (``view_wells`` and its sources).

The seam is a recording stub viewer: ``view_wells`` is driven in-process
and every ``add_image`` call is captured, so the tests touch neither napari
nor PyQt nor a display. The synthetic mini plate (small — the viewer does
no circle fit) supplies the raw frames.
"""

import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from colony_tracking import ViewOptions, view_wells
from colony_tracking.viewer import available_wells, configured_sources
from conftest import build_mini_plate

WELLS = ("-A001", "-A002")


class RecordingViewer:
    """A napari viewer stand-in: records every layer, in order."""

    def __init__(self):
        self.layers = []  # (data, add_image kwargs)

    def add_image(self, data, **kwargs):
        self.layers.append((data, kwargs))
        return kwargs.get("name")


def _plate(tmp_path, wells=WELLS, n_frames=3, size=24):
    """A tiny raw plate (no ring: the viewer never fits the well circle)."""
    return build_mini_plate(
        tmp_path / "raw", layout="flat", wells=wells, n_frames=n_frames,
        size=size, noise_sigma=0.0, rings={well: None for well in wells},
        disks={well: [] for well in wells},
    )


def _options(plate, wells=WELLS):
    return ViewOptions(wells=list(wells), raw_dir=plate.root)


# --- the layer builder --------------------------------------------------------


def test_one_well_opens_as_its_raw_stack(tmp_path):
    plate = _plate(tmp_path, wells=("-A001",))
    viewer = RecordingViewer()

    returned = view_wells(_options(plate, ("-A001",)), viewer=viewer)

    assert returned is viewer  # the one passed in, never a napari session
    (data, kwargs), = viewer.layers
    assert kwargs["name"] == "-A001 raw"
    assert kwargs["colormap"] == "gray"  # greyscale, explicitly
    assert kwargs.get("visible", True)  # every loaded layer is visible
    # Frames are time: a (T, H, W) stack, raw dtype, no pixel conversion.
    assert data.shape == (plate.n_frames, plate.size, plate.size)
    assert data.dtype == tifffile.imread(plate.root / plate.wells["-A001"][0]).dtype
    expected = np.stack(
        [tifffile.imread(plate.root / name) for name in plate.wells["-A001"]]
    )
    assert np.array_equal(data, expected)


def test_several_wells_are_separate_layers(tmp_path):
    plate = _plate(tmp_path)
    viewer = RecordingViewer()

    view_wells(_options(plate), viewer=viewer)

    assert [kwargs["name"] for _, kwargs in viewer.layers] == [
        "-A001 raw", "-A002 raw",
    ]
    for data, _ in viewer.layers:
        assert data.shape == (plate.n_frames, plate.size, plate.size)


def test_no_wells_raises_listing_the_available_ones(tmp_path):
    plate = _plate(tmp_path)
    viewer = RecordingViewer()

    with pytest.raises(ValueError) as excinfo:
        view_wells(ViewOptions(raw_dir=plate.root), viewer=viewer)

    message = str(excinfo.value)
    assert "no wells requested" in message
    assert "-A001" in message and "-A002" in message
    assert viewer.layers == []


def test_unknown_well_raises_listing_the_available_ones(tmp_path):
    plate = _plate(tmp_path)
    viewer = RecordingViewer()

    with pytest.raises(ValueError) as excinfo:
        view_wells(_options(plate, ("-C009",)), viewer=viewer)

    message = str(excinfo.value)
    assert "unknown well(s) -C009" in message
    assert "-A001" in message and "-A002" in message
    assert viewer.layers == []


def test_a_session_with_nothing_to_load_is_an_error(tmp_path):
    viewer = RecordingViewer()

    with pytest.raises(ValueError) as excinfo:
        view_wells(ViewOptions(wells=["-A001"]), viewer=viewer)

    assert "available wells: none" in str(excinfo.value)
    assert viewer.layers == []


# --- the configured sources ---------------------------------------------------


def test_configured_sources_are_the_paths_the_options_carry():
    assert configured_sources(ViewOptions()) == []
    sources = configured_sources(ViewOptions(raw_dir=Path("raw")))
    assert [(source.kind, source.root) for source in sources] == [
        ("raw", Path("raw")),
    ]


def test_available_wells_span_the_configured_sources(tmp_path):
    plate = _plate(tmp_path)
    assert available_wells(_options(plate)) == ["-A001", "-A002"]
    assert available_wells(ViewOptions()) == []


# --- napari stays out of the way ---------------------------------------------


def test_a_stubbed_session_never_opens_a_napari_viewer(monkeypatch, tmp_path):
    plate = _plate(tmp_path, wells=("-A001",))

    def forbidden(*args, **kwargs):
        raise AssertionError("the stub session created a napari Viewer")

    monkeypatch.setitem(sys.modules, "napari", SimpleNamespace(Viewer=forbidden))

    viewer = RecordingViewer()
    assert view_wells(_options(plate, ("-A001",)), viewer=viewer) is viewer
    assert [kwargs["name"] for _, kwargs in viewer.layers] == ["-A001 raw"]


def test_the_viewer_and_cli_import_napari_lazily():
    """No module-level napari in the package (a test run may have napari
    loaded as a pytest plugin; the package must never rely on that)."""
    from colony_tracking import cli, viewer as viewer_module

    assert "napari" not in vars(viewer_module)
    assert "napari" not in vars(cli)


def test_missing_napari_is_a_clear_error(monkeypatch, tmp_path):
    plate = _plate(tmp_path, wells=("-A001",))
    monkeypatch.setitem(sys.modules, "napari", None)  # the "not installed" case

    with pytest.raises(RuntimeError, match="napari is not installed"):
        view_wells(_options(plate, ("-A001",)))


# --- the option dataclass -----------------------------------------------------


def test_every_view_option_has_help_text():
    for f in fields(ViewOptions):
        assert f.metadata.get("help"), f"ViewOptions.{f.name} has no help text"


def test_view_options_defaults():
    options = ViewOptions()
    assert options.wells is None
    assert options.config is None and options.raw_dir is None


def test_view_is_not_a_tui_stage():
    from colony_tracking.tui.screens.stage import STAGES

    assert "view" not in STAGES
