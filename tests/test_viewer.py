"""Contract tests for the viewer (``view_wells`` and its sources).

The seam is a recording stub viewer: ``view_wells`` is driven in-process and
every layer it adds is captured, so the tests touch neither napari nor PyQt
nor a display. The synthetic mini plate (small — the viewer does no circle
fit) supplies the raw frames; the derived trees are written directly in the
on-disk shape the difference, segment, and analyze stages produce, so the
tests assert against the bytes on disk, not against a stage run.
"""

import itertools
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from colony_tracking import ViewOptions, view_wells
from colony_tracking.viewer import available_wells, configured_sources
from conftest import build_mini_plate

WELLS = ("-A001", "-A002")
KINDS = ("raw", "diff", "labels", "qa")
FIELDS = {"raw": "raw_dir", "diff": "diff_dir", "labels": "seg_dir",
          "qa": "analysis_dir"}
#: The name of the segmentation root whose analysis output holds the QA
#: overlays (the QA path is ``<analysis>/<seg-root-name>/qa/<well>_rim.png``).
SEG_ROOT = "seg_growth_diff"


class RecordingViewer:
    """A napari viewer stand-in: records every layer, in order."""

    def __init__(self):
        self.layers = []  # (data, add kwargs)
        self.methods = []  # the add method of each layer, in order

    def add_image(self, data, **kwargs):
        self.methods.append("add_image")
        self.layers.append((data, kwargs))
        return kwargs.get("name")

    def add_labels(self, data, **kwargs):
        self.methods.append("add_labels")
        self.layers.append((data, kwargs))
        return kwargs.get("name")


def _plate(tmp_path, wells=WELLS, n_frames=3, size=24):
    """A tiny raw plate (no ring: the viewer never fits the well circle)."""
    return build_mini_plate(
        tmp_path / "raw", layout="flat", wells=wells, n_frames=n_frames,
        size=size, noise_sigma=0.0, rings={well: None for well in wells},
        disks={well: [] for well in wells},
    )


def _frame_name(well, lo, suffix):
    """A derived frame's file name (the raw vocabulary, plus the suffix)."""
    return f"{well}--PO01--LO{lo:03d}--T{lo * 3_600_000:010d}--WE00001{suffix}"


def _diff_tree(root, wells=WELLS, n_frames=3, size=24):
    """A difference tree as the difference stage writes it: one directory
    per well, uint8 previews named ``<frame-stem>_diff_uint8.tif``."""
    stacks = {}
    for index, well in enumerate(wells):
        well_dir = root / well
        well_dir.mkdir(parents=True)
        frames = []
        for lo in range(1, n_frames + 1):
            frame = np.full((size, size), 128, dtype=np.uint8)  # 128 = no change
            frame[lo % size, :] = 128 + index * 10 + lo
            tifffile.imwrite(well_dir / _frame_name(well, lo, "_diff_uint8.tif"),
                             frame)
            frames.append(frame)
        (well_dir / "scale.txt").write_text("reference\tx\n")
        stacks[well] = np.stack(frames)
    return stacks


def _seg_tree(root, wells=WELLS, n_frames=3, size=24):
    """A segmentation tree as the segment stage writes it: one directory per
    well, uint32 label images named ``<frame-stem>__labels.tif``."""
    stacks = {}
    for index, well in enumerate(wells):
        well_dir = root / well
        well_dir.mkdir(parents=True)
        frames = []
        for lo in range(1, n_frames + 1):
            frame = np.zeros((size, size), dtype=np.uint32)
            frame[: lo + 1, : lo + 1] = index * 100 + lo
            tifffile.imwrite(well_dir / _frame_name(well, lo, "__labels.tif"),
                             frame)
            frames.append(frame)
        # The reference frame's group labels carry no frame number: not a frame.
        tifffile.imwrite(well_dir / "ref__labels.tif",
                         np.ones((size, size), dtype=np.uint32))
        stacks[well] = np.stack(frames)
    return stacks


def _qa_tree(analysis_dir, seg_root=SEG_ROOT, wells=WELLS, size=8):
    """An analysis output root with one QA overlay PNG per well."""
    qa_dir = analysis_dir / seg_root / "qa"
    qa_dir.mkdir(parents=True)
    pixels = {}
    for index, well in enumerate(wells):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[..., 0] = 200
        image[..., 1] = index * 40
        image[:, index % size, 2] = 255
        Image.fromarray(image).save(qa_dir / f"{well}_rim.png")
        pixels[well] = image
    return pixels


def _trees(tmp_path, wells=WELLS):
    """Every derived tree, built and returned by kind (and the raw plate)."""
    return {
        "raw": _plate(tmp_path, wells=wells).root,
        "diff": tmp_path / "difference",
        "labels": tmp_path / SEG_ROOT,
        "qa": tmp_path / "analysis",
    }


def _build(tmp_path, wells=WELLS, n_frames=3, size=24, qa_size=8):
    """Build all four sources and return ``(dirs, arrays)``."""
    dirs = _trees(tmp_path, wells=wells)
    arrays = {
        "diff": _diff_tree(dirs["diff"], wells=wells, n_frames=n_frames,
                           size=size),
        "labels": _seg_tree(dirs["labels"], wells=wells, n_frames=n_frames,
                            size=size),
        "qa": _qa_tree(dirs["qa"], wells=wells, size=qa_size),
    }
    return dirs, arrays


def _options_for(dirs, kinds, wells=WELLS):
    """The options for exactly the configured ``kinds``."""
    return ViewOptions(
        wells=list(wells), **{FIELDS[kind]: dirs[kind] for kind in kinds}
    )


def _options(plate, wells=WELLS):
    return ViewOptions(wells=list(wells), raw_dir=plate.root)


def _layer(viewer, name):
    """The one recorded layer called ``name``."""
    (data, kwargs), = [
        (data, kwargs) for data, kwargs in viewer.layers
        if kwargs["name"] == name
    ]
    return data, kwargs


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


def test_a_complete_run_gives_exactly_the_four_layers(tmp_path):
    dirs, _ = _build(tmp_path)
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, KINDS), viewer=viewer)

    assert [kwargs["name"] for _, kwargs in viewer.layers] == [
        "-A001 raw", "-A001 diff", "-A001 labels", "-A001 qa",
        "-A002 raw", "-A002 diff", "-A002 labels", "-A002 qa",
    ]
    assert viewer.methods == [
        "add_image", "add_image", "add_labels", "add_image",
        "add_image", "add_image", "add_labels", "add_image",
    ]
    for _, kwargs in viewer.layers:
        assert kwargs.get("visible", True)  # every loaded layer is visible
    # Only the derived stacks are greyscale images; the QA bitmap keeps its
    # own colors and the labels carry a Labels layer.
    for name in ("-A001 raw", "-A001 diff"):
        assert _layer(viewer, name)[1]["colormap"] == "gray"
    assert "colormap" not in _layer(viewer, "-A001 qa")[1]


@pytest.mark.parametrize(
    "kinds",
    [combo for n in range(1, len(KINDS) + 1)
     for combo in itertools.combinations(KINDS, n)],
    ids=lambda kinds: "+".join(kinds),
)
def test_every_source_subset_opens(tmp_path, kinds):
    dirs, _ = _build(tmp_path, wells=("-A001",))
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, kinds, wells=("-A001",)), viewer=viewer)

    assert [kwargs["name"] for _, kwargs in viewer.layers] == [
        f"-A001 {kind}" for kind in kinds
    ]
    assert available_wells(_options_for(dirs, kinds)) == ["-A001"]


# --- the derived stacks -------------------------------------------------------


def test_the_diff_layer_is_the_stored_uint8_stack(tmp_path):
    dirs, arrays = _build(tmp_path, wells=("-A001",))
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, ("raw", "diff"), wells=("-A001",)),
               viewer=viewer)

    data, _ = _layer(viewer, "-A001 diff")
    assert data.dtype == np.uint8
    assert np.array_equal(data, arrays["diff"]["-A001"])
    # The stored frame order (LO001 first), not a re-sort toward the raw series.
    on_disk = sorted((dirs["diff"] / "-A001").glob("*_diff_uint8.tif"))
    expected = np.stack([tifffile.imread(path) for path in on_disk])
    assert np.array_equal(data, expected)
    assert np.array_equal(data[0], tifffile.imread(on_disk[0]))
    assert data[0, 1, 0] == 128 + 1  # the per-frame marker, in file order


def test_the_labels_layer_is_the_stored_uint32_stack(tmp_path):
    dirs, arrays = _build(tmp_path, wells=("-A001",))
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, ("labels",), wells=("-A001",)),
               viewer=viewer)

    data, _ = _layer(viewer, "-A001 labels")
    assert viewer.methods == ["add_labels"]
    assert data.dtype == np.uint32
    assert np.array_equal(data, arrays["labels"]["-A001"])
    assert data.shape == (3, 24, 24)  # ref__labels.tif is not a frame


def test_the_qa_layer_is_the_png_on_the_raw_footprint(tmp_path):
    dirs, arrays = _build(tmp_path, wells=("-A001",), size=24, qa_size=8)
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, KINDS, wells=("-A001",)), viewer=viewer)

    data, kwargs = _layer(viewer, "-A001 qa")
    assert np.array_equal(data, arrays["qa"]["-A001"])
    with Image.open(dirs["qa"] / SEG_ROOT / "qa" / "-A001_rim.png") as png:
        assert png.size == (8, 8)
    # H/512, W/512 in the real case; the raw frame's extent over the bitmap.
    assert kwargs["scale"] == (24 / 8, 24 / 8)


def test_the_qa_layer_is_unscaled_without_a_raw_stack(tmp_path):
    dirs, _ = _build(tmp_path, wells=("-A001",), qa_size=8)
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, ("labels", "qa"), wells=("-A001",)),
               viewer=viewer)

    _, kwargs = _layer(viewer, "-A001 qa")
    assert kwargs["scale"] == (1.0, 1.0)


def test_the_qa_layer_follows_the_configured_segmentation_root(tmp_path):
    """The QA path is ``<analysis-dir>/<seg-root-name>/qa``, never a glob
    across the analysis root: another root's overlay is not shown."""
    dirs, _ = _build(tmp_path, wells=("-A001",))
    _qa_tree(dirs["qa"], seg_root="other_seg", wells=("-A001",))
    # The other root's PNG is a different image; make it recognisable.
    other = dirs["qa"] / "other_seg" / "qa" / "-A001_rim.png"
    Image.fromarray(np.full((8, 8, 3), 7, dtype=np.uint8)).save(other)
    viewer = RecordingViewer()

    view_wells(_options_for(dirs, ("labels", "qa"), wells=("-A001",)),
               viewer=viewer)

    data, _ = _layer(viewer, "-A001 qa")
    assert data[0, 0, 0] == 200  # the configured root's overlay


# --- missing artifacts --------------------------------------------------------


def test_a_well_missing_from_a_source_loses_only_that_layer(tmp_path, caplog):
    dirs, _ = _build(tmp_path, wells=("-A001",))
    plate = _plate(tmp_path, wells=("-A002",))
    options = ViewOptions(
        wells=["-A002"], raw_dir=plate.root, diff_dir=dirs["diff"],
        seg_dir=dirs["labels"], analysis_dir=dirs["qa"],
    )
    viewer = RecordingViewer()

    with caplog.at_level("INFO", logger="colony_tracking"):
        view_wells(options, viewer=viewer)

    assert [kwargs["name"] for _, kwargs in viewer.layers] == ["-A002 raw"]
    for kind in ("diff", "labels", "qa"):
        assert f"-A002 {kind}: no artifact" in caplog.text


def test_missing_trees_leave_the_session_open(tmp_path, caplog):
    plate = _plate(tmp_path, wells=("-A001",))
    options = ViewOptions(
        wells=["-A001"], raw_dir=plate.root,
        diff_dir=tmp_path / "no_difference", seg_dir=tmp_path / "no_seg",
        analysis_dir=tmp_path / "no_analysis",
    )
    viewer = RecordingViewer()

    with caplog.at_level("INFO", logger="colony_tracking"):
        view_wells(options, viewer=viewer)

    assert [kwargs["name"] for _, kwargs in viewer.layers] == ["-A001 raw"]
    assert "no_difference" in caplog.text and "no_seg" in caplog.text
    assert "no_analysis" in caplog.text


def test_a_missing_qa_png_leaves_the_other_layers(tmp_path, caplog):
    dirs, _ = _build(tmp_path, wells=("-A001",))
    (dirs["qa"] / SEG_ROOT / "qa" / "-A001_rim.png").unlink()
    options = _options_for(dirs, ("labels", "qa"), wells=("-A001",))
    viewer = RecordingViewer()

    with caplog.at_level("INFO", logger="colony_tracking"):
        view_wells(options, viewer=viewer)

    assert [kwargs["name"] for _, kwargs in viewer.layers] == ["-A001 labels"]
    assert "-A001 qa: no artifact" in caplog.text


def test_a_qa_only_session_finds_a_single_analysis_root(tmp_path):
    """Without ``--seg-dir`` there is no segmentation root name: a well whose
    overlay is in exactly one analysis root's QA directory still loads."""
    dirs, arrays = _build(tmp_path, wells=("-A001",), qa_size=8)
    viewer = RecordingViewer()

    view_wells(ViewOptions(wells=["-A001"], analysis_dir=dirs["qa"]),
               viewer=viewer)

    data, kwargs = _layer(viewer, "-A001 qa")
    assert np.array_equal(data, arrays["qa"]["-A001"])
    assert kwargs["scale"] == (1.0, 1.0)


def test_an_ambiguous_qa_overlay_is_skipped(tmp_path, caplog):
    """Several analysis roots hold the well and none is the configured
    segmentation root: show neither (never the wrong one)."""
    dirs, _ = _build(tmp_path, wells=("-A001",))
    _qa_tree(dirs["qa"], seg_root="other_seg", wells=("-A001",))
    viewer = RecordingViewer()

    with caplog.at_level("INFO", logger="colony_tracking"):
        with pytest.raises(ValueError, match="nothing to load"):
            view_wells(ViewOptions(wells=["-A001"], analysis_dir=dirs["qa"]),
                       viewer=viewer)

    assert viewer.layers == []
    assert "-A001 qa: no artifact" in caplog.text
    assert "ambiguous" in caplog.text


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


def test_configured_sources_cover_every_kind_in_layer_order(tmp_path):
    kinds = KINDS
    dirs = _trees(tmp_path)
    sources = configured_sources(_options_for(dirs, kinds))
    assert [(source.kind, source.root) for source in sources] == [
        (kind, dirs[kind]) for kind in kinds
    ]


def test_available_wells_span_the_configured_sources(tmp_path):
    plate = _plate(tmp_path)
    assert available_wells(_options(plate)) == ["-A001", "-A002"]
    assert available_wells(ViewOptions()) == []


def test_a_well_only_in_one_source_is_available(tmp_path):
    dirs = {
        "raw": _plate(tmp_path, wells=("-A001",)).root,
        "diff": tmp_path / "difference",
        "labels": tmp_path / SEG_ROOT,
        "qa": tmp_path / "analysis",
    }
    _diff_tree(dirs["diff"], wells=("-A002",))
    _seg_tree(dirs["labels"], wells=("-A002",))
    _qa_tree(dirs["qa"], wells=("-A002",))

    options = _options_for(dirs, KINDS)
    assert available_wells(options) == ["-A001", "-A002"]

    viewer = RecordingViewer()
    view_wells(_options_for(dirs, KINDS, wells=("-A002",)), viewer=viewer)
    assert [kwargs["name"] for _, kwargs in viewer.layers] == [
        "-A002 diff", "-A002 labels", "-A002 qa",
    ]


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
    for name in ("config", *FIELDS.values()):
        assert getattr(options, name) is None


def test_view_is_not_a_tui_stage():
    from colony_tracking.tui.screens.stage import STAGES

    assert "view" not in STAGES
