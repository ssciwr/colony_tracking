"""Tests for the shared I/O layer (colony_tracking.io).

Seams under test (per the ticket): the pure file-name parsers, input-tree
discovery in both layouts, the shared 1/99 stretch and 128-neutral
difference mapping, and the on-disk formats (scale.txt, done marker,
colonies.csv with legacy column mapping, well_geometry.csv, label images).
The synthetic fixtures built in conftest are exercised here as well, so
later tickets inherit a known-good builder.
"""

import math

import numpy as np
import pytest
import tifffile

from colony_tracking import io
from conftest import (
    DiskSpec,
    RingSpec,
    build_label_series,
    build_mini_plate,
    disk_mask,
    frame_name,
)

# A realistic raw-plate frame name (real acquisition vocabulary, the
# millisecond T field zero-padded to 10 digits).
RAW_NAME = (
    "-A001--PO01--LO001--CO1--SL001--PX32500--PW0050--IN0020"
    "--TM366--X014222--Y011210--Z220046--T0000000000--WE00001.tif"
)


def _write_frame(directory, name: str, shape=(4, 4)) -> None:
    tifffile.imwrite(directory / name, np.zeros(shape, np.uint16))


HOUR = 3_600_000  # ms; the real plate's frames are hourly


# --- file-name parsing -----------------------------------------------------


def test_parse_well():
    assert io.parse_well(RAW_NAME) == "-A001"
    assert io.parse_well("-C006--PO01--LO001.tif") == "-C006"
    # Not a well field unless it leads the name.
    assert io.parse_well("ref__labels.tif") is None
    assert io.parse_well("x-A001--LO001.tif") is None
    # Exactly three digits, row A..H only.
    assert io.parse_well("-A01--LO001.tif") is None
    assert io.parse_well("-I001--LO001.tif") is None


def test_parse_frame():
    assert io.parse_frame(RAW_NAME) == 1
    assert io.parse_frame(RAW_NAME.replace("--LO001--", "--LO015--")) == 15
    # The LO field of a difference frame (full raw stem embedded).
    assert io.parse_frame(RAW_NAME[:-4] + "__diff_uint8.tif") == 1
    # No LO field (e.g. the reference group labels).
    assert io.parse_frame("ref__labels.tif") is None
    # The digit count is not fixed to three.
    assert io.parse_frame("-A001--PO01--LO7--CO1.tif") == 7


def test_parse_time():
    assert io.parse_time(RAW_NAME) == 0
    assert io.parse_time(RAW_NAME.replace("--T0000000000--", "--T0014399594--")) == 14399594
    assert io.parse_time("ref__labels.tif") is None
    # The TM field must not be mistaken for the T field.
    assert io.parse_time("-A001--TM366--X0.tif") is None


# --- input-tree discovery ---------------------------------------------------


def test_discover_flat_layout(tmp_path):
    # Written out of order on purpose: discovery must sort by frame number.
    for lo, t in [(2, 2 * HOUR), (1, 0), (3, 3 * HOUR)]:
        _write_frame(tmp_path, frame_name("-A001", lo, t))
    for lo, t in [(1, 0), (2, 2 * HOUR)]:
        _write_frame(tmp_path, frame_name("-B004", lo, t))
    # Non-frame files are ignored.
    (tmp_path / "notes.txt").write_text("nope")
    _write_frame(tmp_path, "._" + frame_name("-C001", 1, 0))  # resource fork
    _write_frame(tmp_path, "ref__labels.tif")  # no well, no LO field

    wells = io.discover_wells(tmp_path)
    assert set(wells) == {"-A001", "-B004"}
    assert [f.frame for f in wells["-A001"]] == [1, 2, 3]
    assert [f.t for f in wells["-A001"]] == [0, 2 * HOUR, 3 * HOUR]
    assert wells["-A001"][0].name == frame_name("-A001", 1, 0)
    assert wells["-A001"][0].path == tmp_path / frame_name("-A001", 1, 0)
    assert wells["-A001"][0].well == "-A001"


def test_discover_per_well_layout(tmp_path):
    a = tmp_path / "-A001"
    b = tmp_path / "-B004"
    a.mkdir()
    b.mkdir()
    for lo, t in [(3, 3 * HOUR), (1, 0), (2, 2 * HOUR)]:
        _write_frame(a, frame_name("-A001", lo, t))
    _write_frame(a, "ref__labels.tif")  # no LO field: not a frame
    _write_frame(b, frame_name("-B004", 1, 0))
    # A stray top-level frame is ignored in the per-well layout.
    _write_frame(tmp_path, frame_name("-C001", 1, 0))

    wells = io.discover_wells(tmp_path)
    assert set(wells) == {"-A001", "-B004"}
    assert [f.frame for f in wells["-A001"]] == [1, 2, 3]
    assert wells["-A001"][2].path == a / frame_name("-A001", 3, 3 * HOUR)


def test_discover_wells_filter_and_ordering(tmp_path):
    for lo, t in [(1, 0), (2, 2 * HOUR)]:
        _write_frame(tmp_path, frame_name("-A001", lo, t))
    _write_frame(tmp_path, frame_name("-B004", 1, 0))

    assert set(io.discover_wells(tmp_path, wells=["-B004"])) == {"-B004"}

    # Frames with the same LO order by the T field.
    _write_frame(tmp_path, frame_name("-A001", 2, HOUR + 1))
    ordered = io.discover_wells(tmp_path)["-A001"]
    assert [f.t for f in ordered if f.frame == 2] == [HOUR + 1, 2 * HOUR]


def test_discover_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.discover_wells(tmp_path / "no-such-dir")


# --- 1/99 percentile stretch -------------------------------------------------


def test_stretch_known_values():
    # 100 values: one 0, one 100, ninety-eight 50s.
    # p1  = 0.01*99 = 0.99  -> 0.01*0 + 0.99*50   = 49.5
    # p99 = 0.99*99 = 98.01 -> 50 + 0.01*(100-50) = 50.5
    im = np.full((10, 10), 50.0)
    im[0, 0] = 0.0
    im[0, 1] = 100.0
    out = io.stretch_uint8(im)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0    # (0 - 49.5) / 1 * 255 -> clamped to 0
    assert out[0, 1] == 255  # (100 - 49.5) / 1 * 255 -> clamped to 255
    assert out[5, 5] == 127  # (50 - 49.5) / 1 * 255 = 127.5 -> truncated


def test_stretch_accepts_uint16_and_float():
    u16 = np.array([[0, 1000]], np.uint16)
    out = io.stretch_uint8(u16)
    assert out.dtype == np.uint8
    assert out[0, 0] < out[0, 1]
    assert io.stretch_uint8(np.array([[0.0, 1e4]])).dtype == np.uint8


def test_stretch_flat_frame_is_zero():
    out = io.stretch_uint8(np.full((8, 8), 1234, np.uint16))
    assert out.dtype == np.uint8
    assert (out == 0).all()


# --- 128-neutral difference mapping ------------------------------------------


def test_map_to_uint8_zero_diff_is_128():
    out = io.map_to_uint8(np.zeros((4, 4)), 123.0)
    assert out.dtype == np.uint8
    assert (out == 128).all()


def test_map_to_uint8_known_values():
    diff = np.array([[0.0, 64.0, -64.0, 127.0, -127.0]])
    # limit 127 -> 128 + diff * 127/127 = 128 + diff
    out = io.map_to_uint8(diff, 127.0)
    assert out[0].tolist() == [128, 192, 64, 255, 1]


def test_map_to_uint8_clamps_at_the_edges():
    out = io.map_to_uint8(np.array([[500.0, -500.0]]), 100.0)
    assert out[0].tolist() == [255, 0]


def test_map_to_uint8_rejects_nonpositive_limit():
    with pytest.raises(ValueError):
        io.map_to_uint8(np.zeros(3), 0.0)


# --- display limit -----------------------------------------------------------


def test_display_limit_known_distribution():
    # 100x100 with a stride-8 grid: 13x13 = 169 samples; 4 of them (2.4% >
    # the 0.5% tail) carry the only nonzero value, so the 99.5th percentile
    # is exactly that value.
    diff = np.zeros((100, 100))
    diff[88::8, 88::8] = 1000.0
    assert io.display_limit(diff, 99.5, 8) == 1000.0


def test_display_limit_uses_the_subsampled_grid():
    diff = np.zeros((100, 100))
    diff[99, 99] = 1000.0  # off the stride-8 grid
    assert io.display_limit(diff, 99.5, 8) == 1.0


def test_display_limit_over_several_frames():
    d1 = np.zeros((20, 20))
    d2 = np.zeros((20, 20))
    for y, x in [(0, 0), (0, 2), (2, 0), (2, 2), (4, 4)]:  # stride-2 grid
        d2[y, x] = 2000.0  # 5 of 200 samples (2.5%)
    assert io.display_limit([d1, d2], 99.5, 2) == 2000.0


def test_display_limit_floors_at_one():
    assert io.display_limit(np.zeros((10, 10)), 99.5, 1) == 1.0
    assert io.display_limit([], 99.5, 8) == 1.0


# --- scale.txt (difference output, per well) ---------------------------------


def test_scale_roundtrip(tmp_path):
    well = tmp_path / "-A001"
    well.mkdir()
    io.write_scale(well, RAW_NAME, 3691.0)
    text = (well / "scale.txt").read_text()
    assert f"reference\t{RAW_NAME}\n" in text
    assert "display_limit_int16\t3691.0\n" in text

    scale = io.read_scale(well)
    assert scale["reference"] == RAW_NAME
    assert float(scale["display_limit_int16"]) == 3691.0
    # The reference's frame number is recoverable with the shared parser.
    assert io.parse_frame(scale["reference"]) == 1


def test_scale_reader_tolerates_legacy_lines(tmp_path):
    # The prototype also wrote flatfield/sigma lines; readers must keep them.
    well = tmp_path / "-A001"
    well.mkdir()
    (well / "scale.txt").write_text(
        f"reference\t{RAW_NAME}\nflatfield\tFalse\nsigma\t150.0\n"
        "display_limit_int16\t3691.0\n"
    )
    scale = io.read_scale(well)
    assert scale["reference"] == RAW_NAME
    assert scale["flatfield"] == "False"
    assert scale["sigma"] == "150.0"
    assert float(scale["display_limit_int16"]) == 3691.0


def test_scale_reader_missing_file(tmp_path):
    well = tmp_path / "-A001"
    well.mkdir()
    assert io.read_scale(well) == {}


# --- done marker (per well) ---------------------------------------------------


def test_segment_done_keys_match_spec():
    assert io.SEGMENT_DONE_KEYS == (
        "frames",
        "groups",
        "colonies",
        "pre_existing_colonies",
        "merge_events",
        "forked_segments",
        "discarded_instances",
    )


def test_done_marker_roundtrip(tmp_path):
    well = tmp_path / "-A001"
    well.mkdir()
    counters = {
        "frames": 19, "groups": 4, "colonies": 7, "pre_existing_colonies": 1,
        "merge_events": 2, "forked_segments": 0, "discarded_instances": 5,
    }
    io.write_done(well, counters)
    assert (well / "_done.txt").exists()
    assert io.is_done(well)
    assert io.read_done(well) == counters


def test_done_marker_reader_tolerates_missing_keys(tmp_path):
    well = tmp_path / "-A001"
    well.mkdir()
    (well / "_done.txt").write_text("frames\t8\ncolonies\t3\n")
    assert io.read_done(well) == {"frames": 8, "colonies": 3}
    assert io.is_done(well)


def test_done_marker_absent(tmp_path):
    well = tmp_path / "-A001"
    well.mkdir()
    assert not io.is_done(well)
    with pytest.raises(FileNotFoundError):
        io.read_done(well)


# --- colonies.csv (canonical columns, legacy mapping) ------------------------


def test_colonies_columns_match_spec():
    assert io.COLONIES_COLUMNS == (
        "well", "frame", "t_seconds", "image", "colony_id", "group_id",
        "merged_with", "pre_existing", "area_px", "centroid_y", "centroid_x",
    )


def test_colonies_writer_writes_canonical_columns(tmp_path):
    path = tmp_path / "colonies.csv"
    rows = [
        {
            "well": "-A001", "frame": "7", "t_seconds": "21600043",
            "image": RAW_NAME, "colony_id": "1", "group_id": "1",
            "merged_with": "", "pre_existing": "0", "area_px": "5706",
            "centroid_y": "89.1", "centroid_x": "179.6",
        }
    ]
    io.write_colonies(path, rows)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(io.COLONIES_COLUMNS)
    assert io.read_colonies(path) == rows


def test_colonies_reader_maps_legacy_column_names(tmp_path):
    # A prototype-written colonies.csv: `lo` instead of `frame`, `n_merged`
    # instead of `ancestors`. Values must come through unchanged.
    path = tmp_path / "colonies.csv"
    path.write_text(
        "well,lo,t_seconds,image,colony_id,group_id,n_merged,merged_with,"
        "pre_existing,area_px,centroid_y,centroid_x\n"
        "-A001,007,21600043,IMG,1,1,2,3;4,0,5706,89.1,179.6\n"
        "-A001,008,57599662,IMG,1,1,2,3;4,0,6000,89.2,179.5\n"
    )
    rows = io.read_colonies(path)
    assert rows[0]["frame"] == "007"
    assert rows[0]["ancestors"] == "2"
    assert rows[0]["merged_with"] == "3;4"
    assert "lo" not in rows[0] and "n_merged" not in rows[0]
    assert rows[1]["t_seconds"] == "57599662"


def test_reader_maps_analysis_legacy_column_names(tmp_path):
    # The prototype's analysis outputs: trajectories.csv (lo, n_merged) and
    # growth_rates.csv (first_lo, last_lo, max_n_merged, rim_excluded_lo).
    traj = tmp_path / "trajectories.csv"
    traj.write_text(
        "well,lo,t_hours,colony_id,group_id,merged_with,n_merged,pre_existing,"
        "area_px,age_h,rim_frac,rim_excluded,bad,bad_reason\n"
        "-A001,7,6.0,1,1,,1,0,5706,,1.0,1,0,\n"
    )
    rows = io.read_csv_rows(traj)
    assert rows[0]["frame"] == "7"
    assert rows[0]["ancestors"] == "1"
    assert rows[0]["t_hours"] == "6.0"
    assert "lo" not in rows[0] and "n_merged" not in rows[0]

    rates = tmp_path / "growth_rates.csv"
    rates.write_text(
        "well,colony_id,group_id,first_lo,last_lo,n_frames,n_frames_valid,"
        "area_first,area_last,max_n_merged,rim_excluded_lo,growth_rate_per_h,"
        "growth_doublings_per_day,abs_growth_px_per_h,bad,bad_reason,pre_existing\n"
        "-A002,1,1,11,19,9,9,3202,123328,1,7,0.46984,16.268,15015.8,0,,0\n"
    )
    rows = io.read_csv_rows(rates)
    assert rows[0]["first_frame"] == "11"
    assert rows[0]["last_frame"] == "19"
    assert rows[0]["max_ancestors"] == "1"
    assert rows[0]["rim_excluded_frame"] == "7"
    assert rows[0]["well"] == "-A002"


# --- well_geometry.csv (shared cache in the analysis out root) ---------------


def test_geometry_roundtrip(tmp_path):
    path = tmp_path / "well_geometry.csv"
    geometry = {"-A002": (1029.8, 1079.1, 926.6), "-A001": (1036.44, 1064.76, 941.61)}
    io.write_geometry(path, geometry)
    lines = path.read_text().splitlines()
    assert lines[0] == "well,cy,cx,R"
    # cy, cx, R all at one decimal; wells sorted.
    assert lines[1] == "-A001,1036.4,1064.8,941.6"
    assert lines[2] == "-A002,1029.8,1079.1,926.6"

    back = io.read_geometry(path)
    assert back["-A001"] == (pytest.approx(1036.4), pytest.approx(1064.8), pytest.approx(941.6))
    assert back["-A002"][2] == pytest.approx(926.6)


def test_geometry_read_missing_file(tmp_path):
    assert io.read_geometry(tmp_path / "well_geometry.csv") == {}


def test_geometry_update_read_modify_write(tmp_path):
    path = tmp_path / "well_geometry.csv"
    io.update_geometry(path, "-A001", 1036.4, 1064.7, 941.6)
    io.update_geometry(path, "-B004", 900.0, 900.0, 950.0)
    back = io.read_geometry(path)
    assert set(back) == {"-A001", "-B004"}
    # Updating an existing well replaces it in place.
    io.update_geometry(path, "-A001", 1.0, 2.0, 3.0)
    back = io.read_geometry(path)
    assert back["-A001"] == (1.0, 2.0, 3.0)
    assert back["-B004"] == (900.0, 900.0, 950.0)


# --- label images -------------------------------------------------------------


def test_label_image_roundtrip(tmp_path):
    labels = np.zeros((6, 6), np.uint32)
    labels[1, 1] = 3
    labels[2, 2] = 7
    path = tmp_path / "LO001__labels.tif"
    io.write_label_image(path, labels)
    back = io.read_label_image(path)
    assert back.dtype == np.uint32
    assert (back == labels).all()


# --- the conftest builders (sanity: later tickets inherit these) --------------


def test_frame_name_uses_the_packages_acquisition_vocabulary():
    # The fixture's file names must round-trip through the package parsers.
    name = frame_name("-C006", 15, 14_399_594)
    assert io.parse_well(name) == "-C006"
    assert io.parse_frame(name) == 15
    assert io.parse_time(name) == 14_399_594
    # Millisecond T fields, zero-padded to 10 digits.
    assert "--T0014399594--" in name


def test_mini_plate_flat_layout_and_ground_truth(tmp_path):
    # One "full" well at the real plate size, noise-free (exact arithmetic).
    # The default planting is calibrated for 2048x2048 with the ring at R 920.
    root = tmp_path / "plate"
    plate = build_mini_plate(
        root, layout="flat", wells=("-A001",), n_frames=8,
        size=2048, noise_sigma=0,
    )

    # The tree: all 8 frames of the one well in one directory.
    assert plate.layout == "flat"
    names = plate.wells["-A001"]
    assert len(names) == 8
    for lo in range(1, 9):
        assert (root / names[lo - 1]).is_file()
    assert names[0] == frame_name("-A001", 1, 0)
    # Millisecond T fields (hourly frames).
    assert plate.t_values == [i * 3_600_000 for i in range(8)]
    assert [io.parse_time(n) for n in names] == plate.t_values

    # 2048x2048 uint16 frames, readable.
    img1 = tifffile.imread(root / names[0])
    assert img1.shape == (2048, 2048) and img1.dtype == np.uint16
    assert img1[0, 0] == 5000  # plain background in the corner

    # Discovery of the built tree agrees with the planted ground truth.
    wells = io.discover_wells(root)
    assert set(wells) == {"-A001"}
    assert [f.frame for f in wells["-A001"]] == list(range(1, 9))
    assert [f.t for f in wells["-A001"]] == plate.t_values

    # The planted ring: R ~ 920 at the image centre, visible in frame 1.
    assert plate.ring["-A001"] == (1024.0, 1024.0, 920.0)
    row = img1[1024, :]
    r_min = abs(int(np.argmin(row)) - 1024)
    assert 900 <= r_min <= 940  # the meniscus dip is at R ~ 920
    assert row.min() <= 5000 - 2000  # deep enough for a circle fit

    # The default "full" well planting: a growing pair, a static disk, one
    # present from frame 1, and one that grows into the rim.
    assert [d.name for d in plate.disks["-A001"]] == [
        "pair_a", "pair_b", "static", "from_start", "rim",
    ]
    for d in plate.disks["-A001"]:  # last_frame resolves to the plate's end
        assert d.last_frame == 8

    pair_a = plate.disk("-A001", "pair_a")
    pair_b = plate.disk("-A001", "pair_b")
    assert pair_a.radius_at(1) is None  # first appears at frame 2
    assert pair_a.radius_at(2) == 30.0
    assert pair_a.radius_at(3) > pair_a.radius_at(2)  # grows
    # The pair grows until it touches: the centres are 200 px apart.
    assert 2 * pair_a.radius_at(6) < 200 <= 2 * pair_a.radius_at(7)
    assert (pair_b.cy, pair_b.cx) == (650.0, 950.0)

    static = plate.disk("-A001", "static")
    assert static.radius_at(2) == static.radius_at(8) == 50.0  # no_growth
    # Ground-truth painted area: a full disk, far from the well wall.
    assert (disk_mask(plate, "-A001", "static", 8).sum()
            == pytest.approx(static.area_at(8), rel=0.02))

    from_start = plate.disk("-A001", "from_start")
    assert from_start.radius_at(1) == 30.0  # present from frame 1
    assert disk_mask(plate, "-A001", "from_start", 1) is not None
    assert disk_mask(plate, "-A001", "pair_a", 1) is None

    rim = plate.disk("-A001", "rim")
    assert rim.first_frame == 3
    # The rim disk starts well inside the wall and grows through it.
    assert 800 + rim.radius_at(3) < 860  # outside the 60 px rim band
    assert 800 + rim.radius_at(8) > 920  # crosses the well circle
    # The painted (wall-clipped) area is smaller than the unclipped one.
    assert disk_mask(plate, "-A001", "rim", 8).sum() < rim.area_at(8)

    # The planted disks are actually in the pixels.
    img8 = tifffile.imread(root / names[7])
    assert img8[1450, 1450] == 5000 - 2000  # the static disk (no noise)


def test_mini_plate_per_well_layout(tmp_path):
    # A cheap small plate (no ring, no disks) to check the layout plumbing.
    root = tmp_path / "plate"
    wells = ("-A001", "-B004")
    plate = build_mini_plate(
        root, layout="per_well", wells=wells, n_frames=2, size=256,
        noise_sigma=0, rings={w: None for w in wells}, disks={},
    )

    assert plate.layout == "per_well"
    for w in wells:
        well_dir = root / w
        assert well_dir.is_dir()
        assert plate.wells[w] == [frame_name(w, 1, 0), frame_name(w, 2, 3_600_000)]
        for name in plate.wells[w]:
            img = tifffile.imread(well_dir / name)
            assert img.shape == (256, 256) and img.dtype == np.uint16
            assert (img == 5000).all()  # background only (no planting)
    # No frames at the top level.
    assert [p for p in root.iterdir() if p.is_file()] == []

    # The per-well layout is discovered via the directory names.
    discovered = io.discover_wells(root)
    assert set(discovered) == set(wells)
    for w in wells:
        assert [f.frame for f in discovered[w]] == [1, 2]
        assert [f.t for f in discovered[w]] == [0, 3_600_000]

    # The overrides were honoured: no ring, no disks.
    assert all(plate.ring[w] is None for w in wells)
    assert all(plate.disks[w] == [] for w in wells)
    with pytest.raises(KeyError):
        plate.disk("-A001", "growing")


def test_mini_plate_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        build_mini_plate(tmp_path / "p", layout="grid")
    with pytest.raises(ValueError):
        build_mini_plate(tmp_path / "p", wells=[])
    with pytest.raises(ValueError):
        build_mini_plate(tmp_path / "p", n_frames=0)


def test_build_label_series(tmp_path):
    root = tmp_path / "-A001"
    labels = [np.zeros((8, 8), np.uint32) for _ in range(3)]
    labels[0][2, 2] = 1
    labels[1][2, 2] = 1
    labels[1][5, 5] = 2
    labels[2][3, 3] = 1  # the colony moved

    series = build_label_series(root, labels)

    # Canonical input names: the in-memory frame list later tickets pass on.
    assert series.names == [
        frame_name("-A001", lo, t)
        for lo, t in [(1, 0), (2, 3_600_000), (3, 2 * 3_600_000)]
    ]
    assert [io.parse_frame(n) for n in series.names] == [1, 2, 3]

    # One <frame-stem>__labels.tif per frame, round-tripping the values.
    assert len(series.label_paths) == 3
    for name, path, expected in zip(series.names, series.label_paths, labels):
        assert path == root / (name[:-4] + "__labels.tif")
        assert path.is_file()
        back = io.read_label_image(path)
        assert back.dtype == np.uint32
        assert (back == expected).all()

    with pytest.raises(ValueError):
        build_label_series(root, labels, names=[series.names[0]])
