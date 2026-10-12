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

# A realistic raw-plate frame name (real acquisition vocabulary, the
# millisecond T field zero-padded to 10 digits).
RAW_NAME = (
    "-A001--PO01--LO001--CO1--SL001--PX32500--PW0050--IN0020"
    "--TM366--X014222--Y011210--Z220046--T0000000000--WE00001.tif"
)


def _frame_name(well: str, lo: int, t_ms: int) -> str:
    """Build a canonical raw-plate frame name for discovery tests."""
    return (
        f"{well}--PO01--LO{lo:03d}--CO1--SL001--PX32500--PW0050--IN0020"
        f"--TM366--X014222--Y011210--Z220046--T{t_ms:010d}--WE00001.tif"
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
        _write_frame(tmp_path, _frame_name("-A001", lo, t))
    for lo, t in [(1, 0), (2, 2 * HOUR)]:
        _write_frame(tmp_path, _frame_name("-B004", lo, t))
    # Non-frame files are ignored.
    (tmp_path / "notes.txt").write_text("nope")
    _write_frame(tmp_path, "._" + _frame_name("-C001", 1, 0))  # resource fork
    _write_frame(tmp_path, "ref__labels.tif")  # no well, no LO field

    wells = io.discover_wells(tmp_path)
    assert set(wells) == {"-A001", "-B004"}
    assert [f.frame for f in wells["-A001"]] == [1, 2, 3]
    assert [f.t for f in wells["-A001"]] == [0, 2 * HOUR, 3 * HOUR]
    assert wells["-A001"][0].name == _frame_name("-A001", 1, 0)
    assert wells["-A001"][0].path == tmp_path / _frame_name("-A001", 1, 0)
    assert wells["-A001"][0].well == "-A001"


def test_discover_per_well_layout(tmp_path):
    a = tmp_path / "-A001"
    b = tmp_path / "-B004"
    a.mkdir()
    b.mkdir()
    for lo, t in [(3, 3 * HOUR), (1, 0), (2, 2 * HOUR)]:
        _write_frame(a, _frame_name("-A001", lo, t))
    _write_frame(a, "ref__labels.tif")  # no LO field: not a frame
    _write_frame(b, _frame_name("-B004", 1, 0))
    # A stray top-level frame is ignored in the per-well layout.
    _write_frame(tmp_path, _frame_name("-C001", 1, 0))

    wells = io.discover_wells(tmp_path)
    assert set(wells) == {"-A001", "-B004"}
    assert [f.frame for f in wells["-A001"]] == [1, 2, 3]
    assert wells["-A001"][2].path == a / _frame_name("-A001", 3, 3 * HOUR)


def test_discover_wells_filter_and_ordering(tmp_path):
    for lo, t in [(1, 0), (2, 2 * HOUR)]:
        _write_frame(tmp_path, _frame_name("-A001", lo, t))
    _write_frame(tmp_path, _frame_name("-B004", 1, 0))

    assert set(io.discover_wells(tmp_path, wells=["-B004"])) == {"-B004"}

    # Frames with the same LO order by the T field.
    _write_frame(tmp_path, _frame_name("-A001", 2, HOUR + 1))
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
