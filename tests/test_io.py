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
