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
