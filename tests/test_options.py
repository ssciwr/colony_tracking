"""Tests for the shared options machinery and the stage options dataclasses.

The dataclasses are plain, importable, and their fields and defaults are
the CLI contract of the spec's "Options and CLI contract" section — these
tests transcribe that contract verbatim.
"""

import dataclasses
from dataclasses import fields
from pathlib import Path

import pytest

from colony_tracking._options import (
    AnalysisOptions,
    DifferenceOptions,
    SegmentationOptions,
    parse_reference,
)


# --- parse_reference ---------------------------------------------------------


def test_parse_reference_accepts_the_cli_spellings():
    """``first`` (any case, surrounded by blanks), bare 1-based frame
    numbers, and the file-name ``LO<nnn>`` form — the spellings the CLI's
    ``--reference`` flag accepts."""
    assert parse_reference("first") == "first"
    assert parse_reference(" FIRST ") == "first"
    assert parse_reference("5") == 5
    assert parse_reference(" 005 ") == 5
    assert parse_reference("LO005") == 5
    assert parse_reference("lo5") == 5


def test_parse_reference_rejects_anything_else():
    """Anything else — including an empty string — raises ValueError with
    the parser's message."""
    with pytest.raises(ValueError, match=r"invalid reference 'second'"):
        parse_reference("second")
    with pytest.raises(ValueError, match="expected 'first', a 1-based frame"):
        parse_reference("")


# --- field contracts (names, order, required-ness) -------------------------


def test_difference_options_fields_match_contract():
    names = [f.name for f in fields(DifferenceOptions)]
    assert names == [
        "raw_dir",
        "out_dir",
        "reference",
        "wells",
        "display_percentile",
        "subsample",
        "n_workers",
        "dry_run",
    ]


def test_segmentation_options_fields_match_contract():
    names = [f.name for f in fields(SegmentationOptions)]
    assert names == [
        "input",
        "raw_dir",
        "diff_dir",
        "out_dir",
        "wells",
        "model_type",
        "checkpoint",
        "fg_ref",
        "fg_early",
        "min_size",
        "min_solidity",
        "mask_margin",
        "min_overlap",
        "max_fg_ratio",
        "exclude_before",
        "merge_touching",
        "redo",
        "reid_only",
    ]


def test_analysis_options_fields_match_contract():
    names = [f.name for f in fields(AnalysisOptions)]
    assert names == [
        "seg_dirs",
        "raw_dir",
        "out_root",
        "wells",
        "rim_width_px",
        "rim_overlap_max",
        "max_bad_segments",
        "keep_bad_wells",
        "keep_pre_existing",
        "raw_seg_dir",
        "start_frames",
        "min_growth_2step",
        "refresh_geometry",
        "plots_only",
        "no_plots",
        "no_qa",
        "no_group_alpha",
    ]


def test_required_fields_have_no_default():
    required = {
        DifferenceOptions: ["raw_dir", "out_dir"],
        SegmentationOptions: ["out_dir"],
        AnalysisOptions: ["seg_dirs", "raw_dir", "out_root"],
    }
    for cls, names in required.items():
        by_name = {f.name: f for f in fields(cls)}
        for name in names:
            assert by_name[name].default is dataclasses.MISSING, (
                f"{cls.__name__}.{name} must be required"
            )


def test_missing_required_field_raises():
    with pytest.raises(TypeError):
        DifferenceOptions()
    with pytest.raises(TypeError):
        SegmentationOptions()
    with pytest.raises(TypeError):
        AnalysisOptions()


# --- defaults, exactly as the spec states -----------------------------------


def test_difference_options_defaults():
    opts = DifferenceOptions(raw_dir=Path("raw"), out_dir=Path("out"))
    assert opts.reference == "first"
    assert opts.wells is None
    assert opts.display_percentile == 99.5
    assert opts.subsample == 8
    assert opts.n_workers == 6
    assert opts.dry_run is False


def test_segmentation_options_defaults():
    opts = SegmentationOptions(out_dir=Path("out"))
    assert opts.input == "raw"
    assert opts.raw_dir is None
    assert opts.diff_dir is None
    assert opts.wells is None
    assert opts.model_type == "vit_b_lm"
    assert opts.checkpoint is None
    assert opts.fg_ref == 0.5
    assert opts.fg_early == 0.3
    assert opts.min_size == 200
    assert opts.min_solidity == 0.7
    assert opts.mask_margin == 15
    assert opts.min_overlap == 0.5
    assert opts.max_fg_ratio == 1.5
    assert opts.exclude_before is None
    assert opts.merge_touching is True
    assert opts.redo is False
    assert opts.reid_only is False


def test_analysis_options_defaults():
    opts = AnalysisOptions(seg_dirs=[Path("seg")], raw_dir=Path("raw"), out_root=Path("out"))
    assert opts.wells is None
    assert opts.rim_width_px == 60
    assert opts.rim_overlap_max == 0.1
    assert opts.max_bad_segments == 2
    assert opts.keep_bad_wells is False
    assert opts.keep_pre_existing is False
    assert opts.raw_seg_dir is None
    assert opts.start_frames == 1
    assert opts.min_growth_2step == 0.3
    assert opts.refresh_geometry is False
    assert opts.plots_only is False
    assert opts.no_plots is False
    assert opts.no_qa is False
    assert opts.no_group_alpha is False


# --- per-field help text ----------------------------------------------------


def test_every_option_field_has_help_text():
    for cls in (DifferenceOptions, SegmentationOptions, AnalysisOptions):
        for f in fields(cls):
            assert f.metadata.get("help"), (
                f"{cls.__name__}.{f.name} has no per-field help text"
            )


# --- dict round-tripping (consumed by TOML config and the future TUI) -------


def test_to_dict_converts_paths_to_strings():
    opts = DifferenceOptions(raw_dir=Path("/data/plate"), out_dir=Path("/data/diff"))
    assert opts.to_dict()["raw_dir"] == "/data/plate"
    assert opts.to_dict()["out_dir"] == "/data/diff"


def test_from_dict_restores_paths():
    data = {
        "raw_dir": "/data/plate",
        "out_dir": "/data/diff",
        "reference": 5,
        "wells": ["-A01", "-B03"],
        "dry_run": True,
    }
    opts = DifferenceOptions.from_dict(data)
    assert opts.raw_dir == Path("/data/plate")
    assert opts.out_dir == Path("/data/diff")
    assert opts.reference == 5
    assert opts.dry_run is True


def test_difference_options_roundtrip():
    opts = DifferenceOptions(
        raw_dir=Path("/data/plate"),
        out_dir=Path("/data/diff"),
        reference=5,
        wells=["-A01", "-B03"],
        display_percentile=99.0,
        subsample=4,
        n_workers=2,
        dry_run=True,
    )
    assert DifferenceOptions.from_dict(opts.to_dict()) == opts


def test_analysis_options_roundtrip_with_path_lists():
    opts = AnalysisOptions(
        seg_dirs=[Path("/seg1"), Path("/seg2")],
        raw_dir=Path("/raw"),
        out_root=Path("/out"),
        raw_seg_dir=Path("/seg_raw"),
    )
    data = opts.to_dict()
    assert data["seg_dirs"] == ["/seg1", "/seg2"]
    assert AnalysisOptions.from_dict(data) == opts


def test_segmentation_options_roundtrip():
    opts = SegmentationOptions(
        input="diff",
        diff_dir=Path("/diff"),
        out_dir=Path("/out"),
        checkpoint=Path("/ckpt/vit_b.pt"),
        exclude_before=5,
        merge_touching=False,
        redo=True,
        reid_only=True,
    )
    assert SegmentationOptions.from_dict(opts.to_dict()) == opts
