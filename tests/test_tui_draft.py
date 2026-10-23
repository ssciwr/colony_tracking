"""Unit tests for the TUI's ``ConfigDraft``.

Seams under test: one draft per run type. The pipeline draft keeps the
sectioned chain document (``[pipeline]`` run level plus one stage section
each) with the pipeline's own semantics — plain TOML-shaped values only
(paths as strings, tuples as lists), pipeline-owned fields refused at
``set`` (the chain derives them, never stores them), and
``resolve``/``write`` producing exactly the config
``PipelineConfig.from_dict`` builds from the equivalent TOML document —
so a TUI-written ``pipeline.toml`` is one ``colony-tracking pipeline
--config`` accepts unchanged. The three stage drafts are flat,
fully-explicit documents of their options dataclass — the formerly
pipeline-owned keys (``input``/``diff_dir``, ``raw_dir``, ``reference``)
are ordinary keys — and ``resolve``/``write`` produce the options object
the equivalent flat TOML document builds, so a TUI-written stage TOML is
one a standalone stage run accepts unchanged. ``load`` reads a config
TOML of the draft's run type back (a load resets, it never merges;
structural validation only). Errors are real ``ConfigError``s with the
pipeline's / the machinery's own messages, logged like the pipeline logs
them.
"""

import logging
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from colony_tracking import PipelineConfig
from colony_tracking._options import (
    AnalysisOptions,
    DifferenceOptions,
    SegmentationOptions,
)
from colony_tracking.pipeline import _OWNED, ConfigError, read_config
from colony_tracking.tui.draft import RUN_TYPES, ConfigDraft


def _complete_draft() -> ConfigDraft:
    """A complete draft: the run level plus one tuned option per stage."""
    draft = ConfigDraft()
    draft.set("pipeline", "raw_dir", "/data/raw")
    draft.set("pipeline", "out_root", "/data/out")
    draft.set("pipeline", "reference", 3)
    draft.set("pipeline", "wells", ["-A001", "-B002"])
    draft.set("difference", "display_percentile", 99.9)
    draft.set("segmentation", "min_size", 300)
    draft.set("analysis", "rim_width_px", 40)
    return draft


#: Each stage run type's options dataclass — what a stage draft resolves
#: to and what its written TOML must round-trip through.
_STAGE_DATACLASS: dict[str, type] = {
    "difference": DifferenceOptions,
    "segmentation": SegmentationOptions,
    "analysis": AnalysisOptions,
}

#: A tuned, fully self-contained flat document per stage run type (the
#: brief's TOML shapes: every key an ordinary field, nothing owned,
#: nothing derived from any other run).
_STAGE_DOCS: dict[str, dict[str, Any]] = {
    "difference": {
        "raw_dir": "/data/raw-plate-frames",
        "out_dir": "/data/out/difference_LO003",
        "reference": 3,
        "wells": ["-A001", "-B002"],
        "display_percentile": 99.9,
    },
    "segmentation": {
        "input": "diff",
        "diff_dir": "/data/out/difference_LO003",
        "out_dir": "/data/out/seg_growth_diff_LO003",
        "min_size": 300,
    },
    "analysis": {
        "seg_dirs": ["/data/out/seg_growth_diff_LO003"],
        "raw_dir": "/data/raw-plate-frames",
        "out_root": "/data/out/analysis",
        "rim_width_px": 40,
    },
}


def _complete_stage_draft(run_type: str) -> ConfigDraft:
    """A complete stage draft: every required key of ``run_type``'s
    dataclass plus one tuned option — the standalone run's whole config."""
    draft = ConfigDraft(run_type)
    for key, value in _STAGE_DOCS[run_type].items():
        draft.set(run_type, key, value)
    return draft


#: The sections a stage draft does not hold: its own plus every other run
#: type's (a stage draft edits no run level, no sibling stage).
_FOREIGN = [
    (run_type, other)
    for run_type in ("difference", "segmentation", "analysis")
    for other in ("pipeline", "difference", "segmentation", "analysis")
    if other != run_type
]


# --- the document ---------------------------------------------------------------


def test_document_is_toml_shaped_and_detached():
    draft = _complete_draft()
    doc = draft.document()
    assert doc == {
        "pipeline": {
            "raw_dir": "/data/raw",
            "out_root": "/data/out",
            "reference": 3,
            "wells": ["-A001", "-B002"],
        },
        "difference": {"display_percentile": 99.9},
        "segmentation": {"min_size": 300},
        "analysis": {"rim_width_px": 40},
    }
    for stage, owned in _OWNED.items():  # owned keys are never stored
        assert not (set(doc[stage]) & owned)
    # The document is a copy: editing it cannot corrupt the draft.
    doc["pipeline"]["wells"].append("-C003")
    doc["pipeline"]["raw_dir"] = "elsewhere"
    assert draft.get("pipeline", "wells") == ["-A001", "-B002"]
    assert draft.get("pipeline", "raw_dir") == "/data/raw"


def test_empty_stage_sections_are_omitted_from_the_document():
    draft = ConfigDraft()
    draft.set("pipeline", "raw_dir", "/data/raw")
    draft.set("pipeline", "out_root", "/data/out")
    assert draft.document() == {
        "pipeline": {"raw_dir": "/data/raw", "out_root": "/data/out"},
    }


def test_set_stores_plain_values_only():
    draft = ConfigDraft()
    draft.set("pipeline", "raw_dir", Path("/data/raw"))  # Path -> str
    draft.set("pipeline", "out_root", Path("/data/out"))
    draft.set("pipeline", "wells", ("-A001", "-A002"))  # tuple -> list
    assert draft.get("pipeline", "raw_dir") == "/data/raw"
    assert draft.get("pipeline", "wells") == ["-A001", "-A002"]
    # The plain document resolves to the same config the equivalent
    # literal TOML document builds.
    assert draft.resolve() == PipelineConfig.from_dict(
        {
            "pipeline": {
                "raw_dir": "/data/raw",
                "out_root": "/data/out",
                "wells": ["-A001", "-A002"],
            },
        }
    )


# --- set / get / clear ------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "key"),
    sorted((stage, key) for stage, keys in _OWNED.items() for key in keys),
)
def test_owned_keys_are_refused(stage, key):
    draft = ConfigDraft()
    with pytest.raises(ConfigError, match="owned by the pipeline"):
        draft.set(stage, key, "whatever")
    assert draft.get(stage, key) is None  # nothing was stored


def test_unknown_section_is_refused():
    draft = ConfigDraft()
    for call in (
        lambda: draft.set("train", "n_epochs", 1),
        lambda: draft.get("train", "n_epochs"),
        lambda: draft.clear("train", "n_epochs"),
    ):
        with pytest.raises(ConfigError, match="unknown draft section 'train'"):
            call()


def test_get_and_clear_return_to_the_default():
    draft = ConfigDraft()
    draft.set("segmentation", "min_size", 300)
    assert draft.get("segmentation", "min_size") == 300
    assert draft.get("segmentation", "min_solidity") is None
    assert draft.get("segmentation", "min_solidity", 0.7) == 0.7
    draft.clear("segmentation", "min_size")
    assert draft.get("segmentation", "min_size") is None
    draft.clear("segmentation", "min_size")  # clearing an unset key: no-op


# --- resolve / write --------------------------------------------------------------


def test_resolve_of_an_empty_draft_reports_the_missing_keys(caplog):
    with pytest.raises(ConfigError, match="missing required"):
        ConfigDraft().resolve()
    assert "missing required" in caplog.text  # logged, not just raised


def test_resolve_rejects_unknown_keys_and_bypassed_owned_keys():
    draft = _complete_draft()
    draft.set("difference", "nonsense", 1)
    with pytest.raises(ConfigError, match="unknown key"):
        draft.resolve()
    draft.clear("difference", "nonsense")
    # An owned key placed behind the draft's back (the sections are plain
    # dicts) still fails resolution: resolve validates the real document.
    draft.stages["difference"]["raw_dir"] = "/elsewhere"
    with pytest.raises(ConfigError, match="owned by the pipeline"):
        draft.resolve()


def test_resolve_equals_from_dict_of_the_document():
    draft = _complete_draft()
    config = draft.resolve()
    assert config == PipelineConfig.from_dict(draft.document())
    # The chain wiring resolved exactly as the pipeline derives it.
    assert config.difference.out_dir.name == "difference_LO003"
    assert config.segmentation.input == "diff"
    assert config.segmentation.diff_dir == config.difference.out_dir
    assert config.analysis.seg_dirs == [config.segmentation.out_dir]


def test_written_toml_is_what_the_cli_would_build(tmp_path):
    draft = _complete_draft()
    path = tmp_path / "configs" / "pipeline.toml"  # parents are created
    config = draft.write(path)
    assert config == draft.resolve()
    with open(path, "rb") as handle:
        doc = tomllib.load(handle)
    assert PipelineConfig.from_dict(doc) == config
    assert read_config(path) == config
    for stage, owned in _OWNED.items():  # the written file omits them
        assert not (set(doc.get(stage, {})) & owned)


# --- stage drafts: flat, fully-explicit documents ---------------------------------


def test_stage_document_is_flat_and_detached():
    """A stage draft's document is flat — a detached copy of its own
    section where every dataclass field is an ordinary key (the formerly
    pipeline-owned ``raw_dir``/``reference`` included) — with no run
    level, because the standalone run derives nothing from one."""
    draft = _complete_stage_draft("difference")
    assert draft.is_pipeline is False
    assert draft.run_level == {}
    assert list(draft.stages) == ["difference"]
    doc = draft.document()
    assert doc == _STAGE_DOCS["difference"]  # flat, no [pipeline] section
    # The copy is detached — even a list value will not alias the draft.
    doc["wells"].append("-C003")
    doc["raw_dir"] = "elsewhere"
    assert draft.get("difference", "wells") == ["-A001", "-B002"]
    assert draft.get("difference", "raw_dir") == "/data/raw-plate-frames"


@pytest.mark.parametrize(("run_type", "foreign"), _FOREIGN)
def test_stage_draft_refuses_foreign_sections(run_type, foreign):
    """A stage draft edits its own section only: set/get/clear of any other
    run type's section — the run level included — refuses with the
    machinery's exact independence wording."""
    draft = ConfigDraft(run_type)
    expected = (
        f"unknown draft section {foreign!r} — the {run_type} draft edits "
        f"its [{run_type}] section only (each run type holds an independent "
        f"config)"
    )
    for call in (
        lambda: draft.set(foreign, "nonsense", 1),
        lambda: draft.get(foreign, "nonsense"),
        lambda: draft.clear(foreign, "nonsense"),
    ):
        with pytest.raises(ConfigError) as info:
            call()
        assert str(info.value) == expected


@pytest.mark.parametrize(
    ("run_type", "key", "value", "plain"),
    [
        ("difference", "raw_dir", Path("/data/raw-plate-frames"),
         "/data/raw-plate-frames"),
        ("difference", "reference", 3, 3),
        ("difference", "wells", ("-A001", "-B002"), ["-A001", "-B002"]),
        ("segmentation", "input", "diff", "diff"),
        ("segmentation", "diff_dir", Path("/data/out/difference_LO003"),
         "/data/out/difference_LO003"),
        ("analysis", "raw_dir", Path("/data/raw-plate-frames"),
         "/data/raw-plate-frames"),
        ("analysis", "seg_dirs", (Path("/data/out/seg_growth_diff"),),
         ["/data/out/seg_growth_diff"]),
    ],
    ids=["diff-raw_dir", "diff-reference", "diff-wells", "seg-input",
         "seg-diff_dir", "ana-raw_dir", "ana-seg_dirs"],
)
def test_stage_drafts_store_their_owned_keys(run_type, key, value, plain):
    """On a stage draft every key of its dataclass is storable — the
    standalone run's config is self-contained, with ``input``/``diff_dir``
    /``raw_dir``/``reference`` ordinary keys like any other. (The pipeline
    draft still refuses each of them: the refusal test above.)"""
    draft = ConfigDraft(run_type)
    draft.set(run_type, key, value)  # plain values only, as anywhere else
    assert draft.get(run_type, key) == plain
    assert draft.document() == {key: plain}


def test_unknown_run_type_is_refused():
    """A run type no registry knows refuses to construct a draft (``train``
    is not a config section any more — a run type would own one)."""
    with pytest.raises(ConfigError) as info:
        ConfigDraft("train")
    assert str(info.value) == (
        f"unknown run type 'train' — expected one of {list(RUN_TYPES)}"
    )


# --- stage drafts: resolve ---------------------------------------------------------


@pytest.mark.parametrize(
    ("run_type", "expected"),
    [
        ("difference", "missing required key(s): ['raw_dir', 'out_dir']"),
        ("segmentation", "missing required key(s): ['out_dir']"),
        ("analysis", "missing required key(s): ['seg_dirs', 'raw_dir', 'out_root']"),
    ],
    ids=["difference", "segmentation", "analysis"],
)
def test_stage_resolve_reports_the_missing_requireds_in_declaration_order(
    run_type, expected, caplog
):
    """An empty stage draft refuses to resolve with its dataclass's
    required keys in declaration order — logged with the run type's own
    prefix, ready for a screen's gate to show verbatim."""
    with pytest.raises(ConfigError) as info:
        ConfigDraft(run_type).resolve()
    assert str(info.value) == expected
    assert f"{run_type} config: {expected}" in caplog.text


def test_stage_resolve_rejects_unknown_keys():
    """Unknown keys refuse with the resolve wording — checked before the
    missing requireds (checked first: a typo always wins)."""
    names = sorted(f.name for f in fields(DifferenceOptions))
    expected = f"unknown key(s) ['nonsense'] — expected one key per name: {names}"
    draft = _complete_stage_draft("difference")
    draft.set("difference", "nonsense", 1)
    with pytest.raises(ConfigError) as info:
        draft.resolve()
    assert str(info.value) == expected
    # Unknown keys are checked first — even on an empty draft.
    draft.clear("difference", "raw_dir")
    draft.clear("difference", "out_dir")
    with pytest.raises(ConfigError) as info:
        draft.resolve()
    assert str(info.value) == expected


def test_stage_resolve_rejects_mistyped_values(caplog):
    """A mistyped value refuses with the pipeline's own wording, prefixed
    by the run type — here the tuned ``display_percentile`` typed as text."""
    draft = _complete_stage_draft("difference")
    draft.set("difference", "display_percentile", "abc")
    expected = "difference: key 'display_percentile' must be a number (got 'abc')"
    with pytest.raises(ConfigError) as info:
        draft.resolve()
    assert str(info.value) == expected
    assert f"difference config: {expected}" in caplog.text  # logged


@pytest.mark.parametrize(
    ("run_type", "doc"), sorted(_STAGE_DOCS.items()), ids=list(_STAGE_DOCS)
)
def test_stage_resolve_equals_from_dict_of_the_literal_document(run_type, doc):
    """A stage draft resolves to exactly the options object the equivalent
    literal flat document builds — every key ordinary, defaults filling
    the omitted ones, nothing derived from any other run."""
    draft = ConfigDraft(run_type)
    for key, value in doc.items():
        draft.set(run_type, key, value)
    assert draft.resolve() == _STAGE_DATACLASS[run_type].from_dict(doc)


# --- stage drafts: write --------------------------------------------------------------


@pytest.mark.parametrize("run_type", list(_STAGE_DOCS), ids=list(_STAGE_DOCS))
def test_stage_write_is_the_flat_toml_the_dataclass_round_trips(tmp_path, run_type):
    """A stage draft writes its run type's own flat TOML: defaults filled,
    parents created, and the file reads back to the resolved options
    through the dataclass's ``from_dict``."""
    draft = _complete_stage_draft(run_type)
    path = tmp_path / "configs" / f"{run_type}.toml"  # parents are created
    options = draft.write(path)
    assert options == draft.resolve()
    assert isinstance(options, _STAGE_DATACLASS[run_type])
    with open(path, "rb") as handle:
        doc = tomllib.load(handle)
    assert doc == {k: v for k, v in options.to_dict().items() if v is not None}
    assert _STAGE_DATACLASS[run_type].from_dict(doc) == options


def test_stage_write_omits_the_none_values(tmp_path):
    """An unset (``None``) option stays out of the written file: the
    defaults fill the file, the unset optional keys do not."""
    draft = ConfigDraft("difference")
    draft.set("difference", "raw_dir", "/data/raw-plate-frames")
    draft.set("difference", "out_dir", "/data/out/difference_LO003")
    path = tmp_path / "difference.toml"
    options = draft.write(path)
    assert isinstance(options, DifferenceOptions)
    assert options.wells is None
    with open(path, "rb") as handle:
        doc = tomllib.load(handle)
    assert "wells" not in doc
    assert DifferenceOptions.from_dict(doc) == options


# --- load: reading a config TOML back into a draft -----------------------------------


def test_stage_load_resets_the_draft(tmp_path, caplog):
    """Reading a stage config replaces the draft — a load resets, it never
    merges: every key absent from the file is gone — and returns the
    path, logging the machinery's success line."""
    fixture = tmp_path / "difference.toml"
    fixture.write_text(
        'raw_dir = "/data/raw-plate-frames"\n'
        'out_dir = "/data/out/difference_LO003"\n'
    )
    draft = ConfigDraft("difference")
    draft.set("difference", "display_percentile", 99.9)
    draft.set("difference", "wells", ["-A001"])
    with caplog.at_level(logging.INFO, logger="colony_tracking"):
        returned = draft.load(fixture)
    assert returned == fixture
    assert draft.document() == {
        "raw_dir": "/data/raw-plate-frames",
        "out_dir": "/data/out/difference_LO003",
    }
    assert f"difference config: loaded {fixture}" in caplog.text


def test_stage_load_validates_structure_only(tmp_path):
    """Unknown keys are refused at load (a typo cannot sit in the draft
    unnoticed) with the resolve wording — but missing requireds and
    mistyped values pass and are only reported verbatim by ``resolve``."""
    loaded = ConfigDraft("difference")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(
        'raw_dir = "/data/raw"\nout_dir = "/data/out/diff"\nnonsense = 1\n'
    )
    names = sorted(f.name for f in fields(DifferenceOptions))
    with pytest.raises(ConfigError) as info:
        loaded.load(unknown)
    assert str(info.value) == (
        f"unknown key(s) ['nonsense'] — expected one key per name: {names}"
    )
    # Missing requireds sit in the draft until the next resolve.
    partial = tmp_path / "partial.toml"
    partial.write_text("display_percentile = 99.5\n")
    loaded.load(partial)
    assert loaded.document() == {"display_percentile": 99.5}
    with pytest.raises(ConfigError) as info:
        loaded.resolve()
    assert str(info.value) == "missing required key(s): ['raw_dir', 'out_dir']"
    # Mistyped values ditto.
    mistyped = tmp_path / "mistyped.toml"
    mistyped.write_text(
        'raw_dir = "/data/raw"\nout_dir = "/data/out/diff"\n'
        'display_percentile = "abc"\n'
    )
    loaded.load(mistyped)
    with pytest.raises(ConfigError) as info:
        loaded.resolve()
    assert str(info.value) == (
        "difference: key 'display_percentile' must be a number (got 'abc')"
    )


def test_stage_load_reports_decode_errors(tmp_path, caplog):
    """A file that is not TOML reports under the decode-error wording,
    logged with the run type's own prefix."""
    fixture = tmp_path / "malformed.toml"
    fixture.write_text("this is ] not toml =\n")
    with pytest.raises(ConfigError) as info:
        ConfigDraft("difference").load(fixture)
    assert str(info.value).startswith(f"{fixture}: not valid TOML: ")
    assert f"difference config: {fixture}: not valid TOML: " in caplog.text


def test_pipeline_load_refuses_unknown_sections_and_keys(tmp_path):
    """Structural validation, pipeline edition: unknown top-level sections
    (``train`` again) and unknown keys inside a known section refuse with
    ``from_dict``'s wordings."""
    draft = ConfigDraft()
    fixture = tmp_path / "train.toml"
    fixture.write_text(
        '[pipeline]\nraw_dir = "/r"\nout_root = "/o"\n[train]\nn_epochs = 3\n'
    )
    with pytest.raises(ConfigError) as info:
        draft.load(fixture)
    assert str(info.value) == (
        "unknown section(s) ['train'] — expected one section per name: "
        "['pipeline', 'difference', 'segmentation', 'analysis']"
    )
    fixture.write_text('[pipeline]\nraw_dir = "/r"\nout_root = "/o"\nnonsense = 1\n')
    with pytest.raises(ConfigError) as info:
        draft.load(fixture)
    assert str(info.value) == (
        "unknown key(s) ['nonsense'] in [pipeline] — expected one key per "
        "name: ['raw_dir', 'out_root', 'reference', 'wells']"
    )


def test_pipeline_load_is_structural_only(tmp_path):
    """Owned keys, missing requireds, and mistyped values pass a pipeline
    load and surface verbatim — the pipeline's own wordings — at
    ``resolve``."""
    draft = ConfigDraft()
    owned = tmp_path / "owned.toml"
    owned.write_text(
        '[pipeline]\nraw_dir = "/r"\nout_root = "/o"\n'
        '[difference]\nraw_dir = "/elsewhere"\n'
    )
    draft.load(owned)
    assert draft.document() == {
        "pipeline": {"raw_dir": "/r", "out_root": "/o"},
        "difference": {"raw_dir": "/elsewhere"},
    }
    with pytest.raises(ConfigError) as info:
        draft.resolve()
    assert str(info.value) == (
        "[difference] key(s) are owned by the pipeline — 'raw_dir': "
        "it comes from [pipeline].raw_dir"
    )
    mistyped = tmp_path / "mistyped.toml"
    mistyped.write_text(
        '[pipeline]\nraw_dir = "/r"\nout_root = "/o"\n'
        '[difference]\ndisplay_percentile = "abc"\n'
    )
    draft.load(mistyped)
    with pytest.raises(ConfigError) as info:
        draft.resolve()
    assert str(info.value) == (
        "[difference]: key 'display_percentile' must be a number (got 'abc')"
    )


def test_pipeline_load_resets_the_draft(tmp_path, caplog):
    """A pipeline load replaces the whole document — the run level and
    every stage section not in the file are gone — and logs the success
    line."""
    fixture = tmp_path / "sparse.toml"
    fixture.write_text('[pipeline]\nraw_dir = "/new"\n')
    draft = ConfigDraft()
    draft.set("pipeline", "out_root", "/old-root")
    draft.set("difference", "display_percentile", 99.9)
    draft.set("analysis", "rim_width_px", 40)
    with caplog.at_level(logging.INFO, logger="colony_tracking"):
        draft.load(fixture)
    assert draft.document() == {"pipeline": {"raw_dir": "/new"}}
    assert f"pipeline config: loaded {fixture}" in caplog.text


# --- the four drafts are independent ---------------------------------------------


def test_drafts_of_different_run_types_are_independent():
    """Each run type holds its own draft and nothing else's: editing the
    difference draft leaves the pipeline draft's document untouched and
    vice versa — separate objects that share no sections."""
    difference, pipeline = ConfigDraft("difference"), ConfigDraft()
    difference.set("difference", "raw_dir", "/elsewhere")
    pipeline.set("pipeline", "raw_dir", "/data/pipeline-raw")
    assert difference is not pipeline
    assert difference.document() == {"raw_dir": "/elsewhere"}
    assert pipeline.document() == {"pipeline": {"raw_dir": "/data/pipeline-raw"}}
    # Both drafts happen to hold a section named "difference" — distinct ones.
    assert difference.stages["difference"] is not pipeline.stages["difference"]
