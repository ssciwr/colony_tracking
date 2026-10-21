"""Unit tests for the TUI's ``ConfigDraft`` (ticket 01).

Seams under test: the draft as the TUI's single config source of truth —
plain TOML-shaped values only (paths as strings, tuples as lists),
pipeline-owned fields refused at ``set`` (they are derived, never stored),
and ``resolve``/``write`` producing exactly the config
``PipelineConfig.from_dict`` builds from the equivalent TOML document —
so a TUI-written config file is one ``colony-tracking pipeline --config``
accepts unchanged. Errors are real ``ConfigError``s with the pipeline's
own messages, logged like the pipeline logs them.
"""

import tomllib
from pathlib import Path

import pytest

from colony_tracking import PipelineConfig
from colony_tracking.pipeline import _OWNED, ConfigError, read_config
from colony_tracking.tui.draft import ConfigDraft


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
