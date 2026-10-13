"""Contract tests for the CLI entry point (ticket 09).

Seams under test (per the ticket): ``colony_tracking.cli.main(argv)`` driven
in-process — the exit-code contract (0 success, 1 runtime error, 2 usage
error), the kebab-case flag mapping asserted on the constructed options
dataclass (each ``run_*`` function is monkeypatched to record what it was
given), logging through the package logger (caplog), and the real
difference/analyze runs on the conftest builders' small plates (the segment
and train stages are only exercised through option construction and error
cases — never the model).

Well names start with a dash, so every ``--wells``-style value stays inside
the flag value: both ``--wells=-A001,-A002`` and ``--wells -A001`` must
parse.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

import colony_tracking.cli as cli
from colony_tracking import (
    AnalysisOptions,
    AnalysisSummary,
    DifferenceOptions,
    DifferenceSummary,
    EvaluateOptions,
    EvaluateSummary,
    PipelineConfig,
    PipelineSummary,
    PrepareOptions,
    PrepareSummary,
    SegmentationOptions,
    SegmentationSummary,
    TrainOptions,
    TrainSummary,
    io,
)
from colony_tracking.difference import WellResult
from conftest import (
    DiskSpec,
    RingSpec,
    build_label_series,
    build_mini_plate,
    disk_mask,
)


def _diff_summary(**kwargs) -> DifferenceSummary:
    return DifferenceSummary(
        wells={}, processed=1, skipped=0, failed=0, **kwargs
    )


# --- flag parsing (difference) -------------------------------------------------


def test_difference_flags_map_kebab_case(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_difference",
        lambda options: (calls.append(options), _diff_summary())[1],
    )
    raw, out = tmp_path / "raw", tmp_path / "out"
    code = cli.main([
        "difference", "--raw-dir", str(raw), "--out-dir", str(out),
        "--reference", "LO005", "--wells=-A001,-A002", "--wells", "-B005",
        "--display-percentile", "99.9", "--subsample", "4", "--n-workers", "2",
    ])

    assert code == 0
    (options,) = calls
    assert options == DifferenceOptions(
        raw_dir=raw, out_dir=out, reference=5,
        wells=["-A001", "-A002", "-B005"],
        display_percentile=99.9, subsample=4, n_workers=2,
    )


def test_difference_reference_first_bare_number_and_dry_run(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_difference",
        lambda options: (calls.append(options), _diff_summary())[1],
    )
    base = ["difference", "--raw-dir", str(tmp_path / "raw"),
            "--out-dir", str(tmp_path / "out")]

    assert cli.main([*base, "--reference", "first"]) == 0
    assert calls[-1].reference == "first"
    assert calls[-1].dry_run is False

    assert cli.main([*base, "--reference", "5", "--dry-run"]) == 0
    assert calls[-1].reference == 5
    assert calls[-1].dry_run is True

    assert cli.main([*base]) == 0
    assert calls[-1].reference == "first"  # the spec default


# --- flag parsing (segment) -----------------------------------------------------


def test_segment_flags_map_kebab_case(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_segmentation",
        lambda options: (calls.append(options), SegmentationSummary(
            wells=1, processed=1, skipped=0, failed=0))[1],
    )
    raw, out = tmp_path / "raw", tmp_path / "out"
    code = cli.main([
        "segment", "--input", "raw", "--raw-dir", str(raw),
        "--out-dir", str(out), "--wells", "-A001",
        "--model-type", "vit_t", "--checkpoint", str(tmp_path / "ckpt.pth"),
        "--fg-ref", "0.4", "--fg-early", "0.2", "--min-size", "100",
        "--min-solidity", "0.6", "--mask-margin", "10", "--min-overlap", "0.4",
        "--max-fg-ratio", "2", "--exclude-before", "3",
        "--no-merge-touching", "--redo",
    ])

    assert code == 0
    (options,) = calls
    assert options == SegmentationOptions(
        input="raw", raw_dir=raw, out_dir=out, wells=["-A001"],
        model_type="vit_t", checkpoint=tmp_path / "ckpt.pth",
        fg_ref=0.4, fg_early=0.2, min_size=100, min_solidity=0.6,
        mask_margin=10, min_overlap=0.4, max_fg_ratio=2,
        exclude_before=3, merge_touching=False, redo=True,
    )


def test_segment_defaults_and_diff_input(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_segmentation",
        lambda options: (calls.append(options), SegmentationSummary(
            wells=1, processed=1, skipped=0, failed=0))[1],
    )
    diff, out = tmp_path / "diff", tmp_path / "out"
    code = cli.main([
        "segment", "--input", "diff", "--diff-dir", str(diff),
        "--out-dir", str(out),
    ])

    assert code == 0
    (options,) = calls
    assert options == SegmentationOptions(
        input="diff", diff_dir=diff, out_dir=out,
    )  # every other field at its spec default


# --- flag parsing (analyze) ------------------------------------------------------


def test_analyze_flags_map_kebab_case(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_analysis",
        lambda options: (calls.append(options), AnalysisSummary())[1],
    )
    seg_a, seg_b, raw, out_root = (tmp_path / p for p in
                                   ("seg_a", "seg_b", "raw", "out"))
    code = cli.main([
        "analyze", "--seg-dir", str(seg_a), "--seg-dir", str(seg_b),
        "--raw-dir", str(raw), "--out-root", str(out_root),
        "--wells", "-A001,-B002", "--rim-width-px", "40",
        "--rim-overlap-max", "0.2", "--max-bad-segments", "3",
        "--keep-bad-wells", "--keep-pre-existing",
        "--raw-seg-dir", str(tmp_path / "raw_seg"),
        "--start-frames", "2", "--min-growth-2step", "0.5",
        "--refresh-geometry", "--plots-only", "--no-plots", "--no-qa",
        "--no-group-alpha",
    ])

    assert code == 0
    (options,) = calls
    assert options == AnalysisOptions(
        seg_dirs=[seg_a, seg_b], raw_dir=raw, out_root=out_root,
        wells=["-A001", "-B002"], rim_width_px=40, rim_overlap_max=0.2,
        max_bad_segments=3, keep_bad_wells=True, keep_pre_existing=True,
        raw_seg_dir=tmp_path / "raw_seg", start_frames=2,
        min_growth_2step=0.5, refresh_geometry=True, plots_only=True,
        no_plots=True, no_qa=True, no_group_alpha=True,
    )


def test_analyze_seg_dir_comma_separated_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_analysis",
        lambda options: (calls.append(options), AnalysisSummary())[1],
    )
    seg, raw = tmp_path / "seg_a", tmp_path / "raw"
    code = cli.main([
        "analyze", f"--seg-dir={seg},{tmp_path / 'seg_b'}",
        "--raw-dir", str(raw), "--out-root", str(tmp_path / "out"),
    ])

    assert code == 0
    (options,) = calls
    assert options.seg_dirs == [seg, tmp_path / "seg_b"]
    # The defaults the spec lists for the analyze stage.
    assert options.rim_width_px == 60 and options.rim_overlap_max == 0.1
    assert options.max_bad_segments == 2 and options.start_frames == 1
    assert options.min_growth_2step == 0.3
    assert not (options.keep_bad_wells or options.keep_pre_existing)
    assert not (options.plots_only or options.no_plots or options.no_qa
                or options.no_group_alpha or options.refresh_geometry)
    assert options.raw_seg_dir is None and options.wells is None


# --- pipeline --------------------------------------------------------------------


def _write_config(tmp_path, text: str):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


_CONFIG = """\
[pipeline]
raw_dir = "raw"
out_root = "out"
reference = "first"

[difference]
subsample = 4

[analysis]
rim_width_px = 40
"""


def test_pipeline_config_and_run_level_overrides(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_pipeline",
        lambda config: (calls.append(config), PipelineSummary(status="ok"))[1],
    )
    config = _write_config(tmp_path, _CONFIG)
    raw = tmp_path / "other_raw"
    out_root = tmp_path / "other_out"
    code = cli.main([
        "pipeline", "--config", str(config),
        "--raw-dir", str(raw), "--out-root", str(out_root),
        "--reference", "5", "--wells=-A001,-B002",
    ])

    assert code == 0
    (resolved,) = calls
    # The run-level overrides win over the config file's [pipeline] values...
    assert isinstance(resolved, PipelineConfig)
    assert resolved.raw_dir == raw and resolved.out_root == out_root
    assert resolved.reference == 5 and resolved.wells == ["-A001", "-B002"]
    # ...and they flow into the resolved stages (derived names follow).
    assert resolved.difference.raw_dir == raw
    assert resolved.difference.reference == 5
    assert resolved.difference.out_dir == out_root / "difference_LO005"
    # Per-stage tuning from the config file is kept.
    assert resolved.difference.subsample == 4
    assert resolved.analysis.rim_width_px == 40
    # The segment stage consumes the derived difference tree.
    assert resolved.segmentation.input == "diff"
    assert resolved.segmentation.diff_dir == out_root / "difference_LO005"
    assert resolved.segmentation.exclude_before == 5


def test_pipeline_without_overrides_uses_config_values(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_pipeline",
        lambda config: (calls.append(config), PipelineSummary(status="ok"))[1],
    )
    config = _write_config(tmp_path, _CONFIG)
    code = cli.main(["pipeline", "--config", str(config)])

    assert code == 0
    (resolved,) = calls
    assert resolved.raw_dir == Path("raw") and resolved.out_root == Path("out")
    assert resolved.reference == "first" and resolved.wells is None
    assert resolved.difference.out_dir == Path("out") / "difference"
    assert resolved.difference.subsample == 4


def test_pipeline_missing_config_file_is_runtime_error(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["pipeline", "--config", str(tmp_path / "nope.toml")])

    assert code == 1
    assert "nope.toml" in caplog.text


def test_pipeline_unknown_config_key_is_usage_error(tmp_path, caplog):
    config = _write_config(
        tmp_path,
        _CONFIG.replace('reference = "first"',
                        'reference = "first"\nnonsense = true'),
    )

    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["pipeline", "--config", str(config)])

    assert code == 2
    assert "nonsense" in caplog.text


def test_pipeline_requires_config(monkeypatch):
    monkeypatch.setattr(
        cli, "run_pipeline",
        lambda config: PipelineSummary(status="ok"),
    )
    assert cli.main(["pipeline", "--raw-dir", "x"]) == 2


# --- flag parsing (train) --------------------------------------------------------


def test_train_prepare_flags_and_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_prepare",
        lambda options: (calls.append(options), PrepareSummary())[1],
    )
    raw, label, out = (tmp_path / p for p in ("raw", "labels", "h5"))
    code = cli.main([
        "train", "prepare", "--raw-dir", str(raw), "--label-dir", str(label),
        "--out-dir", str(out), "--val-fraction", "0.3", "--max-val-wells", "2",
        "--min-instances", "2", "--n-workers", "4", "--val-wells=-C006,-D007",
    ])

    assert code == 0
    (options,) = calls
    assert options == PrepareOptions(
        raw_dir=raw, label_dir=label, out_dir=out, val_fraction=0.3,
        max_val_wells=2, min_instances=2, n_workers=4,
        val_wells=["-C006", "-D007"],
    )


def test_train_prepare_spec_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_prepare",
        lambda options: (calls.append(options), PrepareSummary())[1],
    )
    code = cli.main([
        "train", "prepare", "--raw-dir", str(tmp_path / "raw"),
        "--label-dir", str(tmp_path / "labels"),
        "--out-dir", str(tmp_path / "h5"),
    ])

    assert code == 0
    (options,) = calls
    assert options.val_fraction == 0.2 and options.max_val_wells == 3
    assert options.min_instances == 1 and options.n_workers == 8
    assert options.val_wells is None


def test_train_train_flags_and_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_train",
        lambda options: (calls.append(options), TrainSummary())[1],
    )
    code = cli.main([
        "train", "train", "--data-dir", str(tmp_path / "h5"),
        "--out-dir", str(tmp_path / "model"), "--model-type", "vit_b",
        "--configuration", "gtx1080", "--samples-per-image", "5",
        "--val-samples-per-image", "1", "--n-objects-per-batch", "4",
        "--patch-shape", "448", "448", "--n-epochs", "10",
        "--early-stopping", "3", "--lr", "2e-5", "--seed", "7",
    ])

    assert code == 0
    (options,) = calls
    assert options == TrainOptions(
        data_dir=tmp_path / "h5", out_dir=tmp_path / "model",
        model_type="vit_b", configuration="gtx1080", samples_per_image=5,
        val_samples_per_image=1, n_objects_per_batch=4,
        patch_shape=(448, 448), n_epochs=10, early_stopping=3, lr=2e-5,
        seed=7,
    )
    # And the spec's hyperparameter defaults, when no flag is given.
    code = cli.main([
        "train", "train", "--data-dir", str(tmp_path / "h5"),
        "--out-dir", str(tmp_path / "model"),
    ])
    (defaults,) = calls[-1:]
    assert defaults.model_type == "vit_t" and defaults.configuration == "Minimal"
    assert defaults.samples_per_image == 3 and defaults.val_samples_per_image == 2
    assert defaults.n_objects_per_batch == 2
    assert defaults.patch_shape == (512, 512)
    assert defaults.n_epochs == 100 and defaults.early_stopping == 15
    assert defaults.lr == 1e-5 and defaults.seed == 42


def test_train_evaluate_flags(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "run_evaluate",
        lambda options: (calls.append(options), EvaluateSummary())[1],
    )
    checkpoint = tmp_path / "best.pth"
    code = cli.main([
        "train", "evaluate", "--data-dir", str(tmp_path / "h5"),
        "--split", "train", "--checkpoint", str(checkpoint),
        "--model-type", "vit_t", "--patch-shape", "1024", "1024",
        "--n-max-images", "3", "--n-max-instances", "10",
        "--output", str(tmp_path / "report.json"),
    ])

    assert code == 0
    (options,) = calls
    assert options == EvaluateOptions(
        data_dir=tmp_path / "h5", split="train", checkpoint=checkpoint,
        model_type="vit_t", patch_shape=(1024, 1024),
        n_max_images=3, n_max_instances=10,
        output=tmp_path / "report.json",
    )


# --- exit codes and real runs ----------------------------------------------------

def _mini_plate(tmp_path, n_frames=4, size=256):
    """A small raw plate: one ringed well with one growing disk.

    256 px is enough for the difference stage; the analyze stage's
    well-circle fit is calibrated for the real plate size (2048), so tests
    that run analyze for real use the default size.
    """
    return build_mini_plate(
        tmp_path / "raw", layout="flat", wells=("-A001",), n_frames=n_frames,
        size=size, noise_sigma=0.0,
        rings={"-A001": RingSpec(cy=size / 2, cx=size / 2, R=size * 0.45)},
        disks={"-A001": [DiskSpec("growing", cy=size / 2, cx=size / 2,
                                  r0=size * 0.05, growth=1.4,
                                  first_frame=2)]},
    )


def _seg_tree(tmp_path, plate, well="-A001", seg_name="seg_growth"):
    """A synthetic segmentation tree for the mini plate (no model involved)."""
    well_dir = tmp_path / "seg" / seg_name / well
    well_dir.mkdir(parents=True)
    rows = []
    for f in range(1, plate.n_frames + 1):
        mask = disk_mask(plate, well, "growing", f)
        label = np.zeros((plate.size, plate.size), np.uint32)
        if mask is not None:
            label[mask] = 1
            ys, xs = np.nonzero(mask)
        build_label_series(well_dir, [label], well=well,
                           names=[plate.wells[well][f - 1]])
        if mask is not None:
            rows.append({
                "well": well, "frame": f, "t_seconds": plate.t_values[f - 1],
                "image": plate.wells[well][f - 1], "colony_id": 1,
                "group_id": 1, "merged_with": "", "pre_existing": 0,
                "area_px": int(mask.sum()),
                "centroid_y": float(ys.mean()),
                "centroid_x": float(xs.mean()),
            })
    io.write_colonies(well_dir / "colonies.csv", rows)
    return well_dir


def test_difference_success_path(tmp_path, capsys):
    plate = _mini_plate(tmp_path)
    out = tmp_path / "out"
    code = cli.main(["difference", "--raw-dir", str(tmp_path / "raw"),
                     "--out-dir", str(out)])

    assert code == 0
    assert io.is_done(out / "-A001")
    assert list((out / "-A001").glob("*_diff_uint8.tif"))
    assert (out / "-A001" / "scale.txt").exists()
    assert capsys.readouterr().out == ""  # nothing ever prints


def test_difference_dry_run_via_cli(tmp_path, caplog):
    _mini_plate(tmp_path)
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="colony_tracking"):
        code = cli.main(["difference", "--raw-dir", str(tmp_path / "raw"),
                         "--out-dir", str(out), "--dry-run"])

    assert code == 0
    assert not out.exists()  # nothing written
    assert "-A001" in caplog.text and "LO001" in caplog.text  # discovery logged


def test_missing_raw_dir_is_runtime_error(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["difference", "--raw-dir", str(tmp_path / "nope"),
                         "--out-dir", str(tmp_path / "out")])

    assert code == 1
    assert "nope" in caplog.text  # a clear logged error, not a traceback


def test_segment_input_dir_mismatches_are_usage_errors(monkeypatch):
    monkeypatch.setattr(
        cli, "run_segmentation",
        lambda options: SegmentationSummary(
            wells=1, processed=1, skipped=0, failed=0),
    )
    base = ["segment", "--out-dir", "out"]
    assert cli.main(base) == 2  # input=raw without --raw-dir
    assert cli.main([*base, "--input", "diff"]) == 2  # without --diff-dir
    assert cli.main([*base, "--input", "raw", "--raw-dir", "raw"]) == 0


def test_unknown_flag_and_invalid_reference_are_usage_errors(caplog):
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        assert cli.main(["difference", "--nonsense"]) == 2
        assert cli.main(["difference", "--reference", "nonsense"]) == 2
        assert cli.main([]) == 2  # no subcommand
    assert caplog.text.count("usage error") == 3


def test_analyze_run_via_cli(tmp_path):
    plate = _mini_plate(tmp_path, size=2048)  # the circle fit's calibrated size
    _seg_tree(tmp_path, plate)
    out = tmp_path / "out"
    code = cli.main([
        "analyze", "--seg-dir", str(tmp_path / "seg" / "seg_growth"),
        "--raw-dir", str(tmp_path / "raw"), "--out-root", str(out),
        "--rim-width-px", "15",
    ])

    assert code == 0
    trajectories = out / "seg_growth" / "trajectories.csv"
    assert trajectories.exists() and trajectories.stat().st_size > 0
    assert (out / "seg_growth" / "growth_rates.csv").exists()


# --- summary failures are runtime errors -----------------------------------------


def test_failed_difference_summary_is_runtime_error(monkeypatch, tmp_path,
                                                    caplog):
    monkeypatch.setattr(cli, "run_difference", lambda options: DifferenceSummary(
        wells={"-A001": WellResult(well="-A001", status="failed",
                                   n_frames=3, error="unreadable frame")},
        processed=0, skipped=0, failed=1,
    ))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["difference", "--raw-dir", "raw", "--out-dir", "out"])

    assert code == 1
    assert "-A001" in caplog.text and "unreadable frame" in caplog.text


def test_failed_segment_summary_is_runtime_error(monkeypatch, caplog):
    monkeypatch.setattr(cli, "run_segmentation",
                        lambda options: SegmentationSummary(
                            wells=1, processed=0, skipped=0, failed=1,
                            failed_wells=["-A001"]))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["segment", "--input", "raw", "--raw-dir", "raw",
                         "--out-dir", "out"])

    assert code == 1 and "-A001" in caplog.text


def test_failed_analyze_summary_is_runtime_error(monkeypatch, caplog):
    monkeypatch.setattr(cli, "run_analysis",
                        lambda options: AnalysisSummary(
                            failures=["seg_growth: no wells found"]))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["analyze", "--seg-dir", "seg",
                         "--raw-dir", "raw", "--out-root", "out"])

    assert code == 1 and "no wells found" in caplog.text


def test_failed_analyze_wells_are_runtime_error(monkeypatch, caplog):
    monkeypatch.setattr(cli, "run_analysis",
                        lambda options: AnalysisSummary(
                            failed_wells={"seg_growth": ["-H008"]}))
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["analyze", "--seg-dir", "seg",
                         "--raw-dir", "raw", "--out-root", "out"])

    assert code == 1 and "-H008" in caplog.text


def test_failed_pipeline_summary_is_runtime_error(monkeypatch, caplog,
                                                  tmp_path):
    monkeypatch.setattr(cli, "run_pipeline", lambda config: PipelineSummary(
        status="failed", failed_stage="segmentation"))
    config = _write_config(tmp_path, _CONFIG)
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["pipeline", "--config", str(config)])

    assert code == 1 and "segmentation" in caplog.text


def test_finetune_error_is_runtime_error(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(
        cli, "run_prepare",
        lambda options: PrepareSummary(error="no such label directory: x"),
    )
    with caplog.at_level(logging.ERROR, logger="colony_tracking"):
        code = cli.main(["train", "prepare", "--raw-dir", "raw",
                         "--label-dir", "labels", "--out-dir", "h5"])

    assert code == 1 and "no such label directory" in caplog.text


# --- help -------------------------------------------------------------------------


def test_help_lists_flags_with_dataclass_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["difference", "--help"])
    assert excinfo.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--raw-dir" in out and "--display-percentile" in out
    assert "--dry-run" in out
    assert "Percentile of |diff|" in out  # the dataclass per-field help
    assert "Process only these wells" in out  # the --wells help


def test_top_level_and_train_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for command in ("difference", "segment", "analyze", "pipeline", "train"):
        assert command in out

    with pytest.raises(SystemExit):
        cli.main(["train", "--help"])
    out = capsys.readouterr().out
    for command in ("prepare", "train", "evaluate"):
        assert command in out

    with pytest.raises(SystemExit):
        cli.main(["pipeline", "--help"])
    out = capsys.readouterr().out
    assert "--config" in out and "--raw-dir" in out
    assert "--fg-ref" not in out  # no per-stage tuning on the pipeline

    with pytest.raises(SystemExit):
        cli.main(["train", "train", "--help"])
    out = capsys.readouterr().out
    assert "--patch-shape" in out and "--early-stopping" in out