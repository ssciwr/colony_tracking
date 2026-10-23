"""Shared options machinery for every stage.

Each stage is an options dataclass whose fields are exactly the CLI flags of
that stage (defaults per the spec's "Options and CLI contract" section). The
dataclasses here are plain and importable; per the package layout they live
in this private module for now and are re-exported from the package root by
the ticket that wires up the public API.

Every field carries its help text in ``Field(metadata={"help": ...})`` — the
single source for CLI help and the future TUI's option forms. ``to_dict`` /
``from_dict`` round-trip a dataclass through plain (TOML-friendly) values,
which the TOML config file and the future TUI consume.

The dataclasses are ``kw_only``: the spec's field order is the CLI flag
order and some required fields follow defaulted ones, which a plain
positional dataclass could not express.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

#: Reference frame selection: ``"first"`` or a 1-based frame number.
Reference = Literal["first"] | int

#: Segment stage input mode.
InputMode = Literal["raw", "diff"]


def parse_reference(text: str) -> Reference:
    """Parse one reference input: ``first`` (case-insensitive), a bare
    1-based frame number (``5``), or the file-name form ``LO005`` (the
    ``LO`` prefix is case-insensitive, the number may be zero-padded).

    The one parser for user-typed reference text: the TUI's forms and
    quick start parse through it, so they accept exactly what the CLI's
    ``--reference`` flag accepts and reject the rest with the same
    message. Raises ValueError otherwise; blank text is the caller's
    "unset" and never reaches this parser.
    """
    lowered = text.strip().lower()
    if lowered == "first":
        return "first"
    if lowered.startswith("lo"):
        lowered = lowered[2:]
    try:
        return int(lowered)
    except ValueError:
        raise ValueError(
            f"invalid reference {text!r}: expected 'first', a 1-based frame "
            "number, or LO<nnn>"
        ) from None


def _to_plain(value: Any) -> Any:
    """Convert a field value to a plain (TOML-friendly) value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        # Tuples come out as lists (TOML has no tuples).
        return [_to_plain(v) for v in value]
    return value


# Shared internal helper of this package: also imported by pipeline.py and cli.py.
def _union_members(hint: Any) -> tuple[Any, ...] | None:
    """The non-None members of a union hint (``typing.Union`` or
    ``types.UnionType``), else None."""
    if get_origin(hint) not in (Union, UnionType):
        return None
    return tuple(a for a in get_args(hint) if a is not type(None)) or None


def _from_plain(value: Any, type_hint: Any) -> Any:
    """Coerce a plain value back to the field's declared type."""
    if (members := _union_members(type_hint)) is not None and len(members) == 1:
        # ``X | None``: coerce against X.
        return _from_plain(value, members[0])
    origin, args = get_origin(type_hint), get_args(type_hint)
    if origin is list and isinstance(value, list) and args:
        return [_from_plain(v, args[0]) for v in value]
    if origin is tuple and isinstance(value, (list, tuple)) and args:
        return tuple(_from_plain(v, args[0]) for v in value)
    if type_hint is Path and isinstance(value, str):
        return Path(value)
    return value


class Options:
    """Base for the stage options dataclasses (see module docstring)."""

    def to_dict(self) -> dict[str, Any]:
        """Plain, TOML-friendly dict of all fields (paths as strings)."""
        return {f.name: _to_plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Options:
        """Build an instance from a plain dict; omitted keys take defaults."""
        hints = get_type_hints(cls)
        kwargs = {
            f.name: _from_plain(data[f.name], hints[f.name])
            for f in fields(cls)
            if f.name in data
        }
        return cls(**kwargs)


@dataclass(kw_only=True)
class DifferenceOptions(Options):
    """Options for the difference stage (``difference`` subcommand)."""

    raw_dir: Path = field(
        metadata={"help": "Directory of raw plate frames (flat or per-well layout)."}
    )
    out_dir: Path = field(
        metadata={"help": "Directory to write the per-well difference trees into."}
    )
    reference: Reference = field(
        default="first",
        metadata={"help": 'Reference frame: "first" or a 1-based frame number.'},
    )
    wells: list[str] | None = field(
        default=None,
        metadata={
            "help": "Process only these wells (repeatable, comma-separated). "
            "Default: all wells."
        },
    )
    display_percentile: float = field(
        default=99.5,
        metadata={
            "help": "Percentile of |diff| that sets the per-well display limit."
        },
    )
    subsample: int = field(
        default=8,
        metadata={
            "help": "Stride of the subsampled grid used to estimate the display "
            "limit."
        },
    )
    n_workers: int = field(
        default=6, metadata={"help": "Thread-pool size over wells."}
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "Report wells, frames, and reference; write nothing."},
    )


@dataclass(kw_only=True)
class SegmentationOptions(Options):
    """Options for the segment stage (``segment`` subcommand)."""

    input: InputMode = field(
        default="raw",
        metadata={"help": 'Input mode: "raw" frames or "diff" difference frames.'},
    )
    raw_dir: Path | None = field(
        default=None,
        metadata={"help": "Raw plate directory (required when input = 'raw')."},
    )
    diff_dir: Path | None = field(
        default=None,
        metadata={"help": "Difference tree directory (required when input = 'diff')."},
    )
    out_dir: Path = field(
        metadata={"help": "Directory to write the per-well label/CSV/marker trees into."}
    )
    wells: list[str] | None = field(
        default=None,
        metadata={
            "help": "Segment only these wells (repeatable, comma-separated). "
            "Default: all wells."
        },
    )
    model_type: str = field(
        default="vit_b_lm", metadata={"help": "micro-sam model type."}
    )
    checkpoint: Path | None = field(
        default=None,
        metadata={
            "help": "Optional finetuned decoder checkpoint, used in place of the "
            "pretrained model."
        },
    )
    fg_ref: float = field(
        default=0.5,
        metadata={"help": "Decoder foreground threshold for the final frame."},
    )
    fg_early: float = field(
        default=0.3,
        metadata={"help": "Decoder foreground threshold for earlier frames."},
    )
    min_size: int = field(
        default=200,
        metadata={"help": "Drop final-frame instances smaller than this (px)."},
    )
    min_solidity: float = field(
        default=0.7,
        metadata={"help": "Drop final-frame instances with lower solidity."},
    )
    mask_margin: int = field(
        default=15,
        metadata={"help": "Dilation (px) of the final-frame groups forming the "
        "growth mask."},
    )
    min_overlap: float = field(
        default=0.5,
        metadata={
            "help": "Minimum fraction of an instance's pixels that must lie "
            "inside the growth mask and inside a single group."
        },
    )
    max_fg_ratio: float = field(
        default=1.5,
        metadata={
            "help": "Drift gate: skip frames whose elevated-foreground fraction "
            "exceeds this multiple of the reference frame's (0 disables the gate)."
        },
    )
    exclude_before: int | None = field(
        default=None,
        metadata={
            "help": "Flag colonies first detected at or before this frame "
            "number as pre-existing (excluded from the analysis); frames "
            "below it are still segmented and tracked."
        },
    )
    merge_touching: bool = field(
        default=True,
        metadata={"help": "Pre-merge touching segments of one group "
        "(disable with --no-merge-touching)."},
    )
    redo: bool = field(
        default=False,
        metadata={"help": "Reprocess wells even when their done marker is present."},
    )
    reid_only: bool = field(
        default=False,
        metadata={
            "help": "Re-derive segments from the saved label images and re-track "
            "without the model (no GPU)."
        },
    )


@dataclass(kw_only=True)
class AnalysisOptions(Options):
    """Options for the analyze stage (``analyze`` subcommand)."""

    seg_dirs: list[Path] = field(
        metadata={"help": "Segmentation trees to analyze (repeatable)."}
    )
    raw_dir: Path = field(
        metadata={"help": "Raw plate directory (well-circle fit and the "
        "difference-root probe)."},
    )
    out_root: Path = field(
        metadata={"help": "Directory to write the analysis outputs into."}
    )
    wells: list[str] | None = field(
        default=None,
        metadata={"help": "Analyze only these wells (repeatable, comma-separated). "
        "Default: all wells."},
    )
    rim_width_px: int = field(
        default=60,
        metadata={"help": "Width of the rim band inside the well circle (px)."},
    )
    rim_overlap_max: float = field(
        default=0.1,
        metadata={
            "help": "A colony is rim-excluded from the first frame in which more "
            "than this fraction of its pixels lies in the rim."
        },
    )
    max_bad_segments: int = field(
        default=2,
        metadata={"help": "A well with at least this many distinct bad colonies is "
        "a bad well."},
    )
    keep_bad_wells: bool = field(
        default=False, metadata={"help": "Include bad wells in aggregates and plots."}
    )
    keep_pre_existing: bool = field(
        default=False,
        metadata={"help": "Include pre-existing colonies in aggregates and plots."},
    )
    raw_seg_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Raw-root segmentation tree probed to flag pre-existing "
            "colonies of difference roots; without it the raw reference "
            "frame itself is probed (dark centroid window)."
        },
    )
    start_frames: int = field(
        default=1,
        metadata={
            "help": "A colony whose first valid frame lies within this many frames "
            "of the series start is bad (from_start); pre-existing colonies "
            "are exempt."
        },
    )
    min_growth_per_frame: float = field(
        default=0.1,
        metadata={
            "help": "A colony for which any valid frame, compared to the earliest "
            "valid frame >= 2 images ahead, grows by this factor or less per "
            "image is bad (no_growth); pre-existing colonies are exempt."
        },
    )
    refresh_geometry: bool = field(
        default=False,
        metadata={"help": "Re-fit the well circles instead of using the cached "
        "well_geometry.csv."},
    )
    plots_only: bool = field(
        default=False,
        metadata={"help": "Re-render the plots from the existing trajectories CSV "
        "without image IO."},
    )
    no_plots: bool = field(default=False, metadata={"help": "Skip the plots."})
    no_qa: bool = field(
        default=False, metadata={"help": "Skip the per-well QA overlays."}
    )
    no_group_alpha: bool = field(
        default=False,
        metadata={"help": "Do not fade group opacity with member count."},
    )


@dataclass(kw_only=True)
class PrepareOptions(Options):
    """Options for ``train prepare`` (finetune data conversion)."""

    raw_dir: Path = field(
        metadata={"help": "Directory of raw plate frames (flat or per-well layout)."}
    )
    label_dir: Path = field(
        metadata={
            "help": "Directory of ground-truth label images (same layout, file "
            "names matching the raw frames); the label set defines the image set."
        }
    )
    out_dir: Path = field(
        metadata={
            "help": "Directory to write the well-split h5 set (train/ and val/) into."
        }
    )
    val_fraction: float = field(
        default=0.2,
        metadata={
            "help": "Approximate fraction of instances to hold out for validation "
            "(rounded to whole wells)."
        },
    )
    max_val_wells: int = field(
        default=3,
        metadata={
            "help": "Upper bound on the number of wells moved into the "
            "validation split."
        },
    )
    min_instances: int = field(
        default=1,
        metadata={
            "help": "Drop labels with fewer than this many instances (an all-zero "
            "label would abort the micro-sam sampler)."
        },
    )
    n_workers: int = field(
        default=8, metadata={"help": "Threads used for scanning/inspection."}
    )
    val_wells: list[str] | None = field(
        default=None,
        metadata={
            "help": "Explicit wells (e.g. -C006) for validation; overrides "
            "val_fraction."
        },
    )


@dataclass(kw_only=True)
class TrainOptions(Options):
    """Options for ``train train`` (micro-sam decoder training)."""

    data_dir: Path = field(
        metadata={
            "help": "Directory with the train/ and val/ h5 subfolders "
            "(the train prepare output)."
        }
    )
    out_dir: Path = field(
        metadata={
            "help": "Directory for the checkpoints and TensorBoard logs."
        }
    )
    model_type: str = field(
        default="vit_t",
        metadata={
            "help": "SAM variant ('vit_t' is MobileSAM and the lightest option; "
            "the laptop GPU OOMs on vit_b)."
        },
    )
    configuration: str = field(
        default="Minimal",
        metadata={
            "help": "micro-sam hardware preset (Minimal, gtx1080, gtx3080, ...)."
        },
    )
    samples_per_image: int = field(
        default=3,
        metadata={
            "help": "Crops drawn per training image per epoch (micro-sam defaults "
            "to 16, which makes one epoch take ~90 min on a laptop GPU)."
        },
    )
    val_samples_per_image: int = field(
        default=2, metadata={"help": "Crops drawn per validation image per epoch."}
    )
    n_objects_per_batch: int = field(
        default=2,
        metadata={
            "help": "Objects sampled per training step (the Minimal preset asks "
            "for 4, which leaves little VRAM headroom)."
        },
    )
    patch_shape: tuple[int, int] = field(
        default=(512, 512),
        metadata={"help": "Crop (tile) size in px, (height, width)."},
    )
    n_epochs: int = field(default=100, metadata={"help": "Number of training epochs."})
    early_stopping: int = field(
        default=15,
        metadata={
            "help": "Stop after this many epochs without validation improvement "
            "(0 disables it)."
        },
    )
    lr: float = field(default=1e-5, metadata={"help": "Learning rate."})
    seed: int = field(default=42, metadata={"help": "Random seed."})


@dataclass(kw_only=True)
class EvaluateOptions(Options):
    """Options for ``train evaluate`` (held-out IoU benchmark)."""

    data_dir: Path = field(
        metadata={
            "help": "Directory with the train/ and val/ h5 subfolders "
            "(the train prepare output)."
        }
    )
    split: str = field(
        default="val", metadata={"help": "Split to evaluate (val / train)."}
    )
    checkpoint: Path | None = field(
        default=None,
        metadata={
            "help": "Finetuned checkpoint; omit to evaluate the pretrained base model."
        },
    )
    model_type: str = field(
        default="vit_t",
        metadata={"help": "SAM variant the checkpoint was trained with."},
    )
    patch_shape: tuple[int, int] = field(
        default=(512, 512),
        metadata={
            "help": "Tile size in px, (height, width); instances larger than a "
            "tile are flagged clipped."
        },
    )
    n_max_images: int | None = field(
        default=None,
        metadata={
            "help": "Evaluate only the first N h5 images of the "
            "split (a quick smoke run); 0 = uncapped."
        },
    )
    n_max_instances: int | None = field(
        default=None,
        metadata={"help": "Score only the first N instances per image."},
    )
    output: Path | None = field(
        default=None, metadata={"help": "Optional path for a JSON report."}
    )
