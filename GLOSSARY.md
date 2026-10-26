# GLOSSARY

Binding domain vocabulary for `colony_tracking`. These terms are used in
identifiers, help text, and documentation.

Sections: Data / Colonies / Exclusions-and-flags / Analysis / Configuration.

## Data

**plate** — A 96-well culture plate; the unit of one experiment, one raw
input tree, and one pipeline run.

**well** — One of the 96 wells of a plate, named like `-A001` (the
`-<A-H>\d{3}` binding). In a flat
input tree the well is parsed from the leading `-<A-H>\d{3}` of the file
name; in a per-well input tree it is the subdirectory name. All stage
outputs are per-well nested subdirectories.

**frame** — One acquired image of a well. Frame metadata is parsed from the
file name in both input layouts: the frame number from `LO<nnn>`, the
acquisition time from the `T` field.

**frame number** — The 1-based index of a frame within a well, parsed from
`LO<nnn>` in the file name. Written to disk as `frame`.

**reference frame** — The frame all other frames of a well are differenced
against. Selected as `first` or by frame number (1-based).

**time field** — The `T` field of the file name, the acquisition time
(milliseconds in this dataset). Written to `colonies.csv` as `t_seconds`
(the raw value); converted to hours by the magnitude heuristic (values >
1e6 are read as milliseconds).

**difference frame** — `cur − ref` in counts for one well, stored as a uint8
preview with 128 = no change, scaled by the per-well display limit (ADR
0001).

**display limit** — The per-well ceiling that scales difference counts into
the uint8 preview: the `display_percentile` (99.5) of |diff|, estimated on
the subsampled grid. Recorded in the well's `scale.txt` as
`display_limit_int16`.

**input tree layout** — The two accepted layouts of stage inputs: *flat*
(all wells' frames in one directory, the well parsed from the file name)
and *per-well* (one subdirectory per well). All stage outputs are per-well
nested subdirectories.

## Colonies

**segment** — One connected foreground region that the tracking rules
consider. In a full segment run these are the instances that passed the
growth-mask gating; in re-ID mode each saved label value is re-split into
connected components.

**instance** — One connected foreground region the decoder fires on in a
single frame (foreground above the stage's threshold).

**group** — One connected component of the union of the final frame's
surviving instances. Groups are the units the growth mask is built from.

**growth mask** — The union of the final frame's groups, dilated by
`mask_margin` px. Detection in earlier frames is confined to it (ADR 0002).

**colony** — One tracked identity across frames: a series of one or more
segments linked by the continue / merge / fork rules.

**colony ID** — The stable identity number of a colony. A surviving colony
keeps its ID through merges and forks.

**merge** — Two or more colonies combining into one segment. The survivor
keeps its ID; the absorbed colonies are listed in `merged_with`.

**absorbed** — A colony that lost its identity in a merge. If it separates
again it resumes its own ID.

**fork** — An unmatched segment that splits one colony: the same ID
continues (logged and counted in the done marker).

**pre-existing** — A colony whose first detection is at or before the
`exclude_before` frame. Flagged `pre_existing=1`; excluded from aggregates and
plots over its whole lifetime unless `keep_pre_existing`, and never
bad-flagged (so it cannot exclude its well).

## Exclusions-and-flags

**well circle** — The fitted meniscus circle of a well (`well, cy, cx, R`
in `well_geometry.csv`), the anchor of rim exclusion.

**rim** — The band within `rim_width_px` of the well circle.

**rim-excluded** — A colony, from the first frame in which more than
`rim_overlap_max` of its pixels lie in the rim. Exclusion is monotone:
colonies only grow into the rim.

**bad segment** — A colony flagged `from_start` (first valid frame within
`start_frames` of the series start) or `no_growth` (any valid frame,
compared to the earliest valid frame ≥ 2 images ahead, growing by
`min_growth_per_frame` or less per image).

**bad well** — A well with ≥ `max_bad_segments` distinct bad colonies;
excluded from aggregates and plots unless `keep_bad_wells`.

**ancestors** — The transitive, self-inclusive count of the original
colonies combined in a segment (1 = unmerged; an A+B where B had merged
with C ends at 3). Written as `ancestors` (the prototype's `n_merged`).

**drift gate** — The skip of a frame whose elevated-foreground fraction
exceeds `max(max_fg_ratio × reference-frame fraction, 0.05)`; drift bursts
around the reference frame make the decoder hallucinate giant instances.
`max_fg_ratio = 0` disables the gate (ADR 0002).

**exclude_before** — A frame number N that flags colonies first detected at
or before N as pre-existing; frames before N are segmented and tracked like
any other (nothing is blanked).

## Analysis

**growth rate** — The per-colony rates over its valid (not rim-excluded)
frames, computed only when the window is at least three frames wide
(first to last valid frame ≥ 2 images apart — the same window the
per-frame `no_growth` comparison needs; a bare two-point slope cannot be
vetted against the growth floor): the exponential rate (least-squares
slope of ln(area) vs hours, and doublings per day) and the absolute
growth (px/h). Narrower-window colonies (single valid frame, two adjacent
frames) get blank rate cells and are skipped by the pooled aggregates and
plots.

**pooled** — Cross-well statistics grouped by ancestor count: pooled area
statistics per (frame, ancestors) and pooled rate statistics per
max-ancestors.

## Configuration

**default** — The value an option takes when the draft leaves it unset;
resolved at run time and shown gray in an empty TUI field.
