# Image operations through the pipeline

Every pixel-level operation the pipeline applies to the input images, in
execution order, with the exact math and the code line that performs it.

**Input.** Raw plate frames: 2048×2048 `uint16` TIFFs, one time series per
well (hourly acquisition). File names carry the well, the 1-based frame
number (`LO<nnn>`), and the acquisition time `T` (ms in this dataset)
(`src/colony_tracking/io.py:35-39`, parsed at `io.py:42-61`).

Derived images produced along the way:

| Image | DType | Produced by |
| --- | --- | --- |
| difference frame (`*_diff_uint8.tif`) | `uint8`, 128 = no change | difference stage |
| segment input (raw path) | `uint8` percentile-stretched | segment stage, `input="raw"` |
| group / growth-mask images | `uint32` / `bool` | segment stage |
| label images (`*__labels.tif`) | `uint32`, 0 = background | segment stage |
| rim zone, distance map, QA composite | `float32` / `bool` / `uint8` RGB | analyze stage |
| training pair (`raw`, `label` in h5) | `uint8` / `int32` | finetune prepare |

Constants referenced below: `subsample` (default 8), `display_percentile`
(99.5), `fg_ref` (0.5), `fg_early` (0.3), `min_size` (200 px),
`min_solidity` (0.7), `mask_margin` (15 px), `min_overlap` (0.5),
`GAP_FLOOR` (0.5), `MATCH_IOU` (0.3), `ABSORB_COVERAGE` (0.5),
`TOUCH_DILATE` (3 px), `rim_width_px` (60), `rim_overlap_max` (0.1).

---

## 1. Difference stage — raw frame → uint8 difference frame

Module: `src/colony_tracking/difference.py`, helpers in `src/colony_tracking/io.py`.
Per well, per frame, on CPU in a thread pool (`difference.py:247-251`).

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 1.1 | Load frame | `im = np.asarray(tifffile.imread(path), dtype=np.float32)` — uint16 counts to float32 so subtraction cannot wrap | `difference.py:92-94` (`_load`) |
| 1.2 | Subtract reference | `d = cur - ref`, elementwise float32, always `cur - ref` (pre-reference frames come out polarity-flipped by design) | `difference.py:133` (limit pass), `difference.py:149` (write pass) |
| 1.3 | Reference frame placeholder | `diff = np.zeros(ref.shape, dtype=np.float32)` — the reference is written as an all-128 frame so the well stays a complete series | `difference.py:146` |
| 1.4 | Subsample \|d\| for the limit | `np.abs(d)[::subsample, ::subsample].copy()` — strided grid, stride 8 by default; only the strided samples are kept, the full difference is not held | `difference.py:134` |
| 1.5 | Display limit | `limit = max(percentile(concat(strided |d| over all non-reference frames), 99.5), 1.0)` — per frame: `arr = np.abs(np.asarray(d, dtype=np.float32))`, `arr[::subsample, ::subsample].ravel()`; floored at 1.0 so an unchanged well has a defined mapping | `io.py:162-184` (`display_limit`), called at `difference.py:136` |
| 1.6 | Map to uint8 | `scaled = 128.0 + diff * (127.0 / limit)` then `np.clip(scaled, 0, 255).astype(np.uint8)` — ±limit counts hit the 8-bit edges, zero difference maps to exactly 128 (ADR 0001); quantisation cost `2·limit/255` counts | `io.py:149-159` (`map_to_uint8`), called at `difference.py:153` |
| 1.7 | Write | `tifffile.imwrite(out_dir / "<stem>_diff_uint8.tif", …, compression="zlib")` | `difference.py:152-154` |

The per-well `scale.txt` records the reference file name and the float limit
(`io.py:202-214`) so uint8 previews can be converted back to counts.

---

## 2. Segment stage — difference (or raw) frame → label images

Module: `src/colony_tracking/segmentation.py`. One model (micro-sam decoder)
is loaded once per run (`segmentation.py:250-268`, 293).

### 2.1 Loading the frame for the model

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 2.1.1 | Load | `image = tifffile.imread(frame.path)` | `segmentation.py:244` (`_load_frame`) |
| 2.1.2a | `input="diff"`: pass through | `np.asarray(image).astype(np.uint8)` — the uint8 difference is fed as written (128 = neutral) | `segmentation.py:247` |
| 2.1.2b | `input="raw"`: percentile stretch | `im = np.asarray(image, dtype=np.float32)`; `lo, hi = np.percentile(im, (1, 99))`; flat image → all-zero; else `np.clip((im - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)` | `io.py:134-146` (`stretch_uint8`), called at `segmentation.py:246` |

### 2.2 One model call per frame (`_segment_frame`, `segmentation.py:271-310`)

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 2.2.1 | SAM encode | `segmenter.initialize(image)` — image encoder, per-tile (micro-sam internal) | `segmentation.py:293` |
| 2.2.2 | Elevated-foreground fraction (drift gate) | `elevated = float((segmenter._foreground >= fg_threshold).mean())` — fraction of all pixels whose decoder foreground probability ≥ threshold | `segmentation.py:294` |
| 2.2.3 | Drift-gate decision | skip (return all-zero instances) when `elevated > max(max_fg_ratio · ref_frac, 0.05)`; disabled for `max_fg_ratio ≤ 0`; the reference (final) frame is never gated | `segmentation.py:90-113` (`drift_gate_limit`, `drift_gate_skips`), applied at `segmentation.py:295-305` and re-checked at `segmentation.py:437-440` |
| 2.2.4 | Close enclosed sub-threshold gaps | `above = fg >= threshold`; `gaps = binary_fill_holes(above) & ~above`; every gap pixel with `fg >= GAP_FLOOR · threshold` is raised to 1.0 — the strict threshold draws instance boundaries and must not slice holes into an instance's interior; an enclosed region the model scores below the floor (background by the model's own lights) stays out, and a sub-threshold fringe open to the image background is never touched | `segmentation.py:133-170` (`GAP_FLOOR`, `fill_enclosed_gaps`), applied at `segmentation.py:366-368` |
| 2.2.5 | Clear foreground outside the growth mask | `segmenter._foreground[~allowed] = 0.0` — earlier frames only; the final frame is segmented with `allowed=None` | `segmentation.py:369-370` |
| 2.2.6 | Instance segmentation | `segmenter.generate(foreground_threshold=fg, min_size=min_size)` — decoder foreground + watershed (micro-sam internal) | `segmentation.py:371-373` |

### 2.3 Growth mask from the final frame (`build_growth_mask`, `segmentation.py:119-148`)

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 2.3.1 | Junk filter per final-frame instance | keep instance when `region.area ≥ min_size (200)` and `region.solidity ≥ min_solidity (0.7)` (skimage `regionprops`); `keep |= (instances == region.label)` | `segmentation.py:142-144` |
| 2.3.2 | Groups | `groups = ndi.label(keep)[0].astype(np.uint32)` — 8-connected components of the surviving union | `segmentation.py:147` |
| 2.3.3 | Dilation to the growth mask | `mask = ndi.binary_dilation(groups > 0, iterations=mask_margin)` — 15-px binary dilation (scipy default 8-connected structuring element) | `segmentation.py:148` |
| 2.3.4 | Persist group labels | `ref__labels.tif` = group label image (all-zero when nothing survives) | `segmentation.py:403-406` |

### 2.4 Mask-overlap filter for each non-final frame (`filter_to_mask`, `segmentation.py:154-190`)

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 2.4.1 | Per-instance mask overlap | keep when `float((instance & mask).sum()) / region.area ≥ min_overlap (0.5)` | `segmentation.py:177-178` |
| 2.4.2 | Per-instance group overlap | `hits = np.bincount(groups[instance].ravel(), minlength=n_groups + 1)`; keep only if `float(hits[1:].max()) / region.area ≥ min_overlap` — an instance must lie mostly in the growth mask **and** in a single group | `segmentation.py:180-185` |
| 2.4.3 | Relabel kept instances | kept instances relabelled 1..k in `regionprops` order (local IDs, full extent kept) | `segmentation.py:186-187` |

### 2.5 Tracking (segment-level operations, `src/colony_tracking/tracking.py`)

Operates on the boolean segment masks derived above (no pixel intensity math).

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 2.5.1 | Split label image into connected segments | per label value: run-length 4-connected components; per row `edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))`, `starts = flatnonzero(edges == 1)`, `ends = flatnonzero(edges == -1)`, unified vertically by union-find over overlapping runs | `tracking.py:77-136` (`_label_components`), used by `extract_segments` at `tracking.py:168-184` |
| 2.5.2 | Segment centre of mass | `ys, xs = np.nonzero(mask)`; `cy = ys.mean()`, `cx = xs.mean()` | `tracking.py:152-155` (`_center`) |
| 2.5.3 | Pool segments under their group | `hits = np.bincount(groups[s.mask].ravel(), minlength=n_groups + 1)`; a segment joins the group with the most pixels, `per_group[int(np.argmax(hits[1:]) + 1)][t]`; a segment outside every group (only possible in the dilated margin) is dropped with a warning | `tracking.py:313-324` |
| 2.5.4 | Pre-merge touching segments | repeat until stable: merge `s` into `k` when `(_dilate(k.mask, 3) & s.mask).any()` (4-neighbourhood cross dilation, 3 iterations: `grown[1:,:] |= out[:-1,:]` etc.); the union mask keeps the larger segment's centroid; `k.area += s.area` | `tracking.py:139-149` (`_dilate`), `tracking.py:187-236` (`premerge_touching`), applied at `tracking.py:334-335` |
| 2.5.5 | Continue (identity) | `inter = float((s.mask & o["mask"]).sum())`; `iou = inter / (s.area + o["area"] − inter)`; `covered = inter / o["area"]`; a segment continues the colony with the best IoU ≥ 0.3 **or** covering ≥ 0.5 of the colony's mask (the strong-growth case: a colony that more than ~triples in one frame has IoU < 0.3 even for a perfect continuation; among equal-coverage candidates the larger colony has the higher IoU, so the dominant colony of a merge wins; ties: larger colony); the colony's last-known mask is updated to the segment's mask | `tracking.py:350-363` |
| 2.5.6 | Fork | for an unmatched segment, `cov = float((s.mask & start_mask).sum()) / s.area` against every active colony's mask as of the frame's start; if `max cov ≥ 0.5` the segment splits that colony (same ID, mask updated when the host was not continued) | `tracking.py:360-380` |
| 2.5.7 | Absorb-merge | for each segment's surviving colony `o`, absorb every other un-continued colony `c` with `float((s.mask & c["mask"]).sum()) / c["area"] ≥ 0.5`; absorbed IDs are joined `';'` in the frame's `merged_with` | `tracking.py:390-405` |
| 2.5.8 | New colony | any unmatched, non-forking segment gets the next fresh ID | `tracking.py:381-386` |
| 2.5.9 | Write colony IDs to the label image | `labels_out[t][s.mask] = o["id"]` (uint32); gated frames stay all-zero, keeping 1:1 with the input frames | `tracking.py:355`, written at `segmentation.py:341-342` (`_write_outputs`) |
| 2.5.10 | Row geometry for `colonies.csv` | `area_px = s.area`; `centroid_y = round(cy, 1)`, `centroid_x = round(cx, 1)` | `tracking.py:413-415` |
| 2.5.11 | Pre-existing filter | colonies first detected at number ≤ `exclude_before` get `pre_existing = 1`; frames below it are segmented and tracked like any other (nothing is blanked) | `segmentation.py:410-418` (`_exclude_log`), `tracking.py:349-350` |

---

## 3. Analyze stage — raw frames + label images → measurements

Module: `src/colony_tracking/analysis.py`. Image math on raw frames (well
circle, QA overlay) and on label images (rim fractions).

### 3.1 Well-circle fit from the earliest raw frames (`_fit_frame`, `analysis.py:200-232`)

Up to the three earliest raw frames per well, each fitted independently
(`load_geometry`, `analysis.py:969-988`).

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 3.1.1 | Gaussian blur | `sm = ndi.gaussian_filter(tifffile.imread(path).astype(np.float32), 6)` — σ = 6 px | `analysis.py:211` |
| 3.1.2 | Radial profile sampling (stage 1) | 240 angles `angs = np.linspace(0, 2π, 240, endpoint=False)`; radii `rs = np.arange(650, 1120, 2.0)` around (h/2, w/2); sample `ys = (cy + rr·sin θ).astype(int)`, `xs = (cx + rr·cos θ).astype(int)`; out-of-bounds samples NaN → replaced by the angle's row mean; 9-point radial smoothing `ndi.uniform_filter1d(prof, 9, axis=1)`; darkest radius per angle `i = np.nanargmin(prof, axis=1)` | `analysis.py:125-165` (`_radial_minima`), called at `analysis.py:213` |
| 3.1.3 | Local-dip test | neighbourhood mean over ±30 radius steps (±60 px): `nb_mean = prof[k, lo[k]:hi[k]].mean()` with `lo = clip(i−30, 0, …)`, `hi = clip(i+31, 0, …)`; keep the angle only if ≥ 90 % of its samples were in-bounds and `prof[i] ≤ nb_mean − dip·(depth + 1e−9)` with `depth = nanmax − nanmin` of the angle's profile, `dip = 0.08` (stage 1) | `analysis.py:153-165` |
| 3.1.4 | Kasa circle fit | `a = [y, x, 1]`; `c = lstsq(a, y² + x²)`; `cy = c₀/2`, `cx = c₁/2`, `R = √(c₂ + cy² + cx²)` | `analysis.py:168-173` (`_circle_fit`) |
| 3.1.5 | Iterative radius-outlier trimming | up to 3 iterations: `rr = √((y−cy)² + (x−cx)²)`; keep `\|rr − median(rr)\| < tol` (30 px stage 1, 25 px stage 2); stop when all points survive or fewer than 20 remain; final fit on the survivors, reporting radius `std` | `analysis.py:176-197` (`_trim_fit`), called at `analysis.py:216, 223` |
| 3.1.6 | Stage 2 re-lock | re-sample `_radial_minima` in the narrow window `R ± 70 px` around the coarse fit (same centre), `dip = 0.02`; needs ≥ 40 surviving points (stage 1 needs ≥ 30) | `analysis.py:219-221` |
| 3.1.7 | Acceptance | accept only if `700 < R < 1100`, `200 < cy, cx < 1850`, radius `std < 20 px`, and ≥ 8 of 12 angular sectors covered (`cov = len(np.unique((deg(arctan2(y−cy, x−cx)) + 360) % 360 // 30))`) | `analysis.py:227-231` |
| 3.1.8 | Average the frame fits | `med_r = median(R)`; drop fits with `\|R − med_r\| > 40 px`; result = per-coordinate `mean` of the survivors; cached in `well_geometry.csv` | `analysis.py:235-251` (`fit_well_circle`), cache at `io.py:336-373` |

### 3.2 Rim fractions from the label images (`rim_stats`, `analysis.py:273-315`)

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 3.2.1 | Distance map | `yy, xx = np.mgrid[0:h, 0:w]`; `d = np.sqrt((yy − cy)² + (xx − cx)²).astype(np.float32)` — built once per well from the first loaded label image | `analysis.py:302-304` |
| 3.2.2 | Rim zone | `zone = d > (R − rim_width)` — the 60-px band at the inner edge of the well circle | `analysis.py:305` |
| 3.2.3 | Pixel binning per frame | per label image `lab` (uint32): `n_all = np.bincount(lab.ravel(), minlength=mx+1)`, `n_zone = np.bincount(lab[zone].ravel(), minlength=mx+1)` | `analysis.py:309-310` |
| 3.2.4 | Rim fraction per (frame, colony) | `rim_frac = n_zone[cid] / n_all[cid]` | `analysis.py:314` |
| 3.2.5 | Monotone exclusion | a colony is rim-excluded from the first frame with `rim_frac > rim_overlap_max (0.1)` and stays excluded (`excluded` latch in `build_trajectories`) | `analysis.py:383-395` |

### 3.3 Pre-existing probe on the base image (`probe_pre_existing`, `analysis.py:460-560`)

Difference roots only (`*diff_LO<nnn>`). Two probes, in order of preference:

1. **Raw-root probe** (`raw_seg_dir` set): the raw root's label image at the
   reference frame is loaded and each colony's first-detection centroid is
   probed.
2. **Raw-frame probe** (the fallback, the pipeline chain's default path):
   without the raw root — or without its label image — the raw frame at the
   reference frame is probed directly. Background anchor: the median inside
   the fitted well circle; skipped (warned) without the circle or without
   the raw frame.

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 3.3.1 | Centroid window | `win = lab[y−12 : y+13, x−12 : x+13]` (clipped at the image edge); hit when `win.max() > 0` | `analysis.py:497-500` |
| 3.3.2 | Area-equivalent disk | raw-colony discs `(cy, cx, r)` with `r = (area_px / π)^0.5` from the raw root's `colonies.csv` at the reference frame; hit when `(dcy − cy)² + (dcx − cx)² ≤ (drad + 15)²` | `analysis.py:488-490, 502-503` |
| 3.3.3 | Raw-frame darkness | background `= median(raw[inside circle])`; hit when `median(raw[y−12 : y+13, x−12 : x+13]) < PRE_EXISTING_DARK_FRAC (0.5) × background` | `_probe_raw_frame` |

### 3.4 Time and growth-rate math (no images, listed for completeness)

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 3.4.1 | Hours from the `T` field | `t_h = T / 3.6e6` when `max(T) > 1e6` (ms), else `T / 3600` (s) | `analysis.py:257-267` (`t_hours_of`) |
| 3.4.2 | no_growth test | over the valid frames, every exact 2-frame window: `a2/a1` for frame pairs with `l2 − l1 == 2`; bad when `max(ratios) < 1 + min_growth_2step (1.3)` and ≥ 3 valid frames | `analysis.py:414-418` |
| 3.4.3 | Bad-flag exemption | a colony with any `pre_existing = 1` row gets no `from_start`/`no_growth` reasons — it is excluded from the analysis anyway and cannot flag its well bad | `analysis.py:407-418` |
| 3.4.4 | Specific growth rate | least-squares slope of `ln(area)` vs hours: `k = polyfit(t, log(a), 1)` over valid frames with `t[-1] > t[0]` and `a > 0` | `analysis.py:551-556` |
| 3.4.5 | Doublings / absolute growth | `k · 24 / ln 2` (day⁻¹); `(a[-1] − a[0]) / (t[-1] − t[0])` (px/h) | `analysis.py:557, 559-561` |

### 3.5 QA overlay on a raw frame (`qa_overlay`, `analysis.py:870-929`)

The raw frame with the highest frame number that has trajectory rows.

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 3.5.1 | Load + stretch + RGB | `im = tifffile.imread(src).astype(np.float32)`; `u8 = stretch_uint8(im)` (the 1/99 percentile stretch of `io.py:134-146`); `rgb = np.stack([u8]·3, −1)` | `analysis.py:890-892` |
| 3.5.2 | Colony tint | `sel[m] = 0.45` (m = colony mask); `rgb = (rgb·(1 − sel) + color·sel).astype(np.uint8)`; edge = `m & ~ndi.binary_erosion(m)` (1-px 8-connected boundary) painted solid | `analysis.py:899-915` |
| 3.5.3 | Circle + rim rings | `d = √((yy − cy)² + (xx − cx)²)`; `rgb[\|d − R\| < 2.5]` blue (well circle), `rgb[\|d − (R − rim_width)\| < 2.5]` orange (rim boundary) | `analysis.py:924-927` |
| 3.5.4 | Downscale + save | `Image.fromarray(rgb).resize((512, 512), BILINEAR).save(out_path)` | `analysis.py:928-929` |

---

## 4. Finetuning track — raw frames → training pairs

Module: `src/colony_tracking/finetune/_convert.py` (prepare) and
`src/colony_tracking/finetune/__init__.py` (evaluate).

| # | Operation | Exact math | Code |
| --- | --- | --- | --- |
| 4.1.1 | Label inspection | `label = np.asarray(tifffile.imread(label_path))`; instance count = `int(label.max())`; labels with fewer than `min_instances` instances are dropped | `_convert.py:106-109, 117-119` |
| 4.1.2 | Raw frame stretch | `raw = np.asarray(tifffile.imread(rec["raw"]))`; written as `io.stretch_uint8(raw)` — the same 1/99 percentile stretch (`(im − lo)/(hi − lo) · 255`, clipped, uint8) the raw-input segment path uses | `_convert.py:196-204` |
| 4.1.3 | Label cast | `label.astype(np.int32)`; shape mismatch against the raw frame is skipped with a warning | `_convert.py:198-210` |
| 4.1.4 | Write | h5 datasets `raw` (uint8) and `label` (int32), gzip level 4, one file per frame pair under `<out>/{train,val}/` | `_convert.py:202-210` |
| 4.2.1 | Evaluate: tile crop | box-centred tile: `ty = clip((y0+y1)//2 − ph//2, 0, h − ph)`, `tx = clip((x0+x1)//2 − pw//2, 0, w − pw)`; `patch = raw[ty:ty+ph, tx:tx+pw]` (default 512×512) | `__init__.py:282-295` |
| 4.2.2 | SAM prompt | `predictor.set_image(np.repeat(patch[..., None], 3, −1))`; box swapped to `[x0−tx, y0−ty, x1−tx, y1−ty]`; one mask from `predict(box=…, multimask_output=False)` | `__init__.py:299-305` |
| 4.2.3 | Per-instance IoU | `union = logical_or(pred, true).sum()`; `iou = logical_and(pred, true).sum() / union` (0.0 when the union is empty); `true = (patch_gt == iid)` | `__init__.py:307-314` |
| 4.2.4 | Clipped flag | `(y1 − y0) > ph or (x1 − x0) > pw` | `__init__.py:315-317` |

---

## 5. Dtype flow summary

```
raw uint16
  ├─ difference:  → float32 (cur − ref) → 128 + d·127/limit, clip 0..255 → uint8 diff frame
  ├─ segment (raw): float32 → percentile(1,99) stretch → uint8 → model
  ├─ segment (diff): uint8 (as written) → model
  │     decoder foreground (float probabilities) → threshold fg / clear ~mask / watershed
  │     → instance labels (uint32) → growth mask (bool, 15-px dilation)
  │     → boolean segment masks (IoU / coverage / pre-merge in pure numpy)
  │     → colony label images (uint32)
  └─ analyze: raw uint16 → float32 → gaussian σ6 → radial dips → Kasa circle (float)
              label uint32 → distance map float32 → rim zone (bool) → rim fractions
              raw frame → 1/99 stretch → uint8 RGB overlay → 512×512 PNG
```
