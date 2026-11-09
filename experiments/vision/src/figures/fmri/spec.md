# fMRI figure: search, generalization and computational families of the LAION champion

The figure shows that an agentic search on LAION-fMRI finds an interpretable model (erjpu) that beats
interpretable baselines on an unseen dataset (NSD), and that its explicit computations map onto a
difference across visual areas, from corners and contours in early areas to texture statistics in
FFA and PPA. The last panel compares the contributions of texture operations in hV4, FFA and PPA.

- Model: erjpu, frozen copy
  `artifacts/final/laion-visual-encoding-to-nsd-encoding/cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633/model.py`.
- Search run: `artifacts/search/laion__visual__encoding__2026-11-05` (protocol looped-cross-dataset-v4, 600 candidates).
- Style: follow `docs/technical-writing.md` (one message per panel, no top/right spines, colorblind-safe,
  consistent family colors, readable text).

## Layout

| File | Role |
|---|---|
| `b_trajectory.py` | data for panel B -> `artifacts/figures/fmri/data/b.json` |
| `c_nsd.py` | data for panel C -> `artifacts/figures/fmri/data/c.json` |
| `d_families.py` | data for panel D -> `artifacts/figures/fmri/data/d.json` (per-voxel results in `data/d/`) |
| `e_texture.py` | data for panel E -> `artifacts/figures/fmri/data/e.json` |
| `make_figure.py` | reads all data files, writes `artifacts/figures/fmri/fmri_figure.{pdf,svg,png}` and `panels/{b,c,d,e}.{pdf,png}` |

Render the existing JSON artifacts with `make_figure.py`; styling does not require model evaluation.
The assembled figure is 180 × 116 mm. Lowercase panel letters are bold 8 pt, descriptive
titles are 7 pt, body text is 6.5 pt, and keys and supporting notes are 5.8 pt. Use Arial,
then Helvetica, Liberation Sans and DejaVu Sans as fallbacks. Export vector PDF and SVG,
a 450 dpi PNG, and individual B–E PDF and 450 dpi PNG exports with the same typography. Render each panel
independently with bounds fitted to its labels, so neighboring panels cannot enter its crop.

The D and E heatmaps, including feature counts as aligned text, are exported to
`fmri_heatmaps.pdf` with a 450 dpi PNG preview. E uses the main panel's operation order and
plain labels. Use one shared normalization for both heatmaps: 0–60% for nonnegative shares, extending
the range when needed. If either panel has negative shares, both use a zero-centered
diverging scale. The current exports span −10% to 60%; the small negative mean is labeled
−0.4 rather than rounded to zero. Colorbars and cells remain vector objects.
Do not outline selected cells.
All outputs are under `artifacts/` (untracked). `e_surface.py` is a separate surface-shape analysis,
not the source for panel E.

## Panels

**A. LAION dataset overview.** Made separately by the author; `make_figure.py` leaves an empty labeled slot.

**B. Search trajectory.** Message: the search steadily finds better models.
- x: candidate number in evaluation order (created_at); y: encoding r on LAION search data.
- Grey dots: pooled score of every scored candidate. Main line: running best of the pooled score.
  Two thinner, lighter lines: early-stream and ventral-stream scores of that same running-best model.
- erjpu highlighted as “Searched model.” The two regional traces are scores of the model selected
  by the pooled objective, not independently optimized regional running maxima.
- The plotted y-range focuses on the search progression. With the current data, three of the
  600 scored candidates fall below this range; disclose that count below the panel. The four
  unscored database rows are excluded from the scored sequence, not counted as clipped points.

**C. Generalization to NSD versus interpretable baselines.** Message: the searched model beats existing
interpretable models on a dataset the search never used.
- NSD subjects 01, 02, 05, 07 (completed all sessions, saw all 1,000 shared images); one dot per subject.
- Readout fit on each subject's own (unique) images; tested on the shared images that are not in LAION's
  stimulus set. The current artifact contains 998 test images.
- Models: erjpu; interpretable baselines from `src/baselines/` (pixels, gabor, gabor_norm, gist, hmax,
  scattering, texture, curvature); AlexNet conv4 as a grey reference (layer chosen by LAION pooled encoding before NSD scoring).
- Scores for pooled (early + ventral), early stream and ventral stream.

**D. Relative contributions of computational families.** Message: visual areas differ in which computation
explains them beyond edges.
- Categorical contribution profiles. x: V1, V2, V3, hV4, FFA, PPA; y: share of gain beyond edges (%).
  Highlight corners and contours and texture statistics; draw color, luminance curvature and global
  layout in gray, with distinct markers and line styles so their unconnected points remain
  identifiable. Use the same family colors throughout the figure.
- Connect V1 through hV4 within each family. Show FFA and PPA as unconnected points, separated from
  V1–hV4, so their positions do not imply a serial cortical hierarchy.
- Values are exact Shapley contributions to signed r² gain beyond edges, averaged over ROI voxels
  and subjects as recorded in the artifact. Family shares sum to 100% within each ROI. Preserve
  negative contributions; these are signed attribution shares, not fractions of a positive total.

**E. Contributions of texture operations.** Message: texture operations contribute differently in hV4,
FFA and PPA, with variation across subjects.
- Three horizontal dot plots, one each for hV4, FFA and PPA, with a common percentage axis and shared
  operation rows. Show each subject as a gray dot and the mean across subjects as a texture-pink tick. Preserve
  negative values and show a zero reference.
- Use the same operation rows in all three plots: texture boundaries (M), orientation/color
  co-occurrence (V), cross-scale structure (O), geometry co-occurrence (R), and joint feature
  co-occurrence (J). Display the plain labels without keys or feature counts. Report feature counts
  with the full heatmaps in the supplement.
- E is a separate exact Shapley attribution within an edges-plus-texture model: for a subset S of
  texture operations, the value is signed r²(E + S) − signed r²(E), where E denotes edges.
  Normalize by the gain from all texture operations beyond edges. E is not a subdivision of D's
  texture contribution, which is attributed in the presence of other computational families.

D and E use existing artifacts from the legacy `looped-cross-dataset-v4` protocol: all five LAION
subjects (sub-01, sub-03, sub-05, sub-06, sub-07) and all shared test images. Their provenance records
`stage: search`; this describes the historical evaluation and does not assert compliance with the
current subject/image holdout protocol. Preserve this provenance when rendering; do not use these
artifacts for current search feedback, model selection or repair.

## Proposed caption

**Search trajectory, cross-dataset encoding and computational attribution.**
(a) Reserved for the LAION-fMRI dataset overview. (b) LAION search encoding correlations
for 600 scored candidates. The dark trace is the running pooled-best score; early and
ventral traces report that same model's regional scores. The orange point marks erjpu;
three candidates fall below the displayed range. (c) NSD encoding correlations for erjpu,
eight interpretable baselines and AlexNet conv4, whose layer is selected on LAION.
Dots show subjects 01, 02, 05 and 07; ticks show their means. Readouts are fit on unique
images and tested on 998 shared images, excluding LAION stimuli from both sets.
(d) Exact Shapley shares of signed r² gain beyond edges for five computational families.
Lines connect V1–hV4; FFA and PPA are separate categories. (e) Subject scores and mean
shares for five texture operations in hV4, FFA and PPA. This is a separate attribution
within an edges-plus-texture model, not a subdivision of panel d's texture share.
Panels d and e use five LAION subjects and all shared test images under the historical
protocol, rather than the current subject/image holdout protocol. Shares may be negative.

## Data formats

All JSON files carry a `provenance` object (model path, dataset, subjects, protocol, image-ID hash where
relevant). Region keys use the dataset names: `"early and ventral visual streams"` (pooled),
`"early visual stream"`, `"ventral visual stream"`, and area names `V1, V2, V3, hV4, FFA, PPA`.

`b.json`
```
{"run": str, "champion": "erjpu", "n_excluded": int,
 "candidates": [{"index": int, "id": str, "created_at": str,
                 "pooled": float, "early": float, "ventral": float}],   # scored candidates, evaluation order
 "running_best": [{"index": int, "id": str, "pooled": float, "early": float, "ventral": float}],  # one per candidate
 "provenance": {...}}
```

`c.json`
```
{"subjects": ["01", "02", "05", "07"], "n_test_images": int, "test_image_ids_sha256": str,
 "regions": ["early and ventral visual streams", "early visual stream", "ventral visual stream"],
 "models": {name: {"label": str, "kind": "searched" | "baseline" | "reference",
                   "scores": {subject: {region: float}}, "mean": {region: float}}},
 "provenance": {...}}
```

`d.json` (written by `d_families.py`)
```
{"rois": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
 "families": {"K": "color", "S": "luminance curvature", "C": "corners and contours",
              "T": "texture statistics", "G": "global layout"},
 "share_beyond_edges": {family: {roi: float}},                    # columns sum to 1
 "share_beyond_edges_subjects": {family: {roi: {subject: float}}},
 "features_per_family": {family: int},                         # includes edges and auxiliary groups
 "edges_share_of_total": {roi: float}, "gain_beyond_edges_r2": {roi: float},
 "subjects": [...], "provenance": {...}}
```

`e.json` (written by `e_texture.py`; contains all six ROIs, with three shown in the main panel)
```
{"rois": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
 "substats": {operation: {"name": str, "blocks": [str], "description": str, "n_features": int}},
 "share_beyond_edges": {operation: {roi: float}},                # sums to 1 within each ROI
 "share_beyond_edges_subjects": {operation: {roi: {subject: float}}},
 "gain_beyond_edges_r2": {operation: {roi: float}},
 "unique_r2": {operation: {roi: float}},
 "texture_gain_beyond_edges_r2": {roi: float},
 "subjects": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
 "provenance": {...}}
```

Operation keys are M (second-order modulation), V (V1 channel covariance), J (joint feature
covariance), O (cross-scale relations) and R (geometry co-occurrence). The JSON retains these
technical names and feature-block definitions; display labels may use plainer wording without
changing the attribution groups. Both panels convert stored fractional shares to percentages.
