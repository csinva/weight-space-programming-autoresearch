# Paper results for the iWSP visual cortex model

Stored outputs behind the visual cortex section and Figure 3 of the paper. The iWSP model is the
candidate `erjpu`, the best of 600 candidates in the LAION-fMRI search
`laion__visual__encoding__2026-11-05` (pooled early and ventral encoding, five subjects).

This folder mirrors the layout of `artifacts/`, which the scripts in `src/figures/` read and write.
To re-render a figure from these files, link the folder and run the figure script from
`experiments/vision`:

```bash
ln -s results artifacts
python src/figures/combined/make_figure.py
```

| Path | Contents |
| --- | --- |
| `final/laion-visual-encoding-to-nsd-encoding/<sha256>/` | Frozen iWSP model (`model.py`), how it was selected (`selection.json`) and its final NSD evaluation (`evaluation.json`) |
| `figures/combined/` | Figure 3 of the paper (PDF and PNG) and the map data of panel c |
| `figures/fmri/data/` | Panel data: search trajectory (`b.json`), NSD scores for the iWSP model, baselines and AlexNet layers (`c.json`), family shares by area (`d.json`), texture-operation shares (`e.json`), and the family attribution summary (`d/`) |
| `figures/circuit/` | Area readout drivers and tuning to synthetic stimuli (figure and panel data). The tuning panel is not in the paper yet and is a candidate for the supplement |
| `final/attribution_erjpu_2026-11-06/` | Earlier block-level attribution reports for the iWSP model |
| `final/alexnet_comparison_visual_encoding_2026-11-05/` | AlexNet layer selection on LAION-fMRI (conv3 for early, fc6 for ventral) and its report |
| `final/baselines/` | Interpretable baseline scores on LAION-fMRI and NSD |
| `final/compression/` | Rebuilding the model from a short text description |

Not included: the search run itself (candidate models, score database, agent notes and logs),
cached feature matrices, per-voxel arrays, and the area readout weights. They are large or need the
datasets to be useful; the scripts regenerate them.

Notes on what is stored:

- The family and texture attributions and the area readouts use all five LAION-fMRI subjects and
  the shared images under the search protocol of the `erjpu` run (`looped-cross-dataset-v4`), not
  the current held-out-subject protocol described in the main README.
- `figures/fmri/data/c.json` holds NSD scores for four subjects and 998 shared images that are not
  in LAION-fMRI. It lists three AlexNet layers; the paper uses conv3 (early) and fc6 (ventral).
- The `.npz` panel files and the figures contain a small number of LAION-fMRI stimulus images.
- Absolute paths inside the provenance fields were replaced by `<repo>`.
