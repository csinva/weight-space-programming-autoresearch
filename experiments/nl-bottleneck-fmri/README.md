# Text bottleneck for the language fMRI models

Can a short text description carry everything that matters about a hand-written brain encoding model? This
experiment repeats the text-bottleneck test of `experiments/nl-bottleneck` (algorithmic and NLP tasks) on
the story-listening fMRI models of the paper's hidden-set table.

## Models

The eight models come from the search on all TRs (`experiments/runs/neuro/fmri-may27-run1`, subject UTS03),
the same models as the paper's hidden-set table: the best model so far at roughly evenly spaced iterations,
the last model before the scan-timing jump, the first model after it and the final model. They are listed in
`common.py` (`MODELS`). Several snapshots load other snapshot files and the run's main file
`interpretable_transformer.py` (for word lists); `01_select.py` copies each model with every file it loads.

## Procedure

1. **Describe** (`02_summarize.py`). A fresh headless Claude Code session (`claude -p`, Opus 5.5, Read tool
   only, a sandbox holding only the model's files) reads the model and writes a description of at most 200
   words for an engineer who must rebuild a model that predicts the brain about as well and carries the same
   information, without seeing the code. If the text runs over 200 words, the same session is asked to shorten
   it. The descriptions are in `summaries/`.
2. **Rebuild** (`03_reconstruct.py`). For each description, two fresh headless Claude Code sessions (Opus 5.5)
   each get the task, the description, the original's visible performance and a sandbox (`sandbox_template/`)
   with a template `model.py` and a fixed `score.py`. They hand-write a model with the same rules as the
   original search: no training or fitting of any kind, no pretrained models, embeddings or tokenizers, nothing
   fit to the fMRI responses or story statistics, and NLTK WordNet only if the description says the original
   used it. Each session may score at most 6 experiments; `score.py` fits ridge regression on the 8 fitting
   stories and reports the mean voxel correlation on the 2 visible stories. The sessions never see the hidden
   stories. They are told to aim for faithfulness rather than to beat the original, and end by naming their
   final model. Logs, scored runs and snapshots are in `recon/<model>-r<k>/` (logs gzipped).
3. **Evaluate** (`04_evaluate.py`). Originals and each rebuild's final model are scored the way the hidden-set
   table is: ridge regression (the run's own `src/features.py` and `src/encoding.py`, all TRs) fit on the 8
   fitting stories and evaluated on the 2 visible stories and on 10 hidden stories never shown to any agent.
   The originals reproduce the table's values exactly. Two similarity measures compare a rebuild with its
   original on the hidden stories: *prediction similarity*, the per-voxel correlation between the two models'
   predicted responses, averaged over the original's 10% best-predicted voxels; and *map similarity*, the
   correlation across all voxels between their per-voxel hidden performance maps. Results are in
   `originals_eval/`, `recon_eval/` and `results.json` (`05_analyze.py`).
4. **Compare predictions across models** (`06_pred_fractions.py`). Each rebuild's hidden-story predictions are
   compared with every original's, on that original's best 1%, 2%, 5% and 10% of voxels, giving an 8 by 8 matrix
   (rows: originals; columns: rebuilds, averaged over the two seeds) whose diagonal at 10% is the prediction
   similarity above. An exact permutation test over all 40,320 pairings of originals with rebuilt models compares
   the diagonal mean with the off-diagonal mean. Written to `pred_matrix.json`.
5. **Count parameters** (`07_param_counts.py`). Every number a constructed model stores (tensor and array elements
   and numeric constants; word lists and WordNet are not counted), for each original and rebuild, in
   `param_counts.json`. The search's own count (torch parameters only) gives 2 for the WordNet-based models.

## Results

| Iteration | Model | Why selected | Visible: original / rebuilds | Hidden: original / rebuilds | Prediction similarity | Map similarity | Cost |
|---|---|---|---|---|---|---|---|
| 1 | BoCharMean-d256 | first model | 0.0339 / 0.0325 | 0.0164 / 0.0161 | 0.96 | 0.98 | $0.15 |
| 40 | MultiScaleSubword+FuncWordTags | evenly spaced | 0.0413 / 0.0349 | 0.0201 / 0.0165 | 0.71 | 0.88 | $0.55 |
| 76 | AllNewFeats-Words15 | evenly spaced | 0.0428 / 0.0299 | 0.0216 / 0.0177 | 0.63 | 0.84 | $0.57 |
| 152 | ExpDec-R4.0 | evenly spaced | 0.0440 / 0.0354 | 0.0222 / 0.0182 | 0.74 | 0.89 | $0.67 |
| 192 | DenseSemXX-N20 | evenly spaced | 0.0561 / 0.0425 | 0.0257 / 0.0193 | 0.60 | 0.86 | $1.27 |
| 238 | WordNetMorphLingMultiTau | last before the jump | 0.0658 / 0.0613 | 0.0317 / 0.0298 | 0.88 | 0.95 | $0.86 |
| 239 | WordNetMorphLingStoryArc | first after the jump | 0.0984 / 0.0919 | 0.0433 / 0.0409 | 0.91 | 0.97 | $0.78 |
| 242 | WordNetMorphLingNovelty | final model | 0.1087 / 0.0939 | 0.0483 / 0.0441 | 0.85 | 0.94 | $0.89 |

Averaged over the 8 models, rebuilds keep 87% of the original hidden performance.

Both rebuilds of the first model after the jump recreate its story-position features from the description.
The models in the middle of the search, which rely on long hand-made word lists and hashed word identity that a
200-word description can name but not list, lose the most. Each rebuild session used 1 to 4 of its 6 allowed
experiments; the whole experiment cost about $11 (descriptions about $5, rebuilds about $6) and took about two
hours on one laptop.

**Caveat.** Unlike `experiments/nl-bottleneck`, there is no control condition (rebuilds without a
description), so the similarity measures have no baseline: part of the similarity may come from any reasonable
language model predicting the same voxels.

## Reproducing

Set `NLBF_DATA` to a folder with the public release files (`grids/<story>.TextGrid`, `resp/<story>.hf5`,
`respdict.json`, `split.json` with the `fit`, `visible` and `new` stories, and `nltk/` with WordNet), and
`NLBF_WORK` to a scratch folder outside `~/.claude` (Claude Code will not edit files there). Then run
`01_select.py`, `02_summarize.py`, `04_evaluate.py originals`, `03_reconstruct.py --reps 1 2`,
`04_evaluate.py recon`, `05_analyze.py`, `06_pred_fractions.py` and `07_param_counts.py`. Scoring needs about 10 GB of memory, so `score.py` and
`04_evaluate.py` share a lock that runs one scoring at a time.
