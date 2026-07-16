# Evaluation

Evaluate a search champion and the appropriate reference baselines on two independent cohorts:
the subjects used during search and subjects held out from search. Every subject is scored through
the same harness separately, then subjects are averaged with equal weight within each cohort.

| domain | dataset | search subjects | held-out subjects |
| --- | --- | --- | --- |
| vision | NSD | 1, 2 | 5, 7 |
| vision | TVSD | F | N |
| vision | NNN | 1, 2, 3 | 4, 5 |
| language | Huth Lab | UTS01, UTS02 | UTS03 |
| language | iEEG Podcast | 01, 02, 03 | 04, 05 |

`datasets.py` is the single source of truth for cohorts and metrics. `run.py` automatically chooses
the vision or language baselines, prints one search/held-out table, validates the result, and is the
only writer of `evaluation.json`. `vision.py` and `language.py` are scoring tools; they do not write
paper-result files.

```bash
python experiments/evaluation/run.py --dataset nsd
python experiments/evaluation/run.py --dataset tvsd
python experiments/evaluation/run.py --dataset nnn
python experiments/evaluation/run.py --dataset huth
python experiments/evaluation/run.py --dataset ieeg
```

Render any completed evaluation as a publication-style cohort comparison. The plot shows search
subjects on the left and subjects held out from search on the right. Bars are equal-weight cohort
means, points are individual subjects, and the random projection is a cohort-specific dashed line.
Each architecture is labeled once, with trained and untrained variants paired closely on the same
row and distinguished by solid versus hatched bars. The searched champion is highlighted separately
and explicitly labeled as never trained.

```bash
python experiments/evaluation/plot.py --dataset nnn
python experiments/evaluation/plot.py --input path/to/evaluation.json --output figure.pdf
```

By default, `--dataset` looks first in `runs/best/<dataset>/evaluation.json`, then in
`experiments/evaluation/results/<dataset>/evaluation.json`, and writes `evaluation.png` beside the
input. Use `--output` for PNG, vector PDF, or editable SVG.

The committed NSD and Triple-N figures can be regenerated at publication resolution with:

```bash
python experiments/evaluation/plot.py --dataset nsd --output experiments/evaluation/figures/nsd.png
python experiments/evaluation/plot.py --dataset nsd --output experiments/evaluation/figures/nsd.pdf
python experiments/evaluation/plot.py --dataset nnn --output experiments/evaluation/figures/nnn.png
python experiments/evaluation/plot.py --dataset nnn --output experiments/evaluation/figures/nnn.pdf
```

The evaluator never asks which search run to use. The exporter has already made that decision by
selecting the best run matching the dataset's predefined search cohort. Evaluation reads that exact
`runs/best/<dataset>/model.py`, verifies its provenance against `datasets.py`, and writes beside it:

```bash
python runs/best/export_champions.py
python experiments/evaluation/run.py --dataset nsd
```

The published layout is `runs/best/<dataset>/{model.py,evaluation.json}`. The JSON is model-first:
each stable model ID contains individual subject scores and an equal-weight mean for both `search`
and `heldout`. It also records the metric, readout, source run, Git commit, timestamp, and the exact
champion SHA256. There is deliberately no cross-subject gap field.

Vision evaluations contain the champion, random projection, trained and untrained AlexNet,
ResNet-50, and ViT-B/16, plus pretrained CLIP and DINOv2. Language evaluations contain the champion,
random projection, and pretrained GPT-2 XL. The writer always requires the complete modality-specific
roster, so a partial smoke test cannot accidentally replace a publication result.
