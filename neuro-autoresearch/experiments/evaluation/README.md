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
python experiments/evaluation/run.py --run runs/nsd__DATE
python experiments/evaluation/run.py --run runs/tvsd__DATE
python experiments/evaluation/run.py --run runs/nnn__DATE
python experiments/evaluation/run.py --run runs/huth__DATE
python experiments/evaluation/run.py --run runs/ieeg__DATE
```

Development results use the stable path
`experiments/evaluation/results/<dataset>/evaluation.json`. To publish a result beside the committed
champion, first export the champion and then evaluate that exact file:

```bash
python runs/best/export_champions.py
python experiments/evaluation/run.py --run runs/nsd__DATE --model runs/best/nsd/model.py
```

The published layout is `runs/best/<dataset>/{model.py,evaluation.json}`. The JSON is model-first:
each stable model ID contains individual subject scores and an equal-weight mean for both `search`
and `heldout`. It also records the metric, readout, source run, Git commit, timestamp, and the exact
champion SHA256. There is deliberately no cross-subject gap field.

Vision evaluations contain the champion, random projection, trained and untrained AlexNet,
ResNet-50, and ViT-B/16, plus pretrained CLIP and DINOv2. Language evaluations contain the champion,
random projection, and pretrained GPT-2 XL. The writer always requires the complete modality-specific
roster, so a partial smoke test cannot accidentally replace a publication result.
