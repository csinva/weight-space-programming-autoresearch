# neuro-autoresearch

A tiny, **dataset-agnostic** framework for building computational models that
predict neural activity from stimuli — across domains (image, text, …) and
datasets — with extremely simple code.

```
Stimulus ──(extractor)──▶ X ──(your model)──▶ Ŷ  ≈  Y  (measured neural response)
```

The heavy lifting (downloading datasets, extracting stimulus features, reading
neural recordings, aligning them per stimulus) is delegated entirely to
[NeuralSet + NeuralFetch](https://facebookresearch.github.io/neuroai/). This
repo owns only the two things that are *your* science:

1. the **encoding model** `X → Ŷ`, and
2. the **predictivity metric**.

## The entire API

```python
from neuro_autoresearch import load_pairs, train_test_split, RidgeEncoder, evaluate
from neuro_autoresearch.presets import stimulus, neural

pairs = load_pairs(
    "Grootswagers2022HumanSample",   # ← swap dataset (any in ns.Study.catalog())
    stimulus("image"),               # ← swap domain  (image / text / …)
    neural("eeg"),                   # ← swap modality (meg / eeg / fmri)
    trigger="type=='Image'",
)
train, test = train_test_split(pairs)
print(evaluate(RidgeEncoder(), train, test)["predictivity"])
```

* **Switch dataset** → change the string.
* **Switch domain** → change the stimulus preset.
* **Switch your model** → implement `fit(X, Y)` / `predict(X)` (the `Encoder`
  protocol). Nothing else changes.

## Two reference encoders

| Model | Paradigm |
|---|---|
| `RidgeEncoder` | frozen-feature linear probe — *"how brain-predictive are these features?"* (Brain-Score style) |
| `MLPEncoder`   | learned encoder trained end-to-end with SGD |

Both satisfy the same `Encoder` protocol, so swapping is one line.

## Layout

```
neuro_autoresearch/
  data.py        load_pairs() / train_test_split()  — dataset-agnostic data layer
  models.py      Encoder protocol + RidgeEncoder + MLPEncoder
  eval.py        predictivity() + evaluate()
  presets.py     domain (stimulus) & modality (neural) extractor presets
  experiment.py  CLI: one command = one experiment
examples/run_fake.py   offline 10-line demo
tests/test_smoke.py    offline end-to-end test
```

## Install

Requires Python 3.12 and NeuralSet's `[all]` extra (for image/text extractors):

```bash
/usr/bin/python3.12 -m pip install --user --break-system-packages \
    'neuralset[all]' neuralfetch
/usr/bin/python3.12 -m pip install --user --break-system-packages -e .
```

## Run

```bash
# offline smoke run (no downloads)
python3.12 -m neuro_autoresearch.experiment --study Fake2025Meg \
    --domain image-fast --modality meg --model ridge

# tests
python3.12 -m pytest tests/ -q
```

## How it maps onto NeuralSet/NeuralFetch

NeuralSet/NeuralFetch already provide the aligned `(stimulus, neural)` data
interface — one `Segmenter` with two extractors yields both tensors per
stimulus, identically across datasets. They do **not** provide an encoding
model, training loop, or predictivity metric. This repo is exactly that thin
(~250-line) layer.
