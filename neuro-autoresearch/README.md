# neuro-autoresearch

A bare-minimum, AlphaEvolve-style **autoresearch** proof-of-concept for
neural encoding, built on **NeuralSet + NeuralFetch**. Structure mirrors
[`csinva/circuit-building-autoresearch`](https://github.com/csinva/circuit-building-autoresearch)
(`evolve-neuro/`), generalised to any NeuralSet dataset.

## The loop (3 steps)

1. An agent **proposes a computational model** (`model.py`) — weights written
   directly in code, **never trained** (no gradient descent, no optimizer).
2. The fixed harness passes the stimulus through the model → **activations**.
3. Activations are **linearly read out with ridge regression** to predict the
   recorded neural activity. The held-out **`test_corr`** (mean per-target
   Pearson r) is the **reward**. Only the ridge readout is fit.

So `model.py` is the evolved artifact; `test_corr` is the verifiable reward.

## File map

```
config.py      the one active Dataset (edit one line to switch datasets)
src/data.py    fixed: RawPixels extractor + aligned, split (X, Y) tensors
src/eval.py    fixed: predictivity reward + evaluate() + save_results()
model.py       EDITABLE: the model(s) + main that evaluates and saves
results/       overall_results.csv + test_corr.png (written by save_results)
```

## Run (offline, no downloads)

```bash
/usr/bin/python3.12 model.py                       # active model (random_projection)
/usr/bin/python3.12 model.py --model identity      # the trivial baseline
```

Each run prints a `test_corr` summary and **upserts** a row into
`results/overall_results.csv` (keyed by `(dataset, model_name)`), refreshing the
`results/test_corr.png` bar plot. The default `Fake2025Meg` study scores ≈0
(random) by design — the point is that it **runs end-to-end offline**.

## Switch datasets

Edit the single `DATASET = Dataset(...)` line in `config.py` (e.g. set
`study=`, `neural=`, `trigger=`, and `download=True` for a real study). Nothing
else changes — dataset specifics live only in `config.py`.

Credits: data + alignment via NeuralSet/NeuralFetch; loop structure after csinva's
circuit-building-autoresearch.
