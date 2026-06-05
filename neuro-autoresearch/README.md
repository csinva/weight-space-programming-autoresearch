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
evolve/model.py     EDITABLE: the model(s) + main that evaluates and saves
evolve/program.md   the task description / instructions to the agent
harness/config.py   FIXED: the one active Dataset (edit one line to switch datasets)
harness/data.py     FIXED: RawPixels extractor + aligned, split (X, Y) tensors
harness/eval.py     FIXED: predictivity reward + evaluate() + save_results()
runs/               overall_results.csv + test_corr.png (written by save_results)
```

`evolve/` is the only zone the agent edits; `harness/` is the fixed evaluation
harness the agent must not touch.

## Install

```bash
pip install .   # core deps + copilot-keepgoing (the nonstop agent wrapper)
```

## Run (offline, no downloads)

```bash
/usr/bin/python3.12 evolve/model.py                  # active model (random_projection)
/usr/bin/python3.12 evolve/model.py --model identity # the trivial baseline
```

Run from the repo root so results land in `runs/`. Each run prints a `test_corr`
summary and **upserts** a row into `runs/overall_results.csv` (keyed by
`(dataset, model_name)`), refreshing the `runs/test_corr.png` bar plot. The
default `Fake2025Meg` study scores ≈0 (random) by design — the point is that it
**runs end-to-end offline**.

## Switch datasets

Edit the single `DATASET = Dataset(...)` line in `harness/config.py` (e.g. set
`study=`, `neural=`, `trigger=`, and `download=True` for a real study). Nothing
else changes — dataset specifics live only in `config.py`.

## Drive the autoresearch loop nonstop

The Copilot CLI sometimes stops mid-run. Wrap it with `copilot-keepgoing`
(installed via `pip install .`) to auto-nudge it whenever it goes idle:

```bash
copilot-keepgoing -- copilot --autopilot --yolo   # then: "Follow evolve/program.md"
```

Credits: data + alignment via NeuralSet/NeuralFetch; loop structure after csinva's
circuit-building-autoresearch.
