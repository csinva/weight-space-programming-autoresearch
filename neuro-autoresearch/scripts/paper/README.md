# Final paper evaluation

Activate the project environment on the remote machine, then pass the four completed run folders:

```bash
source .venv/bin/activate
python scripts/paper/evaluate.py \
  runs/nsd__YYYY-MM-DD runs/nnn__YYYY-MM-DD \
  runs/ieeg__YYYY-MM-DD runs/huth__YYYY-MM-DD \
  --split huth=UTS01,UTS02:UTS03
```

Omit `--split` for datasets already declared in `scripts/eval/splits.py`. The override must
match the subjects saved in that run's `env.json`. The runner takes each database champion, scores it unchanged
on search and held-out subjects, then runs paired pretrained and random-initialized baselines through
the same readout and metric. It writes detailed JSON under `results/paper/<run>/` and the combined
paper table to `results/paper/summary.csv`.

Jobs are resumable: existing JSON is reused. Use `--force` to recompute it or `--dry-run` to inspect
every command first. The vision suite uses pretrained and random-initialized AlexNet, ResNet-50, and
ViT-B/16, plus pretrained-only CLIP and DINOv2. The language suite is pretrained GPT-2-XL only.

The lower-level benchmark scripts retain their weight-selection options for one-off checks, but the
paper runner launches only this predefined matrix.

## Exact computation

The runner is a thin wrapper over the existing `scripts/eval/vision.py`,
`scripts/eval/language.py`, `scripts/eval/evaluate.py`, and fixed
`harness/` readout. It does not implement a second evaluation path.

For every subject and vision network:

1. Run the frozen network on the dataset's train and test images and collect its configured layer
   taps. Random-initialized variants use the same fixed seed for train and test images.
2. Project a layer wider than 2,048 features to 2,048 dimensions with the same fixed sparse random
   projection.
3. Split only the neural training set 80/20 and select the layer with the highest validation score.
   Selection is per broad region: `early` and `ventral` for NSD; `EVC` and `IT` for NNN. The final
   test responses are not used for layer selection.
4. Fit the same ridge readout used by the search with the selected representation. NSD is scored on
   its fixed held-out images; NNN uses its fixed five-fold image CV. Report the dataset's exact
   noise-ceiling-normalized score.

For GPT-2-XL, each transcript word receives its contextual hidden state at every layer. The best
layer is selected on a train-only temporal validation tail. iEEG then uses the harness's five-fold
temporal CV; Huth applies its existing Lanczos word-to-TR and FIR-delay pipeline before its fixed
held-out-story score. Both language datasets report raw mean correlation.

Every model is evaluated separately for every subject. Subject scores are averaged with equal
weight within the search cohort and within the held-out cohort. The search champion is loaded from
the run database and evaluated unchanged through the same harness.
