# Final paper evaluation

Activate the project environment on the remote machine, then pass the four completed run folders:

```bash
source .venv/bin/activate
python scripts/paper/evaluate.py \
  runs/nsd__YYYY-MM-DD runs/nnn__YYYY-MM-DD \
  runs/ieeg__YYYY-MM-DD runs/huth__YYYY-MM-DD \
  --split huth=UTS01,UTS02:UTS03
```

Omit `--split` for datasets already declared in `scripts/generalization/splits.py`. The override must
match the subjects saved in that run's `env.json`. The runner takes each database champion, scores it unchanged
on search and held-out subjects, then runs paired pretrained and random-initialized baselines through
the same readout and metric. It writes detailed JSON under `results/paper/<run>/` and the combined
paper table to `results/paper/summary.csv`.

Jobs are resumable: existing JSON is reused. Use `--force` to recompute it or `--dry-run` to inspect
every command first. The default suites are AlexNet, ResNet-50, ViT-B/16, CLIP, and DINOv2 for vision,
plus GPT-2 and GPT-2-XL for language.

For language, "untrained" means random transformer weights with the same fixed pretrained tokenizer;
the tokenizer is part of the input representation and is identical in the paired comparison.
