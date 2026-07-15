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
every command first. The vision suite uses pretrained and random-initialized AlexNet, ResNet-50, and
ViT-B/16, plus pretrained-only CLIP and DINOv2. The language suite is pretrained GPT-2-XL only.

The lower-level benchmark scripts retain their weight-selection options for one-off checks, but the
paper runner launches only this predefined matrix.
