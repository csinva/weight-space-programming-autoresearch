# Experiments

Scientific studies built with the autoresearch framework:

- `evaluation/` compares a search champion with modality-appropriate baselines on both search and
  held-out subjects across NSD, TVSD, NNN, Huth, and iEEG.
- `complementarity.py` tests whether the champion explains neural variance missed by AlexNet.
- `imagenet/` trains the champion's ventral weights on ImageNet and re-evaluates the result on NSD.

Run commands from the repository root. Generated outputs stay beside their experiment under an
ignored `results/` directory, or in the source search run when they belong to that run.

```bash
python experiments/evaluation/run.py --run runs/nsd__DATE
python experiments/complementarity.py --champion runs/nsd__DATE/models/MODEL.py
python experiments/imagenet/train.py --epochs 5
python experiments/imagenet/evaluate.py
```
