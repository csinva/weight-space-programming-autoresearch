# Experiments

Scientific studies built with the autoresearch framework:

- `evaluation/` compares a search champion with modality-appropriate baselines on both search and
  held-out subjects across NSD, TVSD, NNN, Huth, and iEEG.
- `complementarity.py` tests whether the champion explains neural variance missed by AlexNet.
- `imagenet/` trains the champion's ventral weights on ImageNet and re-evaluates the result on NSD.

Run commands from the repository root. Evaluation writes beside the exported champion under
`runs/best/<dataset>/`; other generated outputs stay under ignored `results/` directories.

```bash
python experiments/evaluation/run.py --dataset nsd
python experiments/complementarity.py --champion runs/nsd__DATE/models/MODEL.py
python experiments/imagenet/train.py --epochs 5
python experiments/imagenet/evaluate.py
```
