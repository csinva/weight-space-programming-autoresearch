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

`datasets.py` is the single source of truth for these cohorts. `run.py` automatically chooses the
vision or language baselines and prints one search/held-out table. Its JSON output also retains every
individual subject score because scores from different subjects are not directly interchangeable.

```bash
python experiments/evaluation/run.py --run runs/nsd__DATE
python experiments/evaluation/run.py --run runs/tvsd__DATE
python experiments/evaluation/run.py --run runs/nnn__DATE
python experiments/evaluation/run.py --run runs/huth__DATE
python experiments/evaluation/run.py --run runs/ieeg__DATE
```
