# Best models (`runs/best/`)

One champion for each dataset's canonical search cohort. Regenerate the models with
`python runs/best/export_champions.py`, then generate the complete model-first result with
`experiments/evaluation/run.py`. The latter writes `evaluation.json` beside `model.py`.

An existing evaluation is kept only while its champion SHA256 matches the exported model.
NSD and NNN use noise-normalized r2; TVSD, Huth, and iEEG use raw mean Pearson r.

| Dataset | Search subjects | Search reward | Model | Source run | Evaluation |
| --- | --- | ---: | --- | --- | --- |
| huth | UTS01,UTS02 | 0.0629 | `v0hdy` | huth__2026-07-15 | pending |
| ieeg | 01,02,03 | 0.0630 | `plaxh` | ieeg__2026-07-07 | pending |
| nnn | 1,2,3 | 0.2425 | `cosgm` | nnn__2026-07-15 | pending |
| nsd | 1,2 | 0.4348 | `k7y09` | nsd__2026-06-29 | pending |
| tvsd | F | 0.7136 | `7sjzq` | tvsd__2026-06-30 | pending |
