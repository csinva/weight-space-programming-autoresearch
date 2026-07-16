# neuro-autoresearch

**Search for the *computations* that explain the brain.**

An AI agent writes — and rewrites — the code for a function `f: stimulus → representation`. A
ridge readout from that representation to recorded brain activity scores it, and the held-out score
is the agent's reward. The object of the search is not a model that *fits* the brain but the
**function a brain region computes**: given oriented filters, kernels, energy hierarchies, character
features — any computation expressible in code — which one predicts the neural response?

The unit of progress is one file (`model.py`); the unit of truth is one number (`test_corr`).

> **0.44** NSD (r²/NC)  ·  **0.063** iEEG (raw r)  ·  **< 100K** params, enforced  ·  **0** gradient steps  ·  **4** datasets · **2** species · **3** modalities

## The idea

Three things separate this from architecture search or model-fitting:

- **Not architecture search.** The search space is the space of *functions*, not a fixed backbone
  family. A proposal can be Gabor energy, a scattering transform, a hashed lexical bag — anything
  computable under the contract. The NSD champion is an HMAX energy model, not a CNN.
- **Written, not fitted.** Every parameter is analytic or random-but-fixed, written directly in
  code. No gradient descent, and the function never sees the neural target `Y` — so a high score
  reflects the **computation itself**, not weights tuned to the data.
- **An agent as the scientist.** Autonomous LLM agents author, run, and revise the code — reasoning
  about mechanism, reading their own scores, and building on what worked. The search runs
  unsupervised for hours across a diverse pool.

## The loop

Each proposer is an autonomous LLM agent. On every iteration it gets a brief — the I/O contract, the
source of its own current best function, and an anonymized scoreboard of what rival approaches
scored — and authors a new function as a self-contained program (front-end, nonlinearity, pooling,
layout). It submits; the fixed harness returns a scalar **reward** (held-out predictivity) plus a
per-region breakdown of *where* it won and lost. That signal is **not a gradient**: the agent reads
it against its own notes, forms a hypothesis, and writes the next version.

```
X (N,D_in) → f (agent-authored, fixed) → rep (N,D_out) → ridge (the only fit) → brain Y (held-out) → test_corr → back to the agent
```

**Islands keep the pool diverse.** Scores are shared; source code is not. Each agent is an *island*
that may read code only from its own best or one random other island — never a global champion — so
approaches stay genuinely different instead of collapsing onto one recipe. (`VANILLA_AUTORESEARCH=1`
closes the cross-island channels to recover an independent-loops baseline for controlled comparison.)

## The contract — what keeps the reward honest

A self-contained `model.py` defines `class Model` with `__call__(self, X)` and ends with
`MODEL = Model()`:

| | |
| --- | --- |
| signature | `f: X (N, D_in) → (N, D_out)` |
| weights | written in code, never learned |
| params | `< 100,000` (`NEURO_MAX_PARAMS`, enforced robustly) |
| width | `D_out < 10,000` |
| per-sample | no statistics across the batch |
| sees | only `X`, never the target `Y` |
| fit | the ridge readout only — that *is* the evaluation |
| reward | held-out `test_corr` |

`model.py` is the **only** evolving artifact; the harness around it is fixed — if it could be edited,
the reward would be meaningless. Static and runtime guards block pretrained weights, network access,
and reading the target. The repo-root `model.py` is the **random-projection baseline** — the floor
every model should beat and the seed each proposer starts from. (For image datasets `X` is flattened
RGB pixels of a square image, `side = round(sqrt(D_in/3))`.)

**Why written, not learned?** A trained encoding model entangles two questions: does the brain
resemble this *computation*, or this *training set*? Writing the parameters in code answers only the
first — the one we want. The 100K-parameter ceiling is a forcing function for interpretability: every
champion is a small, readable computation you can turn into a mechanistic claim, not a black box.

## The four datasets

**Two vision, two language** — three recording modalities across two species. Each is a preset in
`harness/datasets.py`, chosen by the `NEURO_DATASET` env var (no source edit, so parallel runs never
collide) and loaded **fully offline** from a local copy.

| domain | `NEURO_DATASET` | study | brain · stimulus | reward |
| --- | --- | --- | --- | --- |
| **vision** | `nsd` (default) | Natural Scenes Dataset · human | fMRI · natural images | early+ventral stream, noise-normalized · **0.44** |
| | `nnn` | Triple-N · macaque | spiking · natural images | reliable EVC+IT units, noise-normalized |
| **language** | `huth` | Huth story-listening · human | fMRI · spoken narrative | all cortical voxels, raw *r* |
| | `ieeg` | 'Podcast' ECoG · human | iEEG · spoken words | all electrodes, 5-fold temporal CV · **0.063** |

**Vision** — `X` is the flattened RGB pixels of a square image (`side = round(sqrt(D_in/3))`).
- **`nsd`** — prepared GLMdenoise betas; reward = the joint early+ventral stream (~14.5k reliable
  voxels), with the shared-1000 images held out image-disjoint. Subjects 1, 2 are the search reward;
  5, 7 are held out for generalization (via `NEURO_SUBJECT`).
- **`nnn`** (Triple-N) — macaque spiking to the *same* NSD-1000 images; reward = reliable EVC+IT units
  pooled, with a 5-fold split over the 1000 images. Search monkeys M1–M3; M4, M5 held out.

**Language** — the stimulus is `chars`: each word is the character-id grid of its n-gram, so the model
builds linguistic structure from characters up, with **no pretrained embeddings**.
- **`huth`** — human fMRI while listening to spoken stories (the "Moth Radio Hour"); per-word features
  are resampled onto the fMRI TR grid and FIR-delayed, then scored over all cortical voxels. Subjects
  UTS01–03.
- **`ieeg`** — human intracranial high-gamma (70–200 Hz) to a ~30-min podcast; reward = a rotating
  5-fold temporal CV over all electrodes (a single fixed hold-out is trivially overfit), meaned over
  patients 01–03 (04, 05 held out).

The **vision** rewards are noise-ceiling-normalized (r²/NC over reliably-driven voxels/units, so 0.44
means 44% of the *recoverable* signal); the **language** rewards are raw mean Pearson *r*. The reward
region is a fixed **per-dataset** attribute — no ROI menu, no way to aim the readout at the target.
Splits are cached on disk keyed by the full dataset identity, so reruns load in seconds. *(The harness
also ships a `tvsd` macaque-vision preset, outside the paper's four.)*

## Two champions, read as hypotheses

- **NSD · 0.44 — an HMAX energy hierarchy.** Gabor quadrature-energy (V1 edge power), HMAX
  max-pooling with a compressive nonlinearity (tolerance), random-weight conv layers with an abs
  (phase-invariant) nonlinearity (mid/high features), a cross-orientation Gram (texture), and a
  foveal pool. Scored through the *identical* harness against five frozen pretrained nets (AlexNet,
  ResNet-50, ViT-B/16, CLIP, DINOv2, each given its best layer per stream), it yields a **double
  dissociation**: the untrained champion owns early/mid vision (0.55 vs AlexNet's 0.43) while a
  pretrained net owns the category-selective ventral tip (DINOv2 0.47 vs champion 0.35).
- **iEEG · 0.063 — a hand-built lexical–syntactic feature map** from character codes (no pretrained
  lexicon): a hashed n-gram form of the current word, a recency-weighted context bag that up-weights
  content words over function words, a syntactic-category one-hot, and a bilinear "does this word fit
  the running topic" term — all read against high-gamma ECoG.

## Run (offline, no downloads)

**Install**
```bash
pip install .
```

**Manual single eval** (repo root) — loads the split once, scores `model.py` once, prints the row
(`train_corr` / `test_corr` / timing); persists nothing.
```bash
python -m harness.run                          # score the current model.py on the active dataset
NEURO_DATASET=ieeg python -m harness.run       # pick a preset: nsd | nnn | huth | ieeg
NEURO_READOUT=himalaya python -m harness.run   # reference readout (default = the ~20× faster fast ridge)
```

**Open-ended search** (unattended) — `search/run.sh` scaffolds a dated run and supervises three
decentralized, self-paced loops with no synchronization barrier:
```bash
search/run.sh nsd --hours 12                # 10 proposers + 1 GPU evaluator + a conductor steering every 60m
search/run.sh nsd --hours 12 --effort high  # ONE effort knob (low|medium|high|xhigh|max) for every agent
search/run.sh nsd --hours 12 --epoch-mins 0 # conductor OFF — pure decentralized pool
search/run.sh nnn --hours 6 --gpu 1         # pin to GPU 1 → run a dataset per card in parallel
```
- **Proposer pool** — N headless LLM CLIs, each confined to its own sandbox, editing its own
  `model.py`, proposing it, and blocking only on *its own* score.
- **GPU evaluator** — claims queued models one at a time, scores each with the fixed harness in
  isolation, and writes the result back (`search/evaluator/repair.sh` auto-fixes small mechanical
  faults, ≤2 tries).
- **Conductor** (optional) — every `--epoch-mins` writes one broad, **advisory** `direction.txt` per
  proposer to keep the pool diverse and unstuck; it never gates the queue or scoring.

By default the fleet is spread round-robin across five models (Opus 4.8/4.7/4.6, GPT‑5.5, Gemini 3.1
Pro); override with `NEURO_MODEL_ROSTER`. Everything shared lives in one SQLite file,
`runs/<dataset>__<date>/search.db` — the queue, full history, and scores. Re-launching the same date
**resumes** losslessly; a new date starts fresh.

**The `hub` API/CLI** — agents never touch the DB, GPU, or dataset directly:
```bash
python hub.py propose-model model.py --idea "…"  # snapshot + queue this model → prints its id
python hub.py wait-for-score <id>                # block until the evaluator scores it
python hub.py my-best                            # YOUR island's best model (+source) to build on
python hub.py peek-island                        # one RANDOM other island's idea + notes (no source, no score)
python hub.py random-models -n 5                 # random prior models: scores + ideas (no source)
python hub.py summary                            # one-line DB tally + current champion
```

## Repo layout

```
model.py              EVOLVED — the one model (seed: random-projection baseline)
hub.py                the API + CLI over a run's search.db
harness/              FIXED evaluation — loaders, ridge readout, reward, and manual runner
search/               the core autoresearch framework
  run.sh                the single search entry point
  proposer/             proposer pool, worker, and prompt
  evaluator/            evaluator pool, worker, and repair agent
  conductor/            advisory conductor loop and prompt
dashboard/            live view of a running search
experiments/          scientific studies built with the framework
  evaluation/           baselines and held-out generalization
  complementarity.py    champion + pretrained-model complementarity
  imagenet/             ImageNet training and neural re-evaluation
scripts/              operational support: prepare_data/, benchmark/, simplify/
docs/                 human docs — start at docs/overview.html
runs/<dataset>__<date>/ one search: search.db + models/ + proposers/ + eval/ (gitignored)
```

## The experiments it enables

Three programs build on each other — **search → analyze → dissect**:

1. **Search** for models, either optimizing a single target dataset (vanilla) or *jointly* across
   several at once (multi-dataset). *Are there models good for human vision but poor for monkey — or
   vice versa — and can one be good at both?*
2. **Analyze the population** (~2,000 models): which computational motifs are enriched in the best
   models, and which properties distinguish the ones that generalize across datasets?
3. **Dissect the best**: identify a champion's motifs, perturb them one at a time, and measure which
   are *causally* responsible for brain alignment — moving from *finding* good models to
   understanding *why* they work.

---
*Timeline segmentation and alignment are provided by NeuralSet.*
