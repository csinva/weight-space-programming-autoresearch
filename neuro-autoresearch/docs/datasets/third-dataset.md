# Third Dataset — Le Petit Prince MEG (LPP-Listen)

> The chosen study (`Bel2026PetitListenSample`) landed and is the live `meg` preset
> in `PRESETS` in `harness/datasets.py`. This doc records why it was picked and how
> it loads (download steps + MaxShield gotchas), all still current.

A third famous dataset of a **genuinely different neural modality** so the
autoresearch loop spans modalities. NSD (`NsdSubj01`) is fMRI/vision
and Li2022Petit is fMRI/language; this adds **MEG**.

## Chosen study: `Bel2026PetitListenSample`

| Property | Value |
| --- | --- |
| Study (catalog name) | `Bel2026PetitListenSample` (sample of `Bel2026PetitListen`) |
| Aliases | `LPP-Listen`, `LPP MEG Listen` |
| **Neural modality** | **MEG** (306-channel Elekta Neuromag TRIUX, 1000 Hz) — *different from fMRI* |
| **Stimulus modality** | **text / audio** (spoken French narrative; word + sentence + audio + full-text events) |
| Paradigm | Passive listening to the *Le Petit Prince* audiobook (French) |
| Source | OpenNeuro `ds007523` via `openneuro-py` (already installed) |
| Licence | **CC0-1.0** — fully open, **no AWS, no Terms-&-Conditions gate** |
| DOI | 10.18112/openneuro.ds007523.v1.0.0 |
| Sample scope | subject 26, session 01, run 01 (single MEG run) |
| Local download size | **923 MB** (820 MB MEG `.fif` + 104 MB stimuli) |

### Why this study

- **Different modality, as required.** MEG is electrophysiological (ms-resolution
  magnetic fields), categorically different from the BOLD fMRI of the other two
  datasets. EEG/MEG was the explicit target.
- **Famous.** *Le Petit Prince* is a well-known naturalistic-language neuroimaging
  benchmark; this is the MEG companion (Bel, Bonnaire, King, Pallier, 2026) to the
  fMRI LPP corpus already used here (`Li2022Petit`).
- **Freely downloadable, no gate.** CC0-1.0 on OpenNeuro, fetched with the
  installed `openneuro-py` client — no AWS CLI, no licence acceptance. (The THINGS
  MEG/EEG candidates `Hebart2023ThingsMeg` / `Gifford2022Large` /
  `Grootswagers2022HumanSample` were rejected: their image stimuli require the
  password-gated THINGS-images database — `NEURALFETCH_THINGS_PASSWORD` — which is a
  T&C gate. `Brennan2019Hierarchical` needs a Globus client.)
- **Has a Sample variant** for a fast first download (one subject/run) and the
  sample explicitly bundles the open text + audio stimulus files.

## Exact download steps

The sample auto-downloads via the neuralset/neuralfetch backend; no manual fetch:

```python
import neuralset as ns

DATA_DIR = "/home/t-ymehta/Research/neuralset/data/bel2026petitlisten_sample"
study = ns.Study(name="Bel2026PetitListenSample", path=DATA_DIR)
study.download()        # selective OpenNeuro pull: sub-26 run-01 + stimuli (CC0)
events = study.run()    # -> events DataFrame
```

`download()` nests the data one level deeper under the class name (neuralset's
rule), so files land in `…/bel2026petitlisten_sample/Bel2026PetitListenSample/`:

```
sub-26/ses-01/meg/sub-26_ses-01_task-listen_run-01_meg.fif   (820 MB, MaxShield raw)
sub-26/ses-01/meg/sub-26_ses-01_task-listen_run-01_events.tsv
sourcedata/stimuli/audio/ch1-3.wav                            (spoken stimulus)
sourcedata/stimuli/txt_laser/run1.txt                         (text stimulus)
sourcedata/task-listen_run-01_extra_info.tsv
participants.tsv  dataset_description.json
```

The dataset dir lives under `…/neuralset/data/` which is **gitignored** (not
committed — expected).

## Load verification (measured)

`study.run()` returns **1775 events**:

| Event type | Count |
| --- | --- |
| `Word` | 1614 |
| `Sentence` | 158 |
| `Meg` | 1 |
| `Audio` | 1 |
| `Text` | 1 |

One `Segmenter` segment built with a text stimulus extractor + MEG neural
extractor (trigger on word onsets, 0.5 s epoch, 100 Hz):

```python
from neuralset.extractors.text import WordLength
seg = ns.Segmenter(
    extractors=dict(
        stim=WordLength(aggregation="sum"),                       # text stimulus
        neural=ns.extractors.MegExtractor(frequency=100.0,
                                          allow_maxshield=True),  # MEG neural
    ),
    trigger_query="type=='Word'", start=0.0, duration=0.5,
    drop_incomplete=True,
)
batch = seg.apply(events).load_all()
```

Resulting shapes (verified):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `X` (stim, `WordLength`) | **(1614, 1)** | one scalar feature per word |
| `Y` (neural, `MegExtractor`) | **(1614, 306, 50)** | 1614 words × 306 MEG channels × 50 samples (0.5 s @ 100 Hz) |
| `Y` flattened per segment | **15300** | 306 × 50 |

### Two integration gotchas (needed for the load to succeed)

1. **MaxShield.** The raw `.fif` is Elekta MaxShield (Internal Active Shielding),
   so the MEG extractor must pass `allow_maxshield=True` or it raises.
2. **Overlapping word windows.** Word epochs overlap, so a text extractor on
   `Word` events needs `aggregation="sum"` (or `"mean"`); the default `"single"`
   raises on colliding static slots.

## Live preset (`harness/datasets.py`)

This dataset's stimulus is **text, not image**, so `imsize` is irrelevant and the
harness selects a **text** stimulus extractor (e.g. `WordLength(aggregation="sum")`)
and passes `allow_maxshield=True` to the MEG extractor for the `meg` branch. It is
wired up as the `meg` entry in `PRESETS`:

```python
"meg": Dataset(
    study="Bel2026PetitListenSample",   # Le Petit Prince MEG (LPP-Listen), OpenNeuro ds007523
    trigger="type=='Word'",             # one segment per spoken word onset
    neural="meg",                       # 306-ch MEG (MaxShield raw -> allow_maxshield=True)
    stimulus="audio",
    start=0.0,
    duration=0.5,                       # 0.5 s epoch -> 50 samples @ 100 Hz
    download=False,                     # pre-downloaded at the path below (CC0, no gate)
    path="/home/t-ymehta/Research/neuralset/data/bel2026petitlisten_sample",
),
```

> Stimulus = **text/audio** (language); neural = **MEG**. With NSD (fMRI/vision)
> and Li2022Petit (fMRI/language) already in place, this dataset gives the loop its
> first electrophysiological (MEG) modality.
