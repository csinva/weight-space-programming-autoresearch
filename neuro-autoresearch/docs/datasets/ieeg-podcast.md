# iEEG Dataset — "Podcast" ECoG (`Zada2025Podcast`)

The loop's first **intracranial** neural modality: ECoG (iEEG) recorded while
patients listened to a spoken story. Adds intracranial electrophysiology to the
existing fMRI / MEG / spiking targets.

## Dataset

| Property | Value |
| --- | --- |
| Study (catalog name) | `Zada2025Podcast` |
| Name | "Podcast" ECoG dataset (Zada, Nastase, Goldstein, Hasson) |
| **Neural modality** | **iEEG / ECoG** — 1,330 intracranial electrodes across **9 epilepsy patients**, 512 Hz |
| **Stimulus modality** | **audio** (a 30-minute spoken podcast; >5,000 transcript-aligned words) |
| Paradigm | Passive listening; each aligned **Word** triggers a 0.5 s epoch |
| Source | OpenNeuro `ds005574` |
| Licence | **CC0** — fully open, no gate |
| Size | ~3.5 GB BIDS |
| Sample scope | **single subject** (default `sub-01`) — each patient has a different electrode set, so cross-subject pooling is not possible |

## Where it's staged

`~/Research/neuralset/data/ds005574_podcast` (under the **gitignored**
`~/Research/neuralset/data/`, not committed — expected).

## How it was downloaded

Via the `openneuro` Python package (no AWS, no T&C gate):

```python
import openneuro

openneuro.download(
    dataset="ds005574",
    target_dir="/home/t-ymehta/Research/neuralset/data/ds005574_podcast",
)
```

## How to run

```bash
NEURO_DATASET=ieeg python -m harness.run
```

Structurally identical to the `meg` preset (audio stimulus + `Word` triggers + a
fast electrophysiology signal): `IeegExtractor` picks the ECOG/SEEG channels
(dropping ECG/MISC), `read_raw_bids` marks bad channels from `channels.tsv`, and
the stimulus is the Mel spectrum of the audio in the same `[0, 0.5] s` window.
Single-subject (`Zada2025Podcast`, default `sub-01`).
