"""Process the iEEG audio NeuralSet dataset into a slim, raw-free data/ieeg/ tree.

OPTION A: we do NOT pre-segment or bake the Mel / window. We write only:
  data/ieeg/neural/continuous_ieeg_sub-<NN>.fif  -- the FULL continuous neural signal already
      channel-picked + bad-aware exactly as the raw path, downsampled to 100Hz. Staged from the
      dataset's OWN high-gamma derivative (see below), saved via mne (lossless).
  data/ieeg/neural/meta.json                  -- neural event start (recording-time origin),
      subject, frequency, the trigger query + neural/stim windows (provenance).
  data/ieeg/stimuli/audio.wav                 -- the source podcast audio, copied verbatim.
  data/ieeg/stimuli/words.csv                 -- the word-onset table (start,duration,text,
      language,modality) in recording time.

At LOAD time, harness.loaders.timeline rebuilds an events DataFrame from these files and runs the
EXISTING Segmenter + _build_xy, so the window (start/duration) stays flexible and the output
reproduces the raw path.

Run (on the box where the raw data lives), per patient:
  python -m scripts.process_dataset.audio_neural --dataset ieeg --subject 01
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace

import neuralset as ns

from harness.datasets import PRESETS
from harness.loaders.shared import processed_dir
from harness.loaders import studies  # noqa: F401  (registers the studies)

# Raw source (the offline build reads here once; runtime reads only data/ieeg/).
RAW = {"ieeg": "/home/t-ymehta/Research/neuralset/data/ds005574_podcast"}


def _process(ds_key: str, subject: str = "") -> None:
    """Stage one iEEG patient: write the picked+resampled high-gamma .fif, the copied audio, the word-onset table, and meta.json into data/ieeg/."""
    d = PRESETS[ds_key]
    if subject:
        d = replace(d, subject=subject)          # stage this patient (iEEG is multi-subject)
    out = processed_dir(ds_key)
    neural_dir, stim_dir = os.path.join(out, "neural"), os.path.join(out, "stimuli")
    os.makedirs(neural_dir, exist_ok=True)
    os.makedirs(stim_dir, exist_ok=True)

    study = ns.Study(name=d.study, path=RAW[ds_key])
    events = study.run()

    # --- neural: the dataset's OWN high-gamma derivative (ecogprep: artifact-cleaned, z-scored
    # 70-200 Hz power -- the standard ECoG cognition signal; raw voltage @100Hz discards the HG band,
    # and a hand-rolled Hilbert envelope isn't normalised so slow drift dominates the reward), picked
    # to ECOG channels and resampled to 100Hz. Baked once here so the load path stays a plain
    # pick/resample (_NEURAL["ieeg"]). ---
    import mne
    hg = os.path.join(RAW[ds_key], f"derivatives/ecogprep/sub-{d.subject}/ieeg",
                      f"sub-{d.subject}_task-podcast_desc-highgamma_ieeg.fif")
    raw = mne.io.read_raw_fif(hg, preload=True).pick("ecog").resample(100.0)
    subject, freq, start = f"{d.study}/{d.subject}", 100.0, 0.0
    # Per-patient neural file (iEEG stages several patients into one tree). The stimuli
    # (audio/words) and the shared fields of meta.json are identical across patients.
    fif = os.path.join(neural_dir, f"continuous_{d.neural}_sub-{d.subject}.fif")
    raw.save(fif, overwrite=True)

    # --- stimuli: copy the audio verbatim, write the word-onset table. ---
    audio_row = events[events["type"] == "Audio"].iloc[0]
    shutil.copyfile(audio_row["filepath"], os.path.join(stim_dir, "audio.wav"))

    words = events[events["type"] == "Word"].copy()
    cols = ["start", "duration", "text", "language", "modality"]
    words = words[[c for c in cols if c in words.columns]]
    words.to_csv(os.path.join(stim_dir, "words.csv"), index=False)

    # The audio's recording-time origin + span: words index into the wav relative to audio_start,
    # so it MUST be preserved.
    json.dump(
        {
            "modality": d.neural,
            "subject": subject,
            "frequency": freq,
            "start": start,
            "audio_start": float(audio_row["start"]),
            "audio_duration": float(audio_row["duration"]),
            "trigger": d.timeline.trigger,
            "neural_start": d.timeline.start,
            "neural_duration": d.timeline.duration,
        },
        open(os.path.join(neural_dir, "meta.json"), "w"),
        indent=2,
    )

    print(f"[{ds_key}] wrote {fif} ({raw.info['nchan']} ch @ {raw.info['sfreq']}Hz, "
          f"{raw.n_times} samp), audio.wav, {len(words)} words -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ieeg", choices=["ieeg"])
    p.add_argument("--subject", default="", help="iEEG patient to stage, e.g. 01 (default: the preset's)")
    a = p.parse_args()
    _process(a.dataset, a.subject)
