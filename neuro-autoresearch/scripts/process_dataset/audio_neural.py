"""Process an audio NeuralSet dataset (meg | ieeg) into a slim, raw-free data/<ds>/ tree.

OPTION A: we do NOT pre-segment or bake the Mel / window. We write only:
  data/<ds>/neural/continuous_<modality>.fif  -- the FULL continuous neural signal already
      channel-picked + (for ieeg) bad-aware exactly as the raw path, downsampled to 100Hz.
      Reconstructed from the dataset's own neural extractor (_NEURAL[d.neural]) so it is the
      SAME preprocessing the raw path runs; saved via MneTimedArray.to_native() (lossless).
  data/<ds>/neural/meta.json                  -- neural event start (recording-time origin),
      subject, frequency, the trigger query + neural/stim windows (provenance).
  data/<ds>/stimuli/audio.wav                 -- the source podcast/listen audio, copied verbatim.
  data/<ds>/stimuli/words.csv                 -- the word-onset table (start,duration,text,
      language,modality) in recording time.

At LOAD time, harness.loaders.timeline rebuilds an events DataFrame from these files and runs the
EXISTING Segmenter + Mel extractor + _build_xy, so the window (start/duration) and Mel params stay
flexible and the output reproduces the raw path bit-for-bit (verified for meg).

Run (full, on the box where the raw data lives):
  python -m scripts.process_dataset.audio_neural --dataset meg
  python -m scripts.process_dataset.audio_neural --dataset ieeg
Both source datasets are small enough to fully generate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import pandas as pd
import neuralset as ns

from harness.datasets import PRESETS
from harness.loaders.shared import processed_dir
from harness.loaders import studies  # noqa: F401  (registers the studies)
from harness.loaders.timeline import _NEURAL

# Raw source per dataset (the offline build reads here once; runtime reads only data/<ds>/).
RAW = {
    "meg": "/home/t-ymehta/Research/neuralset/data/bel2026petitlisten_sample",
    "ieeg": "/home/t-ymehta/Research/neuralset/data/ds005574_podcast",
}


def _process(ds_key: str) -> None:
    d = PRESETS[ds_key]
    out = processed_dir(ds_key)
    neural_dir, stim_dir = os.path.join(out, "neural"), os.path.join(out, "stimuli")
    os.makedirs(neural_dir, exist_ok=True)
    os.makedirs(stim_dir, exist_ok=True)

    study = ns.Study(name=d.study, path=RAW[ds_key])
    events = study.run()

    # --- neural: run the dataset's own extractor to get the preprocessed (picked, 100Hz)
    # continuous signal, then dump it as a plain .fif via MneTimedArray.to_native(). ---
    ext = _NEURAL[d.neural](d)
    typed = ext._event_types_helper.extract(events)
    if len(typed) != 1:
        raise RuntimeError(f"expected one {d.neural} event, got {len(typed)}")
    ta = next(ext._get_data([typed[0]]))
    raw = ta.to_native()
    fif = os.path.join(neural_dir, f"continuous_{d.neural}.fif")
    raw.save(fif, overwrite=True)

    # --- stimuli: copy the audio verbatim, write the word-onset table. ---
    audio_row = events[events["type"] == "Audio"].iloc[0]
    shutil.copyfile(audio_row["filepath"], os.path.join(stim_dir, "audio.wav"))

    words = events[events["type"] == "Word"].copy()
    cols = ["start", "duration", "text", "language", "modality"]
    words = words[[c for c in cols if c in words.columns]]
    words.to_csv(os.path.join(stim_dir, "words.csv"), index=False)

    # The audio's recording-time origin + span: words index into the wav relative to
    # audio_start, so it MUST be preserved (MEG's audio starts at 43.9s, not 0).
    json.dump(
        {
            "modality": d.neural,
            "subject": str(getattr(typed[0], "subject", "") or ""),
            "frequency": float(ta.frequency),
            "start": float(ta.start),
            "audio_start": float(audio_row["start"]),
            "audio_duration": float(audio_row["duration"]),
            "trigger": d.trigger,
            "neural_start": d.start,
            "neural_duration": d.duration,
        },
        open(os.path.join(neural_dir, "meta.json"), "w"),
        indent=2,
    )

    print(f"[{ds_key}] wrote {fif} ({raw.info['nchan']} ch @ {raw.info['sfreq']}Hz, "
          f"{raw.n_times} samp), audio.wav, {len(words)} words -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["meg", "ieeg"])
    _process(p.parse_args().dataset)
