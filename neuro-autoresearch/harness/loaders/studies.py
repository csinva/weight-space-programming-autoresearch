"""NeuralSet ``Study`` subclasses (registered on import).

Importing this module registers every class below in ``neuralset.Study.catalog()`` so they can be
loaded by name with ``ns.Study(name=..., ...)``.

``Zada2025Podcast``: the OpenNeuro "Podcast" ECoG dataset (ds005574).

A single subject listened to a ~30-minute spoken podcast while their intracranial
EEG (iEEG/ECoG) was recorded at 512 Hz. This ``Study`` exposes that recording as a
NeuralSet timeline of ``Word`` triggers + a single ``Audio`` stimulus + one ``Ieeg``
neural event. Importing this module registers the class in
``neuralset.Study.catalog()`` under the name ``Zada2025Podcast`` so it can be loaded
with ``ns.Study(name="Zada2025Podcast", ...)``.

The dataset ships nine subjects (``sub-01``..``sub-09``), but each has its OWN
electrode array, so concatenating subjects is invalid. We therefore restrict to one
subject via the ``SUBJECT`` class var, and the study yields exactly ONE timeline. The
word timing is shared across all subjects: a single central transcript
(``stimuli/podcast_transcript.csv``) gives each word's ``start``/``end`` in RECORDING
time (the recording IS the podcast playback, so a word's ``start`` indexes directly
into the iEEG with no extra offset).
"""
from __future__ import annotations

import typing as tp

import mne
import pandas as pd
from neuralset.events import study

# mne/mne-bids are chatty when reading BIDS iEEG; silence them at import.
mne.set_log_level(False)


# Provenance: authored here (no neuralfetch study); a from-scratch NeuralSet Study over BIDS iEEG.
class Zada2025Podcast(study.Study):
    """OpenNeuro Podcast ECoG (ds005574) restricted to one subject (sub-01)."""

    # See module docstring for why this is single-subject.
    SUBJECT: tp.ClassVar[str] = "01"

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        query="type=='Ieeg'",
        num_events_in_query=1,
        event_types_in_query={"Ieeg", "Audio", "Word"},
        data_shape=(124, 921600),
        frequency=512.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        if not self.path.exists():
            raise RuntimeError(f"Missing folder {self.path}")
        yield dict(subject=self.SUBJECT)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        """Read the BIDS iEEG recording for this subject via mne-bids.

        ``read_raw_bids`` applies the channel types (ECOG/ECG/MISC) and bad-channel
        status from the sidecar ``channels.tsv``.
        """
        from mne_bids import BIDSPath, read_raw_bids

        bp = BIDSPath(
            subject=timeline["subject"],
            task="podcast",
            root=self.path,
            datatype="ieeg",
            suffix="ieeg",
            extension=".edf",
        )
        return read_raw_bids(bp, verbose=False)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # One Word event per transcript row (timing is in recording time, so it
        # indexes straight into the iEEG), one Audio event spanning the recording,
        # and one Ieeg event whose data is loaded lazily via the SpecialLoader.
        df = pd.read_csv(self.path / "stimuli" / "podcast_transcript.csv")

        word_events: list[dict[str, tp.Any]] = []
        for r in df.itertuples():
            word = str(r.word)
            if word == "nan" or not word.strip():
                continue
            word_events.append(
                dict(
                    type="Word",
                    start=float(r.start),
                    duration=float(r.end) - float(r.start),
                    text=word,
                    language="english",
                    modality="heard",
                )
            )

        audio_event = dict(
            type="Audio",
            start=0.0,
            duration=float(df.end.max()) + 1.0,
            filepath=str(self.path / "stimuli" / "podcast.wav"),
        )

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        ieeg_event = dict(type="Ieeg", filepath=info, start=0)

        return pd.DataFrame([ieeg_event, audio_event, *word_events])
