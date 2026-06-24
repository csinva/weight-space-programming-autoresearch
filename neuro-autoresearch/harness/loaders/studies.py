"""Per-subject / single-subject NeuralSet ``Study`` subclasses (registered on import).

This module bundles three independent study families; importing it registers every
class below in ``neuralset.Study.catalog()`` so they can be loaded by name with
``ns.Study(name=..., ...)``.

``Li2022PetitSubj``: the Le Petit Prince fMRI corpus restricted to one subject.

A thin subclass of NeuralFetch's ``Li2022Petit`` (OpenNeuro ds003643) that keeps
only ONE English subject (``SUBJECT``, default EN057) and all 9 of its story
sections. Importing this module registers it (and the siblings below) in
``neuralset.Study.catalog()``, e.g. as ``Li2022PetitSubj``, loadable with
``ns.Study(name="Li2022PetitSubj", ...)``.

It also registers one sibling per English subject -- ``Li2022PetitSubj<ID>``
(e.g. ``Li2022PetitSubjEN058``) for every id in ``EN_SUBJECTS`` -- each overriding only
``SUBJECT``. Every subject heard the SAME story, so a model found on one can be re-fit
and scored on any other (the ridge re-fit IS the eval), exactly as NSD's
``NsdSubj01/02/03`` siblings enable. Select one via ``NEURO_SUBJECT`` (see
``harness/datasets.py``).

The parent yields each fMRI run as a path to a full-brain volume (73x90x74 ~ 486k
voxels); a single subject's 9 sections would be a ~60 GB target matrix the ridge
readout cannot hold. So, exactly as ``Allen2022Massive`` (NSD) does for its ROI, we
apply the spatial reduction AT THE STUDY LEVEL: the Fmri event carries a
``SpecialLoader`` that 2x-downsamples each volume onto a fixed grid and keeps only MNI
brain voxels (~29k), so the full volume is never materialised. Run->file mapping and
word/text parsing are reused from the parent unchanged.

``NsdSubj01/02/03``: the Natural Scenes Dataset restricted to one subject each.

Thin subclasses of NeuralFetch's ``Allen2022Massive`` (NSD) that keep a single
subject and all of its sessions. Importing this module registers each subclass in
``neuralset.Study.catalog()`` (e.g. ``NsdSubj01``) so it can be loaded with
``ns.Study(name="NsdSubj01", ...)``; the parent's nsdgeneral / visual-cortex ROI is
inherited unchanged.

Subjects 2 and 3 are the multi-subject ``NEURO_SUBJECT`` siblings: the same
shared-1000 images are shown to every NSD subject, so a model found on subject 1 can
be re-fit and scored on them unchanged.

The parent's StudyInfo counts all 8 subjects (3408 timelines); restricting to one
yields ``sessions x 12 runs`` (the 14-run sessions still yield 12 runs, just shifted
to runs 2-13). NeuralSet validates the discovered timeline count against
``num_timelines``, so it must reflect the subset.

``Zada2025Podcast``: the OpenNeuro "Podcast" ECoG dataset (ds005574).

A single subject listened to a ~30-minute spoken podcast while their intracranial
EEG (iEEG/ECoG) was recorded at 512 Hz. This ``Study`` exposes that recording as a
NeuralSet timeline of ``Word`` triggers + a single ``Audio`` stimulus + one ``Ieeg``
neural event. Importing this module registers the class in
``neuralset.Study.catalog()`` under the name ``Zada2025Podcast`` so it can be loaded
with ``ns.Study(name="Zada2025Podcast", ...)``.

The dataset ships nine subjects (``sub-01``..``sub-09``), but each has its OWN
electrode array, so concatenating subjects is invalid. We therefore restrict to one
subject (exactly as ``NsdSubj01`` / ``Li2022PetitSubj`` do) via the ``SUBJECT`` class
var, and the study yields exactly ONE timeline. The word timing is shared across all
subjects: a single central transcript (``stimuli/podcast_transcript.csv``) gives each
word's ``start``/``end`` in RECORDING time (the recording IS the podcast playback, so a
word's ``start`` indexes directly into the iEEG with no extra offset).
"""
from __future__ import annotations

import typing as tp

import numpy as np
import nibabel
import mne
import pandas as pd
from neuralfetch.studies.li2022petit import Li2022Petit
from neuralfetch.studies.allen2022massive import Allen2022Massive
from neuralset.events import study

# mne/mne-bids are chatty when reading BIDS iEEG; silence them at import.
mne.set_log_level(False)

# Spatial coarsening factor (per axis); shrinks the target matrix ~8x.
_DOWNSAMPLE = 2

# Per-process caches (deterministic, so per-process recomputation is harmless):
#  _GRID -> the fixed downsampled target grid + brain mask shared by every run.
#  _VOLS -> the masked (n_vox, T) array per nii file, so the heavy resample runs
#           once even though the loader is read twice (duration autofill + extract).
_GRID: dict[str, tp.Any] = {}
_VOLS: dict[str, np.ndarray] = {}


def _grid_and_mask(ref: nibabel.Nifti1Image) -> dict[str, tp.Any]:
    """Fixed (downsampled affine, shape, brain mask) shared by all runs.

    Derived once from the reference geometry; every run is resampled onto this
    exact grid so the masked voxel set is identical across runs and processes.
    """
    if not _GRID:
        from nilearn.image import resample_to_img
        from nilearn.datasets import load_mni152_brain_mask

        affine = ref.affine.copy()
        affine[:3, :3] = ref.affine[:3, :3] @ (np.eye(3) * _DOWNSAMPLE)
        shape = tuple(int(np.ceil(s / _DOWNSAMPLE)) for s in ref.shape[:3])
        grid = nibabel.Nifti1Image(np.zeros(shape), affine)
        mask = np.asarray(
            resample_to_img(
                load_mni152_brain_mask(), grid, interpolation="nearest"
            ).get_fdata()
        ) > 0
        _GRID.update(affine=affine, shape=shape, mask=mask)
    return _GRID


class Li2022PetitSubj(Li2022Petit):
    """Le Petit Prince fMRI restricted to a single subject (``SUBJECT``, all 9 sections)."""

    # Which subject this class loads (e.g. "EN057"). Per-subject siblings generated at
    # the bottom of this module override only this attribute.
    SUBJECT: tp.ClassVar[str] = "EN057"

    # One subject x 9 story sections; the parent's 1008-timeline count
    # would otherwise trip the "corrupted dataset" timeline check.
    _info: tp.ClassVar[study.StudyInfo] = Li2022Petit._info.model_copy(
        update={"num_timelines": 9, "num_subjects": 1}
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        want = f"sub-{self.SUBJECT}"
        for timeline in super().iter_timelines():
            if timeline["subject"] == want:
                yield timeline

    def _masked_load(self, timeline: dict[str, tp.Any]) -> nibabel.Nifti2Image:
        """Load one run, 2x-downsample, and keep only brain voxels -> (n_vox, T)."""
        niifile = timeline["niifile"]
        data = _VOLS.get(niifile)
        if data is None:
            from nilearn.image import resample_img

            img = nibabel.load(niifile)
            g = _grid_and_mask(img)
            ds = resample_img(
                img,
                target_affine=g["affine"],
                target_shape=g["shape"],
                interpolation="continuous",
            )
            data = ds.get_fdata(dtype="float32")[g["mask"]]
            _VOLS[niifile] = data
        return nibabel.Nifti2Image(data, np.eye(4))

    def _load_timeline_events(self, timeline: dict[str, tp.Any]):
        # Reuse the parent's run->file mapping and word/text/audio parsing, then
        # swap the single Fmri row's full-volume path for the reducing loader.
        events = super()._load_timeline_events(timeline)
        is_fmri = events["type"] == "Fmri"
        niifile = str(events.loc[is_fmri, "filepath"].iloc[0])
        loader = study.SpecialLoader(
            method=self._masked_load, timeline={"niifile": niifile}
        ).to_json()
        events.loc[is_fmri, "filepath"] = loader
        return events


# # # # # Per-subject siblings (multi-subject NEURO_SUBJECT) # # # # #

# The English arm of ds003643 has 49 subjects: EN057-EN115 with these gaps.
_EN_MISSING: frozenset[int] = frozenset({60, 66, 71, 80, 85, 90, 102, 107, 111, 112})
EN_SUBJECTS: tuple[str, ...] = tuple(
    f"EN{n:03d}" for n in range(57, 116) if n not in _EN_MISSING
)

# Generate one thin ``Li2022PetitSubj<ID>`` subclass per English subject, each only
# overriding ``SUBJECT``. Creating the class auto-registers it in the catalog (via
# ``Study.__init_subclass__``), giving it its own cache key + results file. Pydantic
# requires the overridden ClassVar to be annotated, hence the explicit ``__annotations__``.
for _sub in EN_SUBJECTS:
    _name = f"Li2022PetitSubj{_sub}"
    if _name not in globals():
        globals()[_name] = type(
            _name,
            (Li2022PetitSubj,),
            {
                "__doc__": f"Le Petit Prince fMRI restricted to subject {_sub}.",
                "__module__": __name__,
                "SUBJECT": _sub,
                "__annotations__": {"SUBJECT": tp.ClassVar[str]},
            },
        )


_RUNS_PER_SESSION = 12

# subject id -> (sessions, sessions_with_14_runs).
_SUBJECTS: dict[int, tuple[int, tuple[int, int]]] = {
    1: (40, (21, 38)),
    2: (40, (21, 30)),
    3: (32, (21, 30)),
}

# Generate one thin ``NsdSubj0<id>`` subclass per subject. Creating the class
# auto-registers it in the NeuralSet catalog (via ``Study.__init_subclass__``).
# Pydantic requires overridden ClassVars to be annotated, hence the explicit
# ``__annotations__``.
for _sid, (_sessions, _runs14) in _SUBJECTS.items():
    _name = f"NsdSubj{_sid:02d}"
    if _name not in globals():
        globals()[_name] = type(
            _name,
            (Allen2022Massive,),
            {
                "__doc__": f"NSD restricted to subject {_sid} (all {_sessions} sessions).",
                "__module__": __name__,
                "SESSIONS_PER_SUBJECT": {_sid: _sessions},
                "SESSIONS_WITH_14_RUNS": {_sid: _runs14},
                "_info": Allen2022Massive._info.model_copy(
                    update={
                        "num_timelines": _sessions * _RUNS_PER_SESSION,
                        "num_subjects": 1,
                    }
                ),
                "__annotations__": {
                    "SESSIONS_PER_SUBJECT": tp.ClassVar[dict],
                    "SESSIONS_WITH_14_RUNS": tp.ClassVar[dict],
                },
            },
        )


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
