"""``Li2022PetitSubj``: the Le Petit Prince fMRI corpus restricted to one subject.

A thin subclass of NeuralFetch's ``Li2022Petit`` (OpenNeuro ds003643) that keeps
only ONE English subject (``SUBJECT``, default EN057) and all 9 of its story
sections. Importing this module registers the subclass in
``neuralset.Study.catalog()`` under the name ``Li2022PetitSubj`` so it can be loaded
with ``ns.Study(name="Li2022PetitSubj", ...)``.

For the cross-subject generalization study, importing this module ALSO registers one
thin sibling per English subject -- ``Li2022PetitSubj<ID>`` (e.g. ``Li2022PetitSubjEN058``)
for every id in ``EN_SUBJECTS`` -- each only overriding ``SUBJECT``. Every subject
heard the SAME story, so a model found on one subject can be re-fit and scored on any
other (the ridge readout is re-fit per subject -- that re-fit IS the eval), exactly as
NSD's ``NsdSubj01/02/03`` siblings enable. See ``scripts/generalization_text.py``.

The parent yields each fMRI run as a path to a full-brain volume (73x90x74 ~ 486k
voxels). For a single subject's 9 sections that is ~15k word segments x ~1M
target values => a ~60 GB target matrix that the ridge readout cannot hold. So,
exactly as ``Allen2022Massive`` (NSD) does for its ROI, we apply the spatial
reduction AT THE STUDY LEVEL: the Fmri event carries a ``SpecialLoader`` that
loads the volume, 2x-downsamples it onto a fixed grid and keeps only MNI brain
voxels, yielding pre-reduced inline data (~29k voxels) so the full volume is
never materialised. All run->file mapping and word/text parsing is reused from
the parent unchanged.
"""
from __future__ import annotations

import typing as tp

import numpy as np
import nibabel
from neuralfetch.studies.li2022petit import Li2022Petit
from neuralset.events import study

# Spatial coarsening factor: keeps whole-brain coverage while shrinking the
# target matrix ~8x, so the ridge readout (which upcasts to float64) stays small.
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
        _GRID.update(affine=affine, shape=shape, mask=mask, n=int(mask.sum()))
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
            assert data.shape[0] == g["n"], (data.shape, g["n"])
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


# # # # # Per-subject siblings (cross-subject generalization) # # # # #

# The English arm of ds003643 has 49 subjects: EN057-EN115 with these gaps.
_EN_MISSING: frozenset[int] = frozenset({60, 66, 71, 80, 85, 90, 102, 107, 111, 112})
EN_SUBJECTS: tuple[str, ...] = tuple(
    f"EN{n:03d}" for n in range(57, 116) if n not in _EN_MISSING
)

# Generate one thin ``Li2022PetitSubj<ID>`` subclass per English subject, each only
# overriding ``SUBJECT``. Creating the class auto-registers it in the NeuralSet catalog
# (via ``Study.__init_subclass__``), so it gets its own cache key + results file and can
# be loaded with ``ns.Study(name="Li2022PetitSubj<ID>", path=...)`` -- mirroring
# NsdSubj01/02/03. Pydantic requires the overridden ClassVar to be annotated, hence the
# explicit ``__annotations__``.
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
