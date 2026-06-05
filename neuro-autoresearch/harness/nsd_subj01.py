"""``NsdSubj01``: the Natural Scenes Dataset restricted to subject 1.

A thin subclass of NeuralFetch's ``Allen2022Massive`` (NSD) that keeps only
subject 1 and all 40 of its sessions. Importing this module registers the
subclass in ``neuralset.Study.catalog()`` under the name ``NsdSubj01`` so it can
be loaded with ``ns.Study(name="NsdSubj01", ...)``. The default nsdgeneral /
visual-cortex ROI of the parent class is inherited unchanged.
"""
from __future__ import annotations

from neuralfetch.studies.allen2022massive import Allen2022Massive


class NsdSubj01(Allen2022Massive):
    """NSD restricted to subject 1 (all 40 sessions)."""

    SESSIONS_PER_SUBJECT = {1: 40}
    SESSIONS_WITH_14_RUNS = {1: (21, 38)}

    # The parent's StudyInfo counts all 8 subjects (3408 timelines); restricting
    # to subject 1 yields 480 (40 sessions x 12 runs). NeuralSet validates the
    # discovered timeline count against this, so it must reflect the subset.
    _info = Allen2022Massive._info.model_copy(
        update={"num_timelines": 480, "num_subjects": 1}
    )
