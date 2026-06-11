"""``NsdSubj02``: the Natural Scenes Dataset restricted to subject 2.

A thin sibling of :class:`NsdSubj01` keeping only subject 2 (all 40 sessions). It
exists for the cross-subject generalization study: the same shared-1000 images are
shown to every NSD subject, so a model found on subject 1 can be re-fit and scored
here unchanged. Importing this module registers ``NsdSubj02`` in the catalog.
"""
from __future__ import annotations

from neuralfetch.studies.allen2022massive import Allen2022Massive


class NsdSubj02(Allen2022Massive):
    """NSD restricted to subject 2 (all 40 sessions)."""

    SESSIONS_PER_SUBJECT = {2: 40}
    SESSIONS_WITH_14_RUNS = {2: (21, 30)}

    # 40 sessions x 12 runs each (the 14-run sessions still yield 12 runs, just
    # shifted to runs 2-13) -> 480 timelines for the single subject.
    _info = Allen2022Massive._info.model_copy(
        update={"num_timelines": 480, "num_subjects": 1}
    )
