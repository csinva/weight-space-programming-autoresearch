"""``NsdSubj03``: the Natural Scenes Dataset restricted to subject 3.

A thin sibling of :class:`NsdSubj01` keeping only subject 3 (32 sessions). It exists
for the cross-subject generalization study: the same shared-1000 images are shown to
every NSD subject, so a model found on subject 1 can be re-fit and scored here
unchanged. Importing this module registers ``NsdSubj03`` in the catalog.
"""
from __future__ import annotations

from neuralfetch.studies.allen2022massive import Allen2022Massive


class NsdSubj03(Allen2022Massive):
    """NSD restricted to subject 3 (32 sessions)."""

    SESSIONS_PER_SUBJECT = {3: 32}
    SESSIONS_WITH_14_RUNS = {3: (21, 30)}

    # 32 sessions x 12 runs each (the 14-run sessions still yield 12 runs, just
    # shifted to runs 2-13) -> 384 timelines for the single subject.
    _info = Allen2022Massive._info.model_copy(
        update={"num_timelines": 384, "num_subjects": 1}
    )
