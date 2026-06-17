"""``NsdSubj01/02/03``: the Natural Scenes Dataset restricted to one subject each.

Thin subclasses of NeuralFetch's ``Allen2022Massive`` (NSD) that keep a single
subject and all of its sessions. Importing this module registers each subclass in
``neuralset.Study.catalog()`` (e.g. ``NsdSubj01``) so it can be loaded with
``ns.Study(name="NsdSubj01", ...)``; the parent's nsdgeneral / visual-cortex ROI is
inherited unchanged.

Subjects 2 and 3 exist for the cross-subject generalization study: the same
shared-1000 images are shown to every NSD subject, so a model found on subject 1 can
be re-fit and scored on them unchanged.

The parent's StudyInfo counts all 8 subjects (3408 timelines); restricting to one
yields ``sessions x 12 runs`` (the 14-run sessions still yield 12 runs, just shifted
to runs 2-13). NeuralSet validates the discovered timeline count against
``num_timelines``, so it must reflect the subset.
"""
from __future__ import annotations

import typing as tp

from neuralfetch.studies.allen2022massive import Allen2022Massive

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
