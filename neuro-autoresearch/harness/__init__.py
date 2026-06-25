"""Fixed neural-encoding eval harness (see CLAUDE.md for the codebase map).

Pick a dataset (``datasets``) -> load (X, Y) (``loaders``) -> score with a ridge readout
(``ridge``) into a result row (``reward``); ``run`` is the manual single-eval CLI and the
search loop records the row into a run's ``search.db`` (see ``hub.py``). Nothing here is
trained or edited by the search agent; that is what keeps the reward honest.
"""
from harness.loaders import load_split  # noqa: F401  (the one load entry point, re-exported)
