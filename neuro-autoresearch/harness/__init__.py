"""Fixed neural-alignment harness; see README.md for the scientific contract.

Load each protocol's complete subject cohort, then compute RSA search feedback
or final cross-dataset ridge encoding. ``reward`` builds result rows; ``run``
provides manual evaluation. Only search RSA enters the search database.
"""
from harness.loaders import load_split  # noqa: F401  (the one load entry point, re-exported)
