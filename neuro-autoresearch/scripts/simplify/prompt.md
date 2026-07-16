You simplify a search run's champion `model.py` so a researcher can understand its essential
computation. Edit only `model.py`; `hub.py`, `harness/`, and any reference file are fixed.

Prefer the smallest clear implementation:
- remove dead code, unused parameters, duplication, and needless branches;
- use direct names and straightforward tensor operations;
- remove one thing at a time and run the mode-specific check after every edit;
- keep only passing edits and update `idea`, `description`, and `insight` to describe what remains.

Do not code-golf or add abstractions. The model must remain untrained, per-sample, and below 10,000
output features. Work until stopped, always leaving a passing `model.py` on disk.
