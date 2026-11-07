# LEAN summary figures

Regenerate: `../../.venv/bin/python plot_progress_grid.py` (Figure 1) and
`../../.venv/bin/python make_summary_figures.py` (Figure 2). Plotted numbers: `fig1_data.csv`,
`../varpart-v21/results_*.csv`.

All scores: patient1, fixed 23-chunk speech-rich test set (session 450a36e4, held out of training),
500 training chunks, ridge on 4 FIR delays, canonical per-chunk Pearson r averaged over electrodes.
This test set runs optimistic by ~0.015 r versus random splits; comparisons between models hold.

**Figure 1 — progress of the hand-built encoder (LEAN), 125 surface-grid electrodes.**
Every main-line autoresearch trial in the order it was tried (x) against its test r on the grid
electrodes (y): faint dots are attempts, the black step line is the running best, shaded bands are
the successive search phases, and callouts mark the main feature additions and promoted versions
v16–v20. Dashed line: Whisper encoder features
with the same ridge and delays. Jun 3–5 trials were originally scored on all 166 electrodes and were
re-run from their saved code and rescored on the grid (`runs-wave/rescore-grid/`), keeping only those
that reproduced their logged score within ±0.00015; Jun 6+ trials were scored on the grid at the time.
525 of 601 main-line trials are plotted (the rest had no saved code, or a snapshot overwritten or
reverted after scoring). The clean-slate side workers (A–I) are not on the main line and not shown.

**Figure 2 — LEAN-v21 variance partitioning, grid electrodes.**
Variance explained per electrode = r·|r|, averaged over electrodes; full model 1.89%. "Alone" = a ridge
on that block only; "unique" = full model minus the full model without that block; both as % of the full
model. 95% CIs from a paired bootstrap over the 23 test chunks. (a) Seven feature families (raw dims in
parentheses). (b) Sum of unique shares at three partition granularities; the remainder is shared.
(c) Sixteen sections; open circles are LEAN-v20. (d) The 51 groups: variance alone versus size; blue =
unique variance with a CI above zero.

---

# New held-out test set (rerun Nov 6–7, 2026)

Regenerate: `../../.venv/bin/python plot_progress_newtest.py` and `plot_breakdown_newtest.py`.
Data: `fig1_newtest_data.csv`, `../varpart-v20-newtest/results_*.csv`. Protocol (test set fixed
before scoring): `../newtest/PROTOCOL.md`.

Same 500 training chunks and fits as above; new test set = 317 speech-rich chunks of the held-out
session 450a36e4 that the search never used (≥150 clean words, more than 10 chunks away from the old
podcast block 0190–0212). Metric unchanged: per-chunk Pearson r, averaged over chunks, then over the
125 grid electrodes.

**Figure 1 (new test) — `fig1_progression_newtest`.** Every main-line trial rerun in order and scored
on the new test (dots). Black step line: the model the search was keeping at each point (the best so
far by old-test score), scored on the new test. It is not monotonic: a model that won on the old test
can score lower on the new one. Gray step line: that same model's old-test score (Figure 1 above).
Dashed lines: Whisper encoder features, same ridge and delays, new test (dark) and old test (gray).
Code for 378 trials was rebuilt by replaying the agents' file edits from their logs. Every plotted
trial reproduced its logged old-test score within ±0.00015. 593 of 601 trials are plotted: 3 missed
that tolerance (by 0.00015–0.00027, numerical noise; the code was confirmed by the replay), and 5
had no recoverable code. Across the plotted trials, old- and new-test scores correlate at r = 0.981.

**Figure 2 (new test) — `fig2_breakdown_newtest`.** LEAN-v20 variance partitioning, the same 135
ridge fits as Figure 2's v20 comparison (each refit reproduced its stored old-test result to 1e-13),
scored on the new test; 95% CIs from a paired bootstrap over the 317 test chunks. Open circles: the
same fits on the old 23-chunk test. Full model: 1.48% of variance on the new test vs. 1.89% on the old.
