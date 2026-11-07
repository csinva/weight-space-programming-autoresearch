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
