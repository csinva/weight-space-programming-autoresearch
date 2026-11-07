# New held-out test set for the LEAN autoresearch rerun (fixed 2026-11-06, before any scoring)

**Why.** Every autoresearch trial was selected on the same 23-chunk test set (`450a36e4_0190`–`0212`,
one podcast segment). After ~600 selections the scores on that set can be optimistic. This rerun
scores the same trials, in the same order, on chunks that were never used during the search.

**Training: unchanged.** The harness's 500 training chunks (`train_frozen.json`, sha256 prefix
`7d317eca03e19253`), the same ridge, delays and alphas. Every run asserts that its training list
equals this file. The new test chunks come from session `450a36e4`, which is excluded from training
in every run, so the fits are identical to the originals.

**Test sets (rule fixed in advance, no tuning).**
- Candidates (`test_candidates.json`, 493 chunks): every `450a36e4` chunk in `master_list.json`
  with high-gamma and audio, other than the 23 old test chunks.
- **Primary** (`test_primary.json`, 317 chunks): candidates in `master_list_speech150.json`
  (≥150 clean words; the Feb 2026 list, matching the old set's speech-rich selection), excluding
  indices 180–222 (±10 chunks around the old block, so no continuation of the old podcast is reused).
- Secondary: all 493 candidates (per-chunk r is saved for every candidate).
- Hashes: `frozen_hashes.txt`. Script: `freeze_test_set.py`.

**Audio.** Aligned waveform segments for the candidates were built with the harness's own
`build_segment` into `wave_aligned/`, NOT the harness `WAVE_DIR`: the harness picks training chunks by
shuffling every chunk that has a segment there, so adding files would change the training set. The
extractor first rebuilt an existing segment bit-identically.

**What is rerun.**
1. All 601 logged main-line trials of Figure 1 (596 with recoverable code), in Figure 1 order
   (`trials.json`). Code: the Nov 5 rescore-grid snapshot when it already reproduced its logged score
   (218); otherwise the exact file reconstructed from the agent logs (`replay_snapshots.py`, 378):
   every Edit result in the logs carries the full pre-edit file, so the evaluated file can be rebuilt
   and checked against later file states. Each trial must reproduce its logged score on the old test
   (±0.00015, with the metric it was logged with) to be used.
2. Variance partitioning of LEAN-v20 (`../varpart-v20-newtest/`): the same 135 fits from the same
   training statistics, each asserted to reproduce its stored old-test r, scored on the new test.
3. The Whisper-only reference (`whisper_only_encoder.py`, must reproduce 0.0817 all-166 / 0.0933 grid).

**Reported metric.** Same as before: per-chunk per-electrode Pearson r, mean over chunks, then over
the 125 grid electrodes (Figure 1) or as r·|r| variance shares (Figure 2), primary test set.
