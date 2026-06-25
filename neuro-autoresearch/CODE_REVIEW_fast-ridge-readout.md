# Code Review: `fast-ridge-readout`


## Standalone Processed Loaders

- [ ] **1. [High] NSD off-by-one: raw vs processed paths disagree on the 1-based offset** — `harness/loaders/standalone/nsd.py:64`
Raw `_build_X` derives the offset heuristically (`off=1` only if `max nsd_id >= brick size`), but the new processed `store_stimuli` *unconditionally* does `chunk_ids - 1`. For any subject whose largest `nsd_id < 73000` (the common case — no subject sees all 73k images), the raw path gets `off=0` and **pairs every stimulus with the *next* image's betas**. The two paths silently disagree and the readout fits misaligned X→Y pairs, depressing `test_corr`.

- [ ] **2. [High] NSD half-staged tree crashes instead of falling back** — `harness/loaders/standalone/nsd.py:110`
The processed branch is gated on the neural `.pt` only; `images.h5` is read separately and not in the guard. A tree with the neural `.pt` but missing/incomplete H5 passes the gate then fails deep in image loading with an opaque h5py error, rather than degrading to the raw path.

- [ ] **3. [High] TVSD partial-tree X/Y length mismatch** — `harness/loaders/standalone/tvsd.py:142` (also :121)
Processed branch triggers whenever `data/tvsd/` exists. X row count comes from `len(os.listdir(sdir))` (copied JPEGs) while Y comes from the full `.pt`. A smoke/sample prep run (`--n 5`) yields X=5 rows against Y=22248 — crash on shape mismatch, or worse, silent mis-pairing if counts coincide. No validation.

- [ ] **4. [High] NNN processed path drops the reliability fast-fail** — `harness/loaders/standalone/nnn.py:101`
The raw path raises `RuntimeError("NNN: no units passed reliability>...")` when no units survive; `_load_nnn_processed` has no equivalent and **silently returns a zero-column Y** → NaN reward or opaque downstream crash, masking the real cause (mis-set `NNN_MIN_RELIABILITY`).

- [ ] **5. [Low] Eager `scipy.io` import defeats processed-tree purpose** — `harness/loaders/standalone/nnn.py:91`
`load_nnn` imports `glob/re/scipy.io` unconditionally at the top, before the `if os.path.isdir(proc)` branch. On a machine with the processed tree but **no scipy** (the explicit shipping goal), `NEURO_DATASET=nnn` raises ImportError at entry though every needed byte is present. Move these into `_load_nnn_raw`.

- [ ] **6. [Low] NNN processed path never caches X** — `harness/loaders/standalone/nnn.py:76`
`_load_nnn_processed` re-decodes/resizes all 1000 NSD images via PIL on every load; the raw path caches `nnn_X_im{imsize}.pt`. The "Cached by (reliability, imsize)" docstring is false for the *preferred* branch — every benchmark worker / evaluator restart pays the full decode again.

---

## ROI Selection And Advisory Masks

- [ ] **7. [Medium] `reward_roi` ignored on all standalone datasets** — `harness/roi.py:47` *(PLAUSIBLE)*
`_study_for` returns `None` for any loader != `neuralset`, so for nsd/tvsd/nnn the reward fits **all** columns of Y regardless of `dataset.reward_roi`. Narrowing the ROI on a standalone preset silently has no effect — `test_corr` reflects a different region than configured. (Worth confirming against intended design — the standalone presets may bake the region into Y at load time, but the knob being silently inert is still a footgun.)

- [ ] **8. [Medium] NSD ROI breakdown keyed by content-hash collides** — `harness/loaders/standalone/nsd.py:91`
`_nsd_roi_masks` maps voxels→Y columns via `hash(Y[:,j].tobytes())` as an identity proxy. Two byte-identical beta vectors (duplicated/zeroed voxels) collide — one ROI steals/loses the other's column. Scalar reward unaffected; the **per-ROI advisory the proposer reads is wrong**.

- [ ] **9. [Medium] NNN per-area breakdown silently omits "Unknown" units** — `harness/loaders/standalone/nnn.py:102`
`roi_masks = {a: areas == a for a in ("EVC","IT")}`, but `_nnn_unit_area` returns `'Unknown'` (raw path: `'?'`) for units outside annotated ranges. These units *are* scored (they pass reliability) but match neither mask — no "other" bucket. The advisory breakdown under-counts coverage.

---

## NeuralSet Build Cache

- [ ] **10. [High] Stale build cache shadows the processed tree** — `harness/loaders/neuralset/__init__.py:276` *(verdict: PLAUSIBLE)*
`load_neuralset` checks the on-disk build cache before the processed-tree branch, and `_build_cache_key` hashes only `Dataset` fields — nothing identifying raw-vs-processed source, both written to the **same cache path**. A stale raw-derived cache permanently shadows a freshly-generated `data/<ds>/` tree; only escape is `NEURO_NO_CACHE` or manual clear.

---

## Manual Run And Benchmark Docs

- [x] **11. [Medium] Manual-path results CSV deleted with no replacement** — `harness/reward.py:64` (also `README.md:102`)
`results.py` was deleted and `run.py` dropped `save_results`, so `python -m harness.run` no longer appends to `results/<study>.csv` (iteration counter, best_so_far, frozen `corr_` columns). README:101–102 still documents the CSV write. Any tooling/habit reading that file after a manual eval breaks or reads stale pre-branch data. DB recording now lives *only* in the unattended search loop.

- [x] **12. [Low] Benchmark reuses a hand-copied baseline** — `scripts/benchmark/benchmark.py:45` (also :82, :109)
`_reference_proj` re-implements `model.py`'s seed-0 baseline (same scaling, same passthrough) as a fallback. Can silently diverge from `model.py`; when it does, the benchmark profiles a *different* representation than the harness. (Partly justified — the fallback exists for a broken/mid-edit `model.py` — but the constants should be kept in sync deliberately.)

- [x] **13. [Low] Benchmark docstring names removed symbols** — `scripts/benchmark/benchmark.py:7` & :64 *(reported by 8 finders)*
Docstring/comments reference `harness.reward._ridge_predict` and a `harness.fast_ridge` module — neither exists. The real entry point is `harness.ridge.ridge_predict`. Pure doc drift; code runs fine via re-exports. Trivial fix.


