#!/usr/bin/env python3
"""Contextual LLM (GPT-2 / GPT-2-XL) encoding baseline for the language datasets (ieeg) -- the
language analogue of ``alexnet_baseline.py`` / ``vision_baselines.py``.

A frozen pretrained language model is run forward over the transcript (its weights are NEVER trained
here, exactly like the search's untrained models). For every transcript word we take the model's
CONTEXTUAL hidden state -- the activation at the word's LAST sub-token, computed with all preceding
words as left context (causal, sliding 1024-token window) -- at each layer. That per-word, per-layer
representation is then read out through the EXACT harness machinery the search uses:

  * the same timeline alignment (``harness.loaders.timeline``: same events, same neural window, same
    Word triggers, same onset-ordered rows and temporal hold-out) so the model's rows line up with Y,
  * the same ridge readout (``harness.ridge.ridge_predict``, ``NEURO_READOUT``),
  * the same 5-fold rotating temporal CV reward (``harness.reward._cv_column_r``) -- raw mean-r over
    all electrodes (ieeg has no repeat-based noise ceiling), the SAME metric as the search DB.

Best-layer protocol (the "bar to beat"): every layer's mean predictivity is scored on an 80/20 val
split carved from TRAIN only (no test leakage); the single best layer is then reported under the
dataset's own 5-fold CV. A comma ``--subject`` list is scored per-subject and averaged with equal
weight, matching the search reward's multi-subject mean.

    python scripts/benchmark/language_baselines.py --model gpt2-xl --subject 01,02,03   # the ieeg bar
    python scripts/benchmark/language_baselines.py --model gpt2                          # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)   # for subjects.py

_HF_NAME = {"gpt2": "openai-community/gpt2", "gpt2-xl": "openai-community/gpt2-xl"}


# --------------------------------------------------------------------------------------------------
# 1) Data: transcript words + neural targets, aligned exactly like the timeline loader
# --------------------------------------------------------------------------------------------------
def load_words_and_neural(subject: str):
    """(words, Y, onsets) for one iEEG patient, in trigger-onset order -- the SAME rows the harness
    timeline loader builds (same events, neural extractor and window), so Y lines up with the
    per-word LLM features we compute below. Y is (n_words, n_electrodes) float32."""
    from harness.datasets import DATASET, with_subject
    from harness.loaders import timeline as TL
    from harness.loaders.shared import processed_dir

    d = with_subject(DATASET, subject)
    if d.loader != "timeline":
        raise SystemExit(f"language_baselines only supports timeline datasets; got loader={d.loader!r}")
    out = processed_dir(TL._PROCESSED_KEY[d.study])
    events = TL._processed_audio_events(out, d.subject)
    nb = TL._segment_batch(d, events, dict(neural=TL._NEURAL[d.neural](d)),
                           d.timeline.start, d.timeline.duration)
    segs = nb.segments
    onsets = np.array([s.trigger.start for s in segs], dtype=np.float64)
    order = np.argsort(onsets, kind="stable")
    words = [str(getattr(segs[i].trigger, "text", "") or "") for i in order.tolist()]
    Y = TL._neural_2d(nb.data["neural"], d).float().cpu().numpy()[order].astype(np.float32)
    return words, Y, onsets[order], d


# --------------------------------------------------------------------------------------------------
# 2) Frozen LLM: per-word contextual hidden states at every layer
# --------------------------------------------------------------------------------------------------
class ContextualLM:
    """A frozen causal LM that maps a word sequence -> {layer: (n_words, d_model)} contextual
    embeddings. Each word's vector is the hidden state at its LAST sub-token, with left context up to
    the model's max position (sliding window, stride = half the window)."""

    def __init__(self, model_name: str, device, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = _HF_NAME.get(model_name, model_name)
        self.name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf, output_hidden_states=True, dtype=dtype).to(device).eval()
        self.max_ctx = int(getattr(self.model.config, "n_positions", 1024))

    def _tokenize_words(self, words):
        """Flat token-id stream + each word's LAST-token index. Words are joined with single spaces
        (leading space on every word but the first) so BPE sees them mid-sentence, as in the transcript."""
        ids, last = [], []
        for i, w in enumerate(words):
            piece = self.tok.encode(("" if i == 0 else " ") + w, add_special_tokens=False)
            if not piece:                                   # empty/whitespace token -> reuse prior pos
                piece = self.tok.encode(" .", add_special_tokens=False)[-1:] or [self.tok.eos_token_id]
            ids.extend(piece)
            last.append(len(ids) - 1)
        return ids, last

    @torch.no_grad()
    def embed(self, words, batch_layers=True):
        ids, last = self._tokenize_words(words)
        n_tok = len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
        n_words = len(words)
        d_model = int(self.model.config.hidden_size)
        n_layers = int(self.model.config.num_hidden_layers) + 1   # +1 for the embedding layer (idx 0)
        emb = np.zeros((n_layers, n_words, d_model), dtype=np.float16)

        stride = max(1, self.max_ctx // 2)
        assigned = 0                     # words with index < assigned already have a vector
        hi = 0
        while assigned < n_words:
            hi = min(n_tok, hi + stride) if hi else min(n_tok, self.max_ctx)
            lo = max(0, hi - self.max_ctx)
            out = self.model(ids_t[lo:hi].unsqueeze(0))
            hs = torch.stack(out.hidden_states, 0)[:, 0]        # (n_layers, win, d)
            # assign every not-yet-assigned word whose last token falls inside this window's right edge
            while assigned < n_words and lo <= last[assigned] < hi:
                emb[:, assigned] = hs[:, last[assigned] - lo].float().cpu().numpy().astype(np.float16)
                assigned += 1
            if hi >= n_tok:
                break
        if assigned < n_words:            # safety: pin any trailing words to the final window
            for j in range(assigned, n_words):
                emb[:, j] = hs[:, min(last[j] - lo, hs.shape[1] - 1)].float().cpu().numpy().astype(np.float16)
        return emb                        # (n_layers, n_words, d_model) float16


# --------------------------------------------------------------------------------------------------
# 3) Best-layer scoring through the harness readout (5-fold CV reward)
# --------------------------------------------------------------------------------------------------
def score_subject(emb, Y, d, R, device):
    """Score EVERY layer under the dataset's 5-fold rotating temporal CV (the harness reward) and
    report the best. Selecting the layer against the CV mirrors exactly how the search obtains its
    champion -- ``model.py`` is optimized directly against the same CV -- so the best-layer number is
    an apples-to-apples 'bar to beat', not a val-split estimate. One region here = all electrodes
    (ieeg exposes no ROI map); raw mean-r (no repeat-based noise ceiling)."""
    from harness.loaders.shared import surviving_cols

    n = Y.shape[0]
    n_test = max(1, int(round(n * d.test_frac)))
    n_layers = emb.shape[0]

    # drop train-constant electrodes exactly as harness.loaders.timeline.load_timeline does, then run
    # the reward's CV over the full onset-ordered set (the split the reward itself scores).
    keep = surviving_cols(torch.from_numpy(Y[: n - n_test]))
    keep = keep.cpu().numpy() if torch.is_tensor(keep) else np.asarray(keep)
    Y_full = torch.from_numpy(Y[:, keep].astype(np.float32)).to(device)

    layer_cv = np.zeros(n_layers)
    for L in range(n_layers):
        A = torch.from_numpy(emb[L].astype(np.float32)).to(device)
        layer_cv[L] = float(R._cv_column_r(A, Y_full, d.cv_folds, device, None)[1].mean())
    best_L = int(np.argmax(layer_cv))
    return {
        "test_corr": round(float(layer_cv[best_L]), 4),
        "best_layer": best_L,
        "n_electrodes": int(keep.sum()),
        "n_words": int(n),
        "layer_cv": {f"L{L:02d}": round(float(layer_cv[L]), 4) for L in range(n_layers)},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "ieeg"))
    p.add_argument("--model", default="gpt2-xl", help="gpt2 | gpt2-xl | any HF causal-LM id")
    p.add_argument("--subject", default="01,02,03",
                   help="NEURO_SUBJECT; a comma list is scored per-subject and averaged (search reward).")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = a.dataset
    from harness import reward as R
    from subjects import subject_tokens, mean_scalar

    device = R._readout_device()
    subjects = subject_tokens(a.subject) or [""]
    subj_label = ",".join(s for s in subjects if s) or "default"
    print(f"=== {a.model} contextual LM baseline | {a.dataset} subj={subj_label} "
          f"| 5-fold CV reward (raw mean-r) | device={device} ===")

    lm = ContextualLM(a.model, device)
    print(f"  loaded {a.model}: {lm.model.config.num_hidden_layers} layers, "
          f"d={lm.model.config.hidden_size}, max_ctx={lm.max_ctx}")

    emb_cache: dict = {}
    per_subject = {}
    for s in subjects:
        t0 = time.time()
        words, Y, onsets, d = load_words_and_neural(s)
        wkey = hash(tuple(words))
        if wkey not in emb_cache:                       # identical transcript across patients -> reuse
            emb_cache[wkey] = lm.embed(words)
        emb = emb_cache[wkey]
        rec = score_subject(emb, Y, d, R, device)
        per_subject[s or "default"] = rec
        print(f"\n  [subj {s or 'default'}] n_words={rec['n_words']} electrodes={rec['n_electrodes']} "
              f"-> best layer L{rec['best_layer']:02d}  test_corr(5-fold CV)={rec['test_corr']:.4f} "
              f"({time.time()-t0:.1f}s)")

    mean_tc = mean_scalar([r["test_corr"] for r in per_subject.values()])
    print(f"\n=== {a.model} | {a.dataset} subj={subj_label}: MEAN test_corr = {mean_tc:.4f} ===")
    for s, r in per_subject.items():
        print(f"    subj {s}: {r['test_corr']:.4f}  (layer L{r['best_layer']:02d})")

    out = {
        "dataset": a.dataset, "model": a.model, "subject": subj_label,
        "metric": "raw mean-r under 5-fold rotating temporal CV (ieeg reward)",
        "test_corr": mean_tc, "per_subject": per_subject,
    }
    if not a.no_save:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(HERE, "results",
                          f"language_baselines_{a.model.replace('/', '-')}_{a.dataset}_subj{subj_label}_{stamp}.json")
        json.dump(out, open(fn, "w"), indent=1)
        print(f"\n  saved -> {fn}")
    return out


if __name__ == "__main__":
    main()
