#!/usr/bin/env python3
"""Contextual LLM (GPT-2 / GPT-2-XL) encoding baseline for the language datasets (iEEG and Huth) -- the
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

Best-layer protocol: every layer is selected on a train-only temporal validation tail, then the
winner is scored with the dataset's own final protocol. A comma ``--subject`` list is scored
per-subject and averaged with equal weight, matching the search reward's multi-subject mean.

    python experiments/evaluation/language.py --model gpt2-xl --subject 01,02,03
    python experiments/evaluation/language.py --model gpt2
"""
from __future__ import annotations

import argparse
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

    def __init__(self, model_name: str, device, pretrained=True, seed=0, dtype=torch.float32):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        hf = _HF_NAME.get(model_name, model_name)
        self.name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(hf)
        torch.manual_seed(seed)
        if pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf, output_hidden_states=True, dtype=dtype)
        else:
            config = AutoConfig.from_pretrained(hf, output_hidden_states=True)
            self.model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
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


def _decode_words(X):
    """Recover each row's current word from the harness char N-gram grid (used by baked Huth data)."""
    from harness.loaders.timeline import _CHAR_STOI
    itos = {v: k for k, v in _CHAR_STOI.items()}
    rows = X.cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    text = ["".join(itos.get(int(i), "?") for i in row if i).rstrip() for row in rows]
    return [s.rsplit(" ", 1)[-1] or "." for s in text]


def _embed_stories(lm, words, counts):
    """Embed stories independently so context never crosses a story boundary."""
    out, start = [], 0
    for count in counts:
        out.append(lm.embed(words[start:start + count]))
        start += count
    return np.concatenate(out, axis=1)


# --------------------------------------------------------------------------------------------------
# 3) Best-layer scoring through the harness readout (5-fold CV reward)
# --------------------------------------------------------------------------------------------------
def score_subject(emb, Y, d, R, device):
    """Select the layer on a train-only temporal validation tail, then report the harness CV score."""
    from harness.loaders.shared import surviving_cols
    from harness.ridge import ridge_predict

    n = Y.shape[0]
    n_test = max(1, int(round(n * d.test_frac)))
    n_layers = emb.shape[0]

    n_train = n - n_test
    keep = surviving_cols(torch.from_numpy(Y[:n_train]))
    keep = keep.cpu().numpy() if torch.is_tensor(keep) else np.asarray(keep)
    Y_full = torch.from_numpy(Y[:, keep].astype(np.float32)).to(device)

    cut = max(1, round(0.8 * n_train))
    layer_val, layer_cv = np.zeros(n_layers), np.zeros(n_layers)
    for L in range(n_layers):
        A = torch.from_numpy(emb[L].astype(np.float32)).to(device)
        pred = ridge_predict(A[:cut], A[cut:n_train], Y_full[:cut], device=device)[1]
        layer_val[L] = float(R._corr_per_column(pred, R._to_np(Y_full[cut:n_train])).mean())
        layer_cv[L] = float(R._cv_column_r(A, Y_full, d.cv_folds, device, None)[1].mean())
    best_L = int(np.argmax(layer_val))
    return {
        "test_corr": round(float(layer_cv[best_L]), 4),
        "best_layer": best_L,
        "n_electrodes": int(keep.sum()),
        "n_words": int(n),
        "layer_val": {f"L{L:02d}": round(float(layer_val[L]), 4) for L in range(n_layers)},
        "layer_cv": {f"L{L:02d}": round(float(layer_cv[L]), 4) for L in range(n_layers)},
    }


def score_huth_subject(lm, subject, R, device):
    """Frozen-LM best-layer baseline through Huth's exact Lanczos/FIR feature pipeline."""
    from harness.datasets import DATASET, with_subject
    from harness.loaders import apply_feature_pipeline, load_split
    from harness.ridge import ridge_predict

    d = with_subject(DATASET, subject) if subject else DATASET
    Xtr, Xte, Ytr, Yte, _, _ = load_split(d)
    from harness.loaders.huth import _PIPELINE
    pipe = _PIPELINE[d.study]
    Etr = _embed_stories(lm, _decode_words(Xtr), pipe["train_wc"])
    Ete = _embed_stories(lm, _decode_words(Xte), pipe["test_wc"])
    Ytr, Yte = Ytr.to(device), Yte.to(device)
    cut = max(1, round(0.8 * len(Ytr)))
    val, test = [], []
    for layer in range(Etr.shape[0]):
        Atr = apply_feature_pipeline(d, torch.from_numpy(Etr[layer].astype(np.float32)).to(device), "train")
        Ate = apply_feature_pipeline(d, torch.from_numpy(Ete[layer].astype(np.float32)).to(device), "test")
        pred_val = ridge_predict(Atr[:cut], Atr[cut:], Ytr[:cut], device=device)[1]
        val.append(float(R._corr_per_column(pred_val, R._to_np(Ytr[cut:])).mean()))
        pred_test = ridge_predict(Atr, Ate, Ytr, device=device)[1]
        test.append(float(R._corr_per_column(pred_test, R._to_np(Yte)).mean()))
    best = int(np.argmax(val))
    return {"test_corr": round(test[best], 4), "best_layer": best,
            "n_voxels": int(Ytr.shape[1]), "n_words": int(len(Xtr) + len(Xte)),
            "layer_val": {f"L{i:02d}": round(v, 4) for i, v in enumerate(val)},
            "layer_test": {f"L{i:02d}": round(v, 4) for i, v in enumerate(test)}}


def score_lang_subject(lm, dataset, subject, R, device):
    """One subject's best-layer test_corr: huth via the Lanczos/FIR pipeline, else the char/word CV."""
    if dataset == "huth":
        return score_huth_subject(lm, subject, R, device)
    words, Y, _, d = load_words_and_neural(subject)
    return score_subject(lm.embed(words), Y, d, R, device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "ieeg"))
    p.add_argument("--model", default="gpt2-xl", help="gpt2 | gpt2-xl | any HF causal-LM id")
    p.add_argument("--subject", default="01,02,03",
                   help="NEURO_SUBJECT; a comma list is scored per-subject and averaged (search reward).")
    p.add_argument("--weights", choices=("pretrained", "untrained", "both"), default="both")
    p.add_argument("--seed", type=int, default=0, help="random initialization seed")
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = a.dataset
    from harness import reward as R
    from common import subject_tokens, mean_scalar

    device = R._readout_device()
    subjects = subject_tokens(a.subject) or [""]
    subj_label = ",".join(s for s in subjects if s) or "default"
    variants = ("pretrained", "untrained") if a.weights == "both" else (a.weights,)
    records = {}
    for weights in variants:
        print(f"=== {a.model}-{weights} | {a.dataset} subj={subj_label} | device={device} ===")
        lm = ContextualLM(a.model, device, weights == "pretrained", a.seed)
        emb_cache, per_subject = {}, {}
        for s in subjects:
            t0 = time.time()
            if a.dataset == "huth":
                rec = score_huth_subject(lm, s, R, device)
            else:
                words, Y, _, d = load_words_and_neural(s)
                wkey = hash(tuple(words))
                if wkey not in emb_cache:
                    emb_cache[wkey] = lm.embed(words)
                rec = score_subject(emb_cache[wkey], Y, d, R, device)
            per_subject[s or "default"] = rec
            print(f"  [{s or 'default'}] layer=L{rec['best_layer']:02d} "
                  f"test_corr={rec['test_corr']:.4f} ({time.time()-t0:.1f}s)")
        records[f"{a.model}-{weights}"] = {
            "test_corr": mean_scalar(r["test_corr"] for r in per_subject.values()),
            "weights": weights, "per_subject": per_subject,
        }
        del lm

    out = {"dataset": a.dataset, "model": a.model, "subject": subj_label,
           "seed": a.seed, "records": records}
    return out


if __name__ == "__main__":
    main()
