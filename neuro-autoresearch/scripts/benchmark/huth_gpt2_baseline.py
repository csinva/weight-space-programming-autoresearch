#!/usr/bin/env python3
"""GPT-2 XL encoding baseline for the Huth dataset -- validates that our ds003020 processing
reproduces the reference (circuit-building-autoresearch) GPT-2 XL baseline in report.html.

Runs the reference protocol EXACTLY (src/baseline.py + src/features.py + src/eval.py) but on OUR
ds003020-derived data and through the harness ridge: each word's 10-gram is embedded by GPT-2 XL
(layer 24, final-token hidden state), Lanczos-downsampled onto the TR grid (word/TR midpoints),
z-scored per story, FIR-delayed (4), then ridge-fit to the TR-rate BOLD over all cortical voxels.
num_train train stories + the held-out test stories; edge_trim=30; subject UTS03. Prints the mean
test correlation -- compare to the report's trimmed GPT-2 XL baseline (~0.07-0.08 on UTS03).

    python scripts/benchmark/huth_gpt2_baseline.py --subject UTS03 --num-train 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.process_dataset.huth import (RAW, TR, SKIP_TRS, EDGE_TRIM, NGRAM, REF_TRAIN, REF_TEST,
                                          _word_times, _lanczos, _staged)
from harness.loaders.huth import _zscore, _make_delayed          # the SAME pipeline ops the reward uses
from harness.ridge import ridge_predict, _readout_device, _to_np


def get_ngrams(words, n=NGRAM):
    """word -> 'the word + n preceding words' (reference features.get_ngrams)."""
    return [" ".join(words[max(0, i - n): i + 1]).strip() for i in range(len(words))]


class GPT2XLEmbedder:
    """GPT-2 XL layer-24 final-token embedder (reference src/baseline.py GPT2Embedder)."""

    def __init__(self, layer=24, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.layer, self.device = layer, device
        self.tok = AutoTokenizer.from_pretrained("gpt2-xl")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(
            "gpt2-xl", output_hidden_states=True, torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            b = texts[i: i + bs]
            inp = self.tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            hs = self.model(**inp).hidden_states[self.layer]          # (batch, tokens, 1600)
            last = inp["attention_mask"].sum(1) - 1                   # final non-pad token
            out.append(hs[torch.arange(hs.shape[0]), last].float().cpu().numpy())
        return np.concatenate(out, 0)


def _story_feats_Y(subject, story, embed):
    """(z-scored TR features from GPT-2 XL 10-grams, TR BOLD) for one story -- our exact pipeline."""
    with h5py.File(os.path.join(RAW, "derivatives/preprocessed_data", subject, f"{story}.hf5")) as f:
        data = np.asarray(f["data"], dtype=np.float32)
    Y = data[EDGE_TRIM:-EDGE_TRIM]
    times, words = _word_times(os.path.join(RAW, "derivatives/TextGrids", f"{story}.TextGrid"))
    words = [w.lower() for w in words]                               # DataSequence.from_grid lowercases
    newtime = (np.arange(Y.shape[0]) + SKIP_TRS + EDGE_TRIM) * TR + TR / 2.0
    L = torch.from_numpy(_lanczos(newtime, np.asarray(times))).float()
    emb = torch.from_numpy(embed(get_ngrams(words))).float()        # (n_word, 1600)
    return _zscore(L @ emb), torch.from_numpy(Y)                     # (n_tr, 1600), (n_tr, n_vox)


def _build_split(subject, stories, embed):
    Zs, Ys = [], []
    for s in stories:
        z, y = _story_feats_Y(subject, s, embed)
        Zs.append(z)
        Ys.append(y)
        print(f"    {s}: {z.shape[0]} TRs")
    return _make_delayed(torch.cat(Zs, 0), 4), torch.cat(Ys, 0)     # delays across stories, per reference


def _mean_test_corr(pred, Y):
    P, Yt = _to_np(pred), _to_np(Y)
    a, b = P - P.mean(0), Yt - Yt.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return float(np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--num-train", type=int, default=8)
    a = ap.parse_args()

    dev = _readout_device()
    train = [s for s in REF_TRAIN if _staged(a.subject, s)][:a.num_train]
    test = [s for s in REF_TEST if _staged(a.subject, s)]
    print(f"=== GPT-2 XL baseline | {a.subject} | num_train={len(train)} num_test={len(test)} "
          f"| edge_trim={EDGE_TRIM} | layer 24 | device={dev} ===")

    cache = f"/tmp/huth_gpt2_{a.subject}_nt{len(train)}.pt"
    if os.path.exists(cache):
        Ztr, Ytr, Zte, Yte = torch.load(cache, weights_only=False)
        print(f"  loaded cached features {tuple(Ztr.shape)}")
    else:
        t0 = time.time()
        embed = GPT2XLEmbedder(device=str(dev))
        print("  train stories:")
        Ztr, Ytr = _build_split(a.subject, train, embed)
        print("  test stories:")
        Zte, Yte = _build_split(a.subject, test, embed)
        torch.save((Ztr, Ytr, Zte, Yte), cache)
        print(f"  features {tuple(Ztr.shape)} -> Y {tuple(Ytr.shape)}  (embed {time.time()-t0:.0f}s)")

    pred_tr, pred_te = ridge_predict(Ztr.to(dev), Zte.to(dev), Ytr.to(dev), device=dev)
    train_c, test_c = _mean_test_corr(pred_tr, Ytr), _mean_test_corr(pred_te, Yte)
    readout = os.environ.get("NEURO_READOUT", "fast")
    print(f"\n=== GPT-2 XL ({readout} ridge) | UTS03 num_train={len(train)}: "
          f"train_corr={train_c:.4f}  test_corr={test_c:.4f}  (reference trimmed baseline ~0.07-0.08) ===")


if __name__ == "__main__":
    main()
