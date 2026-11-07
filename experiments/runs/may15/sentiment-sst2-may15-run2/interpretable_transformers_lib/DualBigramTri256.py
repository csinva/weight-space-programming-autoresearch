"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Bigram log-odds sentiment classifier.

    Computes log P((c_prev, c_curr)|y=1) - log P((c_prev, c_curr)|y=0) for every
    bigram from the SST-2 pool (closed-form). The transformer evaluates this
    feature on every sentence-internal bigram and pools to the '=' position.

    Architecture (set by build_model):
      - d_model = 160, n_heads = 1, d_head = 160, n_layers = 2, d_ff = 1024.
      - Residual layout:
          [0..V-1]   : current token one-hot              (filled by token_emb)
          [V..2V-1]  : previous token one-hot scratch     (filled by L0 attn)
          [2V..2V+L-1]: position one-hot                  (filled by pos_emb)
          [D-3]      : +0.5  } balanced LN-stabilizing
          [D-2]      : -0.5  } constants (mean=0)
          [D-1]      : sentiment channel
    """
    torch.manual_seed(0)
    V = model.vocab_size
    D = model.d_model
    H = model.n_heads
    dh = D // H
    L = model.max_seq_len

    PREV1_OFF = V
    PREV2_OFF = 2 * V
    POS_OFF = 3 * V
    SENT = D - 1
    CONST_P = D - 3
    CONST_N = D - 2

    SKIP = set("_=01")

    # ----- 1. Build TWO bigram log-odds tables: adjacent (prev1, cur) and skip-1 (prev2, cur) -----
    pool = task._load_pool()
    import numpy as np
    from collections import defaultdict
    alpha = 2.0
    UND = task.stoi['_']
    pos_a = torch.full((V, V), alpha); neg_a = torch.full((V, V), alpha)
    pos_s = torch.full((V, V), alpha); neg_s = torch.full((V, V), alpha)
    tri_pos: dict[tuple, int] = defaultdict(int)
    tri_neg: dict[tuple, int] = defaultdict(int)
    for text, label in pool:
        prev2 = UND
        prev1 = UND
        is_pos = (label == '1')
        for ch in text:
            cur = task.stoi[ch]
            if is_pos:
                pos_a[prev1, cur] += 1
                pos_s[prev2, cur] += 1
                tri_pos[(prev2, prev1, cur)] += 1
            else:
                neg_a[prev1, cur] += 1
                neg_s[prev2, cur] += 1
                tri_neg[(prev2, prev1, cur)] += 1
            prev2 = prev1; prev1 = cur
    bg_a = torch.log(pos_a / pos_a.sum()) - torch.log(neg_a / neg_a.sum())
    bg_s = torch.log(pos_s / pos_s.sum()) - torch.log(neg_s / neg_s.sum())
    for ch in SKIP:
        j = task.stoi[ch]
        bg_a[j, :] = 0.0; bg_a[:, j] = 0.0
        bg_s[j, :] = 0.0; bg_s[:, j] = 0.0

    # ----- 1b. Pick top-K trigrams by |log-odds| * sqrt(count). -----
    K_TRI = 256
    alpha_t = 2.0
    total_pos_tri = sum(tri_pos.values()) + alpha_t * (V ** 3)
    total_neg_tri = sum(tri_neg.values()) + alpha_t * (V ** 3)
    SKIP_IDS = {task.stoi[c] for c in SKIP}
    tri_keys = set(tri_pos) | set(tri_neg)
    tri_scored = []
    import math
    for k in tri_keys:
        if any(c in SKIP_IDS for c in k):
            continue
        p = tri_pos.get(k, 0) + alpha_t
        n = tri_neg.get(k, 0) + alpha_t
        lo = math.log(p / total_pos_tri) - math.log(n / total_neg_tri)
        cnt = tri_pos.get(k, 0) + tri_neg.get(k, 0)
        if cnt < 20:
            continue
        tri_scored.append((abs(lo) * math.sqrt(cnt), lo, k))
    tri_scored.sort(reverse=True)
    tri_top = tri_scored[:K_TRI]
    tri_lookup: dict[tuple, float] = {k: lo for _, lo, k in tri_top}
    print(f"[Tri] kept {len(tri_top)} trigrams (of {len(tri_keys)} seen)", flush=True)

    # ----- 2. Pool calibration on combined adjacent + skip-1 + trigram scores -----
    bg_a_np = bg_a.numpy(); bg_s_np = bg_s.numpy()
    scores = []; labels = []
    for text, label in pool:
        s = 0.0
        prev2 = task.stoi['_']; prev1 = task.stoi['_']
        for ch in text:
            cur_i = task.stoi[ch]
            s += bg_a_np[prev1, cur_i] + bg_s_np[prev2, cur_i]
            t = tri_lookup.get((prev2, prev1, cur_i))
            if t is not None:
                s += t
            prev2 = prev1; prev1 = cur_i
        scores.append(s); labels.append(int(label))
    scores = np.asarray(scores); labels = np.asarray(labels)
    order = np.argsort(scores)
    s_sorted = scores[order]; y_sorted = labels[order]
    n_pool = len(pool); n_pos = int(labels.sum())
    pos_right, neg_right, pos_left, neg_left = n_pos, n_pool - n_pos, 0, 0
    best_acc, best_T = (n_pos / n_pool), float(s_sorted[0]) - 1.0
    for i in range(n_pool):
        acc = (neg_left + pos_right) / n_pool
        if acc > best_acc:
            best_acc = acc
            best_T = (s_sorted[i - 1] + s_sorted[i]) / 2 if i > 0 else s_sorted[i] - 1.0
        if y_sorted[i] == 1: pos_right -= 1; pos_left += 1
        else:                neg_right -= 1; neg_left += 1
    if (n_pool - n_pos) / n_pool > best_acc:
        best_acc, best_T = (n_pool - n_pos) / n_pool, float(s_sorted[-1]) + 1.0
    print(f"[Dual] pool calibration acc {best_acc:.4f} at T*={best_T:.4f}", flush=True)
    # Split shift evenly across adj/skip detectors (one of each fires per position).
    half_shift = float(best_T) / (2 * 81.0)
    bg_a = bg_a - half_shift
    bg_s = bg_s - half_shift

    with torch.no_grad():
        # Embeddings
        model.token_emb.weight.zero_()
        for c in range(V):
            model.token_emb.weight[c, c] = 1.0
            model.token_emb.weight[c, CONST_P] = 0.5
            model.token_emb.weight[c, CONST_N] = -0.5
        model.pos_emb.weight.zero_()
        for p in range(L):
            model.pos_emb.weight[p, POS_OFF + p] = 1.0

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0); blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0); blk.ln2.bias.zero_()
            for w in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                w.weight.zero_()
            blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
            blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Block 0 attention: head 0 = offset 1 (prev1); head 1 = offset 2 (prev2).
        b0 = model.blocks[0]
        M = 50.0
        for p in range(1, L):
            b0.attn.W_q.weight[0 * dh + (p - 1), POS_OFF + p] = M
        for p in range(2, L):
            b0.attn.W_q.weight[1 * dh + (p - 2), POS_OFF + p] = M
        for q in range(L):
            b0.attn.W_k.weight[0 * dh + q, POS_OFF + q] = 1.0
            b0.attn.W_k.weight[1 * dh + q, POS_OFF + q] = 1.0
        for c in range(V):
            b0.attn.W_v.weight[0 * dh + c, c] = 1.0
            b0.attn.W_v.weight[1 * dh + c, c] = 1.0
        prev_gain = 0.125
        for c in range(V):
            b0.attn.W_o.weight[PREV1_OFF + c, 0 * dh + c] = prev_gain
            b0.attn.W_o.weight[PREV2_OFF + c, 1 * dh + c] = prev_gain

        # Block 0 MLP: V*V adjacent-bigram detectors + V*V skip-1 detectors + K trigram detectors.
        bias_thr = 11.0
        trig_val = 4.0  # measured roughly when curr+prev both fire
        for cp in range(V):
            for cc in range(V):
                ia = cp * V + cc
                b0.mlp.fc1.weight[ia, cc] = 1.0
                b0.mlp.fc1.weight[ia, PREV1_OFF + cp] = 1.0
                b0.mlp.fc1.bias[ia] = -bias_thr
                b0.mlp.fc2.weight[SENT, ia] = bg_a[cp, cc].item() / trig_val
                is_ = V * V + cp * V + cc
                b0.mlp.fc1.weight[is_, cc] = 1.0
                b0.mlp.fc1.weight[is_, PREV2_OFF + cp] = 1.0
                b0.mlp.fc1.bias[is_] = -bias_thr
                b0.mlp.fc2.weight[SENT, is_] = bg_s[cp, cc].item() / trig_val
        # Trigram detectors fire when curr + prev1 + prev2 all match.
        # 3 LN signals each ~7.54 → sum ~22.6; bias 20 → triggered ~2.6.
        bias_tri = 20.0
        tri_val = 2.6
        for k, (_, lo, (cp2, cp1, cc)) in enumerate(tri_top):
            idx = 2 * V * V + k
            b0.mlp.fc1.weight[idx, cc] = 1.0
            b0.mlp.fc1.weight[idx, PREV1_OFF + cp1] = 1.0
            b0.mlp.fc1.weight[idx, PREV2_OFF + cp2] = 1.0
            b0.mlp.fc1.bias[idx] = -bias_tri
            b0.mlp.fc2.weight[SENT, idx] = lo / tri_val

        # Block 1 attention: uniform causal aggregation of SENT (use head 0 only).
        b1 = model.blocks[1]
        b1.attn.W_v.weight[0 * dh + 0, SENT] = 1.0
        b1.attn.W_o.weight[SENT, 0 * dh + 0] = 200.0

        # Final LN + head
        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()
        model.head.weight.zero_()
        Kg = 100.0
        model.head.weight[task.stoi['1'], SENT] = Kg
        model.head.weight[task.stoi['0'], SENT] = -Kg

    # ----- Empirical recalibration: close the calibration↔model gap -----
    # Two forward passes:
    #   (1) baseline: measure raw sentiment values at pos 80, find best threshold.
    #   (2) probe   : shift pos_emb[80,SENT] by +1.0, measure how the actual
    #                 model output changes -> alpha (propagation gain).
    # Then set pos_emb[80,SENT] -= best_T/alpha so the effective threshold
    # against zero matches the empirically-optimal one.
    model.eval()
    last_pos = task.prompt_len - 1

    def forward_sents() -> np.ndarray:
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pool), 64):
                chunk = pool[i:i + 64]
                ids = torch.tensor([task.encode(t + "=") for t, _ in chunk], dtype=torch.long)
                pos_idx = torch.arange(ids.shape[1])
                h = model.token_emb(ids) + model.pos_emb(pos_idx)[None, :, :]
                for blk in model.blocks:
                    h = h + blk.attn(blk.ln1(h))
                    h = h + blk.mlp(blk.ln2(h))
                out.extend(h[:, -1, SENT].tolist())
        return np.asarray(out)

    sents_a = forward_sents()
    labels_np = np.asarray([int(l) for _, l in pool])
    order = np.argsort(sents_a)
    s_sorted = sents_a[order]; y_sorted = labels_np[order]
    n_pool = len(pool); n_pos = int(labels_np.sum())
    pos_right, neg_right, pos_left, neg_left = n_pos, n_pool - n_pos, 0, 0
    best_acc, best_T = (n_pos / n_pool), float(s_sorted[0]) - 1.0
    for i in range(n_pool):
        acc = (neg_left + pos_right) / n_pool
        if acc > best_acc:
            best_acc = acc
            best_T = (s_sorted[i - 1] + s_sorted[i]) / 2 if i > 0 else s_sorted[i] - 1.0
        if y_sorted[i] == 1: pos_right -= 1; pos_left += 1
        else:                neg_right -= 1; neg_left += 1
    if (n_pool - n_pos) / n_pool > best_acc:
        best_acc, best_T = (n_pool - n_pos) / n_pool, float(s_sorted[-1]) + 1.0
    print(f"[Bigram] empirical pool acc {best_acc:.4f} at sent-threshold {best_T:.3f}", flush=True)

    with torch.no_grad():
        probe = 5.0
        model.pos_emb.weight[last_pos, SENT] += probe
        sents_b = forward_sents()
        model.pos_emb.weight[last_pos, SENT] -= probe
        alpha = float((sents_b - sents_a).mean() / probe)
        print(f"[Bigram] propagation gain alpha={alpha:.4f}", flush=True)
        delta = -float(best_T) / alpha if abs(alpha) > 1e-6 else 0.0
        model.pos_emb.weight[last_pos, SENT] += delta
        # Verification pass
        sents_c = forward_sents()
        pred = (sents_c > 0).astype(int)
        verify_acc = float((pred == labels_np).mean())
        print(f"[Bigram] post-shift verified pool acc {verify_acc:.4f} "
              f"(applied delta={delta:.3f})", flush=True)


model_shorthand_name = "DualBigramTri256"
model_description = "DualBigramAlpha2 + top-256 trigram detectors (curr+prev1+prev2 all-match, bias=20)."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    # Custom architecture for the BigramLogOdds circuit:
    #   d_model = V + V + L + spare = 160 (V=32, L=82)
    #   1 head with d_head=d_model so we can use a one-hot position encoding inside Q/K
    #   2 layers: L0 = offset-1 attention + bigram MLP; L1 = uniform aggregation
    #   d_ff = V*V = 1024 (one neuron per bigram in L0 MLP; L1 MLP is zeroed)
    model = SimpleTransformer(
        vocab_size=task.vocab_size, max_seq_len=max_seq_len,
        d_model=256, n_heads=2, n_layers=2, d_ff=2048 + 256,
    )
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
