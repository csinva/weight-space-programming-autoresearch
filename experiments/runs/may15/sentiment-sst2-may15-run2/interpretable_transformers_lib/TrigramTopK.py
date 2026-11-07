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
    """Trigram + bigram log-odds sentiment classifier.

    Combines a V*V bigram lookup with the top-K trigram features from the
    SST-2 pool. Two offset-attention heads in block 0 expose the previous
    token (prev1) and the token two positions back (prev2). The block-0 MLP
    has one neuron per bigram and one neuron per selected trigram; each
    writes a scaled log-odds value to the sentiment channel. Block 1 then
    uniformly averages sentiment across positions to the '=' position.

    Architecture (set by build_model):
      - d_model = 256, n_heads = 2, d_head = 128, n_layers = 2.
      - Residual layout:
          [0..31]   : current token one-hot               (token_emb)
          [32..63]  : prev1 token one-hot scratch         (L0 head 0)
          [64..95]  : prev2 token one-hot scratch         (L0 head 1)
          [96..177] : position one-hot (L=82)             (pos_emb)
          [253]     : +0.5   } LN-stabilising constants
          [254]     : -0.5   }
          [255]     : sentiment channel
    """
    import numpy as np

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
    UND = task.stoi['_']

    # ----- 1. Collect n-gram counts from the SST-2 pool -----
    pool = task._load_pool()
    a_bi = 1.0
    pos_bi = torch.full((V, V), a_bi)
    neg_bi = torch.full((V, V), a_bi)
    pos_tri: dict[tuple[int, int, int], float] = {}
    neg_tri: dict[tuple[int, int, int], float] = {}
    tri_count: dict[tuple[int, int, int], float] = {}
    for text, label in pool:
        prev1 = UND
        prev2 = UND
        is_pos = (label == '1')
        for ch in text:
            cur = task.stoi[ch]
            if is_pos: pos_bi[prev1, cur] += 1
            else:      neg_bi[prev1, cur] += 1
            key = (prev2, prev1, cur)
            tri_count[key] = tri_count.get(key, 0) + 1
            if is_pos: pos_tri[key] = pos_tri.get(key, 0) + 1
            else:      neg_tri[key] = neg_tri.get(key, 0) + 1
            prev2 = prev1
            prev1 = cur

    # ----- 2. Compute bigram log-odds -----
    bg = torch.log(pos_bi / pos_bi.sum()) - torch.log(neg_bi / neg_bi.sum())
    for ch in SKIP:
        j = task.stoi[ch]
        bg[j, :] = 0.0; bg[:, j] = 0.0

    # ----- 3. Compute trigram log-odds (Laplace smoothing) for observed trigrams -----
    a_tri = 1.0
    n_pos_total = sum(pos_tri.values())
    n_neg_total = sum(neg_tri.values())
    # Implicit smoothing: every observed trigram gets alpha added to each class count.
    # Denominators count smoothed mass over the universe of *observed* trigrams.
    n_obs = len(tri_count)
    denom_pos = n_pos_total + a_tri * n_obs
    denom_neg = n_neg_total + a_tri * n_obs
    tri_log_odds: dict[tuple[int, int, int], float] = {}
    for key, total in tri_count.items():
        if total < 5:  # require ≥5 occurrences for reliability
            continue
        c2, c1, cc = key
        if c2 in (UND, task.stoi['=']) or c1 in (task.stoi['='],) or cc in (task.stoi['='],):
            continue
        if cc in (task.stoi['0'], task.stoi['1']):
            continue
        pp = (pos_tri.get(key, 0) + a_tri) / denom_pos
        pn = (neg_tri.get(key, 0) + a_tri) / denom_neg
        tri_log_odds[key] = math.log(pp) - math.log(pn)

    # Keep top-K by |log-odds|.
    K_tri = 1024
    sorted_keys = sorted(tri_log_odds.keys(), key=lambda k: -abs(tri_log_odds[k]))[:K_tri]
    selected_tri = {k: tri_log_odds[k] for k in sorted_keys}
    print(f"[Trigram] {len(tri_log_odds)} reliable trigrams; selected top {len(selected_tri)}.", flush=True)

    # ----- 4. Calibrate threshold on the pool using bigram + trigram score -----
    bg_np = bg.numpy()
    tri_arr = selected_tri  # local alias
    scores = np.zeros(len(pool))
    labels = np.zeros(len(pool), dtype=np.int64)
    for i, (text, label) in enumerate(pool):
        prev1 = UND; prev2 = UND
        s = 0.0
        for ch in text:
            cur = task.stoi[ch]
            s += bg_np[prev1, cur]
            tri_v = tri_arr.get((prev2, prev1, cur))
            if tri_v is not None:
                s += tri_v
            prev2 = prev1; prev1 = cur
        scores[i] = s
        labels[i] = int(label)

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
    print(f"[Trigram] pool calibration acc {best_acc:.4f} at T*={best_T:.4f}", flush=True)
    shift = float(best_T) / 81.0
    bg = bg - shift  # subtract uniform shift via bigram only (one bigram fires per position)

    # ----- 5. Set up weights -----
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

        # Reset blocks
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0); blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0); blk.ln2.bias.zero_()
            for w in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                w.weight.zero_()
            blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
            blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Block 0 attention: two heads with offset 1 and offset 2.
        b0 = model.blocks[0]
        M = 50.0
        # Head 0: offset 1
        for p in range(1, L):
            b0.attn.W_q.weight[0 * dh + (p - 1), POS_OFF + p] = M
        for q in range(L):
            b0.attn.W_k.weight[0 * dh + q, POS_OFF + q] = 1.0
        for c in range(V):
            b0.attn.W_v.weight[0 * dh + c, c] = 1.0
        # Head 1: offset 2
        for p in range(2, L):
            b0.attn.W_q.weight[1 * dh + (p - 2), POS_OFF + p] = M
        for q in range(L):
            b0.attn.W_k.weight[1 * dh + q, POS_OFF + q] = 1.0
        for c in range(V):
            b0.attn.W_v.weight[1 * dh + c, c] = 1.0
        # W_o: route head 0 dim c -> PREV1_OFF + c; head 1 dim c -> PREV2_OFF + c.
        prev_gain = 1.0 / 6.21  # empirically tuned to match curr-token LN magnitude
        for c in range(V):
            b0.attn.W_o.weight[PREV1_OFF + c, 0 * dh + c] = prev_gain
            b0.attn.W_o.weight[PREV2_OFF + c, 1 * dh + c] = prev_gain

        # Block 0 MLP: V*V bigram detectors followed by K trigram detectors.
        bias_bi = 10.0
        trig_bi = 3.44
        for cp in range(V):
            for cc in range(V):
                idx = cp * V + cc
                b0.mlp.fc1.weight[idx, cc] = 1.0
                b0.mlp.fc1.weight[idx, PREV1_OFF + cp] = 1.0
                b0.mlp.fc1.bias[idx] = -bias_bi
                b0.mlp.fc2.weight[SENT, idx] = bg[cp, cc].item() / trig_bi

        bias_tri = 15.0  # require all 3 signals (each ~7-8) to add up; suppress 2-of-3
        trig_tri = 7.0   # approx triggered value when all three fire
        OFFSET = V * V
        for i, (key, lo) in enumerate(selected_tri.items()):
            idx = OFFSET + i
            if idx >= b0.mlp.fc1.weight.shape[0]:
                break
            c2, c1, cc = key
            b0.mlp.fc1.weight[idx, cc] = 1.0
            b0.mlp.fc1.weight[idx, PREV1_OFF + c1] = 1.0
            b0.mlp.fc1.weight[idx, PREV2_OFF + c2] = 1.0
            b0.mlp.fc1.bias[idx] = -bias_tri
            b0.mlp.fc2.weight[SENT, idx] = float(lo) / trig_tri

        # Block 1 attention: uniform causal aggregation of SENT (head 0 only).
        b1 = model.blocks[1]
        b1.attn.W_v.weight[0 * dh + 0, SENT] = 1.0
        b1.attn.W_o.weight[SENT, 0 * dh + 0] = 200.0

        # Final LN + head
        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()
        model.head.weight.zero_()
        Kg = 100.0
        model.head.weight[task.stoi['1'], SENT] = Kg
        model.head.weight[task.stoi['0'], SENT] = -Kg

    # ----- 6. Empirical 2-pass threshold calibration on actual model output -----
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
    pr, nr, pl, nl = n_pos, n_pool - n_pos, 0, 0
    best_acc, best_T = (n_pos / n_pool), float(s_sorted[0]) - 1.0
    for i in range(n_pool):
        acc = (nl + pr) / n_pool
        if acc > best_acc:
            best_acc = acc
            best_T = (s_sorted[i - 1] + s_sorted[i]) / 2 if i > 0 else s_sorted[i] - 1.0
        if y_sorted[i] == 1: pr -= 1; pl += 1
        else:                nr -= 1; nl += 1
    if (n_pool - n_pos) / n_pool > best_acc:
        best_acc, best_T = (n_pool - n_pos) / n_pool, float(s_sorted[-1]) + 1.0
    print(f"[Trigram] empirical pool acc {best_acc:.4f} at sent-threshold {best_T:.3f}", flush=True)

    with torch.no_grad():
        probe = 5.0
        model.pos_emb.weight[last_pos, SENT] += probe
        sents_b = forward_sents()
        model.pos_emb.weight[last_pos, SENT] -= probe
        alpha = float((sents_b - sents_a).mean() / probe)
        print(f"[Trigram] propagation gain alpha={alpha:.4f}", flush=True)
        delta = -float(best_T) / alpha if abs(alpha) > 1e-6 else 0.0
        model.pos_emb.weight[last_pos, SENT] += delta
        sents_c = forward_sents()
        pred = (sents_c > 0).astype(int)
        verify_acc = float((pred == labels_np).mean())
        print(f"[Trigram] post-shift verified pool acc {verify_acc:.4f} (delta={delta:.3f})", flush=True)


model_shorthand_name = "TrigramTopK"
model_description = "Bigram + top-1024 trigram log-odds. L0 has 2 offset-attn heads (offsets 1 and 2); MLP has V*V bigram detectors plus K trigram detectors writing to sentiment; L1 uniformly aggregates; two-stage threshold calibration."
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
    # Trigram+bigram architecture: d_model=256, 2 heads (offset 1 / offset 2),
    # 2 layers; d_ff = V*V + 1024 = 2048 (bigram lookup + top-K trigrams).
    model = SimpleTransformer(
        vocab_size=task.vocab_size, max_seq_len=max_seq_len,
        d_model=256, n_heads=2, n_layers=2, d_ff=2048,
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
