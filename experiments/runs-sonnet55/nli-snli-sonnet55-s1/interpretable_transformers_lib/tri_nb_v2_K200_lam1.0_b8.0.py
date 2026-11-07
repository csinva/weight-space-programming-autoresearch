"""
Interpretable transformer for character-level sequence tasks (task: nli-snli).

Prompt: premise(60 chars, '_' padded) '|' hypothesis(60 chars) '='  -> one digit
0=entailment 1=neutral 2=contradiction.  Weights are written in closed form (no training).
"""

from __future__ import annotations

import argparse
import csv
import math
import math as _m
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
# Architecture (no LayerNorm; per-layer head width / MLP width)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal attention with independent head width d_head (q/k/v/o unconstrained)."""
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)  # no 1/sqrt(dh): scales are set by hand
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).reshape(B, T, H * dh))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """x + attn(x) [+ mlp(x)]. d_ff=0 means no MLP."""
    def __init__(self, d_model, n_heads, d_head, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, layer_cfgs):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, *c) for c in layer_cfgs])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Hand-written weights
# ---------------------------------------------------------------------------
#  L1 attn: two previous-token heads copy chars at offsets -1,-2 -> each position holds its char trigram.
#  L1 mlp : K trigram detectors (AND of 3 chars), each writes its own feature dim.
#  L2 attn: hypothesis position looks for the same trigram in the premise (vs. a '|' "sink") -> overlap flag m.
#  L3 attn: '=' averages m and the trigram features over non-pad hypothesis chars.
#  head   : class logits = bias + a_c * mean-overlap + sum_k w_ck * mean-trigram-k  (naive-Bayes style).

PL, HL = 60, 60
BETA = 8.0
K_TRI = 200
LAMBDA = 1.0


def _stats(task, seeds=(1001, 1002, 1003)):
    exs = {}
    for sd in seeds:
        for e in task.generate_examples(2000, sd):
            exs[e.prompt] = e.answer
    return list(exs.items())


def _hyp_trigrams(prompt):
    return [prompt[i - 2: i + 1] for i in range(PL + 1, PL + 1 + HL) if prompt[i] != "_"]


def write_weights(model, task) -> None:
    V, S = task.vocab_size, task.stoi
    data = _stats(task)
    # ---- closed-form statistics (counts -> log-odds) -------------------
    cnt = {}
    Nc = [0, 0, 0]
    ov = [[], [], []]
    for p, a in data:
        c = int(a)
        prem = {p[i - 2: i + 1] for i in range(2, PL)}
        tris = _hyp_trigrams(p)
        for t in tris:
            cnt.setdefault(t, [0, 0, 0])[c] += 1
        Nc[c] += len(tris)
        ov[c].append(sum(t in prem for t in tris) / max(1, len(tris)))

    def score(t):
        n = cnt[t]; tot = sum(n)
        return sum(x * _m.log((x + 0.5) / (tot / 3 + 0.5)) for x in n)

    cands = [t for t in cnt if sum(cnt[t]) >= 6 and "|" not in t]
    feats = sorted(cands, key=score, reverse=True)[:K_TRI]
    K = len(feats)
    assert K == K_TRI, K
    W = torch.zeros(3, K)
    for j, t in enumerate(feats):
        lr = [_m.log((cnt[t][c] + 1) / (Nc[c] + 1)) for c in range(3)]
        mu = sum(lr) / 3
        for c in range(3):
            W[c, j] = lr[c] - mu
    mu_ov = [sum(o) / len(o) for o in ov]
    var = sum(sum((x - mu_ov[c]) ** 2 for x in ov[c]) for c in range(3)) / sum(len(o) for o in ov)
    n_avg = sum(Nc) / len(data)

    # ---- residual layout ------------------------------------------------
    T = model.max_seq_len
    C0, C1, C2 = 0, V, 2 * V
    POS = 3 * V
    F_PREM, F_SINK, F_OTHER, F_HYP, F_CONST = [POS + T + i for i in range(5)]
    M = F_CONST + 1
    FEAT = M + 1
    PM = FEAT + K
    PF = PM + 1
    D = PF + K
    assert D == model.d_model, (D, model.d_model)

    with torch.no_grad():
        for p_ in model.parameters():
            p_.zero_()
        E, P = model.token_emb.weight, model.pos_emb.weight
        for t in range(V):
            E[t, C0 + t] = 1
        for p in range(T):
            P[p, POS + p] = 1
            P[p, F_CONST] = 1
            if p < PL: P[p, F_PREM] = 1
            elif p == PL: P[p, F_SINK] = 1
            else: P[p, F_OTHER] = 1
            if PL < p <= PL + HL: P[p, F_HYP] = 1

        # L1 attention: previous-token heads (offset 1 and 2)
        a1 = model.blocks[0].attn
        for h, off in enumerate((1, 2)):
            for p in range(T):
                if p >= off:
                    a1.W_q.weight[h * T + p - off, POS + p] = 20.0
                a1.W_k.weight[h * T + p, POS + p] = 1.0
            for t in range(V):
                a1.W_v.weight[h * T + t, C0 + t] = 1.0
                a1.W_o.weight[(C1 if h == 0 else C2) + t, h * T + t] = 1.0
        # L1 MLP: trigram detectors
        m1 = model.blocks[0].mlp
        for j, t in enumerate(feats):
            for off, ch in zip((C2, C1, C0), t):
                m1.fc1.weight[j, off + S[ch]] = 1.0
            m1.fc1.bias[j] = -2.0
            m1.fc2.weight[FEAT + j, j] = 1.0
        # L2 attention: trigram found in premise?
        a2 = model.blocks[1].attn
        for i in range(3 * V):
            a2.W_q.weight[i, i] = BETA
            a2.W_k.weight[i, i] = 1.0
        a2.W_q.weight[3 * V, F_CONST] = 1.0
        a2.W_k.weight[3 * V, F_SINK] = 2.5 * BETA
        a2.W_k.weight[3 * V, F_OTHER] = -100.0
        a2.W_v.weight[0, F_PREM] = 1.0
        a2.W_o.weight[M, 0] = 1.0
        # L3 attention: '=' averages over non-pad hypothesis chars
        a3 = model.blocks[2].attn
        a3.W_q.weight[0, F_CONST] = 20.0
        a3.W_k.weight[0, F_HYP] = 1.0
        a3.W_k.weight[0, F_CONST] = -1.0
        a3.W_k.weight[0, C0 + S["_"]] = -1.0
        a3.W_v.weight[0, M] = 1.0
        a3.W_o.weight[PM, 0] = 1.0
        for j in range(K):
            a3.W_v.weight[1 + j, FEAT + j] = 1.0
            a3.W_o.weight[PF + j, 1 + j] = 1.0
        # Readout
        Hd = model.head.weight
        for c in range(3):
            tok = S[str(c)]
            Hd[tok, F_CONST] = 10.0 - mu_ov[c] ** 2 / (2 * var)
            Hd[tok, PM] = mu_ov[c] / var
            for j in range(K):
                Hd[tok, PF + j] = LAMBDA * n_avg * W[c, j]


VARIANT = "v2"
model_shorthand_name = f"tri_nb_{VARIANT}_K{K_TRI}_lam{LAMBDA}_b{BETA}"
model_description = (f"Prev-token heads -> {K_TRI} trigram detectors (naive-Bayes weights, lambda={LAMBDA}) + premise "
                     f"trigram-overlap attention (beta={BETA}), mean-pooled at '='; no LayerNorm. {VARIANT}")


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
    T = task.seq_len
    V = task.vocab_size
    D = 3 * V + T + 7 + 2 * K_TRI
    model = SimpleTransformer(V, T, D, [
        (2, T, K_TRI),
        (1, 3 * V + 1, 0),
        (1, 1 + K_TRI, 0),
    ])
    write_weights(model, task)
    # convenience: snapshot this source under interpretable_transformers_lib/<name>.py
    import shutil
    lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interpretable_transformers_lib")
    os.makedirs(lib, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(lib, model_shorthand_name + ".py"))
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
