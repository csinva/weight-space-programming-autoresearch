"""Hand-written transformer for 5-digit addition. No training anywhere."""

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
    # Original architecture, unmodified. Only block 0 does anything; blocks 1 and 2
    # have zero W_o and fc2, so the residual stream passes through them untouched.
    # Circuit: attention reads x_j = sum_{k>=j} (a_k + b_k) 10^(j-k), and a ReLU
    # polygon around the mod-10 ring turns x_j into the digit floor(x_j) mod 10.
    model.double()
    for p in model.parameters():
        p.data.zero_()
    for name, p in model.named_parameters():
        if "ln" in name and name.endswith("weight"):
            p.data.fill_(1.0)   # every LayerNorm: unit gain, zero shift

    KEY_SINK, KEY_OFF, VAL, VAL_NEG, BAL, BAL_NEG = 5, 6, 7, 8, 9, 10
    KEY_NEG, FILL, FILL_NEG = 11, 12, 13    # balance dims for the position part
    X, X_NEG, U, V, U_NEG, V_NEG = 14, 15, 16, 17, 18, 19
    ALPHA, OFF, DELTA, EPS = 0.25, -80.0, 5e-6, 1e-5    # EPS is LayerNorm's default

    # LayerNorm taming: every token + position embedding has mean 0 and the same
    # norm, so the first LayerNorm is just division by a constant SIGMA.
    C = 100.0
    pos, tok = model.pos_emb.weight.data, model.token_emb.weight.data
    for t in range(model.vocab_size):
        d = float(t) if t < 10 else 0.0
        b = math.sqrt(C - d * d)
        tok[t, [VAL, VAL_NEG, BAL, BAL_NEG]] = torch.tensor([d, -d, b, -b], dtype=tok.dtype)
    root2 = math.sqrt(2.0)
    def prompt_key(p, c):
        pos[p, c], pos[p, KEY_NEG], pos[p, FILL], pos[p, FILL_NEG] = 1.0, -1.0, root2, -root2
    for c in range(5):
        prompt_key(c, c)
        prompt_key(6 + c, c)
    prompt_key(5, KEY_SINK)
    for i in range(6):
        pos[11 + i, i], pos[11 + i, KEY_OFF], pos[11 + i, KEY_NEG] = 1.0, 1.0, -2.0
    base_sq = 6.0 + 2 * C
    D = model.d_model
    SIGMA = math.sqrt(base_sq / D + EPS)

    # Attention (head 0 only) is the same lookup table as RingAttnD11.
    root_dh = math.sqrt(D / model.n_heads)
    table = torch.full((6, 7), OFF, dtype=torch.float64)
    for i in range(6):
        j = i - 1
        weights = {k: ALPHA * 10.0 ** (j - k) for k in range(max(j, 0), 5)}
        for k, w in weights.items():
            table[i, k] = math.log(w)
        table[i, KEY_SINK] = math.log(1 - 2 * sum(weights.values()))
    attn = model.blocks[0].attn
    for k in range(7):
        attn.W_k.weight.data[k, k] = SIGMA
        attn.W_q.weight.data[k, :6] = root_dh * SIGMA * table[:, k]
    attn.W_v.weight.data[0, VAL] = SIGMA
    attn.W_o.weight.data[X, 0] = 1.0 / ALPHA        # x arrives as a balanced pair,
    attn.W_o.weight.data[X_NEG, 0] = -1.0 / ALPHA   # so the token mean stays 0

    # The second LayerNorm now shows y(x) = x / sqrt((base + 2 x^2) / D + EPS),
    # which is strictly increasing. So the polygon kinks go at y(n - DELTA), and the
    # chord from ring(n) to ring(n + 1) still spans exactly digit n's sector.
    y = lambda x: x / math.sqrt((base_sq + 2 * x * x) / D + EPS)
    ring = lambda n: torch.tensor([math.cos(2 * math.pi * n / 10), math.sin(2 * math.pi * n / 10)], dtype=tok.dtype)
    mlp = model.blocks[0].mlp
    mlp.fc2.bias.data[[U, V, U_NEG, V_NEG]] = torch.cat([ring(0), -ring(0)])
    prev_slope = torch.zeros(2, dtype=tok.dtype)
    for n in range(20):
        kink, next_kink = y(n - DELTA), y(n + 1 - DELTA)
        slope = (ring(n + 1) - ring(n)) / (next_kink - kink)
        mlp.fc1.weight.data[n, X] = 1.0
        mlp.fc1.bias.data[n] = -kink
        change = slope - prev_slope
        mlp.fc2.weight.data[[U, V, U_NEG, V_NEG], n] = torch.cat([change, -change])
        prev_slope = slope

    # Head: digit d owns the ring sector centred on d + 0.5 tenths of a turn.
    for d in range(10):
        model.head.weight.data[d, [U, V]] = ring(d + 0.5)

    _stress_test(model)


def _stress_test(model, n_random: int = 50000) -> None:
    """Check the model on random pairs plus hand-picked long carry chains."""
    gen = torch.Generator().manual_seed(1)
    a = torch.randint(0, 100000, (n_random,), generator=gen).tolist()
    b = torch.randint(0, 100000, (n_random,), generator=gen).tolist()
    edges = [0, 1, 9, 10, 99, 100, 999, 1000, 9999, 10000, 49999, 50000, 50001, 89999, 90000, 90001, 99990, 99998, 99999]
    for p in edges:
        for q in edges:
            a.append(p)
            b.append(q)
    for s in (9999, 10000, 19999, 99999, 100000, 150000, 199998):   # complements
        for p in range(max(0, s - 99999), min(s, 99999) + 1, 997):
            a.append(p)
            b.append(s - p)
    wrong, shown = 0, []
    with torch.no_grad():
        for start in range(0, len(a), 4096):
            aa, bb = a[start:start + 4096], b[start:start + 4096]
            ids = torch.tensor([[int(c) for c in f"{p:05d}"] + [10] + [int(c) for c in f"{q:05d}"] + [11]
                                for p, q in zip(aa, bb)])
            for _ in range(6):
                ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
            for row, p, q in zip(ids[:, 12:].tolist(), aa, bb):
                if "".join(map(str, row)) != f"{p + q:06d}":
                    wrong += 1
                    if len(shown) < 5:
                        shown.append((p, q, row))
    print(f"STRESS: {wrong} wrong of {len(a)}; examples: {shown}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "DefaultArchExact"
model_description = "Original 64-dim, 4-head, 3-layer, ReLU, LayerNorm, residual architecture unmodified: balanced zero-mean constant-norm embeddings make LayerNorm a constant, and ReLU kinks sit at the warped y(n)."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
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
