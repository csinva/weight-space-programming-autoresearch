"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task here: date-normalization
("apr. 9, 1994____...=" -> "1994-04-09").

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.
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
    """LayerNorm removed: every stream dimension is a named, exact quantity."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer: parse layer, then pointer-fetch layer."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 44,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 45,
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

DIMS = ("ONE P P2 D0 Z0 L0 D1 Z1 L1 D2 Z2 L2 D3 Z3 D4 Z4 SL DT US GT "
        "F1 F2 F4 F7 F8 F10 OUT EQ K0 K1 K2 K3 K5 K6 K8 K9 HY YR MO DY A B X").split()
I = {n: i for i, n in enumerate(DIMS)}


def write_weights(model: SimpleTransformer, task) -> None:
    """Stream dims: ONE,P,P2 = (1,i,i^2); D/Z/L = digit value, is-digit, letter code
    of the token itself (0) and of the tokens 1-4 back; SL/DT/US/GT = own-token
    classes; F* = position flags; K*/HY = which output step; YR/MO/DY = field tags
    written on the token ENDING a field; A,B = century; X = scalar digit answer."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dh = model.d_model // model.n_heads
        s = math.sqrt(dh)

        for c, t in task.stoi.items():
            e = model.token_emb.weight[t]
            if c.isdigit():
                e[I["D0"]], e[I["Z0"]] = int(c), 1
            if c.isalpha():
                e[I["L0"]] = ord(c) - 96
            if c in "/._":
                e[I[{"/": "SL", ".": "DT", "_": "US"}[c]]] = 1
            if c in ",- snrt":
                e[I["GT"]] = 1

        for i in range(model.max_seq_len):
            e = model.pos_emb.weight[i]
            e[I["ONE"]], e[I["P"]], e[I["P2"]] = 1, i, i * i
            flags = {"F1": i == 1, "F2": i <= 2, "F4": i >= 4, "F7": i == 7, "F8": i == 8,
                     "F10": i == 10, "OUT": i >= 32, "EQ": i == 31, "HY": i in (35, 38)}
            for n, v in flags.items():
                e[I[n]] = float(v)
            for k in (0, 1, 2, 3, 5, 6, 8, 9):
                if i == 31 + k:
                    e[I[f"K{k}"]] = 1
            if i == 31:
                e[I["X"]] = 2  # default century digit

        # Layer 1 attention: head h reads the token (h+1) back. Score is the
        # Gaussian -G(j-(i-o))^2, bilinear in (1,j,j^2) and (1,i,i^2).
        a = model.blocks[0].attn
        G = 30 * s
        for h in range(4):
            o, b = h + 1, h * dh
            a.W_k.weight[b, I["ONE"]] = a.W_k.weight[b + 1, I["P"]] = a.W_k.weight[b + 2, I["P2"]] = 1
            a.W_q.weight[b, I["P2"]], a.W_q.weight[b, I["P"]], a.W_q.weight[b, I["ONE"]] = -G, 2 * G * o, -G * o * o
            a.W_q.weight[b + 1, I["P"]], a.W_q.weight[b + 1, I["ONE"]] = 2 * G, -2 * G * o
            a.W_q.weight[b + 2, I["ONE"]] = -G
            for r, src in enumerate(("D0", "Z0", "L0")):
                a.W_v.weight[b + r, I[src]] = 1
                dst = f"{'DZL'[r]}{o}"
                if dst in I:
                    a.W_o.weight[I[dst], b + r] = 1

        # Layer 1 MLP: each ReLU unit is a named conjunction.
        fc1, fc2 = model.blocks[0].mlp.fc1, model.blocks[0].mlp.fc2
        n = [0]

        def unit(inp, bias, out):
            for k, w in inp.items():
                fc1.weight[n[0], I[k]] = w
            fc1.bias[n[0]] = bias
            for k, w in out.items():
                fc2.weight[I[k], n[0]] = w
            n[0] += 1

        Z4run = {"Z1": 1, "Z2": 1, "Z3": 1, "F4": 1, "Z0": -1, "OUT": -1}
        unit({**Z4run, "Z4": 1}, -4, {"YR": 1})                 # 4-digit run ended here
        unit({**Z4run, "Z3": 2, "Z4": 2, "D4": -1}, -5, {"A": -1, "B": 9})  # ...and it starts with 1 => 19xx
        unit({"US": 1, "F8": 1, "Z3": -1}, -1, {"YR": 1})       # mm/dd/yy_
        unit({"SL": 1, "F2": 1}, -1, {"MO": 1})                 # first slash
        unit({"SL": 1, "F2": -1}, 0, {"DY": 1})                 # second slash
        unit({"DT": 1, "F7": 1}, -1, {"MO": 1})                 # yyyy.mm.
        unit({"Z1": 1, "GT": 1, "OUT": -1}, -1, {"DY": 1})      # day followed by , space - st nd rd th
        unit({"US": 1, "F10": 1, "Z3": -1}, -1, {"DY": 1})      # yyyy.mm.dd_
        unit({"D2": 1, "F1": 20}, -20, {"D2": -1})              # no token 2 back at i=1

        # Month names: exact bump on the base-27 number of the last 3 letters.
        for m, name in enumerate(task.MONTHS, 1):
            h3 = (ord(name[2]) - 96) + 27 * (ord(name[1]) - 96) + 729 * (ord(name[0]) - 96)
            for dlt, w in ((-1, 1), (0, -2), (1, 1)):
                unit({"L0": 1, "L1": 27, "L2": 729}, -(h3 + dlt),
                     {"MO": w, "D2": w * (m // 10), "D1": w * (m % 10)})

        # Layer 2 attention: output steps fetch value slots from tagged tokens.
        a = model.blocks[1].attn
        G2 = 30 * s
        slots = [("A", {"YR": ["K0"]}), ("B", {"YR": ["K1"]}),
                 ("D2", {"YR": ["K2"], "MO": ["K5"], "DY": ["K8"]}),
                 ("D1", {"YR": ["K3"], "MO": ["K6"], "DY": ["K9"]})]
        for h, (slot, want) in enumerate(slots):
            b = h * dh
            for r, tag in enumerate(("YR", "MO", "DY", "EQ")):
                a.W_k.weight[b + r, I[tag]] = 1
            a.W_q.weight[b + 3, I["ONE"]] = G2  # default: the empty '=' token
            for r, tag in enumerate(("YR", "MO", "DY")):
                for kk in want.get(tag, []):
                    a.W_q.weight[b + r, I[kk]] = G2
                    a.W_q.weight[b + 3, I[kk]] = -G2
            a.W_v.weight[b, I[slot]] = 1
            a.W_o.weight[I["X"], b] = 1

        # Head: parabola tangents, logit_d = 2dX + 1 - d^2 peaks at d = X.
        for d in range(10):
            r = task.stoi[str(d)]
            model.head.weight[r, I["X"]] = 2 * d
            model.head.weight[r, I["ONE"]] = 1 - d * d
        model.head.weight[task.stoi["-"], I["HY"]] = 1000

        debug_check(model, task)


def debug_check(model, task, n=5000, seed=777):
    """Generation check on a non-eval seed, listing failures."""
    ex = task.generate_examples(n, seed=seed)
    ids = torch.tensor([task.encode(e.prompt) for e in ex])
    for _ in range(task.answer_len):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], 1)
    bad = [(e.prompt.rstrip("_="), e.answer, task.decode(r[-task.answer_len:].tolist()))
           for e, r in zip(ex, ids) if task.decode(r[-task.answer_len:].tolist()) != e.answer]
    print("debug acc", 1 - len(bad) / n, bad[:25])


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ptr_v2_d44"
model_description = "As ptr_v1 with the 19xx detector requiring all four look-back digits (fixes mm/d0/yy false century)."


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
