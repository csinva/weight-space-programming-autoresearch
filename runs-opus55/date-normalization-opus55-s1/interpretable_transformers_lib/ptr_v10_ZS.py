"""Hand-written (never trained) transformer for task date-normalization:
"apr. 9, 1994____...=" -> "1994-04-09"."""

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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """No LayerNorm. Layer 1: attention + MLP (parser). Layer 2: attention only (fetch)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 32,
                 n_heads: int = 4, d_ff: int = 42):
        super().__init__()
        self.d_model, self.n_heads, self.max_seq_len = d_model, n_heads, max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.look = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)
        self.fetch = CausalSelfAttention(d_model, n_heads)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        h = h + self.look(h)
        h = h + self.mlp(h)
        h = h + self.fetch(h)
        return self.head(h)


DIMS = ("ONE P P2 D0 Z0 L0 D1 Z1 L1 D2 D3 ZS SL DT US "
        "F1 F2 K0 K1 K2 K3 K5 K6 K8 K9 YR MO DY X").split()
I = {n: i for i, n in enumerate(DIMS)}


def S(w, r, **kv):
    """Set row r of weight matrix w, columns named by stream dimension."""
    for k, v in kv.items():
        w[r, I[k]] = v


def write_weights(model: SimpleTransformer, task) -> None:
    """Stream: ONE,P,P2=(1,i,i^2). D/Z/L=digit value, is-digit, letter code of the token (0)
    and of the tokens 1-4 back. SL/DT/US=own class. F*=position flags. K*/HY=output step.
    YR/MO/DY=field tags on the token ENDING a field. A=-1 if 19xx. X=scalar answer digit."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dh = model.d_model // model.n_heads
        s = math.sqrt(dh)

        for c, t in task.stoi.items():
            w = model.token_emb.weight
            if c.isdigit():
                S(w, t, D0=int(c), Z0=1)
            if c.isalpha():
                S(w, t, L0=ord(c) - 96)
            if c in "/._":
                S(w, t, **{{"/": "SL", ".": "DT", "_": "US"}[c]: 1})

        for i in range(model.max_seq_len):
            w = model.pos_emb.weight
            S(w, i, ONE=1, P=i, P2=i * i, F1=i == 1, F2=i <= 2)
            if i - 31 in (0, 1, 2, 3, 5, 6, 8, 9):
                S(w, i, **{f"K{i - 31}": 1})
            if i == 31:
                S(w, i, X=2)  # default century digit

        # Look-back heads: head h reads the token o=h+1 back with the Gaussian
        # score -G(j-(i-o))^2, which is bilinear in (1,j,j^2) and (1,i,i^2).
        a, G = model.look, 30 * s
        for h, o in enumerate((1, 2, 3)):
            b = h * dh
            S(a.W_k.weight, b, ONE=1), S(a.W_k.weight, b + 1, P=1), S(a.W_k.weight, b + 2, P2=1)
            S(a.W_q.weight, b, P2=-G, P=2 * G * o, ONE=-G * o * o)
            S(a.W_q.weight, b + 1, P=2 * G, ONE=-2 * G * o)
            S(a.W_q.weight, b + 2, ONE=-G)
            S(a.W_v.weight, b, D0=1), S(a.W_v.weight, b + 1, Z0=1), S(a.W_v.weight, b + 2, L0=1)
            for r, ch in enumerate("DZL"):
                name = "ZS" if ch == "Z" and o > 1 else f"{ch}{o}"  # ZS counts digits 2,3 back
                if name in I:
                    a.W_o.weight[I[name], b + r] = 1

        # MLP: each ReLU unit is a named conjunction.
        fc1, fc2, n = model.mlp.fc1, model.mlp.fc2, [0]

        def unit(bias, out, **inp):
            S(fc1.weight, n[0], **inp)
            fc1.bias[n[0]] = bias
            for k, v in out.items():
                fc2.weight[I[k], n[0]] = v
            n[0] += 1

        # (K5, K8 are the steps reading back an answer hyphen, the only output run ends.)
        unit(0, dict(DY=1), Z1=1, Z0=-1, K5=-1, K8=-1)                # any run end: DAY
        unit(-2, dict(YR=1, DY=-1, MO=-1), Z1=1, ZS=1, F2=-1, Z0=-1, K5=-1)  # 3+ digits: YEAR
        unit(-1, dict(MO=1, DY=-1), SL=1, F2=1)                     # first slash: MONTH
        unit(-1, dict(MO=1, DY=-1), DT=1, Z1=1)                       # dot: MONTH
        unit(-1, dict(YR=0.5, DY=-0.5), US=1, Z1=1)                   # padding: weak YEAR, weak DAY
        unit(-20, dict(D2=-1), D2=1, F1=20)                           # nothing 2 back at i=1

        # Month names: exact bump on the base-27 number of letters 2-3. Only "ar" is
        # ambiguous (j-an-u-ar-y): march's tag is weak so "an"/"eb" outrank it.
        for m, name in enumerate(task.MONTHS, 1):
            h2 = (ord(name[2]) - 96) + 27 * (ord(name[1]) - 96)
            t = 0.5 if name == "march" else 1
            for dlt, w in ((-1, 1), (0, -2), (1, 1)):
                unit(-(h2 + dlt), dict(MO=w * t, D2=w * (m // 10), D1=w * (m % 10)), L0=1, L1=27)

        # Fetch heads: each output step queries a tag, else the empty '=' token (K0).
        a, G2 = model.fetch, 30 * s
        heads = [("D3", -1 / 9, dict(YR="K0")), ("D3", 1, dict(YR="K1")),  # hundreds digit 9 means 19xx
                 ("D2", 1, dict(YR="K2", MO="K5", DY="K8")), ("D1", 1, dict(YR="K3", MO="K6", DY="K9"))]
        for h, (slot, scale, want) in enumerate(heads):
            b = h * dh
            q3 = dict(ONE=G2)
            for r, tag in enumerate(("YR", "MO", "DY", "K0")):
                S(a.W_k.weight, b + r, **{tag: 1})
                if tag in want:
                    S(a.W_q.weight, b + r, **{want[tag]: G2})
                    q3[want[tag]] = -G2
            S(a.W_q.weight, b + 3, **q3)
            S(a.W_v.weight, b, **{slot: scale})
            a.W_o.weight[I["X"], b] = 1

        # Head: parabola tangents, logit_d = 2dX + 1 - d^2 peaks at d = X.
        for d in range(10):
            S(model.head.weight, task.stoi[str(d)], X=2 * d, ONE=1 - d * d)
        S(model.head.weight, task.stoi["-"], ONE=1000, K0=-1000, K1=-1000, K2=-1000, K3=-1000,
          K5=-1000, K6=-1000, K8=-1000, K9=-1000)  # hyphen wherever no digit step is active
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


model_shorthand_name = "ptr_v10_ZS"
model_description = "is-digit flags from 2 and 3 back merged into one counter ZS (year = Z1 + ZS == 3)."


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
