"""
edit-distance-8: "aabbccdd,abcdabcd=" -> "4". The Levenshtein DP as a tied-weight cellular automaton.

State: 24 registers per position, all starting at 0: M(i), A(i), B(i) for columns i = 0..7.
Labels: 6 fixed channels per position from the embeddings: row flag, PUNCT, character (2), position (2).
Queries and keys only read labels, so where a head looks never depends on the computation.
One block, applied N_LAYERS times:
  * attention routes:   B(i) := -left.(A(i) + PUNCT)    A(i) := -B(i-1)    M(i) += (s_i == t_j)
  * the MLP is a gate:  z = min(1-m, -A+1, -B+1), added to both A and B
Before the MLP, (A, B) hold (-v, -h); after it they hold (z-v, z-h), which are h_out and v_out.
PUNCT at comma supplies DP row 0 (H=+1). '=' ends with B = -H(row 8); the distance is 8 - sum B.
No training: every weight is written in closed form.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]

N_LAYERS = 16    # 15 anti-diagonal sweeps plus one to copy row 8 to '=' are needed for exactness
USE_UNITS = (0, 1)   # ablation: which of the 4 z units contribute
GAMMA = 100.0    # match heads: score bonus for an equal character
LOCK = 4000.0    # match heads: score for looking at column i's position
SHARP = 1000.0   # left/own heads: neighbours differ by SHARP*(1-cos 20deg)/sqrt(2) nats
SELFCHECK = 20000
N_STATE, N_LABEL = 24, 6


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal scaled dot-product attention. q, k: (B, T, heads, dk). v: (B, T, heads, dv). Returns (B, T, heads*dv)."""
    B, T, H, dk = q.shape
    scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / math.sqrt(dk)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
    return (attn @ v.transpose(1, 2)).transpose(1, 2).contiguous().view(B, T, -1)


class CausalSelfAttention(nn.Module):
    """Three kinds of head: 8 match heads (dk 3, dv 1), one left head (dk 2, dv 8), one own head (dk 2, dv 16)."""

    def __init__(self):
        super().__init__()
        self.Q_match = nn.Linear(N_LABEL, 24, bias=False)
        self.K_match = nn.Linear(N_LABEL, 24, bias=False)
        self.V_match = nn.Linear(N_LABEL, 8, bias=False)
        self.Q_route = nn.Linear(N_LABEL, 4, bias=False)
        self.K_route = nn.Linear(N_LABEL, 4, bias=False)
        self.V_route = nn.Linear(N_STATE + N_LABEL, 24, bias=False)
        self.W_o = nn.Linear(32, N_STATE, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        m = attend(self.Q_match(labels).view(B, T, 8, 3), self.K_match(labels).view(B, T, 8, 3),
                   self.V_match(labels).view(B, T, 8, 1))
        q, k = self.Q_route(labels).view(B, T, 2, 2), self.K_route(labels).view(B, T, 2, 2)
        v = self.V_route(torch.cat([x, labels], -1))
        left = attend(q[:, :, :1], k[:, :, :1], v[..., :8][:, :, None])
        own = attend(q[:, :, 1:], k[:, :, 1:], v[..., 8:][:, :, None])
        return self.W_o(torch.cat([m, left, own], -1))


class SimpleTransformer(nn.Module):
    """Causal transformer without LayerNorm whose single block is applied N_LAYERS times (tied weights)."""

    def __init__(self, vocab_size, max_seq_len=32, n_layers=N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.token_emb = nn.Embedding(vocab_size, 3)    # PUNCT, character (2)
        self.pos_emb = nn.Embedding(max_seq_len, 3)     # row flag, position (2)
        self.attn = CausalSelfAttention()
        self.fc1 = nn.Linear(4, 4)                      # the cell rule, shared by all 8 columns
        self.fc2 = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(9, vocab_size, bias=False)   # reads B(0..7) and PUNCT

    def cell(self, x: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
        m, a, b = x[..., :8], x[..., 8:16], x[..., 16:24]
        z = self.fc2(F.relu(self.fc1(torch.stack([m, a, b, row.expand_as(m)], -1)))).squeeze(-1)
        return torch.cat([torch.zeros_like(m), z, z], -1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        p = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        labels = torch.cat([p[..., :1], self.token_emb(ids), p[..., 1:]], -1)
        x = torch.zeros(B, T, N_STATE, device=ids.device)
        for _ in range(self.n_layers):
            x = x + self.attn(x, labels)
            x = x + self.cell(x, labels[..., :1])
        return self.head(torch.cat([x[..., 16:24], labels[..., 1:2]], -1))


LEN, ANS = 8, 17             # string length; position of '='
ROWS = range(9, 17)          # positions of t_1..t_8
STEP = 2 * math.pi / 18      # angle between neighbouring positions on the position circle

def M(i): return i           # match m(i, this row)
def A(i): return 8 + i       # before the MLP: -v. After it: h_out = z - v
def B(i): return 16 + i      # before the MLP: -h. After it: v_out = z - h
ROW, PUNCT, CH, PC = 0, 1, 2, 4   # label channels: row flag; flag on ',' and '='; character (2); position (2)

# z = z1 - z2 - z3 - z4 is min(1-m, v+1, h+1) on {-1,0,1}^2, with v = -A and h = -B.
Z_UNITS = [  # (weight on A, on B, on M, bias, sign in z)
    (-1.0, -1.0, -3.0, 1.0, 1.0),     # v + h + 1
    (-1.0, -1.0, -2.0, 0.0, -1.0),    # v + h
    (1.0, -1.0, -2.0, -1.0, -1.0),    # h - v - 1
    (-1.0, 1.0, -2.0, -1.0, -1.0),    # v - h - 1
]


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        tok, pos, a = model.token_emb.weight, model.pos_emb.weight, model.attn

        for c, (cx, cy) in {"a": (1, 0), "b": (0, 1), "c": (-1, 0), "d": (0, -1)}.items():
            tok[task.stoi[c], 1], tok[task.stoi[c], 2] = cx, cy           # token_emb = PUNCT, CH
        tok[task.stoi[","], 0] = tok[task.stoi["="], 0] = 1.0
        for p in range(ANS + 1):
            pos[p, 1], pos[p, 2] = math.cos(STEP * p), math.sin(STEP * p)  # pos_emb = ROW, PC
            pos[p, 0] = float(p in ROWS)

        Qm, Km, Vm = a.Q_match.weight, a.K_match.weight, a.V_match.weight
        for i in range(LEN):        # score(s_i) = LOCK + GAMMA*same char; score(',') = LOCK + GAMMA/2 (the sink)
            o, ci, si, c8 = 3 * i, math.cos(STEP * i), math.sin(STEP * i), math.cos(STEP * (LEN - i))
            Qm[o, ROW] = 1.0
            Km[o, PC], Km[o, PC + 1], Km[o, PUNCT] = LOCK * ci, LOCK * si, LOCK * (1 - c8) + GAMMA / 2
            Qm[o + 1, CH], Qm[o + 2, CH + 1], Km[o + 1, CH], Km[o + 2, CH + 1] = GAMMA, GAMMA, 1.0, 1.0
            Vm[i, PC], Vm[i, PC + 1], Vm[i, PUNCT] = ci, si, -c8           # value: 1 at s_i, exactly 0 at the comma
            a.W_o.weight[M(i), i] = 1.0

        Qr, Kr, Vr, O = a.Q_route.weight, a.K_route.weight, a.V_route.weight, a.W_o.weight
        cd, sd = math.cos(STEP), math.sin(STEP)
        Qr[0, PC], Qr[0, PC + 1], Qr[1, PC], Qr[1, PC + 1] = SHARP * cd, SHARP * sd, -SHARP * sd, SHARP * cd  # left
        Qr[2, PC], Qr[3, PC + 1] = SHARP, SHARP                                                                # own
        Kr[0, PC], Kr[1, PC + 1], Kr[2, PC], Kr[3, PC + 1] = 1.0, 1.0, 1.0, 1.0

        lab = N_STATE                # label channels start here in the input of V_route
        for i in range(LEN):
            left, own_a, own_b = i, 8 + i, 16 + i                          # rows of V_route
            Vr[left, A(i)], Vr[left, lab + PUNCT] = 1.0, 1.0
            Vr[own_a, A(i)], Vr[own_a, B(i - 1) if i else lab + ROW] = 1.0, 1.0
            Vr[own_b, B(i)] = 1.0
            O[B(i), 8 + left], O[A(i), 8 + own_a], O[B(i), 8 + own_b] = -1.0, -1.0, -1.0

        # The cell rule, written once. Off-row positions act as m + INERT, so z = 0 there whatever m is.
        INERT = N_LAYERS + 1.0
        for n, (wa, wb, wm, bias, sign) in enumerate(Z_UNITS):
            model.fc1.weight[n] = torch.tensor([wm, wa, wb, -wm * INERT])
            model.fc1.bias[n] = bias + wm * INERT
            model.fc2.weight[0, n] = sign if n in USE_UNITS else 0.0

        # Readout on (B(0..7), PUNCT): S = 8 - sum B. logit[d] = 2dS - d^2 is -(S-d)^2 up to a shared term.
        W = model.head.weight
        W[:, 8] = -1000.0
        for d in range(LEN + 1):
            t = task.stoi[str(d)]
            W[t, :8] = -2.0 * d
            W[t, 8] = 2.0 * d * LEN - d * d
        selfcheck(model, task)


def selfcheck(model, task) -> None:
    """Diagnostic only: compare with task.distance on many strings, half of them near-duplicates."""
    rng, pairs = random.Random(123), []
    for n in range(SELFCHECK):
        s = [rng.choice("abcd") for _ in range(LEN)]
        t = [rng.choice("abcd") for _ in range(LEN)]
        if n % 2 == 0:
            t = list(s)
            for _ in range(rng.randint(0, 4)):
                t[rng.randrange(LEN)] = rng.choice("abcd")
            if rng.random() < 0.5:
                k = rng.randrange(LEN)
                t = t[:k] + t[k + 1:] + [rng.choice("abcd")]
        pairs.append(("".join(s), "".join(t)))
    ids = torch.tensor([[task.stoi[c] for c in f"{s},{t}="] for s, t in pairs])
    pred = model(ids)[:, -1].argmax(-1).tolist()
    want = [task.stoi[str(task.distance(s, t))] for s, t in pairs]
    print(f"SELFCHECK: {sum(a == b for a, b in zip(pred, want))}/{len(pairs)}")
    nz = sum(int((p != 0).sum()) for p in model.parameters())
    print(f"NONZERO WEIGHTS: {nz}")


# Row identifier in results/overall_results.csv.
model_shorthand_name = "e25_abl_no_corner_units"
model_description = "Cell ablation on e23: drops z3 and z4, the corrections at (v,h) = (-1,+1) and (+1,-1), leaving z = clip(v+h+1, 0, 1)."


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
