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


class Sparse(nn.Module):
    """A written nn.Linear/nn.Embedding that keeps only its nonzero weights as parameters."""

    def __init__(self, m):
        super().__init__()
        w, b = m.weight.detach(), getattr(m, "bias", None)
        self.shape, self.emb = tuple(w.shape), isinstance(m, nn.Embedding)
        self.register_buffer("idx", w.nonzero().T)
        self.vals = nn.Parameter(w[w != 0].clone())
        self.bias = nn.Parameter(b.detach().clone()) if b is not None and b.any() else None

    def forward(self, x):
        w = torch.zeros(self.shape).to(self.vals).index_put(tuple(self.idx), self.vals)
        return F.embedding(x, w) if self.emb else F.linear(x, w, self.bias)


class SimpleTransformer(nn.Module):
    """No LayerNorm. Layer 1: attention + MLP (parser). Layer 2: attention only (fetch)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 27,
                 n_heads: int = 3, d_ff: int = 19):
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


DIMS = ("ONE P P2 D0 Z0 C0 S0 D1 Z1 C1 S1 D2 D3 ZS SL DT US "
        "FY FM FD H T O YR MO DY X").split()
I = {n: i for i, n in enumerate(DIMS)}


def angle(c):
    return 2 * math.pi * (ord(c) - 97) / 26


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
                S(w, t, C0=math.cos(angle(c)), S0=math.sin(angle(c)))  # letters on a circle
            if c in "/._":
                S(w, t, **{{"/": "SL", ".": "DT", "_": "US"}[c]: 1})
            if c == "=":
                S(w, t, US=2)  # '=' is "extra padding": the loudest US, used as the empty token

        for i in range(model.max_seq_len):
            w = model.pos_emb.weight
            # Answer steps = field (FY/FM/FD) x digit (H, T, O; thousands = FY with none of them).
            # H doubles as "i==1" and T as "i<=2": the prompt uses those, answer steps ignore it.
            S(w, i, ONE=1, P=i, P2=i * i, FY=31 <= i <= 34, FM=i in (36, 37), FD=i in (39, 40),
              H=i in (1, 32), T=i <= 2 or i in (33, 36, 39), O=i in (34, 37, 40))
            if i == 31:
                S(w, i, X=2)  # default century digit

        # Look-back heads: head h reads the token o=h+1 back with the Gaussian
        # score -G(j-(i-o))^2, which is bilinear in (1,j,j^2) and (1,i,i^2).
        a, G = model.look, 10 * s
        for h, o in enumerate((1, 2, 3)):
            b = h * dh
            S(a.W_k.weight, b, ONE=1), S(a.W_k.weight, b + 1, P=1), S(a.W_k.weight, b + 2, P2=1)
            S(a.W_q.weight, b, P2=-G, P=2 * G * o, ONE=-G * o * o)
            S(a.W_q.weight, b + 1, P=2 * G, ONE=-2 * G * o)
            S(a.W_q.weight, b + 2, ONE=-G)
            for r, src in enumerate(("D0", "Z0", "C0", "S0")):
                S(a.W_v.weight, b + r, **{src: 1})
            for r, ch in enumerate("DZCS"):
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

        # (Answer hyphens are the only output run ends; they arrive at FM/FD tens steps.)
        unit(0, dict(DY=1), Z1=1, Z0=-1, FM=-1, FD=-1)                # any run end: DAY
        unit(-2, dict(YR=1, DY=-1, MO=-1), Z1=1, ZS=1, T=-1, Z0=-1)   # 3+ digits: YEAR
        unit(-1, dict(MO=1, DY=-1), SL=1, T=1)                        # first slash: MONTH
        unit(-1, dict(MO=1, DY=-1), DT=1, Z1=1)                       # dot: MONTH
        unit(-2, dict(YR=0.5, DY=-0.5), US=1, Z1=2)                   # padding: weak YEAR, weak DAY
        unit(-20, dict(D2=-1), D2=1, H=20)                            # nothing 2 back at i=1
        unit(-82, dict(X=1), D0=-9, H=100)                            # hundreds = 18 - 9*thousands

        # Month names: one ReLU per month, firing only when letters 2-3 sit exactly at the
        # template angles (sum of two cosines = 2; the nearest miss is 1 + cos(2pi/26) = 1.971).
        # Only "ar" is ambiguous (j-an-u-ar-y): march's tag is weak so "an"/"eb" outrank it.
        for m, name in enumerate(task.MONTHS, 1):
            a0, a1, g = angle(name[2]), angle(name[1]), 1 / 0.015
            t = 0.5 if name == "march" else 1
            unit(-1.985 * g, dict(MO=t, D2=m // 10, D1=m % 10), C0=g * math.cos(a0),
                 S0=g * math.sin(a0), C1=g * math.cos(a1), S1=g * math.sin(a1))

        # Fetch heads: a head always asks for the current field, but unless the step's digit
        # flag is its own, a twice-as-loud query for the empty '=' token (max US) outvotes that.
        a, G2 = model.fetch, 20 * s
        thousands = dict(FY=1, H=-1, T=-1, O=-1)
        fields = dict(YR=dict(FY=1), MO=dict(FM=1), DY=dict(FD=1))
        heads = [("D3", -1 / 9, thousands, dict(YR=thousands)),  # hundreds digit 9 means 19xx
                 ("D2", 1, dict(T=1), fields), ("D1", 1, dict(O=1), fields)]
        for h, (slot, scale, gate, want) in enumerate(heads):
            b = h * dh
            for r, tag in enumerate(("YR", "MO", "DY", "US")):
                S(a.W_k.weight, b + r, **{tag: 1})
                if tag in want:
                    S(a.W_q.weight, b + r, **{k: G2 * v for k, v in want[tag].items()})
            S(a.W_q.weight, b + 3, ONE=2 * G2, **{k: -2 * G2 * v for k, v in gate.items()})
            S(a.W_v.weight, b, **{slot: scale})
            a.W_o.weight[I["X"], b] = 1

        # Head: parabola tangents, logit_d = 2dX + 1 - d^2 peaks at d = X.
        for d in range(10):
            S(model.head.weight, task.stoi[str(d)], X=2 * d, ONE=1 - d * d)
        S(model.head.weight, task.stoi["-"], ONE=1000, FY=-1000, FM=-1000, FD=-1000)  # no field: hyphen

        # Keep only what was written: n_params now counts actual nonzero weights.
        for parent, names in ((model, ("token_emb", "pos_emb", "head")), (model.mlp, ("fc1", "fc2")),
                              (model.look, ("W_q", "W_k", "W_v", "W_o")),
                              (model.fetch, ("W_q", "W_k", "W_v", "W_o"))):
            for name in names:
                setattr(parent, name, Sparse(getattr(parent, name)))
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


model_shorthand_name = "ptr_v20_fetchG20"
model_description = "Gain sweep: fetch sharpness 20. At 10, weak tags (score 5) lost e^-5 to each of ~25 padding tokens; 20 gives score 10."


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
