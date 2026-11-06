"""
Snapshot of run 1 (date_parser_2L_markers, 97.5%). Identical to
date_parser_2L_markers_v2.py except for the SHORT feature (see marked block),
which required a non-digit at p-3 and so missed "m/d/yyyy" days with 1-digit m and d.
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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int = 48):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous()
        return self.W_o(out.view(B, T, H * dh))


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
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 42, d_model: int = 416,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 160):
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
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model: SimpleTransformer, task) -> None:
    V = task.vocab
    sti = task.stoi
    nV = len(V)
    DIG = [sti[str(d)] for d in range(10)]
    L = model.max_seq_len
    D = model.d_model
    A = 40.0

    TOK = 0
    POS = TOK + nV
    CONST = POS + L
    ISPROMPT, POSLE2, POS7 = CONST + 1, CONST + 2, CONST + 3
    PREV = [None] + [POS7 + 1 + nV * (j - 1) for j in range(1, 5)]
    FEAT = PREV[4] + nV
    F4, U, SHORT, MM, Y8 = FEAT, FEAT + 1, FEAT + 2, FEAT + 3, FEAT + 4
    T = FEAT + 5
    GY = [None] + [T + 12 + 10 * (j - 1) for j in range(1, 5)]
    GM1 = GY[4] + 10
    GM2 = GM1 + 10
    GMT = GM2 + 10
    GD1 = GMT + 12
    GD2 = GD1 + 10
    LOGIT = GD2 + 10
    assert LOGIT + nV <= D, (LOGIT + nV, D)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        for i in range(nV):
            model.token_emb.weight[i, TOK + i] = 1.0
        for p in range(L):
            e = model.pos_emb.weight[p]
            e[POS + p] = 1.0
            e[CONST] = 1.0
            e[ISPROMPT] = 1.0 if p < task.prompt_len - 1 else 0.0
            e[POSLE2] = 1.0 if p <= 2 else 0.0
            e[POS7] = 1.0 if p == 7 else 0.0

        at = model.blocks[0].attn
        dh = at.d_head
        for j in range(1, 5):
            h0 = (j - 1) * dh
            for p in range(L):
                if p - j >= 0:
                    at.W_q.weight[h0 + (p - j), POS + p] += A
                at.W_q.weight[h0 + p, POS + p] += A / 2
            for k in range(L):
                at.W_k.weight[h0 + k, POS + k] = 1.0
            for t in range(nV):
                at.W_v.weight[h0 + t, TOK + t] = 1.0
                at.W_o.weight[PREV[j] + t, h0 + t] = 1.0

        mlp = model.blocks[0].mlp
        W1, b1, W2 = mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight
        u = [0]

        def unit(terms, bias, out_dim, out_val=1.0):
            i = u[0]; u[0] += 1
            for dim, w in terms:
                W1[i, dim] += w
            b1[i] = bias
            W2[out_dim, i] += out_val

        def digs(base):
            return [base + d for d in DIG]

        terms = [(ISPROMPT, 1.0)] + [(d, -1.0) for d in digs(TOK)]
        for j in range(1, 5):
            terms += [(d, 1.0) for d in digs(PREV[j])]
        unit(terms, -4.0, F4)
        unit([(TOK + sti['_'], 1.0)] + [(d, 1.0) for d in digs(PREV[1])], -1.0, U)
        # ---- SHORT (run-1 version: requires non-digit at p-3) ----
        terms = [(ISPROMPT, 1.0)] + [(d, -1.0) for d in digs(TOK)] + \
                [(d, 1.0) for d in digs(PREV[1])] + [(d, -1.0) for d in digs(PREV[3])]
        unit(terms, -1.0, SHORT)
        # ----------------------------------------------------------
        unit([(TOK + sti['/'], 1.0), (POSLE2, 1.0), (ISPROMPT, 1.0)], -2.0, MM)
        unit([(TOK + sti['.'], 1.0), (POS7, 1.0), (ISPROMPT, 1.0)], -2.0, MM)
        unit([(TOK + sti['_'], 1.0), (PREV[3] + sti['/'], 1.0)], -1.0, Y8)
        for k, name in enumerate(task.MONTHS):
            a, b, c = name[:3]
            unit([(PREV[2] + sti[a], 1.0), (PREV[1] + sti[b], 1.0), (TOK + sti[c], 1.0),
                  (ISPROMPT, 1.0)], -3.0, T + k)

        at = model.blocks[1].attn
        dh = at.d_head
        h0 = 0
        at.W_q.weight[h0, CONST] = A
        at.W_k.weight[h0, F4] = 2.0
        at.W_k.weight[h0, U] = 1.0
        for j in range(1, 5):
            for d in range(10):
                at.W_v.weight[h0 + 10 * (j - 1) + d, PREV[j] + DIG[d]] = 1.0
                at.W_o.weight[GY[j] + d, h0 + 10 * (j - 1) + d] = 1.0
        h0 = dh
        at.W_q.weight[h0, CONST] = A
        at.W_k.weight[h0, MM] = 1.0
        for k in range(12):
            at.W_k.weight[h0, T + k] = 1.0
        for d in range(10):
            at.W_v.weight[h0 + d, PREV[1] + DIG[d]] = 1.0
            at.W_o.weight[GM1 + d, h0 + d] = 1.0
            at.W_v.weight[h0 + 10 + d, PREV[2] + DIG[d]] = 1.0
            at.W_o.weight[GM2 + d, h0 + 10 + d] = 1.0
        for k in range(12):
            at.W_v.weight[h0 + 20 + k, T + k] = 1.0
            at.W_o.weight[GMT + k, h0 + 20 + k] = 1.0
        h0 = 2 * dh
        at.W_q.weight[h0, CONST] = A
        at.W_k.weight[h0, SHORT] = 1.0
        at.W_k.weight[h0, MM] = -1.0
        at.W_k.weight[h0, Y8] = -1.0
        for d in range(10):
            at.W_v.weight[h0 + d, PREV[1] + DIG[d]] = 1.0
            at.W_o.weight[GD1 + d, h0 + d] = 1.0
            at.W_v.weight[h0 + 10 + d, PREV[2] + DIG[d]] = 1.0
            at.W_o.weight[GD2 + d, h0 + 10 + d] = 1.0

        mlp = model.blocks[1].mlp
        W1, b1, W2 = mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight
        u[0] = 0
        P0 = task.prompt_len - 1
        lg = lambda ch: LOGIT + sti[ch]
        sum_gy3 = [(GY[3] + d, -1.0) for d in range(10)]
        for d in range(10):
            unit([(POS + P0, 1.0), (GY[4] + d, 1.0)] + [(GY[3] + e, 1.0) for e in range(10)],
                 -2.0, lg(str(d)))
        unit([(POS + P0, 1.0)] + sum_gy3, 0.0, lg('2'))
        for d in range(10):
            unit([(POS + P0 + 1, 1.0), (GY[3] + d, 1.0)], -1.0, lg(str(d)))
        unit([(POS + P0 + 1, 1.0)] + sum_gy3, 0.0, lg('0'))
        for d in range(10):
            unit([(POS + P0 + 2, 1.0), (GY[2] + d, 1.0)], -1.0, lg(str(d)))
            unit([(POS + P0 + 3, 1.0), (GY[1] + d, 1.0)], -1.0, lg(str(d)))
        unit([(POS + P0 + 4, 1.0)], 0.0, lg('-'))
        unit([(POS + P0 + 7, 1.0)], 0.0, lg('-'))
        big = [(GMT + 9, 1.0), (GMT + 10, 1.0), (GMT + 11, 1.0)]
        for d in range(10):
            unit([(POS + P0 + 5, 1.0), (GM2 + d, 1.0)], -1.0, lg(str(d)))
        unit([(POS + P0 + 5, 1.0)] + big, -1.0, lg('1'))
        unit([(POS + P0 + 5, 1.0)] + [(GM2 + d, -1.0) for d in range(10)] +
             [(dim, -1.0) for dim, _ in big], 0.0, lg('0'))
        for d in range(10):
            unit([(POS + P0 + 6, 1.0), (GM1 + d, 1.0)], -1.0, lg(str(d)))
        for k in range(12):
            unit([(POS + P0 + 6, 1.0), (GMT + k, 1.0)], -1.0, lg(str((k + 1) % 10)))
        for d in range(10):
            unit([(POS + P0 + 8, 1.0), (GD2 + d, 1.0)], -1.0, lg(str(d)))
        unit([(POS + P0 + 8, 1.0)] + [(GD2 + d, -1.0) for d in range(10)], 0.0, lg('0'))
        for d in range(10):
            unit([(POS + P0 + 9, 1.0), (GD1 + d, 1.0)], -1.0, lg(str(d)))
        assert u[0] <= model.d_ff, u[0]

        for i in range(nV):
            model.head.weight[i, LOGIT + i] = 1.0


model_shorthand_name = "date_parser_2L_markers"
model_description = ("2-layer no-LN hand-wired parser: L1 copies prev 4 tokens and detects year/month/day "
                     "end markers + month trigrams; L2 attends to markers and reads out digits per output position.")


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
    parser.add_argument("--task", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)
    accuracy, _ = evaluate(model, task, n_samples=args.n_samples, seed=args.seed,
                           device=args.device, verbose=args.verbose)
    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{
        "task": args.task, "accuracy": f"{accuracy:.4f}", "status": "",
        "model_shorthand_name": model_shorthand_name, "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)
    print(f"accuracy: {accuracy:.4f}")
