"""
Interpretable transformer for character-level sequence tasks.

Hand-written (untrained) transformer for `edit-distance-8`:
prompt "AAAAAAAA,BBBBBBBB=" -> answer "D" (Levenshtein distance).

Algorithm: one shared block (width 7, no LayerNorm, 2 scalar-valued heads with
3-dim query/key, 3-unit ReLU MLP) applied 9 times, one DP row per pass.

Residual stream layout (d_model = 7):
  0  W   : column value W[l] = D[i][l] - l  (comma = col 0, b_l = col l, '=' = final min)
           first-string tokens start at 16 so they never win a minimum
  1  Y   : countdown  position - pass  (first-string tokens; comma frozen at 0;
           second-string / '=' tokens parked at -100 so they are never selected)
  2  Y2  : Y^2
  3  ux  : letter on the unit ring (a, b, c, d at 90 degree steps)
  4  uy
  5  Z   : scratch: softmin - match (written by attention, cleared by the MLP)
  6  N   : 1 for every token except the comma (constant / "not comma" flag)

Head 1: soft minimum over strictly earlier tokens: scores -4*W_k, value W_k.
Head 2: selects the first-string letter whose countdown is 0 via -32*Y^2 peak,
        +8 on match / -8 on mismatch, against 0 for the comma; value N -> match bit.
MLP:    W <- min(W+1, softmin - match) = W + 1 - relu(W + 1 - Z);
        countdown advance; Z cleared.
Readout at '=': logits_d = 2 d v - d^2 with v = W + 8  (=> round(v)).

Usage:
    uv run interpretable_transformer.py --task edit-distance-8
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
    """Heads with d_qk-dim queries/keys and a single scalar value each.

    Head 0 attends to strictly earlier tokens (position 0 may see itself);
    the other heads use the ordinary causal mask. Scores are not rescaled.
    """

    def __init__(self, d_model: int, n_heads: int, d_qk: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.d_qk = d_qk
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_qk
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, 1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)                      # (B, H, T, T)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        strict = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
        strict[0, 0] = False                                   # avoid an empty row
        mask = torch.stack([strict] + [causal] * (H - 1))      # (H, T, T)
        scores = scores.masked_fill(mask[None], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """One shared block applied n_passes times, no LayerNorm."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 7,
        n_heads: int = 2,
        n_passes: int = 9,
        d_ff: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_passes = n_passes
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.block = Block(d_model, n_heads, d_ff)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for _ in range(self.n_passes):
            h = self.block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

W_, Y_, Y2_, UX, UY, Z_, N_ = range(7)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form from the DP algorithm."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        stoi = task.stoi

        # ---- token embedding: letters on the unit ring ----
        te = model.token_emb.weight
        for i, ch in enumerate("abcd"):
            ang = math.pi / 2 * i
            te[stoi[ch], UX] = round(math.cos(ang))
            te[stoi[ch], UY] = round(math.sin(ang))

        # ---- positional embedding: W init, countdown, not-comma flag ----
        pe = model.pos_emb.weight
        L = 8
        for p in range(model.max_seq_len):
            if p < L:                 # first string: W=16, Y=pos
                pe[p, W_] = 16.0
                pe[p, Y_] = float(p)
                pe[p, Y2_] = float(p * p)
                pe[p, N_] = 1.0
            elif p == L:              # comma: column 0, W=0, Y frozen at 0
                pe[p, N_] = 0.0
            else:                     # columns 1..8 and '=': W=0, never selected
                pe[p, Y_] = -100.0
                pe[p, Y2_] = 1.0e4
                pe[p, N_] = 1.0

        att = model.block.attn
        dk = att.d_qk
        Wq, Wk, Wv, Wo = att.W_q.weight, att.W_k.weight, att.W_v.weight, att.W_o.weight
        # Head 1 (rows 0..dk-1): score = -4 * W_key ; value = W_key ; out -> +Z
        Wq[0, N_] = 4.0
        Wk[0, W_] = -1.0
        Wv[0, W_] = 1.0
        Wo[Z_, 0] = 1.0
        # Head 2 (rows dk..2dk-1): score = 16 u_q.u_k - 8 N_k - 32 Y_k^2 (query const N_q)
        Wq[dk + 0, UX] = 16.0
        Wq[dk + 1, UY] = 16.0
        Wq[dk + 2, N_] = -1.0
        Wk[dk + 0, UX] = 1.0
        Wk[dk + 1, UY] = 1.0
        Wk[dk + 2, Y2_] = 32.0
        Wk[dk + 2, N_] = 8.0
        Wv[1, N_] = 1.0               # first-string letters 1, comma 0
        Wo[Z_, 1] = -1.0              # Z = softmin - match

        mlp = model.block.mlp
        f1w, f1b, f2w, f2b = mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias
        # u1 = relu(W + 1 - Z): W <- W + 1 - u1 = min(W+1, Z)
        f1w[0, W_] = 1.0; f1w[0, Z_] = -1.0; f1b[0] = 1.0
        # u2 = relu(Y + 20) on first-string tokens only (comma/columns gated off)
        f1w[1, Y_] = 1.0; f1w[1, N_] = 100.0; f1b[1] = -80.0
        # u3 = relu(Z + 30) = Z + 30: clears the scratch
        f1w[2, Z_] = 1.0; f1b[2] = 30.0
        # W update
        f2w[W_, 0] = -1.0; f2b[W_] = 1.0
        # countdown: on first-string tokens u1 == 2 exactly (match bit 1, equal W's),
        # so dY = -u1/2 = -1 and dY2 = -2(Y+20) + 20.5*u1 = -2Y + 1; comma: u1=u2=0.
        f2w[Y_, 0] = -0.5
        f2w[Y2_, 1] = -2.0; f2w[Y2_, 0] = 20.5
        # clear Z
        f2w[Z_, 2] = -1.0; f2b[Z_] = 30.0

        # ---- readout at '=': v = W + 8, logit_d = 2 d v - d^2 ----
        hw = model.head.weight
        for c, i in stoi.items():
            if c.isdigit():
                d = int(c)
                hw[i, W_] = 2.0 * d
                hw[i, N_] = 16.0 * d - d * d
            else:
                hw[i, N_] = -1000.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ed8_softmin_dp_v1"
model_description = ("Shared 7-wide block x9 (no LN, 2 scalar heads, 3 ReLU): W=D-l columns, head1 softmin(-4W) "
                     "over earlier tokens, head2 countdown -32Y^2 letter match vs comma, MLP W<-min(W+1,S-M); "
                     "'=' reads W+8 with parabolic digit logits.")


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
