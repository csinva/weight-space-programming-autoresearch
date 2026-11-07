"""
Interpretable transformer for 5-digit addition (snapshot: D54_Final_gam2_mask30).

Prompt "AAAAA+BBBBB=" (pos 0-11), answer = 6 digits MSB first (pos 12-17).
Output digit k is predicted at pos 11+k; digit 0 = final carry, digit k>=1 is column i=k-1.
Circuit (2 layers, d_model=54, 2 heads, 48k params):
  L1 attn : B-digit position 6+j fetches a_j from position j (offset -6)
  L1 MLP  : 100-entry (a,b) table -> (a+b)%10 one-hot, generate flag (a+b>=10), propagate flag (a+b==9)
  L2 attn : head0 -> carry = generate flag of nearest non-propagate column j>=k (to the right)
            head1 -> fetch (a+b)%10 of column k-1 (offset -6)
  L2 MLP  : digit = (s + carry) % 10
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
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 54,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 100,
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


CFG = dict(name="D54_Final_gam2_mask30", vg=0.28, vs=0.27, t1=9.2, w1=0.55, gam=2.0, mask=30.0, bp=30.0, plus_trick=True, use_mask=True)


def write_weights(model: SimpleTransformer, task) -> None:
    # Residual layout (d=54): 0-11 token one-hot | 12-29 position one-hot | 30-39 fetched a digit
    # (reused later as output-digit logits) | 40-49 (a+b)%10 one-hot | 50 generate flag (a+b>=10)
    # 51 propagate flag (a+b==9) | 52 carry
    for p in model.parameters():
        nn.init.zeros_(p)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
    TOK, POS, A0, S0, G, P, C, O0 = 0, 12, 30, 40, 50, 51, 52, 30  # logits share dims with fetched-a
    npos = min(model.max_seq_len, 18)
    dh = model.d_model // model.n_heads
    with torch.no_grad():
        E = model.token_emb.weight
        for t in range(task.vocab_size):
            E[t, TOK + t] = 1.0
        if CFG["plus_trick"]:
            E[10, S0] = 1.0  # '+' token: "column -1" has sum-digit 0 (so final carry = 0 + carry)
        for p in range(npos):
            model.pos_emb.weight[p, POS + p] = 1.0

        # ---- Layer 1 attention: B-digit position 6+j fetches a_j from position j (offset -6)
        at = model.blocks[0].attn
        for j in range(5):
            at.W_q.weight[j, POS + 6 + j] = 3.0
            at.W_k.weight[j, POS + j] = 1.0
        for a in range(10):
            at.W_v.weight[a, TOK + a] = 1.0 / 5.57  # undo LayerNorm scale -> a_j ~ 1.0
            at.W_o.weight[A0 + a, a] = 1.0
        # ---- Layer 1 MLP: 100-entry table (a,b) -> (a+b)%10, generate, propagate; only at B positions
        mlp = model.blocks[0].mlp
        for a in range(10):
            for b in range(10):
                u = a * 10 + b
                mlp.fc1.weight[u, A0 + a] = 1.0
                mlp.fc1.weight[u, TOK + b] = 1.0
                for j in range(5):
                    mlp.fc1.weight[u, POS + 6 + j] = 1.0
                mlp.fc1.bias[u] = -CFG["t1"]
                mlp.fc2.weight[S0 + (a + b) % 10, u] = CFG["w1"]
                if a + b >= 10:
                    mlp.fc2.weight[G, u] = CFG["w1"]
                if a + b == 9:
                    mlp.fc2.weight[P, u] = CFG["w1"]

        # ---- Layer 2 attention
        at = model.blocks[1].attn
        gam, mask, bp = CFG["gam"], CFG["mask"], CFG["bp"]
        # head 0: carry = generate flag of the nearest non-propagate column j >= k (k = digit index)
        for j in range(5):
            at.W_k.weight[j, POS + 6 + j] = 1.0
        at.W_k.weight[5, P] = 1.0          # propagate flag
        for k in range(6):
            q = POS + 11 + k
            if 11 + k >= npos:
                break
            for j in range(5):
                at.W_q.weight[j, q] = -mask if (j < k and CFG["use_mask"]) else gam * (5 - j)
            # every non-column key scores 0 and carries no generate flag -> carry 0
            at.W_q.weight[5, q] = -bp
        at.W_v.weight[0, G] = CFG["vg"]
        at.W_o.weight[C, 0] = 1.0
        # head 1: fetch (a+b)%10 one-hot of column k-1 (position 5+k, offset -6)
        for k in range(6):
            if 11 + k >= npos:
                break
            at.W_q.weight[dh + k, POS + 11 + k] = 3.0
            at.W_k.weight[dh + k, POS + 5 + k] = 1.0
        for m in range(10):
            at.W_v.weight[dh + m, S0 + m] = CFG["vs"]
            at.W_o.weight[S0 + m, dh + m] = 1.0
        # ---- Layer 2 MLP: digit = (s + carry) % 10
        mlp = model.blocks[1].mlp
        for m in range(10):
            mlp.fc1.weight[m, S0 + m] = 1.0
            mlp.fc1.weight[m, C] = -1.0
            mlp.fc1.bias[m] = -2.0
            mlp.fc2.weight[O0 + m, m] = 0.5
            mlp.fc1.weight[10 + m, S0 + m] = 1.0
            mlp.fc1.weight[10 + m, C] = 1.0
            mlp.fc1.bias[10 + m] = -5.8
            mlp.fc2.weight[O0 + (m + 1) % 10, 10 + m] = 0.5
        for d in range(10):
            model.head.weight[d, O0 + d] = 1.0


model_shorthand_name = CFG["name"]
model_description = ("2-layer: L1 attn fetches a_j to B_j, MLP table gives (a+b)%10/generate/propagate; "
                     "L2 attn finds nearest non-propagate column to the right (carry) and fetches column sum; MLP adds mod 10.")


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
