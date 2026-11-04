"""
Hand-written transformer for word-reversal-3x3: "abc_def_ghi=" -> "ghi_def_abc".

Every answer character is a copy of one prompt character, so the circuit is a
single attention layer acting as a pointer: the position decides WHERE to read
(query), prompt positions are addresses (keys), and values carry the token.
No training anywhere: all weights are written in closed form.
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

# Experiment knobs. dt = token dims, dp = position dims, d_model = dt + dp.
CFG = dict(token="onehot", position="onehot", n_layers=3, d_ff=256, use_ln=True,
           residual=True, gain=10.0, beta=20.0)


def source_of(p: int) -> int:
    """Prompt index to read when the token at position p predicts the next char."""
    word, letter = divmod(p - 11, 4)
    return 4 * (2 - word) + letter if letter < 3 else 7 - 4 * word


def token_codes() -> torch.Tensor:
    """Rows are token vectors; the head decodes by dot product with these rows."""
    if CFG["token"] == "onehot":
        return torch.eye(28)
    if CFG["token"] == "binary":
        bits = torch.tensor([[(c >> b) & 1 for b in range(5)] for c in range(28)], dtype=torch.float32)
        return (2 * bits - 1) / math.sqrt(5)
    angle = 2 * math.pi * torch.arange(28) / 28  # "ring"
    return torch.stack([angle.cos(), angle.sin()], dim=1)


def position_codes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (P, M_k, M_q): position table, key map and query map over position dims."""
    if CFG["position"] == "onehot":
        M_q = torch.zeros(23, 23)
        for p in range(11, 22):
            M_q[source_of(p), p] = 1
        return torch.eye(23), torch.eye(23), M_q
    raise ValueError(CFG["position"])


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


def norm(d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model) if CFG["use_ln"] else nn.Identity()


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = norm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = norm(d_model)
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.ln1(x))
        x = x + a if CFG["residual"] else a
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        d_model = token_codes().shape[1] + position_codes()[0].shape[1]
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, 1, CFG["d_ff"]) for _ in range(CFG["n_layers"])])
        self.final_ln = norm(d_model)
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

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    for param in model.parameters():
        param.zero_()
    TOK = token_codes()
    P, M_k, M_q = position_codes()
    dt, D = TOK.shape[1], model.d_model
    model.token_emb.weight[:, :dt] = TOK
    model.pos_emb.weight[:, dt:] = P
    model.head.weight[:, :dt] = TOK

    # Block 0 is the pointer; any later blocks are left zeroed, i.e. identity.
    attn = model.blocks[0].attn
    attn.W_k.weight[dt:, dt:] = M_k
    attn.W_q.weight[dt:, dt:] = CFG["beta"] * math.sqrt(D) * M_q
    attn.W_v.weight[:dt, :dt] = torch.eye(dt)
    attn.W_o.weight[:dt, :dt] = CFG["gain"] * torch.eye(dt)

    if CFG["use_ln"]:
        # A one-hot token plus a one-hot position has a fixed mean and std, so
        # LayerNorm can be set to undo itself exactly on every input.
        mean = 2 / D
        std = math.sqrt(2 / D - mean ** 2)
        model.blocks[0].ln1.weight.fill_(std)
        model.blocks[0].ln1.bias.fill_(mean)
        for ln in [model.blocks[0].ln2, model.final_ln, *[n for b in model.blocks[1:] for n in (b.ln1, b.ln2)]]:
            ln.weight.fill_(1.0)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "E03_OneHotPtr3L"
model_description = "Default shape: 3 layers with LN and MLP; only block 0 acts. One-hot token + one-hot position; W_q maps position to source address. LN calibrated to identity, MLP zeroed."


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
