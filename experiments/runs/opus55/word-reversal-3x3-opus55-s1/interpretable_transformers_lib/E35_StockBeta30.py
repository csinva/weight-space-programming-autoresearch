"""
Hand-written transformer for word-reversal-3x3: "abc_def_ghi=" -> "ghi_def_abc".

Stock architecture (LayerNorm, residuals, MLP, multi-head attention) carrying a
pointer circuit. Dims 0-3 hold the token as (c, s, -c, -s) on a 28-point ring;
dims 4-7 hold the address the same way. Every input therefore has zero mean and
fixed energy, so stock LayerNorm is a constant rescale that ln1 undoes exactly.
Prompt slots store r(s); answer positions store -r(source), so q = -a, k = a.
Only head 0 of block 0 acts; every other head, the MLPs and later blocks are 0.
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

# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "E35_StockBeta30"
model_description = "E31 at beta=30: score gap to adjacent keys only 1.2, so neighbours and even distant keys leak heavily. Looking for where the vector average finally fails."
CFG = dict(d_model=64, n_heads=4, n_layers=3, d_ff=256, beta=30.0, gain=100.0)


def source_of(p: int) -> int:
    """Prompt index to read when the token at position p predicts the next char."""
    word, letter = divmod(p - 11, 4)
    return 4 * (2 - word) + letter if letter < 3 else 7 - 4 * word


def balanced(angle: float) -> torch.Tensor:
    """A point on the unit circle next to its negation: zero sum, fixed energy."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([c, s, -c, -s])


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
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        d_model = self.d_model = CFG["d_model"]
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, CFG["n_heads"], CFG["d_ff"]) for _ in range(CFG["n_layers"])])
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

@torch.no_grad()
def stress_test(model: SimpleTransformer, task, n: int = 5000) -> None:
    """Report accuracy on a/z-only prompts (worst case for repeats and scalar
    leakage) and on the full alphabet (needed for ring codes, where a, z and _
    are close together). Validation only; nothing is fitted."""
    for name, pool in [("az", [task.stoi["a"], task.stoi["z"]]), ("all", list(range(26)))]:
        gen = torch.Generator().manual_seed(1)
        letters = torch.tensor(pool)[torch.randint(len(pool), (n, 9), generator=gen)]
        sep, eq = torch.full((n, 1), task.stoi["_"]), torch.full((n, 1), task.stoi["="])
        ids = torch.cat([letters[:, :3], sep, letters[:, 3:6], sep, letters[:, 6:], eq], dim=1)
        target = torch.cat([ids[:, 8:11], sep, ids[:, 4:7], sep, ids[:, :3]], dim=1)
        for _ in range(11):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
        print(f"stress_{name}:     {(ids[:, 12:] == target).all(dim=1).float().mean():.4f}")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    for param in model.parameters():
        param.zero_()
    D, dh = model.d_model, model.blocks[0].attn.d_head

    tokens = torch.stack([balanced(2 * math.pi * c / 28) for c in range(28)])
    model.token_emb.weight[:, :4] = tokens
    model.head.weight[:, :4] = tokens
    for p in range(model.max_seq_len):
        slot = p if p < 11 else source_of(min(p, 21))
        code = balanced(math.pi * slot / 11)
        model.pos_emb.weight[p, 4:8] = code if p < 11 else -code

    attn = model.blocks[0].attn
    attn.W_k.weight[:2, 4:6] = torch.eye(2)                                # key = address
    attn.W_q.weight[:2, 4:6] = -CFG["beta"] * math.sqrt(dh) * torch.eye(2) # query = -address
    attn.W_v.weight[:4, :4] = torch.eye(4)                                 # value = token
    attn.W_o.weight[:4, :4] = CFG["gain"] * torch.eye(4)                   # drown out input token

    for block in model.blocks:
        block.ln1.weight.fill_(1.0)
        block.ln2.weight.fill_(1.0)
    model.final_ln.weight.fill_(1.0)
    model.blocks[0].ln1.weight.fill_(math.sqrt(4 / D))  # undo LayerNorm's constant rescale
    stress_test(model, task)


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
