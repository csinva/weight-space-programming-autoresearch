"""
Interpretable transformer for parity of up to 10 bits (weights hand-written, never trained).

Prompt "____101011=" (left-padded with '_') -> answer "0"/"1", the XOR of the bits.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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

# Architecture configuration for this attempt.
D_MODEL, N_HEADS, D_FF = 6, 1, 1
HAS_ATTN = [True] * 10   # per layer: does this block have attention?
HAS_MLP = [False] * 10   # per layer: does this block have an MLP?
USE_LN = False
USE_POS = True


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
    return nn.LayerNorm(d_model) if USE_LN else nn.Identity()


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, has_attn: bool, has_mlp: bool):
        super().__init__()
        self.ln1 = norm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads) if has_attn else None
        self.ln2 = norm(d_model)
        self.mlp = MLP(d_model, d_ff) if has_mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL) if USE_POS else None
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, D_FF, a, m) for a, m in zip(HAS_ATTN, HAS_MLP)]
        )
        self.final_ln = norm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        h = self.token_emb(ids)
        if self.pos_emb is not None:
            h = h + self.pos_emb(torch.arange(T, device=ids.device))[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def exhaustive_check(model: SimpleTransformer, task) -> None:
    """Print accuracy over ALL 2047 possible prompts (read-only sanity check, no fitting)."""
    prompts, targets = [], []
    for n in range(11):
        for bits in itertools.product("01", repeat=n):
            prompts.append("_" * (10 - n) + "".join(bits) + "=")
            targets.append(bits.count("1") % 2)
    ids = torch.tensor([task.encode(p) for p in prompts])
    with torch.no_grad():
        pred = model(ids)[:, -1].argmax(-1)
    print(f"EXHAUSTIVE {(pred == torch.tensor(targets)).sum().item()}/{len(targets)}")


def write_weights(model: SimpleTransformer, task) -> None:
    """Attention-only sign flips: '=' holds a spin; block t negates it iff bit t is 1."""
    KEY, SPIN, ONE, J, J2, EQ = range(6)
    SIGN_GAP = 15.0    # softmax margin for "bit is 1" vs "bit is 0"
    LOCATE = 40.0      # how strongly block t restricts attention to {position t, '='}
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.token_emb.weight[task.stoi["0"], KEY] = 1.0
        model.token_emb.weight[task.stoi["_"], KEY] = 1.0
        model.token_emb.weight[task.stoi["1"], KEY] = -1.0
        model.token_emb.weight[task.stoi["="], SPIN] = 1.0    # the spin lives only on '='
        model.token_emb.weight[task.stoi["="], EQ] = 1.0
        for j in range(11):
            model.pos_emb.weight[j, ONE] = 1.0
            model.pos_emb.weight[j, J] = float(j)
            model.pos_emb.weight[j, J2] = float(j * j)
        for t, blk in enumerate(model.blocks):
            a = blk.attn
            # score(position t) = LOCATE t^2 + SIGN_GAP KEY_t,  score('=') = LOCATE t^2.
            # So '=' looks at bit t when it is 0 and at itself when it is 1.
            a.W_q.weight[0, ONE] = SIGN_GAP
            a.W_k.weight[0, KEY] = 1.0
            a.W_q.weight[1, ONE] = 2.0 * t * LOCATE
            a.W_k.weight[1, J] = 1.0
            a.W_q.weight[2, ONE] = LOCATE
            a.W_k.weight[2, J2] = -1.0
            a.W_q.weight[3, ONE] = LOCATE * (10 - t) ** 2   # lifts '=' to the same height as position t
            a.W_k.weight[3, EQ] = 1.0
            # Value -2 * spin: 0 if it looked at bit t (no spin there), -2 spin if it looked at itself.
            a.W_v.weight[SPIN, SPIN] = -2.0
            a.W_o.weight[SPIN, SPIN] = 1.0
        model.head.weight[task.stoi["0"], SPIN] = 1.0
        model.head.weight[task.stoi["1"], SPIN] = -1.0
    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "spin_flip_noMLP"
model_description = "Attention only, 10 blocks: '=' carries a spin; block t compares bit t's key with its own so that it self-attends (and negates via W_v=-2) iff the bit is 1."


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
