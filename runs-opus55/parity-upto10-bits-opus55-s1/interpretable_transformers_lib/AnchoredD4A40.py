"""
Interpretable transformer for parity of a left-padded bit string.

Rules: no training of any kind; every weight is written in closed form in
`write_weights()`. Prompt "____1011010=" style -> answer "0"/"1".
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
        d_model: int = 4,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 9,
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


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def put(param: torch.Tensor, values) -> None:
    with torch.no_grad():
        param.copy_(torch.as_tensor(values, dtype=param.dtype).reshape(param.shape))


@torch.no_grad()
def exhaustive_check(model: SimpleTransformer, task) -> None:
    """Verification only: score every possible prompt and report the worst logit margin."""
    L = task.prompt_len - 1
    prompts, targets = [], []
    for n in range(1, L + 1):
        for v in range(2 ** n):
            bits = format(v, f"0{n}b")
            prompts.append("_" * (L - n) + bits + "=")
            targets.append(bits.count("1") % 2)
    ids = torch.tensor([task.encode(p) for p in prompts])
    logits = model(ids)[:, -1, :]
    targets = torch.tensor(targets)
    correct = (logits.argmax(-1) == targets).float().mean().item()
    tgt = logits.gather(1, targets[:, None]).squeeze(1)
    rest = logits.scatter(1, targets[:, None], float("-inf")).max(-1).values
    print(f"EXHAUSTIVE: acc={correct:.6f} over {len(prompts)} prompts, min_margin={(tgt - rest).min().item():.4f}")


def write_weights(model: SimpleTransformer, task) -> None:
    """Smallest LayerNorm-keeping circuit: 4 dims, 1 head, 1 block.

    Every token is (s, -s, A, -A), with signal s and large anchor A = 40.
    The mean is 0 and the anchors fix the std at about A / sqrt(D/2) = S, so LayerNorm
    is nearly the linear map x -> x / S, and the weights below undo that scale.

    token_emb: s = 1 for '1', else 0. pos_emb: 0.
    block attn: q = k = 0 gives uniform attention over the 11 visible tokens; W_v
                reads dim 0 and W_o writes +/- 11 * sigma back, adding the count to
                (dim 0, dim 1), where sigma is the exact std of a '1' token.
    block mlp:  the triangle wave of CountTriangle9, antisymmetric on dims 0 and 1.
    final_ln, head: logit('1') = +S * dim0, logit('0') = -S * dim0.
    """
    torch.manual_seed(0)
    D = model.d_model
    A = 40.0
    S = A / math.sqrt(D / 2)                         # std of an anchor-only token
    sigma = math.sqrt(1 + A * A) / math.sqrt(D / 2)  # std of a '1' token

    for p in model.parameters():
        put(p, torch.zeros_like(p))
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            put(m.weight, torch.ones_like(m.weight))

    emb = torch.zeros(task.vocab_size, D)
    emb[:, 2], emb[:, 3] = A, -A
    emb[task.stoi["1"], 0], emb[task.stoi["1"], 1] = 1.0, -1.0
    put(model.token_emb.weight, emb)

    block = model.blocks[0]
    W_v = torch.zeros(D, D)
    W_v[0, 0] = 1.0
    put(block.attn.W_v.weight, W_v)
    W_o = torch.zeros(D, D)
    W_o[0, 0], W_o[1, 0] = task.prompt_len * sigma, -task.prompt_len * sigma
    put(block.attn.W_o.weight, W_o)

    fc1_w, fc1_b = torch.zeros(model.d_ff, D), torch.zeros(model.d_ff)
    fc2_w, fc2_b = torch.zeros(D, model.d_ff), torch.zeros(D)
    for i, k in enumerate(range(1, 10)):
        fc1_w[i, 0], fc1_b[i] = S, -float(k)
        fc2_w[0, i], fc2_w[1, i] = 2.0 * (-1.0) ** k, -2.0 * (-1.0) ** k
    fc2_b[0], fc2_b[1] = -0.5, 0.5
    put(block.mlp.fc1.weight, fc1_w)
    put(block.mlp.fc1.bias, fc1_b)
    put(block.mlp.fc2.weight, fc2_w)
    put(block.mlp.fc2.bias, fc2_b)

    head = torch.zeros(task.vocab_size, D)
    head[task.stoi["1"], 0], head[task.stoi["0"], 0] = S, -S
    put(model.head.weight, head)
    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "AnchoredD4A40"
model_description = "AnchoredD4 sensitivity sweep: A=40, just above the predicted threshold A > 31.6, testing the drift formula c^3/(2A^2) (predicted margin about 0.19)."


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
