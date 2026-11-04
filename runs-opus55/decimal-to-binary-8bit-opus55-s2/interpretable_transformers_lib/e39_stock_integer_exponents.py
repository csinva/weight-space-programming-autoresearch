"""
Interpretable transformer for decimal-to-binary-8bit: "197=" -> "11000101".

The model is never trained. write_weights() sets every parameter in closed form.
Algorithm: greedy binary expansion, MSB first. At output position p (bit k=10-p)
    r = n - sum(emitted bits * 2^j),   emit 1 iff r >= 2^k.
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
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, d_ff: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
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

@torch.no_grad()
def check_all_inputs(model, task) -> None:
    """Greedy-decode all 256 inputs; a test only, the model never sees the targets."""
    good, margin = 0, float("inf")
    for n in range(256):
        ids = task.encode(f"{n:03d}=")
        for _ in range(task.answer_len):
            logits = model(torch.tensor([ids]))[0, -1]
            top2 = logits.topk(2).values
            margin = min(margin, float(top2[0] - top2[1]))
            ids = ids + [int(logits.argmax())]
        good += task.decode(ids[task.prompt_len:]) == f"{n:08b}"
    print(f"exhaustive: {good}/256, min logit margin {margin:.4f}")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    d = model.d_model
    dh = d // model.blocks[0].attn.n_heads
    rt = math.sqrt(dh)
    V = task.vocab_size
    POS = V                      # dims 0..10 token one-hot, 11..21 position one-hot
    OUT, ZERO = POS + 11, POS + 12
    block, attn = model.blocks[0], model.blocks[0].attn
    te, pe = model.token_emb.weight, model.pos_emb.weight
    te[range(V), range(V)] = 1.0
    pe[range(11), [POS + p for p in range(11)]] = 1.0
    # every input has exactly two 1s, so ln1 with gamma=std, beta=mean is the identity
    mu = 2 / d
    std = math.sqrt((2 * (1 - mu) ** 2 + (d - 2) * mu ** 2) / d + 1e-5)
    for b in model.blocks:       # later blocks stay zero in W_o and fc2: exact pass-through
        b.ln1.weight.fill_(1.0)
        b.ln2.weight.fill_(1.0)  # fc2 stays zero, so every MLP adds nothing
    block.ln1.weight.fill_(std)
    block.ln1.bias.fill_(mu)
    model.final_ln.weight.fill_(1.0)
    value = [float(i) for i in range(10)] + [0.5]      # '=' pads with 2^(k+1) * 1/2 = 2^k
    mode = [-1.0] * 3 + [1.0] * 8
    exponent = [math.log2(100), math.log2(10), 0.0] + [float(11 - p) for p in range(3, 11)]
    for t in range(V):
        attn.W_v.weight[0, t] = 111.0 * value[t]
        attn.W_v.weight[dh, t] = 256.0 * value[t]
    for p in range(11):
        attn.W_q.weight[0, POS + p] = rt * mode[p]                   # head 0: a = n
        attn.W_k.weight[0, POS + p] = LN2 * exponent[p] - M * mode[p]
        attn.W_q.weight[dh, POS + p] = rt * mode[p]                  # head 1: b = S + 2^k
        attn.W_k.weight[dh, POS + p] = LN2 * exponent[p] + M * mode[p]
        attn.W_q.weight[dh + 1, POS + p] = rt * LN2 * (exponent[p] - 8 * mode[p])
    attn.W_k.weight[dh + 1, task.stoi["="]] = 1.0
    attn.W_o.weight[OUT, 0] = 1.0
    attn.W_o.weight[OUT, dh] = -1.0
    # logit('1') = r - 2^k + 1/2 from three zero-sum pieces; the position one-hot supplies the 1/2
    one, zero = task.stoi["1"], task.stoi["0"]
    model.head.weight[one, OUT] = 1.0
    model.head.weight[one, POS:POS + 11] = 0.5
    model.head.weight[one, ZERO] = -6.5
    model.head.weight[zero] = -model.head.weight[one]
    check_all_inputs(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
M = 6.0                      # how strongly a head prefers its own region
LN2 = math.log(2)            # the only irrational scale in the attention scores
model_shorthand_name = "e39_stock_integer_exponents"
model_description = "Unmodified default architecture with the final conventions: gain in W_v, so attention outputs n and S+2^k, and exponents stored as integers."


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
