"""
Hand-written transformer for word-reversal-3x3: "abc_def_ghi=" -> "ghi_def_abc".

Stock blocks (LayerNorm, attention, residual, MLP) carrying a readable circuit.
Each position has an address: word (0, 1, 2, or none for separators), letter
slot (0, 1, 2, separator) and phase (+1 prompt, -1 answer). W_q is the whole
program: mirror the word, keep the slot, flip the phase. So answer position
(word w, slot l) attends to prompt position (word 2-w, slot l). Separator
queries carry no word and match every prompt separator, all of which are '_'.

All codes are centred with fixed energy, so stock LayerNorm is a constant
rescale that ln1 undoes exactly. The MLP is zero and unused.
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
model_shorthand_name = "E37_StockWordLetterRing"
model_description = "Stock blocks with the readable word/slot/phase address: W_q is mirror-word, keep-slot, flip-phase. Ring tokens, d_model 14, 1 head, 1 layer, d_ff 1."
CFG = dict(token="ring", n_layers=1, d_ff=1, beta=20.0, gain=100.0)
D_ADDRESS = 10  # word(4) + slot(4) + phase pair(2)


def token_codes() -> torch.Tensor:
    """Zero-sum, fixed-energy token codes; the head decodes by dot product."""
    if CFG["token"] == "onehot":
        return torch.eye(28) - 1 / 28
    angle = 2 * math.pi * torch.arange(28) / 28
    return torch.stack([angle.cos(), angle.sin(), -angle.cos(), -angle.sin()], dim=1)


def address_code(p: int) -> torch.Tensor:
    prompt = p < 11
    word, slot = divmod(p if prompt else p - 11, 4)
    code = torch.zeros(D_ADDRESS)
    code[3 if slot == 3 else word] = 1  # category 3 means "no word"
    code[4 + slot] = 1
    code[:8] -= 0.25
    code[8:] = torch.tensor([1.0, -1.0] if prompt else [-1.0, 1.0])
    return code


def query_map() -> torch.Tensor:
    """The program: mirror the word, keep the slot, flip the phase."""
    M = torch.eye(D_ADDRESS)
    M[:3] = M[[2, 1, 0]]
    M[8:, 8:] *= -1
    return M


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
        d_model = self.d_model = token_codes().shape[1] + D_ADDRESS
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, 1, CFG["d_ff"]) for _ in range(CFG["n_layers"])])
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
    D = model.d_model
    tokens = token_codes()
    dt = tokens.shape[1]
    model.token_emb.weight[:, :dt] = tokens
    model.head.weight[:, :dt] = tokens
    for p in range(model.max_seq_len):
        model.pos_emb.weight[p, dt:] = address_code(p)

    attn = model.blocks[0].attn
    attn.W_k.weight[:D_ADDRESS, dt:] = torch.eye(D_ADDRESS)                     # key = address
    attn.W_q.weight[:D_ADDRESS, dt:] = CFG["beta"] * math.sqrt(D) * query_map() # query = program
    attn.W_v.weight[:dt, :dt] = torch.eye(dt)                                   # value = token
    attn.W_o.weight[:dt, :dt] = CFG["gain"] * torch.eye(dt)                     # outvote input token

    for block in model.blocks:
        block.ln2.weight.fill_(1.0)
    model.final_ln.weight.fill_(1.0)
    energy = tokens[0].pow(2).sum() + address_code(0).pow(2).sum()
    model.blocks[0].ln1.weight.fill_(math.sqrt(energy / D))  # undo LayerNorm's constant rescale
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
