"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
        # LayerNorm removed (identity): keeps one-hot features exact so the hand-built circuit is exact.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """5-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 42,
        n_heads: int = 1,
        n_layers: int = 5,
        d_ff: int = 6,
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
        self.final_ln = nn.Identity()
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

# Knobs for diagnostic ablations (defaults = the working circuit). Changing one knob at a time.
KNOBS = dict(
    beta=100.0,          # attention sharpness: score on a position match, before 1/sqrt(d_head)
    gate_pos=True,       # include pos==2L as an input of the MLP AND-gates
    copy_op=True,        # attention copies the operator token into o_L
    copy_x0=True,        # layer 1 copies the raw bit x0 into g_1
    one_bias=True,       # output bias via the constant ONE dim (logit0 = 1 - g5)
    drop_units=(),       # indices of MLP gate units to delete (0 AND, 1 OR-a, 2 OR-b, 3 OR-c, 4 XOR-a, 5 XOR-b)
    out_scale=1.5,       # MLP output weight per firing gate (2.0 -> value 1 after the 1/2 attention averaging)
)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Hand-built left-to-right evaluator for boolean chains like "1|0&1^1=".
    Sequence: idx 0..8 = x0 op1 x1 op2 x2 op3 x3 op4 x4, idx 9 = '='. Answer is the pure
    left-to-right fold (((x0 op1 x1) op2 x2) ...), checked on 3000 generated examples.

    Residual stream (d_model=42):
      dims  0..5   TOK  one-hot token id ('0','1','&','|','^','=')
      dims  6..21  POS  one-hot position
      dim   22     ONE  constant 1 (for the output bias)
      dim   23     S    running value: S_L = result after operand L (a single bit)
      per layer L: g_L (1 dim, gathered previous value), o_L (3 dims, gathered operator)

    Layer L=1..4 writes S_L at operand position 2L:
      head 0 attends to i-2 and copies S_{L-1} into g_L (layer 1 copies the raw bit x0);
      head 1 attends to i-1 and copies the operator token into o_L;
      MLP (6 ReLU units, one per AND/OR/XOR case that outputs 1) gates on op, (prev, x), and pos==2L.
    Layer 5 at position 9: head 1 copies S_4 from position 8 into g_5; the output reads g_5.
    """
    torch.manual_seed(0)
    stoi = task.stoi
    assert task.vocab_size == 6 and model.n_heads == 1 and model.n_layers == 5 and model.d_ff == 6
    D = model.d_model
    dh = D // model.n_heads  # 42 (single head)
    TOK, POS, ONE, S = 0, 6, 22, 23
    B0 = TOK + stoi["0"]
    B1 = TOK + stoi["1"]
    AND_D = TOK + stoi["&"]
    OR_D = TOK + stoi["|"]
    XOR_D = TOK + stoi["^"]
    BETA = KNOBS["beta"] * math.sqrt(dh)  # q.k = BETA on a position match -> attention score = KNOBS["beta"]

    def slot(L):
        base = 24 + 4 * (L - 1)
        return base, base + 1  # (g_L dim, o_L first dim; o_L = o..o+2)

    # MLP units: each is a list of (input dim, weight). A unit fires iff all +1 inputs are 1
    # and all -1 inputs are 0. We use bias = -(n_plus - 0.5) and output weight 2 -> value 1 on match.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(model.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1.0
            model.token_emb.weight[t, ONE] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS + p] = 1.0

        for L in range(1, 6):
            blk = model.blocks[L - 1]
            attn = blk.attn
            g, o = slot(L)
            # One head: query at i matches both i-1 and i-2 (score BETA each), so the softmax
            # splits 1/2 each. Values are scaled by 2 in W_o to undo the 1/2 weighting.
            for s in (1, 2):
                for k in range(16 - s):
                    attn.W_q.weight[k, POS + k + s] = BETA
            for k in range(16):
                attn.W_k.weight[k, POS + k] = 1.0
            vg, vo = 16, 17  # value dims: gathered previous value (g), gathered operator (o)
            if L == 1:
                if KNOBS["copy_x0"]:
                    attn.W_v.weight[vg, B1] = 1.0  # x0 == '1' (only operand positions carry B1)
            else:
                attn.W_v.weight[vg, S] = 1.0  # previous running value (op positions carry S = 0)
            attn.W_o.weight[g, vg] = 2.0
            if L <= 4 and KNOBS["copy_op"]:
                for b, src in enumerate([AND_D, OR_D, XOR_D]):
                    attn.W_v.weight[vo + b, src] = 1.0
                    attn.W_o.weight[o + b, vo + b] = 2.0

            if L <= 4:
                mlp = blk.mlp
                pos_dim = POS + 2 * L
                AND_o, OR_o, XOR_o = o, o + 1, o + 2
                units = [
                    # AND: prev=1, x=1
                    [(g, 1.0), (B1, 1.0), (AND_o, 1.0), (pos_dim, 1.0)],
                    # OR: (prev=1, x=0), (prev=0, x=1), (prev=1, x=1)
                    [(g, 1.0), (B0, 1.0), (OR_o, 1.0), (pos_dim, 1.0)],
                    [(g, -1.0), (B1, 1.0), (OR_o, 1.0), (pos_dim, 1.0)],
                    [(g, 1.0), (B1, 1.0), (OR_o, 1.0), (pos_dim, 1.0)],
                    # XOR: (prev=1, x=0), (prev=0, x=1)
                    [(g, 1.0), (B0, 1.0), (XOR_o, 1.0), (pos_dim, 1.0)],
                    [(g, -1.0), (B1, 1.0), (XOR_o, 1.0), (pos_dim, 1.0)],
                ]
                for u, unit in enumerate(units):
                    if u in KNOBS["drop_units"]:
                        continue
                    unit = [(d, w) for d, w in unit if KNOBS["gate_pos"] or d != pos_dim]
                    n_plus = sum(1 for _, w in unit if w > 0)
                    for dim, w in unit:
                        mlp.fc1.weight[u, dim] = w
                    mlp.fc1.bias[u] = -(n_plus - 0.5)
                    mlp.fc2.weight[S, u] = KNOBS["out_scale"]

        # Output: logit('1') = g_5, logit('0') = 1 - g_5.
        g5, _ = slot(5)
        model.head.weight[stoi["1"], g5] = 1.0
        model.head.weight[stoi["0"], g5] = -1.0
        if KNOBS["one_bias"]:
            model.head.weight[stoi["0"], ONE] = 1.0

    # Self-test (diagnostic only, does not touch the model): all 2592 expressions, plain left-to-right.
    import itertools
    cases, answers = [], []
    for ops in itertools.product(["&", "|", "^"], repeat=4):
        for bits in itertools.product([0, 1], repeat=5):
            s, v = str(bits[0]), bits[0]
            for i, op in enumerate(ops):
                s += op + str(bits[i + 1])
                b = bits[i + 1]
                v = (v & b) if op == "&" else (v | b) if op == "|" else (v ^ b)
            cases.append([stoi[c] for c in s + "="])
            answers.append(v)
    with torch.no_grad():
        pred = model(torch.tensor(cases)).argmax(-1)[:, 9].tolist()
    print(f"SELFTEST exhaustive: {sum(p == a for p, a in zip(pred, answers))}/{len(answers)}")
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "scan5_1head_outscale1p5"
model_description = "scan5_1head with the MLP output weight 1.5: fails (0.65). A firing gate writes 1.5, which the next layer reads as prev=1.5 and the fixed AND/OR/XOR thresholds mis-fire."


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
