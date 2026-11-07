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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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

def write_weights(model: SimpleTransformer, task) -> None:
    """The circuit "x0 o1 x1 o2 x2 o3 x3 o4 x4 =" is evaluated strictly left to right.

    Residual stream slots (bits are bipolar: value 0 -> (+A,-A), value 1 -> (-A,+A)):
      D(0,1)=digit bit, OP(2..4)=one-hot of & | ^, POS(5..15)=one-hot position,
      CONST(16,17)=+-K (pins the LayerNorm scale), S_1..S_3 (18..23)=running value,
      OPin (24..26)=operator copied from p-1, OPin4 (27..29)=operator at p-2 (last block),
      Spv_l (30..37)=S_{l-1} copied from p-2 (p-3 in last block), DIN (38,39), FIN (40,41)=answer.
    Block l=1..3 (at digit position p=2l): head1 fetches S_{l-1}[p-2] (S_0 = the digit x0 itself),
      block 1's head0 fetches the operator at p-1; the MLP has one ReLU AND-neuron per
      (S_prev, op, digit) triple (12) and writes S_l = op(S_prev, digit). Only the diagonal
      S_1[2] -> S_2[4] -> S_3[6] is meaningful; other positions hold junk that is never read.
    Block 4 (at '='): head1 fetches S_3[6] (p-3), head0 attends p-1 and p-2 at once and fetches
      the last digit and the last operator (disjoint slots); the MLP applies the final op into FIN.
    """
    # K dominates the LayerNorm scale; a one-hot bipolar entry of height A has LN'd amplitude
    # a ~ 0.95 (A/K ~ 0.233 matters: 0.2 and 0.25 both break the single-ReLU gates).
    A, K, c, a0 = 7.0, 30.0, 40.0, 1.0
    D, OP, POS, CK = 0, 2, 5, 16

    def Sd(l): return 0 if l == 0 else 18 + 2 * (l - 1)      # S_1..S_3 (18..23)
    def OPin(l): return 27 if l == 4 else 24                  # op at p-1 (blocks 1-3, written once) / p-2 (block 4)
    def Spv(l): return 30 + 2 * (l - 1)                       # S_{l-1} copy (30..37)
    DIN, FIN = 38, 40
    dh = model.d_model // model.n_heads
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        st = task.stoi
        for ch in "01":  # bipolar bit code: value v -> (+A,-A) if v==0 else (-A,+A); keeps LN mean ~0
            model.token_emb.weight[st[ch], D + int(ch)] = A
            model.token_emb.weight[st[ch], D + 1 - int(ch)] = -A
        for k, ch in enumerate("&|^"):
            model.token_emb.weight[st[ch], OP + k] = A
        for i in range(model.max_seq_len):
            if i <= 10:
                model.pos_emb.weight[i, POS + i] = A
            model.pos_emb.weight[i, CK] = K
            model.pos_emb.weight[i, CK + 1] = -K
        for ln in [model.final_ln] + [x for b in model.blocks for x in (b.ln1, b.ln2)]:
            ln.weight.fill_(1.0)

        def head_shift(attn, h, shift, src, dst):
            """Head h at position p attends (equally) to position(s) p-shift; copies src->dst slots."""
            shifts = [shift] if isinstance(shift, int) else list(shift)
            for j in range(11):
                for sh in shifts:
                    if j + sh <= 10:
                        attn.W_q.weight[h * dh + j, POS + j + sh] = c
                attn.W_k.weight[h * dh + j, POS + j] = 1.0
            for m, (s_, d_) in enumerate(zip(src, dst)):
                attn.W_v.weight[h * dh + m, s_] = len(shifts) * A / a0   # compensate softmax averaging
                attn.W_o.weight[d_, h * dh + m] = 1.0

        def op_fn(o, s, d):
            return [s & d, s | d, s ^ d][o]

        for l in range(1, 5):
            blk = model.blocks[l - 1]
            last = (l == 4)  # last block runs at '=' and applies the final operator directly
            Dsrc, out = (DIN, FIN) if last else (D, Sd(l))
            if last:
                # head 0 attends p-1 and p-2 at once: the digit and operator tokens occupy disjoint
                # slots, so one head fetches both (digit -> DIN, operator -> OPin)
                head_shift(blk.attn, 0, (1, 2), [OP, OP + 1, OP + 2, D, D + 1],
                           [OPin(l), OPin(l) + 1, OPin(l) + 2, DIN, DIN + 1])
                head_shift(blk.attn, 1, 3, [Sd(l - 1), Sd(l - 1) + 1], [Spv(l), Spv(l) + 1])
            else:
                if l == 1:  # the operator at p-1 is the same for blocks 1-3: copy once, reuse
                    head_shift(blk.attn, 0, 1, [OP, OP + 1, OP + 2], [OPin(l), OPin(l) + 1, OPin(l) + 2])
                head_shift(blk.attn, 1, 2, [Sd(l - 1), Sd(l - 1) + 1], [Spv(l), Spv(l) + 1])
            n = 0

            def bit(slot, v):
                """Linear read-out of 'bit slot == v': +-a depending on match."""
                return [(slot + v, 0.5), (slot + 1 - v, -0.5)]

            def gate(n, inputs, r, theta, eps):
                """One ReLU AND-neuron: relu(sum(inputs) - theta); fires only when all inputs match
                (sum ~ 4a) and writes bit value r (bipolar, height ~A) into the output slot."""
                for i, w in inputs:
                    blk.mlp.fc1.weight[n, i] = w
                blk.mlp.fc1.bias[n] = -theta
                # single ReLU: output ~ A when the gate input is ~ (theta + eps)
                blk.mlp.fc2.weight[out, n] = (A / eps) * (1 - 2 * r)
                blk.mlp.fc2.weight[out + 1, n] = -(A / eps) * (1 - 2 * r)

            for o in range(3):
                for s_ in range(2):
                    for d in range(2):
                        # op counts double: wrong op -> <=2a, all match -> 4a
                        gate(n, [(OPin(l) + o, 2.0)] + bit(Spv(l), s_) + bit(Dsrc, d), op_fn(o, s_, d), 2.4, 1.4)
                        n += 1
        model.head.weight[st["0"], FIN] = 1.0
        model.head.weight[st["1"], FIN + 1] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "chain4L_final_v2"
model_description = "Final LTR chain (A=7,K=30,c=40, theta=2.4, eps=1.4): 4 blocks, d_model=42, 2 heads, 12 single-ReLU lookup neurons/block; clean docstring."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, d_model=42, n_heads=2, n_layers=4, d_ff=12)
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
