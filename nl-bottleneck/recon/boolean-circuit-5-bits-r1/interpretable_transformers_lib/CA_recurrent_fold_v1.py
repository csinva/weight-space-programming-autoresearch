"""
Interpretable transformer for character-level sequence tasks.

Recurrent "cellular automaton" transformer for boolean-circuit-5-bits.
No training: every parameter is set in closed form in `write_weights()`.

Usage:
    uv run interpretable_transformer.py --task boolean-circuit-5-bits
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
# Each position holds one scalar state s = S - 0.5 (S = expression value so far).
# One tied block (1 attention head, 2-neuron ReLU MLP) is applied N_PASSES times.
# Per pass, cell i builds h_i = emb_i + Attn(previous cell) in channels [b, A, X],
# then the MLP folds h_i into the new state of cell i.

N_PASSES = 10


class CARecurrentTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16, n_passes: int = N_PASSES):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_passes = n_passes
        # token embedding into [b, A, X]
        self.token_emb = nn.Embedding(vocab_size, 3)
        # fixed sinusoidal positions, 30 deg per step (buffer, not a parameter)
        ang = torch.arange(max_seq_len, dtype=torch.float32) * (math.pi / 6)
        self.register_buffer("pos", torch.stack([torch.cos(ang), torch.sin(ang)], -1))
        # query map on positions (no key matrix: keys are the raw positions)
        self.W_q = nn.Linear(2, 2, bias=False)
        # value map: [s, A, X] of attended cell -> [b, A, X]
        self.W_v = nn.Linear(3, 3, bias=False)
        # MLP: [b, A, X] -> 2 ReLU -> new centred state
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)
        # initial centred state
        self.s0 = nn.Parameter(torch.zeros(1))
        # readout: state -> logits
        self.head = nn.Linear(1, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        E = self.token_emb(ids)                       # (B,T,3) [b,A,X]
        p = self.pos[:T]                              # (T,2)
        q = self.W_q(p)                               # (T,2)
        scores = q @ p.t()                            # (T,T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
        s = self.s0.expand(B, T).clone()              # (B,T)
        for _ in range(self.n_passes):
            v_in = torch.cat([s[..., None], E[..., 1:]], -1)   # [s, A, X]
            v = self.W_v(v_in)                                 # -> [b, A, X]
            h = E + attn[None] @ v                             # own embedding + prev cell
            s = self.fc2(F.relu(self.fc1(h)))[..., 0]
        return self.head(s[..., None])


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model, task) -> None:
    st = task.stoi
    with torch.no_grad():
        # embeddings [b, A, X]
        E = torch.zeros(task.vocab_size, 3)
        E[st["1"], 0] = 1.0
        E[st["&"]] = torch.tensor([1.0, 1.0, 0.0])   # AND flag + identity bit 1
        E[st["|"]] = torch.tensor([0.0, 0.0, 0.0])   # OR: no flags, identity 0
        E[st["^"]] = torch.tensor([0.0, 0.0, 1.0])   # XOR flag, identity 0
        # '0' and '=' have zero embedding
        model.token_emb.weight.copy_(E)
        # query = position rotated back one step (-30 deg), scaled by 200
        th = -math.pi / 6
        R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        model.W_q.weight.copy_(200.0 * R)
        # values: previous state -> bit channel, previous flags -> flags
        model.W_v.weight.copy_(torch.eye(3))
        # MLP fold step on u = b + s (s centred in {-0.5, 0.5}):
        #   h1 = relu(u - A), h2 = relu(2u + X/2 - A - 2); s' = 2(h1 - h2) - 1/2
        #   no flags -> OR, A -> AND, X -> XOR
        model.fc1.weight.copy_(torch.tensor([[1.0, -1.0, 0.0], [2.0, -1.0, 0.5]]))
        model.fc1.bias.copy_(torch.tensor([0.0, -2.0]))
        model.fc2.weight.copy_(torch.tensor([[2.0, -2.0]]))
        model.fc2.bias.copy_(torch.tensor([-0.5]))
        # state starts at S = 0 (centred -0.5)
        model.s0.fill_(-0.5)
        # readout: logits = +-4 * centred state
        W = torch.zeros(task.vocab_size, 1)
        W[st["1"], 0] = 4.0
        W[st["0"], 0] = -4.0
        model.head.weight.copy_(W)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "CA_recurrent_fold_v1"
model_description = ("Recurrent cellular-automaton transformer: scalar centred state per cell, one tied block "
                     "(keyless 30deg sinusoidal prev-cell attention x200, 2-ReLU fold MLP) applied 10 times; "
                     "op cells carry flags+identity bit; readout +-4*state.")


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



def build_model(task):
    max_seq_len = max(task.seq_len, 16)
    model = CARecurrentTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
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
