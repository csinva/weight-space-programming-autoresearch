"""
Snapshot of run 1 (RecurrentCA_v1, 48.5%). BUGGY: the fold MLP used
s_new = relu(w - A) - relu(2w + 0.5X - A - 2) - 0.5 on centred w, which is off
by 0.5 for OR/XOR. Fixed in RecurrentCA_v2. Otherwise identical to v2.
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


class RecurrentCA(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, n_passes: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_passes = n_passes
        self.token_emb = nn.Parameter(torch.zeros(vocab_size, 3))   # [bit, A, X]
        self.W_q = nn.Parameter(torch.zeros(2, 2))
        self.W_v = nn.Parameter(torch.zeros(3, 3))
        self.W_o = nn.Parameter(torch.zeros(3, 3))
        self.fc1_w = nn.Parameter(torch.zeros(2, 3))
        self.fc1_b = nn.Parameter(torch.zeros(2))
        self.fc2_w = nn.Parameter(torch.zeros(1, 2))
        self.fc2_b = nn.Parameter(torch.zeros(1))
        self.head = nn.Parameter(torch.zeros(vocab_size, 1))
        ang = torch.arange(max_seq_len, dtype=torch.float32) * (math.pi / 6)
        self.register_buffer("pos", torch.stack([ang.cos(), ang.sin()], -1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        E = self.token_emb[ids]
        pos = self.pos[:T]
        q = pos @ self.W_q.T
        scores = q @ pos.T
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
        s = torch.full((B, T, 1), -0.5, device=ids.device)
        for _ in range(self.n_passes):
            z = torch.cat([s, E[..., 1:3]], -1)
            v = z @ self.W_v.T
            h = E + (attn @ v) @ self.W_o.T
            s = F.relu(h @ self.fc1_w.T + self.fc1_b) @ self.fc2_w.T + self.fc2_b
        return s @ self.head.T


def write_weights(model, task) -> None:
    st = task.stoi
    with torch.no_grad():
        E = torch.zeros(task.vocab_size, 3)
        E[st['1'], 0] = 1.0
        E[st['&'], 0] = 1.0; E[st['&'], 1] = 1.0
        E[st['^'], 2] = 1.0
        model.token_emb.copy_(E)
        th = math.pi / 6
        R = torch.tensor([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        model.W_q.copy_(200.0 * R)
        model.W_v.copy_(torch.eye(3))
        model.W_o.copy_(torch.eye(3))
        model.fc1_w.copy_(torch.tensor([[1.0, -1.0, 0.0], [2.0, -1.0, 0.5]]))
        model.fc1_b.copy_(torch.tensor([0.0, -2.0]))
        model.fc2_w.copy_(torch.tensor([[1.0, -1.0]]))
        model.fc2_b.copy_(torch.tensor([-0.5]))
        H = torch.zeros(task.vocab_size, 1)
        H[st['1'], 0] = 4.0
        H[st['0'], 0] = -4.0
        model.head.copy_(H)


model_shorthand_name = "RecurrentCA_v1"
model_description = ("Recurrent cellular-automaton transformer: 1 tied block (keyless sinusoidal prev-cell attention, "
                     "2-ReLU fold MLP) applied 10x; centred scalar state; readout +/-4*state.")


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
    model = RecurrentCA(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)
    accuracy, _ = evaluate(model, task, n_samples=args.n_samples, seed=args.seed,
                           device=args.device, verbose=args.verbose)
    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{
        "task": args.task, "accuracy": f"{accuracy:.4f}", "status": "",
        "model_shorthand_name": model_shorthand_name, "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)
    print(f"accuracy: {accuracy:.4f}")
