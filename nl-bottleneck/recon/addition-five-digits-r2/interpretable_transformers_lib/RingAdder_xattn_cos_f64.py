"""
Interpretable transformer for 5-digit addition: one cross-attention read that
turns A+B into an angle on a ten-sector ring, read out with a cosine.

No training: every parameter is written in closed form in `write_weights`.
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

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (NO TRAINING)
# ---------------------------------------------------------------------------

class RingAdder(nn.Module):
    """One cross-attention read + cosine readout on a ten-sector ring.

    Parameters (all written in closed form):
      pos_table (12 x 2): rows 0..4  = key for place index c (c=0 is the most
                          significant digit), shared by a_c and b_c: (1, -c ln10)
                          row 5      = '+' sink key at the origin
                          rows 6..11 = query for output step i: (ln t_i, 1)
      value_scale (1):    digit value = digit * value_scale = digit * 2pi/10/1e-5
    Computation is float64.
    """

    def __init__(self, vocab_size: int, max_seq_len: int = 32, dtype=torch.float64):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dt = dtype
        self.pos_table = nn.Parameter(torch.zeros(12, 2, dtype=dtype))
        self.value_scale = nn.Parameter(torch.zeros(1, dtype=dtype))
        # fixed index maps (buffers, not parameters): prompt position -> key row
        key_rows = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
        self.register_buffer("key_rows", torch.tensor(key_rows, dtype=torch.long))
        self.register_buffer("digit_of_token", torch.tensor(
            [float(i) if i < 10 else 0.0 for i in range(vocab_size)], dtype=dtype))
        self.register_buffer("is_digit", torch.tensor(
            [i < 10 for i in range(vocab_size)], dtype=torch.bool))
        self.shift = 5e-6

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        out = torch.zeros(B, T, self.vocab_size, dtype=self.dt, device=ids.device)
        out[:, :, ~self.is_digit] = -1e9          # non-digit logits masked
        if T < 12:
            return out
        keys = self.pos_table[self.key_rows]                         # (11, 2)
        n_q = min(T - 11, 6)
        qidx = torch.arange(n_q, device=ids.device)
        queries = self.pos_table[6 + qidx]                           # (n_q, 2)
        scores = queries @ keys.T                                    # ln(t_i 10^-c); sink 0
        attn = scores.softmax(dim=-1)                                # only the 11 prompt keys
        vals = self.digit_of_token[ids[:, :11]] * self.value_scale   # (B, 11); '+' -> 0
        z = vals @ attn.T                                            # (B, n_q) angle
        d = torch.arange(10, dtype=self.dt, device=ids.device)
        phase = 2 * math.pi / 10 * (d + 0.5 - self.shift)
        logits = torch.cos(z[:, :, None] - phase[None, None, :])     # (B, n_q, 10)
        out[:, 11:11 + n_q, :10] = logits
        return out


# ---------------------------------------------------------------------------
# Closed-form weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: RingAdder, task) -> None:
    ln10 = math.log(10.0)
    dt = model.dt
    with torch.no_grad():
        for c in range(5):
            model.pos_table[c] = torch.tensor([1.0, -c * ln10], dtype=dt)
        model.pos_table[5] = torch.tensor([0.0, 0.0], dtype=dt)  # '+' sink at origin
        S = sum(10.0 ** (-c) for c in range(5))  # sum_c 10^-c per operand
        for i in range(6):
            # softmax weight on place c: t_i 10^-c / Z, Z = 1 + 2 S t_i (sink gives the 1).
            # Want t_i/Z = 10^(i-6) so that output = 1e-5 (A+B) / 10^(5-i).
            r = 10.0 ** (i - 6)
            t = r / (1.0 - 2.0 * S * r)
            model.pos_table[6 + i] = torch.tensor([math.log(t), 1.0], dtype=dt)
        model.value_scale.fill_(2 * math.pi / 10 / 1e-5)


model_shorthand_name = "RingAdder_xattn_cos_f64"
model_description = ("One cross-attention read over the 11 prompt positions with 2-D place-value keys (1,-c ln10) "
                     "and queries (ln t_i,1), '+' as zero sink; digit*2pi/10/1e-5 values give an angle; "
                     "cosine readout over 10 sectors with 5e-6 shift; float64.")


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



def build_model(task) -> RingAdder:
    max_seq_len = max(task.seq_len, 16)
    model = RingAdder(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
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
