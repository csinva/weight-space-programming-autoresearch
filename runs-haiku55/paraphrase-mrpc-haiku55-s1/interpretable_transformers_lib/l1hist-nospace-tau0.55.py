"""
Interpretable transformer for paraphrase-mrpc (GLUE MRPC, 60-char sentences).

Prompt "<sent1 padded to 60>|<sent2 padded to 60>=" (122 chars), answer "0"/"1".
Rules: no training; weights are written by hand in write_weights().
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
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 66,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 70,
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

TAU = 0.55         # predict '1' iff L1 distance between the two char histograms < TAU
STEP_K = 50.0      # sharpness of the threshold ramp
ATTN_SCORE = 12.0  # attention logit for keys in the attended sentence (others ~0)
SWAP = False       # if True: predict '1' iff S >= TAU (reversed direction)
SKIP_SPACE = True  # if True: leave the space character out of both histograms


def write_weights(model: SimpleTransformer, task) -> None:
    """Predict '1' (paraphrase) iff the two sentences have similar character histograms.

    Layer 1 attention (query = '=' at position 121) attends uniformly to the sentence-1
    positions (head A) and to the sentence-2 positions (head B), so each head outputs the
    character histogram of its sentence. W_o writes +p1 and -p2 into the same residual dims,
    so the residual holds diff = p1 - p2 (LayerNorm offsets cancel).
    Layer 1 MLP: relu(diff_c) and relu(-diff_c) per character, summed into residual dim RS,
    giving S = sum_c |p1_c - p2_c|.
    Layer 2 MLP: two ramps on S give h = 1[S < TAU] (dim RH) and 1 - h (dim RNH).
    Head: logit['1'] = h + const, logit['0'] = (1 - h) + const, all other logits 0.
    """
    D, H = model.d_model, model.n_heads
    dh = D // H
    V = task.vocab_size          # dims 0..V-1 are character one-hots (dim = token id)
    assert dh == V and model.n_layers >= 2 and model.d_ff >= 2 * V + 4
    FS1, FS2, FQ = V, V + 1, V + 2   # segment flags (sentence 1, sentence 2) and query flag
    RS, RH, RNH = V + 3, V + 4, V + 5  # residual dims: S, h, 1-h
    stoi = task.stoi
    ln = lambda x: F.layer_norm(x, (D,))
    M = D / math.sqrt(D - 1)     # flag magnitude: makes LN of the query/flag residual have unit std
    mu = M / D
    sig = M * math.sqrt(D - 1) / D

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)

        # LayerNorm statistics of the prototype vectors (exact, eps included)
        x = torch.zeros(D); x[0] = 1.0; x[FS1] = 1.0
        v = ln(x)
        v_hi, v_lo, f_hi = v[0].item(), v[1].item(), v[FS1].item()
        x = torch.zeros(D); x[FQ] = M
        q_hi = ln(x)[FQ].item()
        wk = ATTN_SCORE * math.sqrt(dh) / (q_hi * f_hi)
        g = 1.0 / (v_hi - v_lo)
        CHARS = [c for c in range(V) if not (SKIP_SPACE and c == stoi[" "])]

        # Token embedding: one-hot character dims (letters, space, apostrophe, underscore)
        for ch, i in stoi.items():
            if ch not in "|=01":
                model.token_emb.weight[i, i] = 1.0
        # Position embedding: sentence-1 flag, sentence-2 flag, query flag at '='
        for p in range(0, 60):
            model.pos_emb.weight[p, FS1] = 1.0
        for p in range(61, 121):
            model.pos_emb.weight[p, FS2] = 1.0
        model.pos_emb.weight[121, FQ] = M

        # Layer 1 attention: head A (dims 0..V-1) -> sentence 1, head B (dims V..2V-1) -> sentence 2
        A = model.blocks[0].attn
        A.W_q.weight[0, FQ] = 1.0
        A.W_q.weight[dh, FQ] = 1.0
        A.W_k.weight[0, FS1] = wk
        A.W_k.weight[dh, FS2] = wk
        for c in CHARS:
            A.W_v.weight[c, c] = 1.0
            A.W_v.weight[dh + c, c] = 1.0
            A.W_o.weight[c, c] = g        # +p1
            A.W_o.weight[c, dh + c] = -g  # -p2

        # Layer 1 MLP: |diff_c| summed into residual dim RS
        m1 = model.blocks[0].mlp
        for c in CHARS:
            m1.fc1.weight[c, c] = 1.0 / sig
            m1.fc1.bias[c] = mu / sig
            m1.fc1.weight[V + c, c] = -1.0 / sig
            m1.fc1.bias[V + c] = -mu / sig
            m1.fc2.weight[RS, c] = 1.0
            m1.fc2.weight[RS, V + c] = 1.0

        # Layer 2 MLP: step on S with threshold TAU (ramps of width 1/K)
        a = (1 - 1 / D) / sig
        c0 = mu / sig
        K = STEP_K
        b0 = K * (TAU - c0 / a)
        m2 = model.blocks[1].mlp
        m2.fc1.weight[2 * V, RS] = -K / a
        m2.fc1.bias[2 * V] = b0 + 0.5
        m2.fc1.weight[2 * V + 1, RS] = -K / a
        m2.fc1.bias[2 * V + 1] = b0 - 0.5
        m2.fc1.weight[2 * V + 2, RS] = K / a
        m2.fc1.bias[2 * V + 2] = -b0 + 0.5
        m2.fc1.weight[2 * V + 3, RS] = K / a
        m2.fc1.bias[2 * V + 3] = -b0 - 0.5
        m2.fc2.weight[RH, 2 * V] = 1.0
        m2.fc2.weight[RH, 2 * V + 1] = -1.0
        m2.fc2.weight[RNH, 2 * V + 2] = 1.0
        m2.fc2.weight[RNH, 2 * V + 3] = -1.0

        # Output head: '1' and '0' each get a constant bias from FQ so they beat the zero logits
        model.head.weight[stoi["1"], RNH if SWAP else RH] = 1.0
        model.head.weight[stoi["1"], FQ] = 1.0
        model.head.weight[stoi["0"], RH if SWAP else RNH] = 1.0
        model.head.weight[stoi["0"], FQ] = 1.0


model_shorthand_name = "l1hist-nospace-tau0.55"
model_description = "2-layer L1 char-histogram distance without the space character; predict '1' iff distance < 0.55."


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
