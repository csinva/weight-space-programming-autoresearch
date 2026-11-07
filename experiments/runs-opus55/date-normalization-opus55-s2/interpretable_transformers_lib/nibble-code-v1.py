"""
Interpretable transformer for date normalization. All weights are hand-written; nothing is trained.

Tokens are a small readable code: a 4-bit digit value, an is-digit flag, a 5-bit letter index, and
flags for word-like characters, '/', '.', '_'. BOS and '=' are the zero vector.

Block 1  attention: 4 heads form a shift register; each token sees tokens at offsets 1..4 behind it.
         MLP: one conjunction neuron per parsing rule tags the token that *terminates* a field
         (YEAR / MONTH / DAY); 12 trigram neurons recognise month words and write their number.
Block 2  attention only: each output step queries a role, and the head for that digit offset copies
         the digit nibble from the tagged token's window. Idle heads query '=', whose window is 0.
Output   nearest code: logit(c) = 2 c.b - |c|^2 for digit nibbles and the dash.
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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x) if self.mlp else x


class SimpleTransformer(nn.Module):
    """Causal transformer without LayerNorm; a BOS token is prepended inside forward()."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 72,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 19,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff), Block(d_model, n_heads, 0)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        ids = torch.cat([ids.new_full((B, 1), self.vocab_size), ids], dim=1)
        pos = torch.arange(T + 1, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)[:, 1:]


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def bits(value: int, width: int) -> list:
    """Most significant bit first."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    for p in model.parameters():
        nn.init.zeros_(p)
    s = task.stoi

    # Token code fields, as (slot) indices of the state.
    NIB, ISD, LET, SEP, SLASH, DOT, UNDER = 0, 4, 5, 10, 11, 12, 13
    COS, SIN, PROMPT, NULL, STEP = 14, 15, 16, 17, 18
    nib, let = list(range(NIB, NIB + 4)), list(range(LET, LET + 5))
    # Which token fields each shift-register offset keeps.
    fields = {1: nib + [ISD] + let, 2: nib + [ISD, SLASH] + let, 3: nib + [ISD, SLASH, DOT], 4: nib + [ISD]}
    WIN, next_slot = {}, STEP + 10
    for k, f in fields.items():
        WIN[k] = {slot: next_slot + i for i, slot in enumerate(f)}
        next_slot += len(f)
    Y, M, D, OUT, DASH, ON = next_slot, next_slot + 1, next_slot + 2, next_slot + 3, next_slot + 7, next_slot + 8
    assert ON < model.d_model

    with torch.no_grad():
        # Token embedding: the readable code. BOS and '=' stay zero.
        emb = model.token_emb.weight
        for ch, i in s.items():
            if ch.isdigit():
                emb[i, ISD] = 1.0
                emb[i, nib] = torch.tensor(bits(int(ch), 4), dtype=emb.dtype)
            if ch.isalpha():
                emb[i, let] = torch.tensor(bits(ord(ch) - 96, 5), dtype=emb.dtype)
            emb[i, SEP] = float(ch.isalpha() or ch in " ,-")
            emb[i, SLASH], emb[i, DOT], emb[i, UNDER] = float(ch == "/"), float(ch == "."), float(ch == "_")

        # Position embedding. idx = original position + 1 (BOS is idx 0).
        theta = 2 * math.pi / 64
        pe = model.pos_emb.weight
        for t in range(pe.shape[0]):
            pe[t, COS], pe[t, SIN] = math.cos(theta * t), math.sin(theta * t)
            pe[t, PROMPT], pe[t, ON] = float(1 <= t <= 31), 1.0
        pe[32, NULL] = 1.0  # '=': dates are at most 27 chars, so its window holds only padding
        for step in range(10):
            pe[32 + step, STEP + step] = 1.0
        pe[32 + 4, DASH] = pe[32 + 7, DASH] = 1.0

        # Block 1 attention: head k-1 rotates the position code back k steps and copies that token.
        a1 = model.blocks[0].attn
        dh = a1.d_head
        B1 = 200.0
        for k, f in fields.items():
            base = (k - 1) * dh
            kx, ky = base + dh - 2, base + dh - 1
            c, sn = math.cos(k * theta), math.sin(k * theta)
            a1.W_k.weight[kx, COS] = B1
            a1.W_k.weight[ky, SIN] = B1
            a1.W_q.weight[kx, COS], a1.W_q.weight[kx, SIN] = B1 * c, B1 * sn
            a1.W_q.weight[ky, COS], a1.W_q.weight[ky, SIN] = -B1 * sn, B1 * c
            for i, slot in enumerate(f):
                a1.W_v.weight[base + i, slot] = 1.0
                a1.W_o.weight[WIN[k][slot], base + i] = 1.0

        # Block 1 MLP: conjunction neurons.
        fc1, fc2 = model.blocks[0].mlp.fc1, model.blocks[0].mlp.fc2
        neuron = [0]

        def rule(inputs: dict, threshold: float, outputs: dict) -> None:
            n = neuron[0]
            neuron[0] += 1
            for slot, w in inputs.items():
                fc1.weight[n, slot] += w
            fc1.bias[n] = -threshold
            for slot, w in outputs.items():
                fc2.weight[slot, n] += w

        # YEAR: four digits just behind a prompt token.
        rule({WIN[1][ISD]: 1, WIN[2][ISD]: 1, WIN[3][ISD]: 1, WIN[4][ISD]: 1, PROMPT: 1}, 4, {Y: 1})
        # YEAR (mm/dd/yy): padding three after a slash; force the window's century digit to '2'.
        force_2 = {WIN[4][nib[i]]: 3.0 * (2 * b - 1) for i, b in enumerate(bits(2, 4))}
        rule({UNDER: 1, WIN[3][SLASH]: 1}, 1, {Y: 1, **force_2})
        # MONTH: first slash; second dot of yyyy.mm.dd.
        rule({SLASH: 1, WIN[2][SLASH]: -1, WIN[3][SLASH]: -1}, 0, {M: 1})
        rule({DOT: 1, WIN[3][DOT]: 1}, 1, {M: 1})
        # DAY: second slash; end of yyyy.mm.dd; a digit followed by a word/space/comma/dash.
        rule({SLASH: 1, WIN[2][SLASH]: 1, WIN[3][SLASH]: 1}, 1, {D: 1})
        rule({UNDER: 1, WIN[3][DOT]: 1}, 1, {D: 1})
        rule({WIN[1][ISD]: 1, SEP: 1, PROMPT: 1}, 2, {D: 1})
        # MONTH words: a trigram detector writes the month number into the (digit-free) window.
        for m, name in enumerate(task.MONTHS, start=1):
            inputs, n_ones = {}, 0
            for slots, ch in ((list(WIN[2][x] for x in let), name[0]), (list(WIN[1][x] for x in let), name[1]), (let, name[2])):
                code = bits(ord(ch) - 96, 5)
                n_ones += sum(code)
                inputs.update({slot: 2 * b - 1 for slot, b in zip(slots, code)})
            outs = {M: 1}
            outs.update({WIN[1][nib[i]]: b for i, b in enumerate(bits(m % 10, 4)) if b})
            if m >= 10:
                outs[WIN[2][nib[3]]] = 1
            rule(inputs, n_ones - 1, outs)
        assert neuron[0] == model.d_ff

        # Block 2 attention: heads read window offsets 4,3,2,1; keys are roles; steps choose queries.
        a2 = model.blocks[1].attn
        B2 = 300.0
        plan = {0: (0, Y), 1: (1, Y), 2: (2, Y), 3: (3, Y), 5: (2, M), 6: (3, M), 8: (2, D), 9: (3, D)}
        roles = [Y, M, D, NULL]
        for h in range(4):
            base = h * dh
            for i, role in enumerate(roles):
                a2.W_k.weight[base + 4 + i, role] = 1.0
            for step in range(10):
                role = plan[step][1] if step in plan and plan[step][0] == h else NULL
                a2.W_q.weight[base + 4 + roles.index(role), STEP + step] = B2
            for b in range(4):
                a2.W_v.weight[base + b, WIN[4 - h][nib[b]]] = 1.0
                a2.W_o.weight[OUT + b, base + b] = 1.0

        # Unembedding: nearest code. Digits use the 4 nibble bits, the dash its own bit.
        for d in range(10):
            code = bits(d, 4)
            model.head.weight[s[str(d)], OUT:OUT + 4] = 2.0 * torch.tensor(code, dtype=emb.dtype)
            model.head.weight[s[str(d)], ON] = -float(sum(code))
        model.head.weight[s["-"], DASH] = 2.0
        model.head.weight[s["-"], ON] = -1.0

    _dev_check(model, task)


def _dev_check(model, task, n: int = 5000) -> None:
    """Development only: exact-match accuracy on fresh examples, listing a few failures."""
    exs = task.generate_examples(n, seed=98765)
    ids = torch.tensor([task.encode(e.prompt) for e in exs])
    with torch.no_grad():
        for _ in range(task.answer_len):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    preds = [task.decode(r[task.prompt_len:].tolist()) for r in ids]
    bad = [(e.prompt.rstrip("_="), e.answer, p) for e, p in zip(exs, preds) if p != e.answer]
    nnz = sum(int((p != 0).sum()) for p in model.parameters())
    print(f"DEV {n - len(bad)}/{n} nonzero={nnz}", bad[:8])


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "nibble-code-v1"
model_description = "Replaces one-hot tokens with a readable 14-dim code (digit nibble, letter index, punctuation flags) and nearest-code output decoding; d_model 248 -> 72."


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
