"""
Interpretable transformer for decimal-to-binary-8bit: "197=" -> "11000101" (MSB first).

Rules of the game: hand-written weights only, never trained.
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
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 10,
                 n_heads: int = 2, n_layers: int = 1, d_ff: int = 1):
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
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

# Token channels are VAL, ALT, -VAL, -ALT; position channels are OUT, U, SINK, C1, C2, C3.
VAL, ALT, NVAL, NALT, OUT, U, SINK, C1, C2, C3 = range(10)
OFF = 20.0            # attention key magnitude that silences a position
OUT0 = 0.5            # half-integer offset that prevents ties
SINK_FLAG = 0.25      # small, because SINK is also the head's reference channel
TOKEN_C = 100.0       # every token has n^2 + a^2 = TOKEN_C
POS_NORM2 = 1000.0    # every position has squared norm POS_NORM2


def verify_all_inputs(model, task) -> None:
    """Sanity check only: greedy-decode every n in 0..255 and count exact matches."""
    ok = 0
    with torch.no_grad():
        for n in range(256):
            ids = task.encode(f"{n:03d}=")
            for _ in range(task.answer_len):
                ids = ids + [int(model(torch.tensor([ids]))[0, -1].argmax())]
            ok += task.decode(ids[task.prompt_len:]) == format(n, "08b")
    seen = [int(e.prompt[:-1]) for e in task.generate_examples(5000, 1)]
    print(f"VERIFY exhaustive {ok}/256; sampled n range {min(seen)}..{max(seen)}")


def write_weights(model: SimpleTransformer, task) -> None:
    """Trial subtraction with stock pre-norm blocks: emit '1' at bit k iff N >= S + 2^k.

    Head A reads N = 111 * mean(digits) with attention weights 100:10:1.
    Head B reads (S + 2^k) / 256 with weights 2^bit on emitted bits; '=' is the sink.

    LayerNorm is made an exact identity: every token vector and every position vector is
    zero-mean with a fixed norm on disjoint channels, so std(x) never changes, and setting
    LayerNorm's gain to that std cancels the division. The MLP is written as a no-op.
    """
    D = model.d_model
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for b in model.blocks:
            b.ln1.weight.fill_(1.0)
            b.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)
        attn = model.blocks[0].attn
        dh = attn.d_head
        s = math.sqrt(dh)  # cancels the sqrt(dh) in scaled dot-product attention
        head_a, head_b = 0, dh  # first output index of each head

        # Tokens: [n, a, -n, -a] is zero-mean with squared norm 2 * TOKEN_C for every token.
        values = {str(d): float(d) for d in range(10)}
        values["="] = 0.5
        for c, n in values.items():
            a = math.sqrt(TOKEN_C - n * n)
            model.token_emb.weight[task.stoi[c], [VAL, ALT, NVAL, NALT]] = torch.tensor([n, a, -n, -a])

        # Positions: the circuit's channels plus three compensators fixing the mean and norm.
        P = model.pos_emb.weight
        for t in range(model.max_seq_len):
            if t < 3:
                u = -(OFF + math.log([100.0, 10.0, 1.0][t]))
            elif 4 <= t <= 10:
                u = (11 - t) * math.log(2.0)
            else:
                u = 0.0
            flag = SINK_FLAG if t == 3 else 0.0
            sm = OUT0 + u + flag
            rest = POS_NORM2 - (OUT0 ** 2 + u ** 2 + flag ** 2)
            a = (-4 * sm + math.sqrt(16 * sm ** 2 - 24 * (sm ** 2 - rest))) / 12  # 2a + c = -sm
            P[t, [OUT, U, SINK, C1, C2, C3]] = torch.tensor([OUT0, u, flag, a, a, -sm - 2 * a])

        # Pre-attention LayerNorm: identity, since std(x) is fixed by the two norms above.
        sigma = math.sqrt((2 * TOKEN_C + POS_NORM2) / D)
        model.blocks[0].ln1.weight.fill_(sigma)

        # Head A: query -OUT, key U, so attention is 100:10:1 on the digits.
        attn.W_q.weight[head_a, OUT] = -s / OUT0
        attn.W_k.weight[head_a, U] = 1.0
        attn.W_v.weight[head_a, VAL] = 1.0
        # Head B: query (+OUT, U), key (U, SINK), so bits get 2^bit and the sink gets U_p.
        attn.W_q.weight[head_b, OUT] = s / OUT0
        attn.W_q.weight[head_b + 1, U] = s
        attn.W_k.weight[head_b, U] = 1.0
        attn.W_k.weight[head_b + 1, SINK] = 1.0 / SINK_FLAG
        attn.W_v.weight[head_b, VAL] = 1.0
        attn.W_o.weight[OUT, head_a] = 111.0    # + N
        attn.W_o.weight[OUT, head_b] = -256.0   # - (S + 2^k)

        # Output: the final LayerNorm preserves the sign of OUT - SINK.
        model.head.weight[task.stoi["1"], [OUT, SINK]] = torch.tensor([1.0, -1.0])
        model.head.weight[task.stoi["0"], [OUT, SINK]] = torch.tensor([-1.0, 1.0])
    verify_all_inputs(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e15_stock_block_d10"
model_description = "Full stock block (ln1, attention, ln2, MLP) at d=10: equal-norm zero-mean embeddings make the pre-attention LayerNorm an exact identity; MLP is a no-op."


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
