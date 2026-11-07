"""Hand-written transformer for MRPC paraphrase detection (nothing is trained).

"<s1>|<s2>=" -> "1"/"0". L1 copies the previous chars into each position, L2 asks
whether this 3-gram occurs in s1 (null slot = no), L3 pools those bits over s2 at
'=', and the head thresholds the resulting overlap. Attention is causal.
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

# Hand-set experiment configuration.
N = 4                       # shingle length
THRESHOLD = 0.40            # answer "1" iff overlap > THRESHOLD
MARGIN = 5                  # bits of disagreement tolerated by the null slot
SHARP = 10.0                # softmax sharpness of the match attention


class R:
    """Register map: which slice of the d_model state holds what."""
    BITS, D = 5, 28
    WIN = 0                 # [0, 20): char codes at offsets 0..N-1
    POS = 20                # [20, 22): (cos, sin) of position
    IS_S2 = 22              # position lies in sentence 2
    NOTPAD, BIAS = 23, 24   # character is not '_' / constant 1
    MATCH = 25              # this shingle occurs in s1
    OVERLAP = 26            # fraction of s2 shingles found in s1


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, null_slot: bool = False):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.null_score = nn.Parameter(torch.zeros(n_heads)) if null_slot else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        if self.null_score is not None:
            null = self.null_score.view(1, H, 1, 1).expand(B, H, T, 1)
            attn = torch.cat([scores, null], dim=-1).softmax(dim=-1)[..., :-1]
        else:
            attn = scores.softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, D))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        self.token_emb = nn.Embedding(vocab_size, R.D)
        self.pos_emb = nn.Embedding(max_seq_len, R.D)
        self.window = Attention(R.D, 4)
        self.match = Attention(R.D, 1, null_slot=True)
        self.pool = Attention(R.D, 1)
        self.head = nn.Linear(R.D, vocab_size, bias=False)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1]))[None]
        for layer in (self.window, self.match, self.pool):
            h = h + layer(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(ids))


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter from constants. No training."""
    for p in model.parameters():
        p.zero_()
    theta, sent = 2 * math.pi / 256, task.SENT_LEN

    # Embeddings: +-1 bit code per character (pad is all zeros) and flags.
    te, pe = model.token_emb.weight, model.pos_emb.weight
    for i in range(task.vocab_size):
        te[i, R.WIN:R.WIN + R.BITS] = torch.tensor(
            [1.0 if (i >> b) & 1 else -1.0 for b in range(R.BITS)])
    te[:, R.NOTPAD] = 1.0
    te[task.stoi["_"], R.WIN:R.WIN + R.BITS] = 0.0
    te[task.stoi["_"], R.NOTPAD] = 0.0
    te[task.stoi["|"]] = te[task.stoi[" "]]  # s2 starts at a word boundary
    for p in range(model.max_seq_len):
        pe[p, R.POS], pe[p, R.POS + 1] = math.cos(theta * p), math.sin(theta * p)
        pe[p, R.IS_S2] = float(sent < p <= 2 * sent)
        pe[p, R.BIAS] = 1.0

    # L1 window: head k looks k chars back by rotating its position query.
    a = model.window
    dh, g = a.d_head, math.sqrt(1e5 * math.sqrt(a.d_head))
    for k in range(1, N):
        h = (k - 1) * dh
        c, s = math.cos(k * theta), math.sin(k * theta)
        a.W_q.weight[h, R.POS], a.W_q.weight[h, R.POS + 1] = g * c, g * s
        a.W_q.weight[h + 1, R.POS], a.W_q.weight[h + 1, R.POS + 1] = -g * s, g * c
        a.W_k.weight[h, R.POS] = a.W_k.weight[h + 1, R.POS + 1] = g
        for b in range(R.BITS):
            a.W_v.weight[h + b, R.WIN + b] = 1.0
            a.W_o.weight[R.WIN + k * R.BITS + b, h + b] = 1.0

    # L2 match: shingle dot product counts agreeing bits, s2 keys are vetoed,
    # and the null slot sits just below a perfect score.
    a = model.match
    dh, width = a.d_head, N * R.BITS
    g = math.sqrt(SHARP * math.sqrt(dh))
    for j in range(width):
        a.W_q.weight[j, R.WIN + j] = a.W_k.weight[j, R.WIN + j] = g
    a.W_q.weight[width, R.BIAS] = g
    a.W_k.weight[width, R.IS_S2] = -100.0 * g
    a.W_v.weight[0, R.BIAS] = 1.0
    a.W_o.weight[R.MATCH, 0] = 1.0
    a.null_score[0] = SHARP * (width - MARGIN)

    # L3 pool: the single head averages match bits over non-pad s2 positions.
    a = model.pool
    dh, gain = a.d_head, 30.0
    g = math.sqrt(gain * math.sqrt(dh))
    a.W_q.weight[0, R.BIAS] = g
    a.W_k.weight[0, R.NOTPAD] = a.W_k.weight[0, R.IS_S2] = g
    a.W_v.weight[0, R.MATCH] = 1.0
    a.W_o.weight[R.OVERLAP, 0] = 1.0

    # Head: logit("1") - logit("0") = 20 * (overlap - THRESHOLD).
    w, one, zero = model.head.weight, task.stoi["1"], task.stoi["0"]
    w[:, R.BIAS] = -100.0
    w[zero, R.BIAS] = 0.0
    w[one, R.BIAS] = -20.0 * THRESHOLD
    w[one, R.OVERLAP] = 20.0
    dev_report(model, task)


@torch.no_grad()
def dev_report(model: SimpleTransformer, task) -> None:
    """Diagnostics only; never feeds the weights. Eval prompts are removed unread."""
    held = {e.prompt for e in task.generate_examples(200, seed=0)}
    dev = {e.prompt: e for e in task.generate_examples(2000, seed=12345)
           if e.prompt not in held}
    ids = torch.tensor([task.encode(p) for p in dev])
    y = torch.tensor([int(e.answer) for e in dev.values()])
    h = torch.cat([model.features(b)[:, -1] for b in ids.split(256)])
    one, zero = task.stoi["1"], task.stoi["0"]
    logits = model.head(h)
    pred = logits.argmax(-1) == one
    print(f"DEV n={len(y)} acc={(pred.long() == y).float().mean():.4f} "
          f"pred_pos={pred.float().mean():.4f} "
          f"auc={auc((logits[:, one] - logits[:, zero]).tolist(), y.tolist()):.4f}")


def auc(scores: list, labels: list) -> float:
    """Rank AUC: chance that a random positive outscores a random negative."""
    pairs = sorted(zip(scores, labels))
    i, rank_sum, n_pos = 0, 0.0, sum(labels)
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(l for _, l in pairs[i:j]) * (i + j + 1) / 2
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos))


model_shorthand_name = "E36_FourGramMargin5"
model_description = "D28 circuit with 4-grams tolerating two flipped bits, threshold 0.40: wider tolerance scaled to the longer shingle."


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
