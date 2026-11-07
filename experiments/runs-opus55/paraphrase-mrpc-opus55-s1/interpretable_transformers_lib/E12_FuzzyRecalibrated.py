"""Hand-written transformer for MRPC paraphrase detection (nothing is trained).

"<s1>|<s2>=" -> "1"/"0". L1 copies the last K chars into each position, L2 finds
exact n-gram matches in the other sentence (null slot = no match), L3 pools the
match bits at '=', and the head thresholds the pooled overlap.
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
NGRAMS = (3, 4, 6)          # shingle length of each match head
WEIGHTS = (0.5, 0.5, 0.0)   # head weight of each pooled overlap fraction
THRESHOLD = 0.42            # answer "1" iff weighted overlap > THRESHOLD
MARGIN = 3                  # bits of disagreement tolerated by the null slot
BIDIR = True                # also match s1 against s2 (L2 ignores causality)
WORD_START = (False, True, True)  # head counts only word-initial shingles
DECAY = 0.0                 # extra pooling log-weight of early positions
W_FILL = 0.0                # head weight on how full s2 is


class R:
    """Register map: which slice of the d_model state holds what."""
    BITS, K, D = 6, 6, 120
    WIN = 0                 # [0, 36): char codes at offsets 0..K-1
    POS = 36                # [36, 38): (cos, sin) of position
    IS_S1, IS_S2 = 38, 39   # position lies in sentence 1 / 2
    NOTPAD, BIAS = 40, 41   # character is not '_' / constant 1
    MATCH = 42              # [42, 45): match bit per n-gram
    POOLED = 45             # [45, 48): overlap fraction per n-gram
    SPACE, RAMP = 48, 49    # char is a word boundary / 1 at s2 start, falling
    WORD = 50               # [50, 53): shingle n starts at a word boundary
    FILL = 53               # fraction of positions that are non-pad


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, null_slot: bool = False, causal: bool = True):
        super().__init__()
        self.n_heads, self.d_head, self.causal = n_heads, d // n_heads, causal
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
        if self.causal:
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
        self.window = Attention(R.D, 20)
        self.match = Attention(R.D, 3, null_slot=True, causal=not BIDIR)
        self.pool = Attention(R.D, 12)
        self.head = nn.Linear(R.D, vocab_size, bias=False)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1]))[None]
        for layer in (self.window, self.match, self.pool):
            h = h + layer(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(ids))


def char_codes(task) -> torch.Tensor:
    """+-1 binary code of each vocabulary index; pad is the zero vector, and
    '|' shares the space code so the first word of s2 starts at a boundary."""
    codes = torch.tensor([[1.0 if (i >> b) & 1 else -1.0 for b in range(R.BITS)]
                          for i in range(task.vocab_size)])
    codes[task.stoi["_"]] = 0.0
    codes[task.stoi["|"]] = codes[task.stoi[" "]]
    return codes


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter from constants. No training."""
    for p in model.parameters():
        p.zero_()
    theta, sent = 2 * math.pi / 256, task.SENT_LEN

    # Embeddings.
    te, pe = model.token_emb.weight, model.pos_emb.weight
    te[:, R.WIN:R.WIN + R.BITS] = char_codes(task)
    te[:, R.NOTPAD] = 1.0
    te[task.stoi["_"], R.NOTPAD] = 0.0
    te[task.stoi[" "], R.SPACE] = te[task.stoi["|"], R.SPACE] = 1.0
    for p in range(model.max_seq_len):
        pe[p, R.POS], pe[p, R.POS + 1] = math.cos(theta * p), math.sin(theta * p)
        pe[p, R.IS_S1], pe[p, R.IS_S2] = float(p < sent), float(sent < p <= 2 * sent)
        pe[p, R.BIAS], pe[p, R.RAMP] = 1.0, 1.0 - (p - sent) / sent

    # L1 window: head k looks k chars back by rotating its position query.
    a = model.window
    dh, g = a.d_head, math.sqrt(1e5 * math.sqrt(a.d_head))
    for k in range(1, R.K):
        h = (k - 1) * dh
        c, s = math.cos(k * theta), math.sin(k * theta)
        a.W_q.weight[h, R.POS], a.W_q.weight[h, R.POS + 1] = g * c, g * s
        a.W_q.weight[h + 1, R.POS], a.W_q.weight[h + 1, R.POS + 1] = -g * s, g * c
        a.W_k.weight[h, R.POS] = a.W_k.weight[h + 1, R.POS + 1] = g
        for b in range(R.BITS):
            a.W_v.weight[h + b, R.WIN + b] = 1.0
            a.W_o.weight[R.WIN + k * R.BITS + b, h + b] = 1.0
    for i, n in enumerate(NGRAMS):  # copy the flag "char n-1 back is a boundary"
        h, hk = (R.K + i) * dh, (n - 2) * dh
        a.W_q.weight[h:h + 2] = a.W_q.weight[hk:hk + 2]
        a.W_k.weight[h:h + 2] = a.W_k.weight[hk:hk + 2]
        a.W_v.weight[h, R.SPACE] = 1.0
        a.W_o.weight[R.WORD + i, h] = 1.0

    # L2 match: shingle dot product counts agreeing bits; same-side keys are
    # vetoed, and the null slot sits just below a perfect score.
    a = model.match
    dh, sharp = a.d_head, 10.0
    g = math.sqrt(sharp * math.sqrt(dh))
    for i, n in enumerate(NGRAMS):
        h = i * dh
        for j in range(n * R.BITS):
            a.W_q.weight[h + j, R.WIN + j] = a.W_k.weight[h + j, R.WIN + j] = g
        for side, flag in enumerate((R.IS_S1, R.IS_S2)):
            a.W_q.weight[h + 36 + side, flag] = g
            a.W_k.weight[h + 36 + side, flag] = -100.0 * g
        a.W_v.weight[h, R.IS_S1] = a.W_v.weight[h, R.IS_S2] = 1.0
        a.W_o.weight[R.MATCH + i, h] = 1.0
        a.null_score[i] = sharp * (n * R.BITS - MARGIN)

    # L3 pool: uniform attention over non-pad positions reads the match bits.
    a = model.pool
    dh, gain = a.d_head, 30.0
    g = math.sqrt(gain * math.sqrt(dh))
    for i in range(len(NGRAMS)):
        h = i * dh
        a.W_q.weight[h, R.BIAS] = g
        a.W_k.weight[h, R.NOTPAD] = g
        a.W_k.weight[h, R.IS_S2] = g
        a.W_k.weight[h, R.IS_S1] = g if BIDIR else 0.0
        a.W_k.weight[h, R.RAMP] = g * DECAY / gain
        a.W_k.weight[h, R.WORD + i] = g if WORD_START[i] else 0.0
        a.W_v.weight[h, R.MATCH + i] = 1.0
        a.W_o.weight[R.POOLED + i, h] = 1.0
    h = 3 * dh  # fill: mean of NOTPAD over all of s2
    a.W_q.weight[h, R.BIAS] = a.W_k.weight[h, R.IS_S2] = g
    a.W_v.weight[h, R.NOTPAD] = 1.0
    a.W_o.weight[R.FILL, h] = 1.0

    # Head: logit("1") - logit("0") = 20 * (weighted overlap - THRESHOLD).
    w, one, zero = model.head.weight, task.stoi["1"], task.stoi["0"]
    w[:, R.BIAS] = -100.0
    w[zero, R.BIAS] = 0.0
    w[one, R.BIAS] = -20.0 * THRESHOLD
    for i, fw in enumerate(WEIGHTS):
        w[one, R.POOLED + i] = 20.0 * fw
    w[one, R.FILL] = 20.0 * W_FILL
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
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    for i, n in enumerate(NGRAMS):
        f = h[:, R.POOLED + i]
        print(f"DEV n={n} auc={auc(f.tolist(), y.tolist()):.4f} "
              f"q0 {torch.quantile(f[y == 0], qs).numpy().round(3)} "
              f"q1 {torch.quantile(f[y == 1], qs).numpy().round(3)}")


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


model_shorthand_name = "E12_FuzzyRecalibrated"
model_description = "Fuzzy gram + stem blend (margin 3) with the threshold raised to 0.42 to match the higher fuzzy overlap scale."


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
