"""
Interpretable transformer for character-level sequence tasks (task: trec-coarse).
Snapshot kw_priority_v26 (debug helpers stripped).

Circuit (2 layers, hand-written weights, no training):
  Layer 1 attention : 6 heads, head h copies the token that is h+1 positions back
                      (position match done with Fourier position codes) into its own
                      slot -> every position sees its last 7 characters (+ a START flag).
  Layer 1 MLP       : one neuron per KEYWORD (n-gram, <=7 chars, optionally anchored to the
                      question start). Fires iff the last characters equal the keyword;
                      writes a 1 into that keyword's indicator dimension.
  Layer 2 attention : the final '=' position attends to positions whose keyword fired,
                      with a score equal to the keyword's PRIORITY (tier) minus a small
                      "later in the sentence" penalty; value = the keyword's class vote.
                      => "highest priority (then earliest) keyword wins".
  Head              : class dims -> digit token; constant dim gives default class.
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


# Keyword table: (pattern, class digit, tier). Higher tier wins. Patterns <= 7 chars
# (longer ones are truncated to their last 7 chars). '^' = anchored to question start.
KEYWORDS = [
    # tier 0: weak defaults from the opening words
    ("what is ", 2, 0), ("what are ", 2, 0), ("what was ", 1, 0), ("what does ", 2, 0),
    ("what do ", 1, 0), ("what did ", 1, 0), ("what 's ", 1, 0), ("how ", 2, 0),
    ("name ", 3, 0),
    ("what are the ", 1, 0.5), ("name of ", 1, 0.7),
    ("what kind", 1, 1.3), ("what type", 1, 1.3), ("what color", 1, 1.3), ("what causes", 2, 1),
    ("what caused", 2, 1), ("what happened", 2, 1), ("what prompted", 2, 1), ("why ", 2, 1),
    # tier 2: wh-words and numeric / location / person question words
    ("^who ", 3, 2), ("^whose ", 3, 2), ("^when ", 5, 2), ("^where ", 4, 2),
    ("who ", 3, 0.8), ("when ", 5, 0.8), ("where ", 4, 0.8),
    ("how many", 5, 2), ("how much", 5, 2), ("how long", 5, 2), ("how far", 5, 2),
    ("how old", 5, 2), ("how big", 5, 2), ("how high", 5, 2), ("how tall", 5, 2),
    ("how fast", 5, 2), ("how hot", 5, 2), ("how large", 5, 2), ("how deep", 5, 2),
    ("how wide", 5, 2), ("how heavy", 5, 2),
    ("what country", 4, 2), ("what city", 4, 2), ("what year", 5, 2), ("what percent", 5, 2),
    ("company ", 3, 1), ("team ", 3, 1), ("baseball", 3, 1), ("cartoon", 3, 1),
    ("organiz", 3, 1), ("names ", 3, 1), ("died ", 3, 1),
    ("fear of", 1, 1.2),
    ("london", 4, 1), ("mississ", 4, 1), ("internet", 4, 1), ("reims", 4, 1), ("court ", 4, 1),
    ("home ", 4, 1), ("address", 4, 1), ("website", 4, 1), ("mounta", 4, 1),
    ("had ", 3, 1), (" her ", 3, 1), (" real ", 3, 1), ("john ", 3, 1),
    ("ways ", 1, 1), ("ferent", 1, 1.1), ("called ", 1, 1), ("level ", 1, 1),
    ("history", 2, 1.1), ("purpose", 2, 1.1), ("functio", 2, 1.1), ("nature ", 2, 1.1),
    ("design ", 2, 1.1), ("issue ", 2, 1.1), ("religio", 2, 1.1), ("believe", 2, 1.1),
    ("conceptio", 2, 1.1), ("princip", 2, 1.1), ("define ", 2, 1.1), ("makes ", 2, 1.1),
    (" rate ", 5, 1.1), (" toll ", 5, 1.1), ("odds ", 5, 1.1), ("size ", 5, 1.1),
    ("wingspan", 5, 1.1), ("latitude", 5, 1.1), ("equival", 5, 1.1), ("months ", 5, 1.1),
    ("find ", 2, 1), ("baby ", 2, 1),
    ("charact", 3, 1), ("known ", 3, 1), ("doonesb", 3, 1), ("werewol", 3, 1),
    ("mascot", 1, 1), ("univers", 1, 1), ("latin", 1, 1),
    ("signatu", 1, 1), ("events", 1, 1), ("notre", 1, 1), ("erm for", 1, 1.2),
    ("own for", 2, 1.3), ("us for ", 2, 1.3), ("ll form", 0, 1.3),
    ("birthda", 5, 1.1), ("horsepo", 5, 1.1), ("income", 5, 1.1), ("probabi", 5, 1.1), ("capita", 5, 1.1),
    ("hearing", 2, 1), ("receive", 2, 1), ("nobel", 2, 1), ("appeara", 2, 1),
    ("quantit", 5, 1.1), ("statist", 5, 1.1), ("lengths", 5, 1.1), ("season ", 5, 1.1),
    ("surname", 3, 1), ("ranger ", 3, 1), (" army ", 3, 1), ("called ", 1, 1),
    ("occupat", 3, 1), ("explore", 3, 1), ("host ", 3, 1), ("yankee", 3, 1), ("protago", 3, 1),
    ("villain", 3, 1), ("astrona", 3, 1), ("whom ", 3, 1), ("police", 3, 1),
    ("constel", 4, 1), ("buildin", 4, 1), ("valley", 4, 1), (" area ", 4, 1), ("librar", 4, 1),
    ("desert", 4, 1), ("nother ", 1, 1.2),
    ("substan", 1, 1), ("soap ", 1, 1), ("show ", 1, 1), (" war ", 1, 1), ("become", 1, 1),
    ("business", 3, 1), ("introdu", 2, 1), (" make ", 2, 1), ("claim ", 2, 1),
    (" day ", 5, 1), (" born ", 5, 1), ("dubbed", 3, 1),
    ("how do ", 2, 1.5), ("how can ", 2, 1.5), ("how did ", 2, 1.5), ("how does ", 2, 1.5),
    ("how is ", 2, 1.5), ("how to ", 2, 1.5),
    # tier 1: topic nouns anywhere in the question
    ("actor ", 3, 1), ("actress ", 3, 1), ("singer ", 3, 1), ("author ", 3, 1),
    ("writer ", 3, 1), ("composer ", 3, 1), ("president ", 3, 1), ("leader ", 3, 1),
    ("king ", 3, 1), ("queen ", 3, 1), ("poet ", 3, 1),
    ("painter ", 3, 1), ("artist ", 3, 1), ("scientist ", 3, 1), ("inventor ", 3, 1),
    ("player ", 3, 1), ("woman ", 3, 1), ("person ", 3, 1), ("daughter ", 3, 1),
    ("nickname ", 3, 1), ("pseudonym ", 3, 1), ("star ", 3, 1), ("judge ", 3, 1),
    (" state ", 4, 1), (" states ", 4, 1), (" country ", 4, 1), ("countries ", 4, 1),
    (" city ", 4, 1), ("cities ", 4, 1), ("ocean ", 4, 1), ("river ", 4, 1),
    (" lake ", 4, 1), ("mountain ", 4, 1), ("island ", 4, 1), ("capital ", 4, 1),
    ("continent ", 4, 1), (" town ", 4, 1), ("airport", 4, 1), ("water ", 4, 1),
    ("year ", 5, 1), ("years ", 5, 1), ("percent", 5, 1), ("speed ", 5, 1),
    ("population", 5, 1), ("frequency ", 5, 1), ("number ", 5, 1), ("fraction ", 5, 1),
    ("average ", 5, 1), ("expectancy", 5, 1), ("wage ", 5, 1), ("cost ", 5, 1),
    ("price ", 5, 1), ("weight ", 5, 1), ("height ", 5, 1), ("distance ", 5, 1),
    ("temperature ", 5, 1), (" age ", 5, 1), ("date ", 5, 1), ("chance ", 5, 1),
    ("salary ", 5, 1), ("length ", 5, 1),
    ("animal", 1, 1), ("game ", 1, 1), ("sport ", 1, 1), ("language ", 1, 1),
    ("food ", 1, 1), ("fruit ", 1, 1), ("breed ", 1, 1), ("novel ", 1, 1),
    ("book ", 1, 1), ("film ", 1, 1), ("movie ", 1, 1), ("song ", 1, 1),
    ("sitcom ", 1, 1), ("plant ", 1, 1), ("flower ", 1, 1), ("drug ", 1, 1),
    ("disease ", 1, 1), ("word ", 1, 1), ("metal ", 1, 1), ("element ", 1, 1),
    ("currency ", 1, 1),
    ("mean ", 2, 1), ("origin ", 2, 1), ("meaning ", 2, 1),
    ("reason ", 2, 1), ("effects ", 2, 1), ("definit", 2, 1), ("ference", 2, 1.1),
    ("stand for", 0, 1), ("abbrevi", 0, 1),
]

L_CTX = 7      # characters visible to each keyword neuron (offsets 0..6)
N_FREQ = 8     # Fourier position-code frequencies
FREQ_M = [1, 2, 4, 7, 11, 16, 22, 29]
PRIO_BASE, PRIO_STEP = 8.0, 16.0
RAMP_GAIN = 8.0


def n_keywords() -> int:
    return len(KEYWORDS)


def build_model_args(task):
    K = n_keywords()
    V = task.vocab_size
    H = L_CTX - 1
    base = V + 2 + 2 * N_FREQ + 1 + (V + 2) * (L_CTX - 1) + K + 6 + 2
    d_model = int(math.ceil(base / H) * H)
    assert d_model // H >= max(V + 2, 2 * N_FREQ)
    return dict(d_model=d_model, n_heads=H, n_layers=2, d_ff=K)


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    V = task.vocab_size
    D, H, dh = model.d_model, model.n_heads, model.d_model // model.n_heads
    K = n_keywords()
    stoi = task.stoi
    T = model.max_seq_len

    # Residual layout. Each slot = V token dims + START flag (that slot's token is at
    # position 0) + NOTSTART flag (the opposite): exactly one is 1, so norms stay constant.
    TOK = 0
    POS = V + 2                   # 2*N_FREQ Fourier dims
    CONST = POS + 2 * N_FREQ      # constant 1
    def BLK(t):                   # slot holding the token t positions back
        return TOK if t == 0 else CONST + 1 + (V + 2) * (t - 1)
    IND = CONST + 1 + (V + 2) * (L_CTX - 1)   # keyword indicators
    CLS = IND + K                       # 6 class dims (digits 0..5)
    ZERO = CLS + 6                     # always-zero dim (mean-shift compensation)
    NOTSTART = TOK + V + 1
    RAMP = ZERO + 1                    # position / 80: lets earlier keywords outrank later ones

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for ln in [model.final_ln] + [b.ln1 for b in model.blocks] + [b.ln2 for b in model.blocks]:
            ln.weight.fill_(1.0)

        # ---- embeddings ----
        for c, i in stoi.items():
            model.token_emb.weight[i, TOK + i] = 1.0
        for p in range(T):
            for f, m in enumerate(FREQ_M):
                w = 2 * math.pi * m / 82.0
                model.pos_emb.weight[p, POS + 2 * f] = math.cos(w * p)
                model.pos_emb.weight[p, POS + 2 * f + 1] = math.sin(w * p)
            model.pos_emb.weight[p, CONST] = 1.0
            model.pos_emb.weight[p, RAMP] = p / 80.0
        model.pos_emb.weight[0, TOK + V] = 1.0
        model.pos_emb.weight[1:, NOTSTART] = 1.0

        # ---- layer 1 attention: head h copies token at offset h+1 ----
        blk = model.blocks[0]
        S = 6.0 * math.sqrt(dh)
        for h in range(H):
            j = h + 1
            for f, m in enumerate(FREQ_M):
                w = 2 * math.pi * m / 82.0
                c, s = math.cos(w * j), math.sin(w * j)
                r = dh * h + 2 * f
                blk.attn.W_q.weight[r, POS + 2 * f] = S * c
                blk.attn.W_q.weight[r, POS + 2 * f + 1] = S * s
                blk.attn.W_q.weight[r + 1, POS + 2 * f] = -S * s
                blk.attn.W_q.weight[r + 1, POS + 2 * f + 1] = S * c
                blk.attn.W_k.weight[r, POS + 2 * f] = 1.0
                blk.attn.W_k.weight[r + 1, POS + 2 * f + 1] = 1.0
            for a in range(V + 2):      # token one-hot + START/NOTSTART flags
                blk.attn.W_v.weight[dh * h + a, TOK + a] = 1.0
                blk.attn.W_o.weight[BLK(j) + a, dh * h + a] = 1.0

        # ---- layer 1 MLP: keyword n-gram detectors ----
        for k, (pat, cls, tier) in enumerate(KEYWORDS):
            anchored = pat.startswith("^")     # '^' = pattern must start at position 0
            pat = pat.lstrip("^")[-L_CTX:]     # neuron sees only the last L_CTX characters
            L = len(pat)
            for t in range(L):
                ch = pat[L - 1 - t]
                blk.mlp.fc1.weight[k, BLK(t) + stoi[ch]] = 1.0
                if ch == " ":   # a word boundary is a space or the '_' padding
                    blk.mlp.fc1.weight[k, BLK(t) + stoi["_"]] = 1.0
            if anchored:
                blk.mlp.fc1.weight[k, BLK(L - 1) + V] = 1.0
            blk.mlp.fc1.bias[k] = -(L + int(anchored) - 0.5)
            blk.mlp.fc2.weight[IND + k, k] = 2.0

        # ---- LayerNorm calibration: LN gain = typical std of its input so LN(x) ~= x ----
        calib = torch.tensor([task.encode(e.prompt) for e in task.generate_examples(32, seed=123)])
        b0 = model.blocks[0]
        x = model.token_emb(calib) + model.pos_emb(torch.arange(calib.shape[1]))[None]
        b0.ln1.weight.fill_(x.std(-1, unbiased=False).mean().item())
        x = x + b0.attn(b0.ln1(x))
        b0.ln2.weight.fill_(x.std(-1, unbiased=False).mean().item())
        x = x + b0.mlp(b0.ln2(x))

        # ---- layer 2 attention: priority-weighted vote ----
        blk = model.blocks[1]
        blk.ln1.weight.fill_(x.std(-1, unbiased=False).mean().item())
        blk.attn.W_q.weight[0, CONST] = math.sqrt(dh)
        for k, (pat, cls, tier) in enumerate(KEYWORDS):
            blk.attn.W_k.weight[0, IND + k] = PRIO_BASE + PRIO_STEP * tier
            blk.attn.W_v.weight[cls, IND + k] = 1.0
        blk.attn.W_k.weight[0, RAMP] = -RAMP_GAIN      # earlier keyword wins ties within a tier
        for c in range(6):
            blk.attn.W_o.weight[CLS + c, c] = 2.0

        # ---- head: class dims -> digit tokens; default class via constant dim ----
        for c in range(6):
            model.head.weight[stoi[str(c)], CLS + c] = 1.0
        model.head.weight[stoi["1"], CONST] = 0.2
        model.head.weight[stoi["2"], CONST] = 0.1

        # Mean-shift immunity for EVERY readout of a LayerNorm output: make each weight row
        # sum to zero by putting its negative sum on the always-zero ZERO dim.
        mats = [model.head.weight]
        for b in model.blocks:
            mats += [b.attn.W_q.weight, b.attn.W_k.weight, b.attn.W_v.weight, b.mlp.fc1.weight]
        for W in mats:
            W[:, ZERO] = 0.0
            W[:, ZERO] = -W.sum(dim=1)


model_shorthand_name = "kw_priority_v26"
model_description = "Keyword-priority circuit (shift heads, n-gram keyword neurons, priority attention vote, anchored openers, earlier-wins ramp); dropped superlative/group rules, added 'known for', 'full form', birthday/income NUM cues."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, **build_model_args(task))
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
