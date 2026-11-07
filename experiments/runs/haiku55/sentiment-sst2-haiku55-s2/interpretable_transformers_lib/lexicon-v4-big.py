"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The task is sentiment-sst2: the prompt is an
80-char lowercased sentence right-padded with '_', followed by '='; the
answer is '1' (positive) or '0' (negative).

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task sentiment-sst2 --n-samples 500
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
# Hand-written sentiment lexicon (general English sentiment words)
# ---------------------------------------------------------------------------

POS_WORDS = """good great best better love loved loves lovely like liked likes nice fine fun funny excellent
wonderful beautiful brilliant amazing perfect enjoy enjoyed enjoyable delightful charming smart strong touching
moving powerful rich warm pleasure entertaining fresh engaging clever superb impressive solid glorious masterpiece
memorable hilarious sweet interesting poignant gripping heartwarming satisfying terrific fantastic joy happy
recommend favorite witty vibrant riveting stunning inventive compelling fascinating laugh laughs engrossing
thoughtful intelligent winning gorgeous honest sharp effective pleasing uplifting graceful elegant thrilling
lively admirable captivating superior
pleasant pleasurable beauty brilliantly delight entertains entertained entertainment satisfy satisfies impress
impressed impresses moved touched tender tenderly earnest sincere heartfelt lush vivid dazzling dazzle dazzled
original originality unique crisp playful winsome charm charms charmed chuckle chuckles smile smiles smiling
laughed laughing cute lovable endearing remarkable treat treasure gem triumph triumphant powerfully accomplished
assured agreeable favourite pleased wise wisdom mature rewarding rewarded affecting affectionate brightly thrills
thrilled exciting excitement excited celebrate celebration dynamic eloquent eloquently resonant stellar tasty
strength beautifully greatest outstanding remarkably praise praised worthy worthwhile valuable wonderfully amusing
amuses amused funniest winner engrossed absorbing absorbed captures captured vividly zesty buoyant sparkling
sparkle ingenious ingenuity charismatic generous smartly warmth poetic lyrical hopeful uplifting""".split()

NEG_WORDS = """bad worst worse terrible awful boring bore bores dull flat mess messy waste weak stupid dumb poor poorly
lame fails failed failure fail unfunny tedious tiresome annoying ugly horrible painful predictable pointless
empty trite sloppy lifeless forgettable disappointing disappointment bland mediocre hate hated dreadful lacks
lack unpleasant incoherent confusing clumsy shallow stale grim numbing insult silly lousy flawed muddled
pretentious vulgar vapid dreary forced
rubbish garbage junk trash crap stinks stink unwatchable unbearable unimaginative uninspired uninteresting
unconvincing unengaging unlikable unlikeable unappealing unsatisfying unbelievable nonsense boredom bored yawn
yawns yawning sleepy tired exhausting ridiculous ridiculously absurd absurdity cheesy cheap crude crudely mindless
idiotic idiots moronic sad sadly depressing depressed bleak gloomy gross nasty cruel creepy disgusting disgust
offensive offends offended insipid tepid tiring fizzle fizzles fizzled flop flops misfire misfires misguided
misses miss missed wasted wasting wasteful clunky awkward overlong bloated dragging drags drag plodding plods
plodded fumbling hollow hollowness emptiness vacuous vain pathetic pitiful laughable lazy lazily formulaic stilted
stiff uneven unnecessary obnoxious cliched cliche phony fake problematic failing boringly stupidly dully weakly
weaker sour sucks suck sucked annoy annoyed irritating irritate grating grates silliness ugliness abysmal
atrocious appalling embarrassing embarrassment regret regrettable regrets disappoint disappoints disappointed
dissatisfied""".split()

SENTIMENT: dict[str, int] = {}
for _w in POS_WORDS:
    SENTIMENT.setdefault(_w, 1)
for _w in NEG_WORDS:
    SENTIMENT.setdefault(_w, -1)

MAX_WORD = max(len(w) for w in SENTIMENT)
# Copy heads cover offsets 0..MAX_WORD+4 (word + boundary + 3-char negation prefix).
N_HEADS = MAX_WORD + 5
# Four neurons per word: plain match, "not"/"n't" negated, "no" negated, sentence-initial.
N_NEURONS = 4 * len(SENTIMENT)
POS_ANGLE_N = 82  # must equal max_seq_len used by build_model (max(seq_len, 16))
DELIMS = [" ", "_", "="]


def make_layout(V: int) -> dict[str, int]:
    """Residual-stream channel layout."""
    lay = {}
    lay["CHAR"] = 0                      # channels 0..V-1: one-hot current char
    lay["CONST"] = V                     # constant 1 (bias channel)
    lay["POS"] = V + 1                   # 2 channels: (cos, sin) of position angle
    lay["COPY"] = V + 3                  # N_HEADS blocks of V channels: char at offset m
    lay["SENT"] = lay["COPY"] + V * N_HEADS  # accumulated sentiment score
    lay["OUT"] = lay["SENT"] + 1         # sentence-level average (layer 2 output)
    lay["D"] = lay["OUT"] + 1
    return lay


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal multi-head attention with separate q/k width and value width."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_qk = d_qk
        self.d_v = d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_qk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm block with LayerNorm removed (identity) so weights stay exactly linear."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_qk: int, d_v: int):
        super().__init__()
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads, d_qk, d_v)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer. Width is derived from the channel layout."""

    def __init__(self, vocab_size: int, max_seq_len: int = 82, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_qk = 2
        self.d_v = vocab_size
        self.n_heads = N_HEADS
        self.d_ff = N_NEURONS
        self.d_model = make_layout(vocab_size)["D"]
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(max_seq_len, self.d_model)
        self.blocks = nn.ModuleList([
            Block(self.d_model, self.n_heads, self.d_ff, self.d_qk, self.d_v)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(self.d_model, vocab_size, bias=False)

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

def _neuron_specs():
    """Yield (constraints, output_value, start_pos). constraints[m] = allowed chars at offset m
    (offset m = the character m positions before the current delimiter). start_pos is None,
    or the position index p where the word starts at the very beginning of the sentence."""
    for w, s in SENTIMENT.items():
        L = len(w)
        # Plain match: delimiter, then word letters, then a space before the word.
        base = [DELIMS] + [[w[L - m]] for m in range(1, L + 1)] + [[" "]]
        yield base, float(s), None
        # Negated match: same, plus "not " or "n't " immediately before the word.
        neg = base + [["t"], ["o", "'"], ["n"]]
        yield neg, -2.0 * float(s), None
        # "no " negation: offsets L+2 = 'o', L+3 = 'n'.
        neg_no = base + [["o"], ["n"]]
        yield neg_no, -2.0 * float(s), None
        # Sentence-initial match: no preceding space exists, so pin the delimiter position to L.
        start = [DELIMS] + [[w[L - m]] for m in range(1, L + 1)]
        yield start, float(s), L


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    torch.manual_seed(0)
    V, N, H = model.vocab_size, model.max_seq_len, model.n_heads
    lay = make_layout(V)
    ci = task.stoi
    dq = model.d_qk
    beta = 5000.0  # hard-attention sharpness

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- Embeddings -------------------------------------------------------
        for ch, i in ci.items():
            model.token_emb.weight[i, lay["CHAR"] + i] = 1.0
        model.token_emb.weight[:, lay["CONST"]] = 1.0
        for p in range(N):
            ang = 2.0 * math.pi * p / N
            model.pos_emb.weight[p, lay["POS"]] = math.cos(ang)
            model.pos_emb.weight[p, lay["POS"] + 1] = math.sin(ang)

        # --- Layer 0 attention: head k copies the char at offset k ------------
        # Query for position j in head k is the position angle of (j - k);
        # causal masking + sharp softmax makes each head attend to exactly j-k.
        att = model.blocks[0].attn
        dlt = 2.0 * math.pi / N
        sq = beta * math.sqrt(dq)  # compensates the 1/sqrt(d_qk) inside attention
        for k in range(H):
            c, s = math.cos(k * dlt), math.sin(k * dlt)
            r = dq * k
            att.W_q.weight[r, lay["POS"]] = c * sq
            att.W_q.weight[r, lay["POS"] + 1] = s * sq
            att.W_q.weight[r + 1, lay["POS"]] = -s * sq
            att.W_q.weight[r + 1, lay["POS"] + 1] = c * sq
            att.W_k.weight[r, lay["POS"]] = 1.0
            att.W_k.weight[r + 1, lay["POS"] + 1] = 1.0
            for i in range(V):
                att.W_v.weight[k * V + i, lay["CHAR"] + i] = 1.0
                att.W_o.weight[lay["COPY"] + k * V + i, k * V + i] = 1.0

        # --- Layer 0 MLP: word-ending detectors ------------------------------
        mlp = model.blocks[0].mlp
        big = 1000.0  # position-match strength: off-by-one position costs ~2.9 in pre-activation
        for n, (cons, val, start) in enumerate(_neuron_specs()):
            R = len(cons)
            for m, allowed in enumerate(cons):
                for ch in allowed:
                    mlp.fc1.weight[n, lay["COPY"] + m * V + ci[ch]] = 1.0
            if start is None:
                mlp.fc1.bias[n] = -(R - 1)  # fires (=1) iff every constraint matches
            else:
                ang = 2.0 * math.pi * start / N
                mlp.fc1.weight[n, lay["POS"]] = big * math.cos(ang)
                mlp.fc1.weight[n, lay["POS"] + 1] = big * math.sin(ang)
                mlp.fc1.bias[n] = 1 - R - big
            mlp.fc2.weight[lay["SENT"], n] = val

        # --- Layer 1 attention: uniform average of SENT over all positions ----
        # (q/k weights are zero -> uniform attention over the causal prefix)
        att2 = model.blocks[1].attn
        att2.W_v.weight[0, lay["SENT"]] = 1.0
        att2.W_o.weight[lay["OUT"], 0] = 1.0

        # --- Readout: logit('1') - logit('0') = 81 * mean(SENT) + 0.5 ---------
        model.head.weight[ci["1"], lay["OUT"]] = 81.0
        model.head.weight[ci["1"], lay["CONST"]] = 0.5
        model.head.weight[ci["0"], lay["OUT"]] = -81.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "lexicon-v4-big"
model_description = (
    "lexicon-v3-no with the sentiment lexicon expanded from ~150 to ~400 general sentiment words."
)


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
