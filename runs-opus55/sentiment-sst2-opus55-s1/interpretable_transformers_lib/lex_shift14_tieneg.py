"""
Interpretable transformer for character-level sentiment (sentiment-sst2).

Hand-written weights, no training. Block 1 is a character shift register
(attention) plus one AND-gate MLP unit per lexicon pattern. Block 2 pools the
evidence at the final token with uniform attention.
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
# Hand-written lexicon. "word" = whole word, "stem*" = word starting with stem.
# ---------------------------------------------------------------------------

POSITIVE = """
amazing astonish* astounding beautiful* beauty best brilliant* captivat* charm* clever* compelling
delight* dazzling engag* enjoy* entertain* excellent exceptional exhilarat* extraordinar* fantastic
fascinat* finest fresh* fun funny gem gorgeous* graceful* great greatest gripping heartfelt
heartwarming hilarious* honest* imaginative impressive inspir* insightful intelligent* intriguing
inventive joy* likable lively love* lovely luminous magnificent marvel* masterful masterpiece
memorable moving nice observant original perfect* pleasant pleasure poignant polished powerful
profound provocative refreshing* remarkable* rewarding rich riveting romantic satisfying sensitive
sincere* smart solid sparkling spirited splendid stunning* stylish subtle superb* sweet tender
terrific thrilling* touching triumph* unforgettabl* uplifting vibrant vivid warm* wise witty
wonderful* wondrous
""".split()

NEGATIVE = """
absurd* annoy* appalling atrocious awful* awkward* bad badly banal bland bore* boring botched cheap*
clich* clumsy* confus* contrived convoluted crap* crude cynical disappoint* disaster* disgust*
disjointed dismal dreadful* dreary dull dumb embarrass* empty excruciating fail* failure fake flat
flawed flimsy forced forgettable formulaic gimmick* gratuitous grating hackneyed hollow horrible
horrid idiotic implausible inane incoherent inept insipid insult* irritating junk lack* lame
laughable lazy lifeless limp listless lousy mediocre mess* mindless misguided monotonous muddled
murky nasty nonsense numbing obnoxious offensive overwrought painful* pathetic plodding pointless
poor* predictable preten* ridiculous* rotten rubbish sadly shallow sill* simplistic slapdash sloppy
slow* sluggish smug soulless stale stiff stinks stupid* tedious* terrible* thin tiresome tired trite
unfunny ugly uneven unfortunate* uninspired unintelligib* uninterest* unpleasant unwatchable
vapid waste* wasted weak worse worst worthless wrong
""".split()

LEXICON = [(w, 1.0) for w in POSITIVE] + [(w, -1.0) for w in NEGATIVE]
TIE_BIAS = -0.01  # no evidence at all -> answer negative

# Layout: 15 heads of 32 dims. Head i < W holds the character at shift i.
C, W = 32, 14
H = W + 1
D = C * H
SCRATCH = C * W
POS0, POS1, SENT, TOTAL = SCRATCH, SCRATCH + 1, SCRATCH + 2, SCRATCH + 3
OMEGA = math.pi / 90
SHARP = 3e5


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
    """Pre-norm is dropped so that every channel keeps an exact logical meaning."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = D,
        n_heads: int = H,
        n_layers: int = 2,
        d_ff: int = len(LEXICON),
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
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def compile_pattern(entry: str, stoi: dict) -> list[list[int]]:
    """Return, for each shift, the character ids accepted at that shift."""
    stem = entry.endswith("*")
    word = entry.rstrip("*")
    slots = [[stoi[c]] for c in reversed(word)] + [[stoi[" "]]]  # left boundary
    if not stem:
        slots = [[stoi[" "], stoi["_"]]] + slots  # right boundary
    assert len(slots) <= W, entry
    return slots


@torch.no_grad()
def dev_report(model, task, show_errors: int = 0) -> None:
    """Accuracy on task examples that are not in the 200 eval examples."""
    model.eval()
    held = {e.prompt for e in task.generate_examples(200, seed=0)}
    dev = [e for e in task.generate_examples(2000, seed=1) if e.prompt not in held]
    ids = torch.tensor([task.encode(e.prompt) for e in dev])
    logits = torch.cat([model(b)[:, -1] for b in ids.split(256)])
    pred = [task.itos[i] for i in logits.argmax(-1).tolist()]
    right = [p == e.answer for p, e in zip(pred, dev)]
    pos = [r for r, e in zip(right, dev) if e.answer == "1"]
    neg = [r for r, e in zip(right, dev) if e.answer == "0"]
    print(f"DEV acc {sum(right)/len(dev):.4f} n={len(dev)} "
          f"tpr {sum(pos)/max(len(pos),1):.3f} tnr {sum(neg)/max(len(neg),1):.3f}")
    wrong = [e for r, e in zip(right, dev) if not r]
    for e in wrong[:show_errors]:
        print("ERR", e.answer, e.prompt.rstrip("_="))


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    stoi = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: one-hot character in head 0, rotating position in scratch.
        for c in range(task.vocab_size):
            model.token_emb.weight[c, c] = 1.0
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, POS0] = math.cos(OMEGA * t)
            model.pos_emb.weight[t, POS1] = math.sin(OMEGA * t)
        # Position 0 also looks like a space, so a word can start the sentence.
        model.pos_emb.weight[0, stoi[" "]] = 1.0

        # Block 1 attention: head i reads the character i steps back.
        a1 = model.blocks[0].attn
        for i in range(1, W):
            base = C * i
            co, si = math.cos(OMEGA * i), math.sin(OMEGA * i)
            a1.W_q.weight[base, POS0] = SHARP * co
            a1.W_q.weight[base, POS1] = SHARP * si
            a1.W_q.weight[base + 1, POS0] = -SHARP * si
            a1.W_q.weight[base + 1, POS1] = SHARP * co
            a1.W_k.weight[base, POS0] = 1.0
            a1.W_k.weight[base + 1, POS1] = 1.0
            for c in range(C):
                a1.W_v.weight[base + c, c] = 1.0
                a1.W_o.weight[base + c, base + c] = 1.0

        # Block 1 MLP: one AND gate per lexicon pattern.
        m1 = model.blocks[0].mlp
        for u, (entry, polarity) in enumerate(LEXICON):
            slots = compile_pattern(entry, stoi)
            for shift, chars in enumerate(slots):
                for c in chars:
                    m1.fc1.weight[u, C * shift + c] = 1.0
            m1.fc1.bias[u] = -(len(slots) - 1.0)
            m1.fc2.weight[SENT, u] = polarity

        # Block 2 attention: q = k = 0 gives uniform attention, so sum = mean * T.
        a2 = model.blocks[1].attn
        a2.W_v.weight[TOTAL, SENT] = 81.0
        a2.W_o.weight[TOTAL, TOTAL] = 1.0

        # Output: sign of the total evidence picks '1' or '0'.
        model.head.bias.fill_(-1e3)
        model.head.bias[stoi["1"]] = TIE_BIAS
        model.head.bias[stoi["0"]] = 0.0
        model.head.weight[stoi["1"], TOTAL] = 1.0
        model.head.weight[stoi["0"], TOTAL] = -1.0

    dev_report(model, task, show_errors=60)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "lex_shift14_tieneg"
model_description = "Same circuit as lex_shift14_flat, but phrases with no lexicon evidence default to negative instead of positive."


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
