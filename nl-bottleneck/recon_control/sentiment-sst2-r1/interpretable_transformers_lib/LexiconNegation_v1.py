"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) lexicon-based sentiment classifier for sentiment-sst2.

Architecture (no LayerNorm, all weights closed form):
  residual = [token one-hot | position one-hot | K "look-back" slots of char one-hots | score | pooled]
  Layer 1  attention: K heads; head i attends (via position one-hots) exactly to position t-i
           and copies that token's one-hot into slot i.  -> each position sees its last K chars.
  Layer 1  MLP: one ReLU neuron per lexicon entry (word / negated phrase). The neuron fires (=1)
           at the delimiter after the word iff every char matches and the word is bounded by
           delimiters (space/_/= or start of sentence). fc2 writes the entry's polarity to `score`.
  Layer 2  attention: a single uniform head (zero Q/K) averages `score` over the prefix into `pooled`.
  Unembed: logit('1') - logit('0') = C * pooled + small positive bias.
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
# Lexicon (written from general knowledge of English sentiment words)
# ---------------------------------------------------------------------------

POS_STRONG = """
great best excellent masterpiece wonderful brilliant brilliantly superb terrific outstanding
beautiful beautifully amazing remarkable stunning delightful gorgeous fantastic marvelous
masterful exhilarating extraordinary exquisite perfect perfectly loved love lovely enjoyable
hilarious charming moving touching triumph gem treasure riveting
""".split()

POS = """
good better fine nice loves enjoy enjoyed enjoying fun funny entertaining entertain entertains
charm charms delight delights powerful compelling engaging engrossing fascinating intelligent smart
clever witty wit sweet warm heartfelt tender poignant honest sincere thoughtful insightful rich
solid strong worth worthy worthwhile recommend pleasure pleasant pleasing joy joyous joyful
satisfying success successful treat thrilling exciting captivating absorbing haunting beauty
spirited lively energetic vibrant inspiring inspired uplifting affecting laugh laughs laughter
humor humour humorous effective accomplished intriguing provocative sharp gripping suspenseful
elegant graceful grace lyrical luminous radiant glorious awesome cool classic unique special
admirable admire appealing likable likeable winning wins deft deftly skillful skillfully assured
confident ambitious epic wise wonder wonders magic magical marvel superior playful endearing
irresistible generous humane compassionate truth original inventive imaginative creative fresh
refreshing memorable impressive impressed well cinematic vivid ingenious brave bold resonant
rewarding rewards gentle hopeful heart hearts soul soulful passion passionate respect fine
breathtaking dazzling enchanting fascinating gift genius glad happy hearty highlight
intelligence keen masterfully nuanced polished potent precious quirky rare refreshingly
splendid stylish subtle surprising surprisingly thoughtfully timely tour unforgettable
unpretentious vital warmth wondrous zest enjoyably delicious delicate tasty pure
""".split()

NEG_STRONG = """
bad worst awful terrible horrible boring dull tedious mess waste wasted stupid pointless
disappointing dreadful unfunny lame mediocre unbearable insufferable excruciating unwatchable
worthless garbage trash disaster laughable atrocious ugly
""".split()

NEG = """
worse bored bore bores tiresome tired dumb silly messy poor poorly weak flat bland fails fail
failed failure flawed problem problems disappointment disappoint disappointed annoying irritating
pretentious predictable cliched cliche cliches formulaic contrived forced shallow empty hollow
lifeless plodding sluggish slow overlong lacks lacking lack mindless idiotic ridiculous absurd
preposterous embarrassing embarrassment unpleasant painful suffer suffers sloppy clumsy clumsily
inept incoherent confusing confused muddled uninspired unoriginal generic routine stale trite
banal numbing sappy cheesy cheap junk misses miss missing devoid falls obnoxious grating
repetitive monotonous lousy crap crappy stinks sucks shoddy amateurish hokey convoluted
overwrought overblown manipulative bloated unnecessary superficial depressing dreary nasty
vulgar offensive gross hate hated unfortunately badly barely mistake mistakes
tedium boredom dumber dud flop flops loses lost meandering mediocrity mess muddled
nonsense numbingly pathetic pity sadly shame shapeless slight sloppily squander squanders
tasteless thin ugly unconvincing undone uneven uninteresting uninvolving unlikable unsatisfying
vapid wearisome wears worn annoyingly awkward awkwardly bothered careless clueless dim
drag drags dragged exhausting fake feeble flimsy fluff frustrating gimmicky hackneyed
derivative incompetent inane insipid irritatingly limp listless lackluster misfire mishmash
murky neither nor overdone overwrought padded pedestrian plodding ponderous preachy
problematic pretension retread schmaltzy sour stiff stilted strained tepid tired tiresome
unimaginative unremarkable wooden heavy handed
""".split()

WEAK_NEG = """
too not n't no nothing never should only instead less least yet
""".split()

NEGATORS = ["not", "n't", "never", "no", "hardly", "barely"]


def build_lexicon():
    lex = {}
    for w in NEG:
        lex[w] = -1.0
    for w in POS:
        lex[w] = 1.0
    for w in NEG_STRONG:
        lex[w] = -2.0
    for w in POS_STRONG:
        lex[w] = 2.0
    for w in WEAK_NEG:
        lex.setdefault(w, -0.4)
    entries = dict(lex)
    # negated phrases: "not good" etc. flip the polarity of the following sentiment word.
    for neg in NEGATORS:
        for w, s in lex.items():
            if abs(s) >= 1.0:
                entries[neg + " " + w] = -1.6 * s  # net = s - 1.6 s = -0.6 s
    return entries


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard causal multi-head attention with arbitrary head dim."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """attn(look-back) -> MLP(word detectors) -> attn(uniform pooling) -> unembed. No LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, n_back: int = 24, d_ff: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_back = n_back
        V, P, K = vocab_size, max_seq_len, n_back
        self.off_tok = 0
        self.off_pos = V
        self.off_slot = V + P
        self.i_score = V + P + K * V
        self.i_pool = self.i_score + 1
        self.d_model = self.i_pool + 1
        D = self.d_model
        self.token_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(P, D)
        self.attn1 = MultiHeadAttention(D, K, P)
        self.mlp1 = MLP(D, d_ff)
        self.attn2 = MultiHeadAttention(D, 1, 1)
        self.head = nn.Linear(D, V)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        h = h + self.attn1(h)
        h = h + self.mlp1(h)
        h = h + self.attn2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    V, P, K, D = model.vocab_size, model.max_seq_len, model.n_back, model.d_model
    stoi = task.stoi
    entries = build_lexicon()
    entries = {w: s for w, s in entries.items() if len(w) + 2 <= K and all(c in stoi for c in w)}
    words = list(entries.keys())
    F_ = len(words)

    # resize MLP to the number of lexicon entries
    model.mlp1 = MLP(D, F_)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # embeddings: token one-hot, position one-hot
        for c in range(V):
            model.token_emb.weight[c, model.off_tok + c] = 1.0
        for p in range(P):
            model.pos_emb.weight[p, model.off_pos + p] = 1.0

        # layer-1 attention: head i attends to position t-i, copies token one-hot into slot i
        BIG = 60.0
        a = model.attn1
        dh = P
        scale = BIG * math.sqrt(dh)
        for i in range(K):
            for p in range(P):
                if p - i >= 0:
                    a.W_q.weight[i * dh + (p - i), model.off_pos + p] = scale
                a.W_k.weight[i * dh + p, model.off_pos + p] = 1.0
            for c in range(V):
                a.W_v.weight[i * dh + c, model.off_tok + c] = 1.0
                a.W_o.weight[model.off_slot + i * V + c, i * dh + c] = 1.0

        # layer-1 MLP: word detectors
        delims = [stoi[c] for c in " _="]
        m = model.mlp1
        for j, w in enumerate(words):
            n = len(w)
            # trailing delimiter at offset 0
            for c in delims:
                m.fc1.weight[j, model.off_slot + 0 * V + c] = 1.0
            # characters: w[k] at offset n-k
            for k, ch in enumerate(w):
                m.fc1.weight[j, model.off_slot + (n - k) * V + stoi[ch]] = 1.0
            # leading delimiter at offset n+1 (space only; '_' / '=' never precede a word) or start
            m.fc1.weight[j, model.off_slot + (n + 1) * V + stoi[" "]] = 1.0
            if n < P:
                m.fc1.weight[j, model.off_pos + n] = 1.0
            m.fc1.bias[j] = -(n + 1.0)
            m.fc2.weight[model.i_score, j] = entries[w]

        # layer-2 attention: uniform average of score -> pooled
        a2 = model.attn2
        a2.W_v.weight[0, model.i_score] = 1.0
        a2.W_o.weight[model.i_pool, 0] = 1.0

        # unembedding: only '0' / '1' are competitive
        C = 200.0
        model.head.bias.fill_(-1e4)
        model.head.bias[stoi["0"]] = 0.0
        model.head.bias[stoi["1"]] = 0.01
        model.head.weight[stoi["1"], model.i_pool] = C
        model.head.weight[stoi["0"], model.i_pool] = -C


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "LexiconNegation_v1"
model_description = ("Lookback attention (24 heads) + MLP word detectors for hand-written sentiment lexicon "
                     "with negation phrases, uniform pooling of polarity, sign -> 0/1.")


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
