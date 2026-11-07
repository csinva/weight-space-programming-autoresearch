"""Hand-written transformer for trec-coarse: a keyword decision list. No training.

Block 1 (window attention + MLP): head h reads the character h steps back via a Gaussian
position kernel; each MLP neuron is an AND gate over one keyword pattern.
Block 2 (decision attention): '=' attends to the position with the highest-priority keyword
(earliest wins ties) and copies that keyword's class vote. Nothing fired -> default class.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter

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
# Rules: (priority, class, keywords). Keywords get word boundaries on both sides;
# '^' anchors to the question start, trailing '*' makes it a stem.
# Classes: 0 ABBR, 1 ENTY, 2 DESC, 3 HUM, 4 LOC, 5 NUM
# ---------------------------------------------------------------------------
DEFAULT_CLASS = 1
RULES = [
    (9, 0, "stand for|stands for|abbreviat*|full form"),
    (9, 2, "come from|comes from"),
    (8, 3, "^who|^whom|^whose"),
    (8, 4, "where"),
    (8, 5, "^when"),
    (8, 2, "why"),
    (8, 1, "you say"),
    (7, 2, "definition|define|meaning|origin|mean|difference*|history|causes|caused|effects|reason|happened|prompted|purpose|describe"),
    (6, 2, "^how did|^how does|^how is|^how was|^how do|^how can|^how would|what makes"),
    (5, 5, "how"),          # 'how' with no verb after it asks for a quantity: how many, how far, how old, ...
    (5, 1, "made of|made from|consist of|kinds"),
    (5, 3, "last name|real name|first name|second name|maiden name|middle name|names of"),
    (5, 1, "film*|movie*|novel*|book*|song*|game*|sport*|drug*|disease*|animal*|dog*|cat|cats|bird*|fish*|food*|drink*|album*|show|series|magazine*|newspaper*|play|poem*|painting*|instrument*|car|cars|ship*|horse*|plant*|tree*|flower*|fruit*|vegetable*|metal*|element*|language*|religion*|product*|event*|war|battle*|award*|prize*|kind|type|word*|term*|color*|symbol*"),
    (5, 3, "person*|man|men|woman|women|people|boy*|girl*|actor*|actress*|president*|king*|queen*|emperor*|pope*|author*|writer*|poet*|novelist*|singer*|inventor*|compan*|team*|comedi*|player*|manufacturer*|organi*|leader*|nickname*|lawyer*|astronaut*|designer*|explorer*|artist*|painter*|composer*|scientist*|governor*|general|minister*|dictator*|director*|producer*|coach*|pitcher*|quarterback*|boxer*|golfer*|athlete*|detective*|doctor*|band|group*|firm*|husband*|wife|son|daughter*|brother*|sister*|father*|mother*|family|host*|profession*|job|character*|star|stars|business*|party|college*|universit*|school*|station*|soldier*|hero*|villain*|killer*|assassin*|founder*|occupation*|professor*|government*|agency|department*"),
    (5, 4, "city|cities|countr*|state|states|river*|mountain*|capital*|lake*|island*|continent*|ocean*|sea|county|province*|airport*|desert*|seaport*|region*|street*|building*|peak*|address*|stadium*|canal*|bridge*|park|museum*|hotel*|church*|temple*|tower*|volcano*|planet*|place*|location*|bay|gulf*|harbor*|town*|nation|nations|nationality|website*|site*|home page*|home of"),
    (5, 5, "year*|date*|day|population*|percent*|number*|cost*|price*|temperature*|speed*|length*|average*|size*|age|toll|probability*|rate|weight*|amount*|income*|value*|worth*|time*|centur*|decade*|month*|expectancy*|quantity*|fraction*|frequency*|dimension*|score*|odds|fare|season*"),
    (5, 2, "nature|function*|facts|information|believe*"),
    (4, 1, "fear of|called|call|what kind|what type|what color"),
    (4, 1, "^what is the|^what are the|^what are some*|^what does a|^what does an|^what does the*"),
    (3, 2, "^what is|^what are|^what do|^what does|^what 's a|^what 's an"),
]

LETTERS = "abcdefghijklmnopqrstuvwxyz"
BITS = 5        # letters a..z are coded 1..26 in binary; 0 means "not a letter"
BOUND, START, T, T2, ONE = 5, 6, 7, 8, 9   # input channels after the 5 letter bits
D_IN = 10
D_VAL = 8       # letter bits, boundary, start, position t
WINDOW = 14
GAMMA, BETA, EPS = 10.0, 100.0, 1.0


def compile_rules():
    out = []
    for prio, label, words in RULES:
        for w in words.split("|"):
            stem = w.endswith("*")
            w = w.rstrip("*")
            w = w if w.startswith("^") else " " + w
            out.append((w if stem else w + " ", label, prio))
    assert max(len(p) for p, _, _ in out) <= WINDOW
    return out


PATTERNS = compile_rules()
N_RULES = len(PATTERNS)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head attention; heads share one key and one value projection and are concatenated."""

    def __init__(self, d_in: int, n_heads: int, d_key: int, d_val: int):
        super().__init__()
        self.n_heads, self.d_key = n_heads, d_key
        self.W_q = nn.Linear(d_in, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_in, d_key, bias=False)
        self.W_v = nn.Linear(d_in, d_val, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, dk = self.n_heads, self.d_key
        q = self.W_q(x).view(B, L, H, dk).transpose(1, 2)
        scores = q @ self.W_k(x).unsqueeze(1).transpose(-2, -1)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        out = attn @ self.W_v(x).unsqueeze(1)       # B, H, L, d_val
        return out.transpose(1, 2).reshape(B, L, H * out.shape[-1])


class MLP(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 82):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_IN)
        self.pos_emb = nn.Embedding(max_seq_len, D_IN)
        self.window_attn = CausalSelfAttention(D_IN, WINDOW, 2, D_VAL)
        self.detect = MLP(WINDOW * D_VAL, N_RULES + 1, 8)      # -> [salience, 6 votes, 1]
        self.decide_attn = CausalSelfAttention(8, 1, 1, 6)
        self.head = nn.Linear(6, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        h = self.detect(self.window_attn(h))
        return self.head(self.decide_attn(h))


# ---------------------------------------------------------------------------
# Weights (closed form, from RULES)
# ---------------------------------------------------------------------------
@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    for tok, i in task.stoi.items():
        if tok in LETTERS:
            code = LETTERS.index(tok) + 1
            for b in range(BITS):
                model.token_emb.weight[i, b] = (code >> b) & 1
        else:
            model.token_emb.weight[i, BOUND] = 1
    for p in range(model.pos_emb.num_embeddings):
        model.pos_emb.weight[p, [T, T2, ONE]] = torch.tensor([p, p * p, 1.0])
    model.pos_emb.weight[0, [BOUND, START]] = 1     # the start acts as a word boundary

    # Head h: score(p) = -gamma (p - (t - h))^2 up to a constant, bilinear in (t, 1) and (p, p^2).
    a = model.window_attn
    for h in range(WINDOW):
        a.W_q.weight[2 * h, T] = 2 * GAMMA
        a.W_q.weight[2 * h, ONE] = -2 * GAMMA * h
        a.W_q.weight[2 * h + 1, ONE] = -GAMMA
    a.W_k.weight[0, T] = 1
    a.W_k.weight[1, T2] = 1
    for c in range(D_VAL):
        a.W_v.weight[c, c] = 1

    # Detector neurons: AND over the characters of each pattern.
    m = model.detect
    for n, (pat, label, prio) in enumerate(PATTERNS):
        ones = 0            # the neuron's activation is 1 only when every required bit matches
        for i, ch in enumerate(pat):
            slot = (len(pat) - 1 - i) * D_VAL
            if ch in LETTERS:
                code = LETTERS.index(ch) + 1
                for b in range(BITS):
                    bit = (code >> b) & 1
                    m.fc1.weight[n, slot + b] = 1 if bit else -1
                    ones += bit
            else:
                m.fc1.weight[n, slot + (START if ch == "^" else BOUND)] = 1
                ones += 1
        m.fc1.bias[n] = -(ones - 1)
        m.fc2.weight[0, n] = BETA * prio
        m.fc2.weight[1 + label, n] = prio
    m.fc1.weight[N_RULES, T] = 1            # neuron reading the position t (head 0)
    m.fc2.weight[0, N_RULES] = -EPS         # earlier keywords win ties
    m.fc2.bias[1 + DEFAULT_CLASS] = 0.5
    m.fc2.bias[7] = 1

    d = model.decide_attn
    d.W_q.weight[0, 7] = 1
    d.W_k.weight[0, 0] = 1
    for c in range(6):
        d.W_v.weight[c, 1 + c] = 1
        model.head.weight[task.stoi[str(c)], c] = 1
    if __name__ == "__main__":
        dev_report(model, task)


# ---------------------------------------------------------------------------
# Dev report: pool minus the held-out draw. Rule design only looks at these errors.
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, task, exs):
    ids = torch.tensor([task.encode(e.prompt) for e in exs])
    return [task.decode(row) for row in model(ids)[:, -1].argmax(-1, keepdim=True).tolist()]


def dev_report(model, task, n_show=12):
    held = task.generate_examples(200, seed=0)
    held_prompts = {e.prompt for e in held}
    dev = list({e.prompt: e for e in task.generate_examples(2000, seed=3) if e.prompt not in held_prompts}.values())
    preds = predict(model, task, dev)
    good = sum(p == e.answer for p, e in zip(preds, dev))
    print(f"DEV acc {good / len(dev):.4f} ({good}/{len(dev)})")
    print("DEV gold>pred:", sorted(Counter(e.answer + ">" + p for p, e in zip(preds, dev) if p != e.answer).items(), key=lambda x: -x[1])[:14])
    errs = [(e, p) for e, p in zip(dev, preds) if p != e.answer]
    step = max(1, len(errs) // n_show)
    for e, p in errs[::step][:n_show]:
        print(f"  {e.answer}>{p} {e.prompt.split('_')[0][:72]}")
    print("HELD draw count check:", sum(p == e.answer for p, e in zip(predict(model, task, held), held)))


model_shorthand_name = "e42_shared_key"
model_description = "All attention heads share one key projection (p, p^2) instead of keeping separate identical copies."


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
