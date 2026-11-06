"""Snapshot KeywordDecisionList_v1 (accuracy 0.795 on trec-coarse).

Identical model code to interpretable_transformer.py at the time of run 1
(harness section omitted from this snapshot; import it from the main file).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

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
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_head: int = 8):
        super().__init__()
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int | None = None,
                 n_heads: int | None = None, n_layers: int = 2, d_ff: int | None = None):
        super().__init__()
        d_model = d_model or D_MODEL
        n_heads = n_heads or N_OFFSETS
        d_ff = d_ff or len(PATTERNS)
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.n_heads, self.n_layers, self.d_ff = n_heads, n_layers, d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, d_head=8),
            Block(d_model, 1, 1, d_head=8),
        ])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


ABBR, ENTY, DESC, HUM, LOC, NUM = 0, 1, 2, 3, 4, 5


def _word(w):
    return [" " + w + " ", "^" + w + " "]


def _wstart(w):
    return [" " + w, "^" + w]


def _stem(w):
    return [w]


def _init(w):
    return ["^" + w + " "]


_RULES = []


def _add(prio, cls, *groups):
    pats = []
    for g in groups:
        pats.extend(g)
    _RULES.append((prio, cls, pats))


_add(14, ABBR, _stem("stands for"), _stem("stand for"), _stem("abbreviat"))
_add(13, DESC, _stem("comes from"))
_add(12, HUM, _init("who"))
_add(11, LOC, _word("where"))
_add(10, NUM, _init("when"))
_add(9, DESC, _word("why"))
_add(8, DESC, _wstart("mean"), _stem("origin"), _wstart("reason"), _wstart("defin"),
     _wstart("cause"), _wstart("purpose"), _wstart("explain"), _wstart("describe"),
     _wstart("differen"))
_add(7, DESC, _word("how did"), _word("how does"), _word("how is"), _word("how do"))
_add(6, NUM, _word("how"))
_add(5, DESC, ["^what is ????????_"])
_add(4, HUM, _word("man"), _wstart("woman"), _wstart("person"), _wstart("president"),
     _wstart("author"), _wstart("actor"), _wstart("actress"), _wstart("singer"),
     _wstart("king"), _wstart("queen"), _wstart("emperor"), _wstart("inventor"),
     _wstart("leader"), _wstart("player"), _wstart("writer"), _wstart("company"),
     _wstart("companies"), _wstart("team"), _wstart("composer"), _wstart("painter"),
     _wstart("artist"), _wstart("scientist"), _wstart("astronaut"), _wstart("director"),
     _wstart("founder"), _wstart("poet"), _wstart("senator"), _wstart("governor"),
     _wstart("pope"), _wstart("minister"), _wstart("comedian"), _wstart("musician"),
     _word("band"), _wstart("nickname"), _wstart("explorer"), _wstart("philosopher"),
     _wstart("general"), _wstart("husband"), _wstart("wife"), _wstart("son of"),
     _wstart("father"), _wstart("mother"), _wstart("character"))
_add(4, LOC, _wstart("countr"), _wstart("city"), _wstart("cities"), _wstart("state"),
     _wstart("capital"), _wstart("river"), _wstart("mountain"), _wstart("island"),
     _wstart("continent"), _wstart("ocean"), _word("sea"), _wstart("lake"),
     _wstart("located"), _wstart("province"), _wstart("county"), _wstart("nation"),
     _wstart("town"), _wstart("desert"), _wstart("airport"), _wstart("street"),
     _wstart("region"), _wstart("border"), _word("park"), _wstart("volcano"),
     _wstart("place"), _wstart("peninsula"), _wstart("canal"), _wstart("hemisphere"),
     _wstart("planet"), _wstart("village"), _wstart("territor"), _wstart("bridge"),
     _wstart("building"), _wstart("museum"))
_add(4, NUM, _wstart("year"), _wstart("date"), _wstart("population"), _wstart("number"),
     _wstart("percent"), _wstart("distance"), _wstart("temperature"), _wstart("speed"),
     _wstart("cost"), _wstart("price"), _word("age"), _word("old"), _wstart("height"),
     _word("tall"), _wstart("weigh"), _wstart("mile"), _wstart("degree"),
     _wstart("centur"), _wstart("average"), _word("day"), _wstart("length"),
     _word("size"), _word("rate"), _wstart("salary"), _wstart("dollar"), _word("worth"),
     _word("time"), _wstart("amount"), _wstart("depth"), _wstart("diameter"),
     _wstart("birthday"), _wstart("decade"), _wstart("hours"), _wstart("minutes"),
     _wstart("feet"), _wstart("pounds"), _word("area"), _wstart("lifespan"))
_add(4, ENTY, _wstart("animal"), _wstart("color"), _wstart("colour"), _wstart("food"),
     _wstart("book"), _wstart("movie"), _wstart("film"), _wstart("sport"), _wstart("game"),
     _wstart("language"), _wstart("instrument"), _wstart("disease"), _wstart("drug"),
     _wstart("bird"), _wstart("dog"), _word("cat"), _wstart("plant"), _wstart("flower"),
     _word("car"), _word("cars"), _wstart("song"), _wstart("novel"), _word("play"),
     _word("war"), _wstart("currency"), _wstart("religion"), _wstart("fruit"),
     _wstart("vegetable"), _wstart("fish"), _wstart("tree"), _wstart("breed"),
     _wstart("product"), _wstart("brand"), _wstart("drink"), _wstart("beer"),
     _wstart("wine"), _wstart("dish"), _wstart("symbol"), _wstart("award"),
     _wstart("prize"), _wstart("invention"), _wstart("material"), _wstart("metal"),
     _wstart("element"), _wstart("chemical"), _wstart("substance"), _wstart("weapon"),
     _wstart("vehicle"), _wstart("ship"), _wstart("boat"), _wstart("plane"),
     _wstart("aircraft"), _wstart("magazine"), _wstart("newspaper"), _wstart("show"),
     _wstart("program"), _word("term"), _wstart("word"), _wstart("medicine"),
     _wstart("virus"), _wstart("insect"), _word("organ"), _wstart("horse"),
     _wstart("snake"), _wstart("mammal"), _wstart("music"), _wstart("dance"),
     _wstart("toy"), _wstart("cocktail"), _wstart("sauce"), _wstart("cheese"),
     _wstart("gas"), _wstart("mineral"), _wstart("stone"), _wstart("gem"),
     _wstart("hobby"), _wstart("technique"), _wstart("method"),
     _wstart("system"), _wstart("computer"), _wstart("software"), _wstart("painting"),
     _wstart("statue"), _wstart("sculpture"), _wstart("poem"), _wstart("opera"),
     _wstart("album"), _wstart("series"), _wstart("cartoon"), _wstart("comic"),
     _wstart("battle"), _wstart("treaty"), _wstart("dessert"))
_add(3, ENTY, _stem("called"), _wstart("what kind"), _wstart("what type"))
_add(2, ENTY, _word("what are the"))
_add(1, DESC, _init("what is"))


def _parse(p):
    toks, i = [], 0
    while i < len(p):
        c = p[i]
        if c == "^":
            toks.append(("L", p[i + 1], True)); i += 2; continue
        if c == " ":
            toks.append(("B", None, False))
        elif c == "_":
            toks.append(("P", None, False))
        elif c == "?":
            toks.append(("W", None, False))
        else:
            toks.append(("L", c, False))
        i += 1
    return toks


PATTERNS = []
for _prio, _cls, _pats in _RULES:
    for _p in _pats:
        PATTERNS.append((_prio, _cls, _parse(_p), _p))

N_OFFSETS = 20
N_FEAT = 8
F_BITS, F_BND, F_START, F_PAD = 0, 5, 6, 7
D_ONE, D_POS, D_POS2 = 8, 9, 10
D_COPY = 11
D_SAL = D_COPY + N_OFFSETS * N_FEAT
D_VOTE = D_SAL + 1
D_OUT = D_VOTE + 6
D_MODEL = D_OUT + 6


def _code(ch):
    if "a" <= ch <= "z":
        return ord(ch) - ord("a") + 1
    if ch == "'":
        return 27
    return 0


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        E = model.token_emb.weight
        for tok, idx in task.stoi.items():
            c = _code(tok)
            for b in range(5):
                E[idx, F_BITS + b] = 1.0 if (c >> b) & 1 else -1.0
            E[idx, F_BND] = 1.0 if tok in " _=" else -1.0
            E[idx, F_PAD] = 1.0 if tok == "_" else -1.0
        P = model.pos_emb.weight
        for j in range(model.max_seq_len):
            P[j, F_START] = 1.0 if j == 0 else -1.0
            P[j, D_ONE] = 1.0
            P[j, D_POS] = float(j)
            P[j, D_POS2] = float(j * j)
        b1 = model.blocks[0]
        A = b1.attn
        dh = A.d_head
        beta = 20.0 * math.sqrt(dh)
        for h in range(N_OFFSETS):
            r = h * dh
            A.W_q.weight[r + 0, D_POS] = 2 * beta
            A.W_q.weight[r + 0, D_ONE] = -2 * beta * h
            A.W_q.weight[r + 1, D_ONE] = -beta
            A.W_k.weight[r + 0, D_POS] = 1.0
            A.W_k.weight[r + 1, D_POS2] = 1.0
            for f in range(N_FEAT):
                A.W_v.weight[r + f, f] = 1.0
                A.W_o.weight[D_COPY + h * N_FEAT + f, r + f] = 1.0
        W1, B1, W2 = b1.mlp.fc1.weight, b1.mlp.fc1.bias, b1.mlp.fc2.weight
        for n, (prio, cls, cons, _) in enumerate(PATTERNS):
            L = len(cons)
            M = 0
            for k, (kind, ch, start) in enumerate(cons):
                off = L - 1 - k
                base = D_COPY + off * N_FEAT
                if kind == "L":
                    c = _code(ch)
                    for b in range(5):
                        W1[n, base + F_BITS + b] = 1.0 if (c >> b) & 1 else -1.0
                    M += 5
                    if start:
                        W1[n, base + F_START] = 1.0
                        M += 1
                elif kind == "B":
                    W1[n, base + F_BND] = 1.0; M += 1
                elif kind == "P":
                    W1[n, base + F_PAD] = 1.0; M += 1
            B1[n] = -(M - 1)
            W2[D_SAL, n] = 300.0 * prio
            W2[D_VOTE + cls, n] = 1.0
        A2 = model.blocks[1].attn
        lam = 10.0 * math.sqrt(A2.d_head)
        A2.W_q.weight[0, D_ONE] = lam
        A2.W_k.weight[0, D_SAL] = 1.0
        A2.W_k.weight[0, D_POS] = -3.0
        for c in range(6):
            A2.W_v.weight[c, D_VOTE + c] = 1.0
            A2.W_o.weight[D_OUT + c, c] = 1.0
        Hd = model.head.weight
        for c in range(6):
            Hd[task.stoi[str(c)], D_OUT + c] = 1.0
        Hd[task.stoi[str(ENTY)], D_ONE] = 0.5


model_shorthand_name = "KeywordDecisionList_v1"
model_description = ("2-block hand-built keyword decision list: 20 Gaussian copy heads + ~400 exact AND-gate "
                     "keyword neurons; '=' attends to max 300*priority-3*pos and copies the class vote; default ENTY.")
