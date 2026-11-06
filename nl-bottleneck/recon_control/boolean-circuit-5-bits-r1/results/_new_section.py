# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------
# No LayerNorm, so the residual stream holds exact 0/1 features.
# Attention: 9 heads; head j copies the token at position j into its own
#            residual slot (a positional "gather" at the '=' token).
# MLP_1..MLP_4: MLP i computes r_i = op_i(r_{i-1}, b_i). Each of the 12
#            (op, r, b) combos is detected by a single ReLU unit.
# Head: logit('1') = G*r_4, logit('0') = G*(1 - r_4).

class GatherAttention(nn.Module):
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
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# residual layout
D_TOK = 0          # 6 dims: token one-hot (vocab "01&|^=")
D_CONST = 6        # 1 dim: constant 1
D_POS = 7          # 16 dims: position one-hot
D_BIT = 23         # 5 dims: gathered bits b_0..b_4
D_OP = 28          # 4 x 3 dims: gathered op one-hots for op_1..op_4
D_R = 40           # 4 dims: r_1..r_4 (r_0 = b_0)
D_MODEL = 48
N_HEADS, D_HEAD = 9, 3
N_STEPS = 4


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.attn = GatherAttention(D_MODEL, N_HEADS, D_HEAD)
        self.mlps = nn.ModuleList([MLP(D_MODEL, 12) for _ in range(N_STEPS)])
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        h = h + self.attn(h)
        for mlp in self.mlps:
            h = h + mlp(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    s = task.stoi
    # embeddings: token one-hot + constant 1, position one-hot
    for c, i in s.items():
        model.token_emb.weight[i, D_TOK + i] = 1.0
        model.token_emb.weight[i, D_CONST] = 1.0
    for p in range(model.max_seq_len):
        model.pos_emb.weight[p, D_POS + p] = 1.0

    # gather attention: head j's score is BIG * [key pos == j]
    BIG = 60.0
    ops = ["&", "|", "^"]
    a = model.attn
    for j in range(9):
        base = j * D_HEAD
        a.W_q.weight[base, D_CONST] = BIG
        a.W_k.weight[base, D_POS + j] = 1.0
        if j % 2 == 0:  # bit position -> b_{j/2}
            a.W_v.weight[base, D_TOK + s["1"]] = 1.0
            a.W_o.weight[D_BIT + j // 2, base] = 1.0
        else:           # op position -> one-hot of op_{(j+1)/2}
            i = (j - 1) // 2
            for t, o in enumerate(ops):
                a.W_v.weight[base + t, D_TOK + s[o]] = 1.0
                a.W_o.weight[D_OP + 3 * i + t, base + t] = 1.0

    # MLP steps: r_i = op_i(r_{i-1}, b_i)
    truth = {
        "&": lambda x, y: x & y,
        "|": lambda x, y: x | y,
        "^": lambda x, y: x ^ y,
    }
    for i in range(1, N_STEPS + 1):
        m = model.mlps[i - 1]
        r_dim = D_BIT if i == 1 else D_R + (i - 2)
        b_dim = D_BIT + i
        u = 0
        for t, o in enumerate(ops):
            for rv in (0, 1):
                for bv in (0, 1):
                    # unit = relu(op + [r==rv] + [b==bv] - 2) is 1 iff exact match
                    m.fc1.weight[u, D_OP + 3 * (i - 1) + t] = 1.0
                    m.fc1.weight[u, r_dim] = 1.0 if rv else -1.0
                    m.fc1.weight[u, b_dim] = 1.0 if bv else -1.0
                    m.fc1.bias[u] = -2.0 + (0.0 if rv else 1.0) + (0.0 if bv else 1.0)
                    if truth[o](rv, bv):
                        m.fc2.weight[D_R + (i - 1), u] = 1.0
                    u += 1

    # readout (other tokens get logit 0 < G)
    G = 10.0
    model.head.weight[s["1"], D_R + N_STEPS - 1] = G
    model.head.weight[s["0"], D_R + N_STEPS - 1] = -G
    model.head.weight[s["0"], D_CONST] = G


model_shorthand_name = "GatherThenChainMLP"
model_description = ("No-LN; 9-head positional gather attention copies bits/ops to the '=' slot, "
                     "then 4 ReLU MLPs each compute r_i=op_i(r_{i-1},b_i) via 12 combo-detector units.")


