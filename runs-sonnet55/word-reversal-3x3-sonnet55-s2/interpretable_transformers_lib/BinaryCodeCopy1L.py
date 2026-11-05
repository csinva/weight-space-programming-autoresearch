# Snapshot (excerpt, not the full file): BinaryCodeCopy1L, accuracy 1.0000 (200/200), 8.24e3 params.
# Only build_model and write_weights differ from the template; the architecture classes are unchanged.
#
# build_model:
#     max_seq_len = task.seq_len - 1
#     SimpleTransformer(vocab_size=V, max_seq_len=max_seq_len, d_model=8 + V,
#                       n_heads=1, n_layers=1, d_ff=0)

def write_weights(model, task) -> None:
    # dims [0,4) key code, [4,8) pointer code (each +-1 bits of a prompt position),
    # [8,8+V) token one-hot.
    V = model.vocab_size
    N = 11
    TOK = 8
    C = 1.0

    def code(x):
        return [1.0 if (x >> b) & 1 else -1.0 for b in range(4)]

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(N):
            model.pos_emb.weight[p, 0:4] = torch.tensor(code(p))   # prompt position p carries its code as key
        blk = model.blocks[0]
        for j in range(N):
            s = {3: 3, 7: 7}.get(j, 8 - 4 * (j // 4) + j % 4)   # source prompt position for answer step j
            model.pos_emb.weight[N + j, 4:8] = torch.tensor(code(s))   # position 11+j carries the code of its source
        for b in range(4):
            blk.attn.W_q.weight[b, 4 + b] = C            # pointer code -> query code
            blk.attn.W_k.weight[b, b] = 1.0              # key code -> key
        for t in range(V):
            blk.attn.W_v.weight[TOK + t, TOK + t] = 1.0   # copy token identity
            blk.attn.W_o.weight[TOK + t, TOK + t] = 1.0
            model.head.weight[t, TOK + t] = 1.0
        model.final_ln.weight.fill_(1.0)
        for ln in (blk.ln1, blk.ln2):
            ln.weight.fill_(1.0)
