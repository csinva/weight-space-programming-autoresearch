# Snapshot (excerpt, not the full file): PointerKeyCopy1L, accuracy 1.0000 (200/200), 1.42e4 params.
# Only build_model and write_weights differ from the template; the architecture classes are unchanged.
#
# build_model:
#     max_seq_len = task.seq_len - 1
#     SimpleTransformer(vocab_size=V, max_seq_len=max_seq_len, d_model=max_seq_len + V,
#                       n_heads=1, n_layers=1, d_ff=0)

def write_weights(model, task) -> None:
    # Layout: dims [0,11) "key id" (prompt position p), [11,22) "pointer"
    # (answer step j: which prompt position to read), [22,22+V) token one-hot.
    V = model.vocab_size
    N = 11
    TOK = 2 * N
    C = 5.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(N):
            model.pos_emb.weight[p, p] = 1.0            # prompt position p carries key id p
        blk = model.blocks[0]
        for j in range(N):
            s = {3: 3, 7: 7}.get(j, 8 - 4 * (j // 4) + j % 4)   # source prompt position for answer step j
            model.pos_emb.weight[N + j, N + j] = 1.0     # position 11+j carries pointer j
            blk.attn.W_q.weight[s, N + j] = C            # pointer j -> query for key id s
        for p in range(N):
            blk.attn.W_k.weight[p, p] = 1.0              # key id -> key
        for t in range(V):
            blk.attn.W_v.weight[TOK + t, TOK + t] = 1.0   # copy token identity
            blk.attn.W_o.weight[TOK + t, TOK + t] = 1.0
            model.head.weight[t, TOK + t] = 1.0
        model.final_ln.weight.fill_(1.0)
        for ln in (blk.ln1, blk.ln2):
            ln.weight.fill_(1.0)
