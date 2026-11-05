# Snapshot (excerpt, not the full file): PosCopy1L_T22_dff0_C5, accuracy 1.0000 (200/200), 1.42e4 params.
# Only build_model and write_weights differ from the template; the architecture classes are unchanged.
#
# build_model:
#     max_seq_len = task.seq_len - 1
#     SimpleTransformer(vocab_size=V, max_seq_len=max_seq_len, d_model=max_seq_len + V,
#                       n_heads=1, n_layers=1, d_ff=0)

def write_weights(model, task) -> None:
    # Residual layout: dims [0,T) one-hot position, [T,T+V) one-hot token.
    T, V = model.max_seq_len, model.vocab_size
    TOK = T
    C = 5.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(T):
            model.pos_emb.weight[p, p] = 1.0
        # source position for query position q = 11 + j (prediction of answer char j)
        src = {}
        for j in range(11):
            src[11 + j] = {3: 3, 7: 7}.get(j, 8 - 4 * (j // 4) + j % 4)
        blk = model.blocks[0]
        for q, s in src.items():
            blk.attn.W_q.weight[s, q] = C      # query asks for position s(q)
        for q in range(T):
            blk.attn.W_k.weight[q, q] = 1.0    # key advertises its own position
        for t in range(V):
            blk.attn.W_v.weight[TOK + t, TOK + t] = 1.0   # copy token identity
            blk.attn.W_o.weight[TOK + t, TOK + t] = 1.0
            model.head.weight[t, TOK + t] = 1.0
        model.final_ln.weight.fill_(1.0)
        for ln in (blk.ln1, blk.ln2):
            ln.weight.fill_(1.0)
