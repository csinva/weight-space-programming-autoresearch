# Snapshot: PosCopy1L (accuracy 1.0000, 200/200, 3.19e4 params)
# Only write_weights / build_model differ from the template; architecture classes unchanged.
#
# build_model:
#     SimpleTransformer(vocab_size=V, max_seq_len=max(seq_len,16), d_model=max_seq_len+2*V,
#                       n_heads=1, n_layers=1, d_ff=1)
#
def write_weights(model, task) -> None:
    # Residual layout: dims [0,T) one-hot position, [T,T+V) one-hot token,
    # [T+V,T+2V) "output" slot where attention writes the copied token.
    T, V = model.max_seq_len, model.vocab_size
    TOK, OUT = T, T + V
    C = 10.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(T):
            model.pos_emb.weight[p, p] = 1.0
        # source position for query position q (prediction of char at q+1)
        src = {q: q for q in range(T)}
        for j in range(11):
            src[11 + j] = {3: 3, 7: 7}.get(j, 8 - 4 * (j // 4) + j % 4)
        blk = model.blocks[0]
        for q, s in src.items():
            blk.attn.W_q.weight[s, q] = C      # query asks for position s(q)
            blk.attn.W_k.weight[q, q] = 1.0    # key advertises its own position
        for t in range(V):
            blk.attn.W_v.weight[OUT + t, TOK + t] = 1.0   # copy token identity
            blk.attn.W_o.weight[OUT + t, OUT + t] = 1.0
            model.head.weight[t, OUT + t] = 1.0
        model.final_ln.weight.fill_(1.0)
        for ln in (blk.ln1, blk.ln2):
            ln.weight.fill_(1.0)
