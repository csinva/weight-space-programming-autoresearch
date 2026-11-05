# Snapshot (architecture defaults d_model=44, n_heads=1, n_layers=1, d_ff=40; harness unchanged, omitted)
def write_weights(model, task):
    """Sort = k-th smallest digit at answer step k.
    Residual: tok one-hot [0:11], pos one-hot [11:22], const [22], bag counts [23:33], logits [33:43].
    Attn (1 head): positions >=5 average the 5 prompt digits -> counts.
    MLP: digit d with cum_lt(d) <= k < cum_le(d); logit_d = step(k-cum_lt(d)) + step(cum_le(d)-k-1)."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        D = model.d_model
        TOK, POS, C, BAG, LOG = 0, 11, 22, 23, 33
        T = P = Cv = 10.0
        m1 = (T + P + Cv) / D
        s1 = math.sqrt((T * T + P * P + Cv * Cv) / D - m1 * m1 + 1e-5)
        m2 = (T + P + Cv + 5) / D
        for t in range(11):
            model.token_emb.weight[t, TOK + t] = T
        for p in range(model.pos_emb.num_embeddings):
            model.pos_emb.weight[p, POS + p] = P
            model.pos_emb.weight[p, C] = Cv
        blk = model.blocks[0]
        at = blk.attn
        at.W_q.weight[0, C] = 1.0
        for p in range(5, 11):
            at.W_k.weight[0, POS + p] = -30.0
        g1 = s1 / T
        h = m1 * s1 / (T * (Cv - m1))
        for e in range(10):
            at.W_v.weight[BAG + e, TOK + e] = g1
            at.W_v.weight[BAG + e, C] = h
            at.W_o.weight[BAG + e, BAG + e] = 5.0
        specs = (("A", 1.0, 1.0), ("A", 0.0, -1.0), ("B", 0.0, 1.0), ("B", -1.0, -1.0))
        mlp = blk.mlp
        u = 0
        for d in range(10):
            for typ, delta, out_sign in specs:
                sign = 1.0 if typ == "A" else -1.0
                w = torch.zeros(D)
                for k in range(6):
                    w[POS + 5 + k] = sign * k / P
                for e in range(10):
                    if typ == "A" and e < d:
                        w[BAG + e] = -1.0
                    if typ == "B" and e <= d:
                        w[BAG + e] = 1.0
                w[C] = (delta + m2 * w.sum()) / (Cv - m2)
                mlp.fc1.weight[u] = w
                mlp.fc2.weight[LOG + d, u] = out_sign
                u += 1
        for d in range(10):
            model.head.weight[d, LOG + d] = 1.0
        model.head.weight[10, C] = -1.0
