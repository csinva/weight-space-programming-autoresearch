# Snapshot of final_ff9_geom6 (acc 1.0, 200/200, 6.44e3 params). Architecture: d_model=33, n_heads=1, n_layers=1, d_ff=9.
# Harness / SimpleTransformer class unchanged apart from these defaults (omitted here).
def write_weights(model, task):
    """Sort = k-th smallest digit at answer step k.
    Residual: tok one-hot [0:11], K (k+1 at answer positions) [11], const [12], bag counts [13:23], logits [23:33].
    Attn (1 head): answer positions average the 5 prompt digits -> counts (K masks answer keys).
    MLP: unit d = relu(cum_le(d) - k) weighted 6^(9-d) into logit d; smallest d with cum_le(d)>k wins; 9 is default."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        D = model.d_model
        TOK, K, C, BAG, LOG = 0, 11, 12, 13, 23
        T = P = Cv = 10.0
        m1 = (T + Cv) / D
        s1 = math.sqrt((T * T + Cv * Cv) / D - m1 * m1 + 1e-5)
        c0 = T + Cv + 5
        for t in range(11):
            model.token_emb.weight[t, TOK + t] = T
        for p in range(model.pos_emb.num_embeddings):
            model.pos_emb.weight[p, C] = Cv
            if p >= 5:
                model.pos_emb.weight[p, K] = P * (p - 4)
        blk = model.blocks[0]
        at = blk.attn
        at.W_q.weight[0, C] = 1.0
        at.W_k.weight[0, K] = -30.0
        g1 = s1 / T
        h = m1 * s1 / (T * (Cv - m1))
        for e in range(10):
            at.W_v.weight[BAG + e, TOK + e] = g1
            at.W_v.weight[BAG + e, C] = h
            at.W_o.weight[BAG + e, BAG + e] = 5.0
        mlp = blk.mlp
        for d in range(9):
            sign, delta = -1.0, 0.0
            w = torch.zeros(D)
            for e in range(d + 1):
                w[BAG + e] = 1.0
            d_eff = delta - sign
            S = (w.sum() + sign / P + d_eff / Cv) / (1 - 1 / D - c0 / (D * Cv))
            w[K] = sign / P + S / D
            w[C] = (d_eff + c0 * S / D) / Cv
            mlp.fc1.weight[d] = w
            mlp.fc2.weight[LOG + d, d] = 6.0 ** (9 - d)
        for d in range(10):
            model.head.weight[d, LOG + d] = 1.0
        model.head.weight[9, C] = 0.05
        model.head.weight[10, C] = -1.0
