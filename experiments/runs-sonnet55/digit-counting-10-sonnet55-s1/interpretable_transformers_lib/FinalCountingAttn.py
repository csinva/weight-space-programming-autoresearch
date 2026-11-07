# Snapshot of FinalCountingAttn (100% on digit-counting-10, 7.29e3 params).
# Architecture: d_model=27, n_heads=1, n_layers=2, d_ff=1 (MLP weights all zero => no MLP), same harness as interpretable_transformer.py.
# Earlier variants differ only in constants (M, Q, w, gam, L1 score, d_model, MLP vs parabola head); see results/overall_results.csv.
def write_weights(model, task):
    d = model.d_model
    for prm in model.parameters():
        prm.data.zero_()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
    M = 30.0
    sig = math.sqrt(d) / math.sqrt(2 * M * M + 8)  # LayerNorm scale (constant pair dominates norm)
    T, ISD, P10, P11, QD, F, BAL = 2, 12, 13, 14, 15, 25, 26
    stoi = task.stoi
    dig = [stoi[str(j)] for j in range(10)]
    for j in range(10):
        model.token_emb.weight.data[dig[j], T + j] = 1.0
        model.token_emb.weight.data[dig[j], BAL] = -1.0
    model.token_emb.weight.data[stoi["="], P11] = 1.0
    model.token_emb.weight.data[stoi["="], BAL] = -1.0
    pe = model.pos_emb.weight.data
    for p in range(pe.shape[0]):
        pe[p, 0], pe[p, 1] = M, -M
        if p <= 12:
            bal = 0.0
            if p == 10:
                pe[p, P10] = 1.0
                bal += 1.0
            if p <= 10:
                pe[p, ISD] = 1.0
                bal += 1.0
            if p == 12:
                pe[p, T + 0] = -1.0
                bal -= 1.0
            pe[p, BAL] = -bal
    a1 = model.blocks[0].attn
    a1.W_q.weight.data[0, 0] = 15 * math.sqrt(d) / (M * sig * sig)
    a1.W_k.weight.data[0, P10] = 1.0
    for j in range(10):
        a1.W_v.weight.data[QD + j, T + j] = 1 / sig
        a1.W_o.weight.data[QD + j, QD + j] = 1.0
        a1.W_o.weight.data[BAL, QD + j] = -1.0
    a2 = model.blocks[1].attn
    Q, w = 10.0, 1 / 12
    P = Q + math.log(w)
    amp = math.sqrt(P * math.sqrt(d)) / sig
    for j in range(10):
        a2.W_q.weight.data[j, QD + j] = amp
        a2.W_k.weight.data[j, T + j] = amp
    a2.W_q.weight.data[10, 0] = 1.0
    a2.W_k.weight.data[10, ISD] = -Q * math.sqrt(d) / (M * sig * sig)
    a2.W_v.weight.data[F, ISD] = 1 / sig
    a2.W_o.weight.data[F, F] = 1.0
    a2.W_o.weight.data[BAL, F] = -1.0
    gam = 1000.0
    hw = model.head.weight.data
    for dd in range(10):
        th = (dd + 1) * w / ((dd + 1) * w + 2)
        hw[dig[dd], F] = gam * 2 * th / sig
        hw[dig[dd], 0] = (100 - gam * th * th) / (M * sig)
    hw[dig[0], P11] = 1000 / sig

model_shorthand_name = "FinalCountingAttn"
