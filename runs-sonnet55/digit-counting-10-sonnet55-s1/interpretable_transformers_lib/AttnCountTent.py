"""
Snapshot AttnCountTent (full harness omitted from snapshot header; see interpretable_transformer.py for harness).
Task digit-counting-10: 'ddddddddddq=' -> 2-digit count of q among the 10 digits.

Model: d_model=64, 1 head, 2 layers, d_ff=20.
Residual layout: 0/1 = +M/-M constant pair, 2..11 token one-hot, 12 isdata, 13..25 pos one-hot,
 26..35 query digit (copied by L1), 36 F (L2 attention match fraction), 37..46 digit logits, 63 balance.
"""
# write_weights (copy):
def write_weights(model, task):
    d = model.d_model
    for prm in model.parameters():
        prm.data.zero_()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
    M = 100.0
    sig = math.sqrt(d) / math.sqrt(2 * M * M + 8)
    T, ISD, POS, QD, F, OUT, BAL = 2, 12, 13, 26, 36, 37, 63
    stoi = task.stoi
    dig = [stoi[str(j)] for j in range(10)]
    for j in range(10):
        model.token_emb.weight.data[dig[j], T + j] = 1.0
        model.token_emb.weight.data[dig[j], BAL] = -1.0
    pe = model.pos_emb.weight.data
    for p in range(pe.shape[0]):
        pe[p, 0], pe[p, 1] = M, -M
        if p <= 12:
            pe[p, POS + p] = 1.0
            bal = 1.0
            if p <= 10:
                pe[p, ISD] = 1.0
                bal += 1.0
            if p == 12:
                pe[p, T + 0] = -1.0
                bal -= 1.0
            pe[p, BAL] = -bal
    a1 = model.blocks[0].attn
    a1.W_q.weight.data[0, 0] = 25 * 8 / (M * sig * sig)
    a1.W_k.weight.data[0, POS + 10] = 1.0
    for j in range(10):
        a1.W_v.weight.data[QD + j, T + j] = 1 / sig
        a1.W_o.weight.data[QD + j, QD + j] = 1.0
        a1.W_o.weight.data[BAL, QD + j] = -1.0
    a2 = model.blocks[1].attn
    Q, w = 30.0, 1 / 3
    P = Q + math.log(w)
    amp = math.sqrt(P * 8) / sig
    for j in range(10):
        a2.W_q.weight.data[j, QD + j] = amp
        a2.W_k.weight.data[j, T + j] = amp
    a2.W_q.weight.data[10, 0] = 1.0
    a2.W_k.weight.data[10, ISD] = -Q * 8 / (M * sig * sig)
    a2.W_v.weight.data[F, ISD] = 1 / sig
    a2.W_o.weight.data[F, F] = 1.0
    a2.W_o.weight.data[BAL, F] = -1.0
    gam = 40.0
    mlp = model.blocks[1].mlp
    for dd in range(10):
        th = (dd + 1) / (dd + 7)
        mlp.fc1.weight.data[2 * dd, F] = gam / sig
        mlp.fc1.bias.data[2 * dd] = -gam * th
        mlp.fc1.weight.data[2 * dd + 1, F] = -gam / sig
        mlp.fc1.bias.data[2 * dd + 1] = gam * th
        mlp.fc2.weight.data[OUT + dd, 2 * dd] = -1.0
        mlp.fc2.weight.data[OUT + dd, 2 * dd + 1] = -1.0
        mlp.fc2.weight.data[BAL, 2 * dd] = 1.0
        mlp.fc2.weight.data[BAL, 2 * dd + 1] = 1.0
    hw = model.head.weight.data
    for dd in range(10):
        hw[dig[dd], OUT + dd] = 1 / sig
        hw[dig[dd], 0] = 100 / (M * sig)
    hw[dig[0], POS + 11] = 1000 / sig

model_shorthand_name = "AttnCountTent"
