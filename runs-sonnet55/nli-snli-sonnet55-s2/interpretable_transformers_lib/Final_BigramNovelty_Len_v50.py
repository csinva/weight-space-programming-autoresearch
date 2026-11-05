"""
Snapshot: Final_BigramNovelty_Len_v50 (nli-snli, 0.52 on 200 samples; ~0.49-0.51 expected on larger samples).
Architecture: d_model=144, 2 heads (72 each), 3 layers, d_ff=4, no LayerNorm. Only write_weights shown;
the rest of the file (model classes, harness) is identical to interpretable_transformer.py.

  Residual layout: 0..33 TOK | 34 isHyp | 35 isPrem | 36 const | 37 is60 | 38 p | 39 p^2
                   40..73 PREV | 74 FOUND | 75 NOVFLAG | 76 REAL | 77 NOVMEAN | 78 REALMEAN
  L1 head0: previous-token head, score = -(j-(i-1))^2 from p, p^2 (pos 60 attends itself).
  L2 head0: query (TOK_i, PREV_i) vs key (TOK_j, PREV_j); full bigram match 2, partial <=1, sink '|' = SINK(1.5),
            non-premise keys -10 -> FOUND ~ 1 iff the hypothesis bigram occurs in the premise.
     MLP:   NOVFLAG = relu(1 - [tok=='_'] - FOUND), REAL = 1 - [tok=='_'].
  L3 head0: uniform mean over hypothesis positions of NOVFLAG and REAL.
  Unembed:  score_k = 60*(NOV[k]*NOVMEAN + LEN[k]*REALMEAN) + BIAS[k] + 100.
"""
import torch

SINK = 1.5
NOV, LEN, BIAS = [0, .25, .55], [0, .28, -.1875], [0, -14, 0]


def write_weights(model, task) -> None:
    stoi = task.stoi
    D, V = model.d_model, model.vocab_size
    SL = 60
    PAD = stoi["_"]
    TOK, ISHYP, ISPREM, CONST, IS60, P, P2 = 0, 34, 35, 36, 37, 38, 39
    PREV, FOUND, NOVF, REAL, NOVM, REALM = 40, 74, 75, 76, 77, 78
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for i in range(V):
            model.token_emb.weight[i, TOK + i] = 1.0
        pe = model.pos_emb.weight
        for pos in range(model.max_seq_len):
            pe[pos, CONST] = 1.0
            pe[pos, P] = float(pos)
            pe[pos, P2] = float(pos * pos)
            if pos < SL:
                pe[pos, ISPREM] = 1.0
            elif pos == SL:
                pe[pos, IS60] = 1.0
            elif pos <= 2 * SL:
                pe[pos, ISHYP] = 1.0
        a = model.blocks[0].attn
        G1 = 160.0
        a.W_q.weight[0, P] = G1; a.W_q.weight[0, CONST] = -G1; a.W_q.weight[0, IS60] = G1
        a.W_q.weight[1, CONST] = G1
        a.W_k.weight[0, P] = 2.0; a.W_k.weight[1, P2] = -1.0
        for i in range(V):
            a.W_v.weight[i, TOK + i] = 1.0
            a.W_o.weight[PREV + i, i] = 1.0
        a = model.blocks[1].attn
        G2 = 320.0
        for i in range(V):
            a.W_q.weight[i, TOK + i] = G2
            a.W_q.weight[V + i, PREV + i] = G2
            a.W_k.weight[i, TOK + i] = 1.0
            a.W_k.weight[V + i, PREV + i] = 1.0
        a.W_q.weight[2 * V, CONST] = G2
        a.W_k.weight[2 * V, CONST] = -10.0
        a.W_k.weight[2 * V, ISPREM] = 10.0
        a.W_k.weight[2 * V, IS60] = 10.0 + SINK
        a.W_v.weight[0, ISPREM] = 1.0
        a.W_o.weight[FOUND, 0] = 1.0
        m = model.blocks[1].mlp
        m.fc1.weight[0, TOK + PAD] = -1.0; m.fc1.weight[0, FOUND] = -1.0; m.fc1.bias[0] = 1.0
        m.fc2.weight[NOVF, 0] = 1.0
        m.fc1.weight[1, TOK + PAD] = -1.0; m.fc1.bias[1] = 1.0
        m.fc2.weight[REAL, 1] = 1.0
        a = model.blocks[2].attn
        a.W_q.weight[0, CONST] = 160.0
        a.W_k.weight[0, ISHYP] = 1.0
        a.W_v.weight[0, NOVF] = 1.0; a.W_v.weight[1, REAL] = 1.0
        a.W_o.weight[NOVM, 0] = 1.0; a.W_o.weight[REALM, 1] = 1.0
        for k in range(3):
            t = stoi[str(k)]
            model.head.weight[t, NOVM] = NOV[k] * SL
            model.head.weight[t, REALM] = LEN[k] * SL
            model.head.weight[t, CONST] = BIAS[k] + 100.0
