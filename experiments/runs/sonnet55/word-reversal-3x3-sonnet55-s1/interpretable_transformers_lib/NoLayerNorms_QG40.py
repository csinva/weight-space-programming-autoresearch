"""Snapshot: NoLayerNorms_QG40 (acc 1.0). Architecture: 1 layer, 1 head, d_model = V+14 = 42,
Block = x + attn(x) (no LayerNorm, no MLP), final_ln = Identity, head = Linear(d_model, V, bias=False).
Only write_weights is shown; the rest is the template with those architecture edits.
"""
import torch


def write_weights(model, task):
    V, P = task.vocab_size, task.prompt_len
    TOK, POS, OUT = 0, V, 0
    SRC = [8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2]   # prompt position read at answer step j (words reversed)
    QG, OG = 40.0, 2.0                          # query gain (sharp attention), output gain
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for i in range(V):
            model.token_emb.weight[i, TOK + i] = 1.0
        # factorised position code (word one-hot 3 | char one-hot 4)
        for t in range(len(SRC)):
            model.pos_emb.weight[t, POS + t // 4] = 1.0
            model.pos_emb.weight[t, POS + 3 + t % 4] = 1.0
        for j, s in enumerate(SRC):
            model.pos_emb.weight[P - 1 + j, POS + 7 + s // 4] = 1.0
            model.pos_emb.weight[P - 1 + j, POS + 10 + s % 4] = 1.0
        blk = model.blocks[0]
        for d in range(7):
            blk.attn.W_k.weight[POS + d, POS + d] = 1.0
            blk.attn.W_q.weight[POS + d, POS + 7 + d] = QG
        for i in range(V):
            blk.attn.W_v.weight[OUT + i, TOK + i] = 1.0
            blk.attn.W_o.weight[OUT + i, OUT + i] = OG
            model.head.weight[i, OUT + i] = 1.0
