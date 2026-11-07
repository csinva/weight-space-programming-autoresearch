"""Snapshot: OneHeadPosCopy (accuracy 1.0 on word-reversal-3x3). Same file as interpretable_transformer.py;
architecture classes unchanged from the template (CausalSelfAttention, MLP, Block, SimpleTransformer).
Only the parts that differ from the template are reproduced below.

build_model:
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len,
                              d_model=2 * task.vocab_size + max_seq_len, n_heads=1, n_layers=1, d_ff=1)
"""

def write_weights(model, task) -> None:
    import torch
    V, P = task.vocab_size, task.prompt_len  # 28 tokens, prompt length 12
    T = task.seq_len                          # 23
    # residual layout: [0,V) token one-hot | [V,V+T) position one-hot | [V+T,2V+T) copied token
    TOK, POS, OUT = 0, V, V + T
    # answer index j (output predicted at position P-1+j) copies from prompt position SRC[j]
    SRC = [8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for ln in [model.final_ln] + [l for b in model.blocks for l in (b.ln1, b.ln2)]:
            ln.weight.fill_(1.0)
        for i in range(V):
            model.token_emb.weight[i, TOK + i] = 1.0
        for t in range(T):
            model.pos_emb.weight[t, POS + t] = 1.0
        blk = model.blocks[0]
        for t in range(T):
            blk.attn.W_k.weight[POS + t, POS + t] = 1.0          # key: my position
        for j, s in enumerate(SRC):
            blk.attn.W_q.weight[POS + s, POS + P - 1 + j] = 1.0  # query: position I want to read
        for i in range(V):
            blk.attn.W_v.weight[OUT + i, TOK + i] = 1.0          # value: token identity
            blk.attn.W_o.weight[OUT + i, OUT + i] = 1.0
            model.head.weight[i, OUT + i] = 1.0                  # unembed copied token
