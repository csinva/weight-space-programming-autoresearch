# Snapshot (excerpt, not the full file): BinaryCodeAll1L_C10, accuracy 1.0000 (200/200), 2.83e3 params.
# Only build_model and write_weights differ from the template; the architecture classes are unchanged.
#
# build_model:
#     max_seq_len = task.seq_len - 1
#     SimpleTransformer(vocab_size=V, max_seq_len=max_seq_len, d_model=18,
#                       n_heads=1, n_layers=1, d_ff=0)

def write_weights(model, task) -> None:
    # Fully binary-coded layout (all codes are +-1 bits):
    #   dims [0,4)   key code of a prompt position (4 bits)
    #   dims [4,8)   pointer code: code of the prompt position to copy (answer positions)
    #   dims [8,13)  token code (5 bits)
    #   dims [13,18) output slot: attention writes the copied token code here
    V = model.vocab_size
    N = 11
    TOK, OUT = 8, 13
    C = 10.0

    def code(x, nbits):
        return torch.tensor([1.0 if (x >> b) & 1 else -1.0 for b in range(nbits)])

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK:TOK + 5] = code(t, 5)
            model.head.weight[t, OUT:OUT + 5] = code(t, 5)   # logit_t = <code_t, copied code>
        for p in range(N):
            model.pos_emb.weight[p, 0:4] = code(p, 4)   # prompt position p carries its code as key
        blk = model.blocks[0]
        for j in range(N):
            s = {3: 3, 7: 7}.get(j, 8 - 4 * (j // 4) + j % 4)   # source prompt position for answer step j
            model.pos_emb.weight[N + j, 4:8] = code(s, 4)       # position 11+j carries the code of its source
        for b in range(4):
            blk.attn.W_q.weight[b, 4 + b] = C            # pointer code -> query code
            blk.attn.W_k.weight[b, b] = 1.0              # key code -> key
        for b in range(5):
            blk.attn.W_v.weight[OUT + b, TOK + b] = 1.0   # copy token code
            blk.attn.W_o.weight[OUT + b, OUT + b] = 1.0
        model.final_ln.weight.fill_(1.0)
        for ln in (blk.ln1, blk.ln2):
            ln.weight.fill_(1.0)
