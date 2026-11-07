# Snapshot (excerpt, not the full file): ConstWeightCode1L_d15, accuracy 1.0000 (200/200), 2.18e3 params.
# Only build_model and write_weights differ from the template; the architecture classes are unchanged.
#
# build_model:
#     max_seq_len = task.seq_len - 1
#     SimpleTransformer(vocab_size=V, max_seq_len=max_seq_len, d_model=15,
#                       n_heads=1, n_layers=1, d_ff=0)

def write_weights(model, task) -> None:
    #   dims [0,4)   key code of a prompt position (4 bits, +-1)
    #   dims [4,8)   pointer code: code of the prompt position to copy (answer positions)
    #   dims [8,15)  token code (7 bits, exactly 3 are +1); attention adds the copied code here too
    V = model.vocab_size
    N = 11
    TOK = 8
    OUT = TOK   # no separate output slot
    C = 100.0

    def code(x, nbits):
        return torch.tensor([1.0 if (x >> b) & 1 else -1.0 for b in range(nbits)])

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # constant-weight token codes so LayerNorm's mean shift adds the same amount to every logit
        import itertools
        tok_codes = []
        for combo in itertools.combinations(range(7), 3):
            tok_codes.append(torch.tensor([1.0 if b in combo else -1.0 for b in range(7)]))
        for t in range(V):
            model.token_emb.weight[t, TOK:TOK + 7] = tok_codes[t]
            model.head.weight[t, OUT:OUT + 7] = tok_codes[t]   # logit_t = <code_t, copied code>
        for p in range(N):
            model.pos_emb.weight[p, 0:4] = code(p, 4)   # prompt position p carries its code as key
        blk = model.blocks[0]
        for j in range(N):
            s = {3: 3, 7: 7}.get(j, 8 - 4 * (j // 4) + j % 4)   # source prompt position for answer step j
            model.pos_emb.weight[N + j, 4:8] = code(s, 4)       # position 11+j carries the code of its source
        for b in range(4):
            blk.attn.W_q.weight[b, 4 + b] = C            # pointer code -> query code
            blk.attn.W_k.weight[b, b] = 1.0              # key code -> key
        for b in range(7):
            blk.attn.W_v.weight[OUT + b, TOK + b] = 40.0   # copy token code (gain 40 dominates own token)
            blk.attn.W_o.weight[OUT + b, OUT + b] = 1.0
        model.final_ln.weight.fill_(1.0)
        for ln in (blk.ln1, blk.ln2):
            ln.weight.fill_(1.0)
