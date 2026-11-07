"""Snapshot ConstOne: baseline that ignores the prompt and always emits 001.
Architecture/harness identical to interpretable_transformer.py (SimpleTransformer, 3 layers, d=64);
only write_weights differs:

def write_weights(model, task):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for ln in [model.final_ln] + [m for b in model.blocks for m in (b.ln1, b.ln2)]:
            ln.weight.fill_(1.0)
        # position p in {7,8,9} lights up pos dim p; head reads it and emits "0","0","1"
        for p, digit in zip((7, 8, 9), (0, 0, 1)):
            model.pos_emb.weight[p, p] = 1.0
            model.head.weight[digit, p] = 1.0

accuracy 0.59 (118/200)
"""
