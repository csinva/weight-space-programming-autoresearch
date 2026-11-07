import math
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB

def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size
        d_ff = model.blocks[0].mlp.fc1.out_features
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head

        # LayerNorm pass-through trick
        C = 100000.0
        S = C * math.sqrt(2.0 / d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            token_emb[i, i] = 1.0 # 0..49
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1003] = float(p)**2
            pos_emb[p, 1004] = 1.0
            
            pos_emb[p, 1005] = -float(p)
            pos_emb[p, 1006] = -float(p)**2
            pos_emb[p, 1007] = -1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # Let's take the best performing WordBoundaryFeatures (scored 0.0405)
        # but scale its representation cleanly.
        
        # In WordBoundaryFeatures, they did NOT use the LayerNorm pass-through trick.
        # They just used standard MLPs with very high variance.
        # The exact trick we are doing might be over-fitting? 
        # Notice that train_corr=0.4893 but test_corr=0.0148 for LexiconBoundaryMix!
        # This is MASSIVE overfitting.
        
        # WordBoundaryFeatures:
        # train_corr=0.2004, test_corr=0.0405
        
        # The problem with LayerNorm pass-through + large random weights is that it perfectly memorizes
        # the training set. It memorizes every single exact char sequence.
        
        # If we want to beat WordBoundaryFeatures, we should build on it.
        # Let's write the exact WordBoundaryFeatures logic, but improve it.
        # WordBoundaryFeatures just had:
        # pos_emb = 100.0
        # W_q = slope
        # W_k = look for spaces
        # MLP1/MLP2 = standard gaussian with high variance.
