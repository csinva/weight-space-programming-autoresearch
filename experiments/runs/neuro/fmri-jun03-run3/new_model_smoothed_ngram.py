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

        # Let's try combining the High Variance MLP from WordBoundaryFeatures
        # with the exact position trick.
        # If we use *extreme* variance on the random projection, we get a highly distinguishing hash.
        
        l1_attn = model.blocks[0].attn
        beta = 100.0
        
        for k in range(n_heads):
            # Exact Relative Position (k-th character from end)
            l1_attn.W_q.weight[k*d_head + 0, 1002] = S * 2.0 * beta
            l1_attn.W_q.weight[k*d_head + 0, 1004] = S * (-2.0 * beta * k)
            l1_attn.W_q.weight[k*d_head + 2, 1004] = S * 1.0
            
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
            l1_attn.W_k.weight[k*d_head + 1, 1004] = S * 1.0
            l1_attn.W_k.weight[k*d_head + 2, 1003] = S * (-beta)
            
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0

        # L1 MLP: Extreme High Variance Hash
        # No lexicon matching, just massive random noise like WordBoundaryFeatures
        mlp1 = model.blocks[0].mlp
        
        # We need this to NOT overfit perfectly to the train set.
        # How does WordBoundaryFeatures not overfit?
        # It relies on broad, smooth features (space counting, char frequency).
        # When we provide EXACT 10-char sequences, the MLP acts as a perfect hashing function
        # mapping every 10-gram to an orthogonal vector.
        # This means the regression model perfectly learns the train set and fails on the test set.
        
        # To fix this, we should NOT provide exact 10-char sequences!
        # We should provide smooth bags of characters, like WordBoundaryFeatures.
        
        # So wait, the "math trick" to exact-extract characters actually HURTS generalization
        # because it allows perfect memorization!
        
        # Let's completely drop the exact extraction. Let's make "BagOfNgrams"
        # We'll have heads that look for character bigrams or trigrams, but smoothly over the whole word.
        
        # Actually, WordBoundaryFeatures scored 0.0405.
        # Our recreation "WordBoundaryFeaturesRecreated" scored 0.0247.
        # Wait... our recreation was much worse! Why?
