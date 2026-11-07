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
            
            # Balance out the sum to keep LayerNorm mean ≈ 0
            pos_emb[p, 1005] = -float(p)
            pos_emb[p, 1006] = -float(p)**2
            pos_emb[p, 1007] = -1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # L1 Attention: Word Boundary Extraction
        l1_attn = model.blocks[0].attn
        space_idx = VOCAB.index(' ')
        
        for k in range(n_heads):
            # Q looks for spaces that are approximately distance `k*4` away
            # We want to find the exact character just after the `k`th previous space?
            # Or just spread attention over the last word.
            # Let's do a smoothed lookback from the high-variance `WordBoundaryFeatures` model.
            
            # Instead of exact math trick, let's use the standard "distance from space" trick
            # but scale the distance
            
            # Head k looks for spaces, weighted by distance.
            # Actually, WordBoundaryFeatures (which scored 0.0405) used:
            # Q = distance preference
            # K = space detection
            # Let's recreate the essence of WordBoundaryFeatures but with random MLPs instead of linear projection
            
            # Q[k] is a simple slope
            slope = -1.0 + (k / max(1, n_heads - 1)) * 2.0
            l1_attn.W_q.weight[k*d_head + 0, 1002] = S * slope
            
            # K[k] responds strongly to spaces
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
            l1_attn.W_k.weight[k*d_head + 0, space_idx] = S * 10.0 # strong space affinity
            
            # V passes through all characters
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0

        # L1 MLP: Semantic Projection
        # Project 10 heads * 50 characters = 500 dimensions into 4000
        mlp1 = model.blocks[0].mlp
        random_proj = torch.randn(d_ff, 500) * 1.0
        
        for k in range(10):
            for i in range(50):
                mlp1.fc1.weight[:, k*50 + i] = random_proj[:, k*50 + i]
                
        mlp1.fc1.bias.data.zero_()
        mlp1.fc2.weight.data.copy_(torch.randn(d_model, d_ff) * 1.0)
        
        # Set LayerNorms
        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "WordBoundaryHash"
model_description = "Uses the successful WordBoundary space-seeking attention mechanism combined with a random 2-layer MLP to create a distributed semantic hash."
