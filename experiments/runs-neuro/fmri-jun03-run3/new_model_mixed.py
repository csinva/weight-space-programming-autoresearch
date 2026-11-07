import math
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB
from top_words import TOP_WORDS

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

        # L1 Attention: Exact Relative Position for heads 0-4, Word Boundary for heads 5-9
        l1_attn = model.blocks[0].attn
        beta = 100.0
        space_idx = VOCAB.index(' ')
        
        for k in range(n_heads):
            if k < 5:
                # Exact Relative Position (k-th character from end)
                # Q = S * [2*beta*p, -2*beta*k, 1]
                l1_attn.W_q.weight[k*d_head + 0, 1002] = S * 2.0 * beta
                l1_attn.W_q.weight[k*d_head + 0, 1004] = S * (-2.0 * beta * k)
                l1_attn.W_q.weight[k*d_head + 2, 1004] = S * 1.0
                
                # K = S * [m, 1, -beta*m^2]
                l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
                l1_attn.W_k.weight[k*d_head + 1, 1004] = S * 1.0
                l1_attn.W_k.weight[k*d_head + 2, 1003] = S * (-beta)
            else:
                # Word Boundary spaces
                slope = -1.0 + ((k-5) / 4) * 2.0
                l1_attn.W_q.weight[k*d_head + 0, 1002] = S * slope
                l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0
                l1_attn.W_k.weight[k*d_head + 0, space_idx] = S * 10.0 # strong space affinity
            
            # V passes through characters
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0

        # L1 MLP: Mixed Lexicon Match and Random Projection
        mlp1 = model.blocks[0].mlp
        
        # We have 4000 hidden units.
        # First 2000: Matched filter for words up to 5 chars (using heads 0-4)
        for w_idx, W in enumerate(TOP_WORDS[:2000]):
            W_padded = ' ' + W
            chars_from_end = list(reversed(W_padded))[:5]
            
            for k, c in enumerate(chars_from_end):
                if c in VOCAB:
                    idx = VOCAB.index(c)
                    dim = k * 50 + idx
                    mlp1.fc1.weight[w_idx, dim] = S * 1.0
                
            expected_sum = len(chars_from_end) + 1 # k=0 adds ~2.0
            mlp1.fc1.bias[w_idx] = -(expected_sum - 0.5)
            
            mlp1.fc2.weight[:, w_idx] = torch.randn(d_model) * 10.0
            
        # Next 2000: Random semantic hash of word boundary features (heads 5-9)
        # Heads 5-9 occupy dims 250..499
        random_proj = torch.randn(2000, 250) * 1.0
        
        for idx in range(2000):
            for i in range(250):
                mlp1.fc1.weight[2000 + idx, 250 + i] = random_proj[idx, i]
                
        mlp1.fc1.bias[2000:].data.zero_()
        mlp1.fc2.weight[:, 2000:].data.copy_(torch.randn(d_model, 2000) * 5.0)
        
        # Set LayerNorms
        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "LexiconBoundaryMix"
model_description = "Combines exact 5-char lexicon matched filtering (for the top 2000 words) with a random semantic projection of word boundary features."
