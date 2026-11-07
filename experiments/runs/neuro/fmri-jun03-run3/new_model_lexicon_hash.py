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

        # We will use LayerNorm pass-through trick
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

        # L1 Attention: Exact Relative Position (heads 0-9)
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
            
            # Pass through the exact char
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0
                l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0

        # L1 MLP: Exact Dictionary Matching + Exact Positional Hashing
        mlp1 = model.blocks[0].mlp
        
        # 1000 neurons for Dictionary Match Filter
        for w_idx, W in enumerate(TOP_WORDS[:1000]):
            W_padded = ' ' + W
            chars_from_end = list(reversed(W_padded))[:10]
            
            for k, c in enumerate(chars_from_end):
                if c in VOCAB:
                    idx = VOCAB.index(c)
                    dim = k * 50 + idx
                    mlp1.fc1.weight[w_idx, dim] = S * 1.0
                
            expected_sum = len(chars_from_end) + 1 # k=0 adds 2.0
            mlp1.fc1.bias[w_idx] = -(expected_sum - 0.5)
            
            # Random projection out
            mlp1.fc2.weight[:, w_idx] = torch.randn(d_model) * 10.0
            
        # 3000 neurons for pure random hashing
        # This will randomly hash the exact n-gram characters
        random_proj = torch.randn(3000, 500) * 0.5
        for idx in range(3000):
            for i in range(500):
                mlp1.fc1.weight[1000 + idx, i] = random_proj[idx, i]
                
        # The key to stop the n-gram hash from overfitting is giving it a random bias
        # so only a few neurons activate per n-gram
        mlp1.fc1.bias[1000:].data.copy_(torch.randn(3000) - 2.0)
        
        # Project back
        mlp1.fc2.weight[:, 1000:].data.copy_(torch.randn(d_model, 3000) * 1.0)
        
        # L2 Attention: pass through
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            l2_attn.W_q.weight[i, 1002] = S * 1.0
            l2_attn.W_k.weight[i, 1002] = S * 1.0
            l2_attn.W_v.weight[i, i] = S * 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        # L2 MLP: further hashing
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.5)

        # Set LayerNorms
        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "LexiconRandomHash"
model_description = "Uses the exact character math trick to feed 1000 dictionary match filters and 3000 random n-gram hash filters, preventing over-fitting with bias/variance tuning and a 2nd layer deep hash."
