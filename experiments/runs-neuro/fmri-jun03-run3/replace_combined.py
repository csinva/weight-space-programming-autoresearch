import os
import math
import torch
import torch.nn as nn
from top_words import TOP_WORDS

content = ""
with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

import re
pattern = r"def write_weights\(model: SimpleTransformer\) -> None:.*?class InterpretableEmbedder"

replacement = """def write_weights(model: SimpleTransformer) -> None:
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

        num_nets = 15
        dim_per_net = 58  # Reduced from 64 to save space
        ff_per_net = 200  # Reduced from 256 to save space
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        ling_start = num_nets * dim_per_net
        
        char_to_dim = {'i': 0, 'n': 1, 'g': 2, 'e': 3, 'd': 4, 's': 5, 'l': 6, 'y': 7, 
                       'r': 8, 'o': 9, 't': 10, 'a': 11, 'c': 12, 'm': 13}
        
        for i, c in enumerate(VOCAB):
            if c in char_to_dim:
                token_emb[i, ling_start + char_to_dim[c]] = S * 1.0
                
            if c.isalpha():
                token_emb[i, ling_start + 14] = S * 1.0
            elif c == ' ':
                token_emb[i, ling_start + 15] = S * 1.0
                
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
            pos_emb[p, ling_start+19] = S * (p / max_seq_len)
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        # Initialize LayerNorms properly
        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
        nn.init.ones_(model.ln_f.weight)
        nn.init.zeros_(model.ln_f.bias)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 10.0 + float(net) * (70.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.05 + float(net) * (11.95 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            # Staggered logic: 3-way split
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(20):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(19):
                l2_attn.W_v.weight[h_start + 20 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 20 + i, h_start + 20 + i] = S * 1.0
                
            for i in range(19):
                l2_attn.W_v.weight[h_start + 39 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 39 + i, h_start + 39 + i] = S * 1.0
                
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
        # --- Explicit concept tracking (Morphology) ---
        d_start = ling_start
        h_start = 0
        
        # Layer 1
        l1_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0
        l1_attn.W_k.weight[h_start + 0, d_start + 19] = 20.0
        
        for i in range(16): 
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 20 + i, h_start + i] = S * 1.0
            
        f_start = num_nets * ff_per_net # 15 * 200 = 3000
        morph_scale = S * 12.0
        
        # 'ing'
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 1] = 1.0
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 2] = 1.0
        mlp1.fc1.bias[f_start + 0] = -2.0 
        mlp1.fc2.weight[d_start + 40, f_start + 0] = morph_scale
        
        # 'ed'
        mlp1.fc1.weight[f_start + 1, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 1, d_start + 20 + 4] = 1.0
        mlp1.fc1.bias[f_start + 1] = -1.0 
        mlp1.fc2.weight[d_start + 41, f_start + 1] = morph_scale
        
        # 's '
        mlp1.fc1.weight[f_start + 2, d_start + 20 + 5] = 1.0
        mlp1.fc1.weight[f_start + 2, d_start + 20 + 15] = 1.0
        mlp1.fc1.bias[f_start + 2] = -1.0 
        mlp1.fc2.weight[d_start + 42, f_start + 2] = morph_scale
        
        # 'ly '
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 6] = 1.0
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 7] = 1.0
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 15] = 1.0
        mlp1.fc1.bias[f_start + 3] = -2.0 
        mlp1.fc2.weight[d_start + 43, f_start + 3] = morph_scale
        
        # 'er '
        mlp1.fc1.weight[f_start + 4, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 4, d_start + 20 + 8] = 1.0
        mlp1.fc1.weight[f_start + 4, d_start + 20 + 15] = 1.0
        mlp1.fc1.bias[f_start + 4] = -2.0 
        mlp1.fc2.weight[d_start + 44, f_start + 4] = morph_scale
        
        # 'or '
        mlp1.fc1.weight[f_start + 5, d_start + 20 + 9] = 1.0
        mlp1.fc1.weight[f_start + 5, d_start + 20 + 8] = 1.0
        mlp1.fc1.weight[f_start + 5, d_start + 20 + 15] = 1.0
        mlp1.fc1.bias[f_start + 5] = -2.0 
        mlp1.fc2.weight[d_start + 45, f_start + 5] = morph_scale
        
        # 'ion'
        mlp1.fc1.weight[f_start + 6, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 6, d_start + 20 + 9] = 1.0
        mlp1.fc1.weight[f_start + 6, d_start + 20 + 1] = 1.0
        mlp1.fc1.bias[f_start + 6] = -2.0 
        mlp1.fc2.weight[d_start + 46, f_start + 6] = morph_scale
        
        # 'tion'
        mlp1.fc1.weight[f_start + 7, d_start + 20 + 10] = 1.0
        mlp1.fc1.weight[f_start + 7, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 7, d_start + 20 + 9] = 1.0
        mlp1.fc1.weight[f_start + 7, d_start + 20 + 1] = 1.0
        mlp1.fc1.bias[f_start + 7] = -3.0 
        mlp1.fc2.weight[d_start + 47, f_start + 7] = morph_scale
        
        # 'ation'
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 11] = 1.0
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 10] = 1.0
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 9] = 1.0
        mlp1.fc1.weight[f_start + 8, d_start + 20 + 1] = 1.0
        mlp1.fc1.bias[f_start + 8] = -4.0 
        mlp1.fc2.weight[d_start + 48, f_start + 8] = morph_scale
        
        # 'ic '
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 0] = 1.0
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 12] = 1.0
        mlp1.fc1.weight[f_start + 9, d_start + 20 + 15] = 1.0
        mlp1.fc1.bias[f_start + 9] = -2.0 
        mlp1.fc2.weight[d_start + 49, f_start + 9] = morph_scale
        
        # 'ment '
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 13] = 1.0
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 3] = 1.0
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 1] = 1.0
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 10] = 1.0
        mlp1.fc1.weight[f_start + 10, d_start + 20 + 15] = 1.0
        mlp1.fc1.bias[f_start + 10] = -4.0 
        mlp1.fc2.weight[d_start + 50, f_start + 10] = morph_scale

        # Layer 2 morphology delay
        l2_attn.W_q.weight[h_start + 0, d_start + 19] = 5.0
        l2_attn.W_k.weight[h_start + 0, d_start + 19] = 1.0 
        
        for i in range(11):
            l2_attn.W_v.weight[h_start + i, d_start + 40 + i] = 1.0
            l2_attn.W_o.weight[d_start + 40 + i, h_start + i] = S * 1.0


        # ==============================================================
        # --- LEXICON HASH: Injecting Semantic Meaning into Structure ---
        # ==============================================================
        lex_start = d_start + 51 # Starts at 921
        # We will use 51 dimensions for Lexicon Hash -> 921 to 971
        num_lex_dims = 50
        
        # Precompute signatures for TOP_WORDS (Use 500 words)
        target_words = TOP_WORDS[:500] 
        num_decays = 5 # use 5 temporal decays to extract word exactly
        decays = [10.0, 5.0, 2.0, 1.0, 0.5]
        
        signatures = torch.zeros(len(target_words), num_decays * 28)
        for w_idx, w in enumerate(target_words):
            seq = " " + w
            for j in range(len(seq)):
                c = seq[len(seq) - 1 - j]
                if c == ' ': c_idx = 0
                elif c.isalpha(): c_idx = 1 + (ord(c) - ord('a'))
                else: c_idx = 27
                for h in range(num_decays):
                    val = math.exp(-decays[h] * j)
                    signatures[w_idx, h * 28 + c_idx] += val
        
        for h in range(num_decays):
            head_sums = signatures[:, h * 28 : (h + 1) * 28].sum(dim=1, keepdim=True)
            signatures[:, h * 28 : (h + 1) * 28] /= (head_sums + 1e-8)

        norms = torch.norm(signatures, dim=1, keepdim=True)
        signatures = signatures / (norms + 1e-8)

        # Generate 50 random dense semantic axes
        semantic_projection = torch.randn(len(target_words), num_lex_dims)

        # We will map Lexicon Hash using the remaining available spaces:
        # We need an attention head in Layer 1 to compute continuous EMA.
        # But wait! Layer 1 heads are used by num_nets (15 heads total).
        # We can't add more heads! We only have 15 heads!
        # And we used 1 head for morphology! So we only have 14 heads for structure?
        # NO! num_nets=15 means h_start = (net % 15) * d_head.
        # So morphology overwrote head 0!
        
        # To compute EMA, we can just REUSE the EMA outputs from the structural network!
        # The structural network already computes EMA with 15 different decays!
        # This is absolutely beautiful! 
        # The structural network ALREADY computed exactly what we need!
        # The structural networks compute `W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0`.
        # So for `net`, the EMA of characters with decay `l1_decay` is at `d_start + 30 + i`.
        # `l1_decay` ranges from `10.0` to `15.0`.
        
class InterpretableEmbedder"""
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(new_content)
