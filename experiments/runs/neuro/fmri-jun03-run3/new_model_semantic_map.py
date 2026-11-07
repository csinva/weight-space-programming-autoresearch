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

        # Let's try something conceptually different from random ensembles.
        # "Character-to-Meaning Map"
        # Since we operate on characters, we can't easily map words to meanings.
        # BUT we can map *character combinations* to a vast array of random semantic axes.
        # If we use a very wide hidden layer (d_ff = 4000) we can act as a massive random lookup table.
        # Network: 1 large network.
        # Layer 1: Attends to current word (since last space)
        # Layer 1 MLP: Vast expansion mapping character sums to arbitrary features.
        # Layer 2: Temporal decay of these word-level features to create a smooth context.
        
        dim_net = 512
        ff_net = 3840 # use almost all of the 4096 capacity
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 # space flag
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 # char flag
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 # char identity
                
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len) # Global pos
            pos_emb[p, 29] = S * 1.0
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        h_start = 0
        d_start = 0
        f_start = 0
        
        # --- Layer 1 Attention: Gather characters in the current word ---
        # Query wants anything since the last space.
        # This is a bit tricky with just 1 head. Let's do a simple recent-character decay 
        # that ignores spaces.
        
        l1_attn.W_q.weight[0, 29] = 5.0
        l1_attn.W_k.weight[0, 28] = 2.0 # Moderate decay (about 3-4 chars)
        
        # Don't attend to spaces
        l1_attn.W_k.weight[0, 0] = -10.0
        
        # Value pulls up char identity
        for i in range(26):
            l1_attn.W_v.weight[i, 2 + i] = 1.0
            l1_attn.W_o.weight[32 + i, i] = S * 1.0
            
        # --- Layer 1 MLP: Randomly mix character counts into "semantic" features ---
        # The input is basically a "bag of characters" for the current word.
        # We expand 26 chars to 3840 features using a sparse or normal projection.
        std_dev = 1.0
        
        # To make it more like a lookup table, we increase variance so ReLUs fire rarely
        # (simulating specific n-gram / char-combination detectors)
        nn.init.normal_(mlp1.fc1.weight[0:ff_net, 32:58], std=2.0)
        nn.init.zeros_(mlp1.fc1.bias)
        # bias negative so it acts like an AND gate (only fires if multiple chars are present)
        mlp1.fc1.bias.data[0:ff_net] = -2.0 
        
        nn.init.normal_(mlp1.fc2.weight[58:58+dim_net, 0:ff_net], std=1.0 * S / math.sqrt(ff_net))

        # --- Layer 2 Attention: Exponential decay / integration of semantic features over time ---
        
        l2_attn.W_q.weight[0, 29] = 5.0
        l2_attn.W_k.weight[0, 28] = 0.5 # Slow decay (integrate context)
        
        for i in range(dim_net):
            l2_attn.W_v.weight[i, 58 + i] = 1.0
            l2_attn.W_o.weight[58 + i, i] = S * 1.0
            
        # --- Layer 2 MLP: Refine features ---
        nn.init.normal_(mlp2.fc1.weight[0:ff_net, 58:58+dim_net], std=1.0)
        nn.init.normal_(mlp2.fc2.weight[58:58+dim_net, 0:ff_net], std=1.0 * S / math.sqrt(ff_net))

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "Char_Combinations_Map"
model_description = "Uses L1 to pool recent chars into a 'bag of chars' word rep, then a huge wide MLP with negative bias to detect specific character combinations (pseudo-words), and L2 to temporally integrate these."
