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

        # Ensembles combined with different temporal decaying mechanisms hit 0.0363!
        # Now let's try something different: explicit extraction of N-grams and specific character transitions.
        # We can use the variance anchor for safety.
        # Let's allocate the first 15 sub-networks to n-grams.
        # Network 1: Bigrams
        # Network 2: Trigrams
        # Network 3: Skip-bigrams (t, t-2)
        # We can implement this via positional shifts in attention.
        
        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
                
        # Copy token embeddings to all networks
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            # Variance anchor
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            # Subnetwork position embeddings for different shifts
            for net in range(num_nets):
                start = net * dim_per_net
                
                # Each net gets a different frequency/decay for position
                decay = 1.0 + (net / 5.0) 
                
                # Base position tracking
                pos_emb[p, start + 28] = S * (p / max_seq_len) # Global pos
                pos_emb[p, start + 29] = S * 1.0 # Constant for shifting
                
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            shift = (net % 3) + 1 # Shifts of 1, 2, 3
            decay_scale = 1.0 + float(net) / 3.0
            
            # Setup attention to look 'shift' steps back using positional embeddings
            # (Approximated by heavy decay)
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0 * shift
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = decay_scale * shift
            
            for i in range(28):
                # Value extracts the character identity
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                
                # Output writes it to a shifted dimension so we have current char AND prev chars
                # We put the shifted char at 30+i
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            # Now MLP1 has access to:
            # d_start:d_start+28 = current character
            # d_start+30:d_start+58 = previous character(s)
            
            # Let's initialize MLP1 to explicitly find combinations (N-grams)
            # We use moderate variance so it mixes them randomly
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # Layer 2 just passes through / mixes further
            for i in range(dim_per_net):
                l2_attn.W_q.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_k.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "Ngram_Ensemble"
model_description = "Uses attention to shift characters and creates random MLP combinations of current and previous characters to form random N-gram and skip-gram features."
