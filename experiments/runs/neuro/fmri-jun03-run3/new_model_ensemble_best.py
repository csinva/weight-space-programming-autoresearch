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

        # PerfectEnsembleWordBoundary got 0.0355!
        # It had 15 identical networks except for decay scale.
        # It used decay_scale = 1.0 + float(net) / 2.0 (range 1.0 to 8.0)
        # It used Q=5.0, K=pos_index * decay_scale.
        # Let's expand this to 15 networks, ALL space-seeking, but with much richer orthographic representations.
        
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
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # The ONLY difference between the nets is their attention scale
            # We found space-seeking is the best.
            # But wait, original WordBoundary is space-seeking?
            # Original: W_q[29] = 5.0, W_k[28] = 1.0. 
            # 29 is the bias, 28 is the pos index!
            # So original WordBoundary DOES NOT seek spaces! It's pure distance decay!
            # Wait, the original has token_emb[i, 0] = 1.0 for space, but it's NEVER used in attention!
            # It's only passed through W_v!
            # The MLPs are what learned to use the space token!
            
            # Okay, let's keep the pure distance decay, but vary the slope slightly.
            decay_scale = 1.0 + float(net) / 4.0
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = decay_scale
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            # Random MLP initialized differently for each net
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=0.5)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=0.5 * S)
            
            for i in range(dim_per_net):
                l2_attn.W_q.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_k.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=0.5)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=0.5 * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "EnsembleWB_FinelyTuned"
model_description = "Tiles 15 identical independent networks but with much finer decay variations (1.0 to 4.75) compared to PerfectEnsembleWordBoundary."
