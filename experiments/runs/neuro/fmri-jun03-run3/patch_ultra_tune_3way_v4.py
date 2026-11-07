import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# Okay, it dropped to 0.0393. The L1 upper bound of 80 IS important.
# The 0.0398 run used:
# L1: 10 to 70... Wait, 10 to 80.
# Let's check exactly what the 0.0398 run used:
# "Uses the exact optimal scales of UltraTune, but changes the staggering from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)"
# The exact optimal scales from UltraTune were:
# L1: 10.0 + (70.0 / 14.0) -> 10 to 80.0
# L2: 0.05 + (11.95 / 14.0) -> 0.05 to 12.0
# Stdev2: 0.01 + (3.99 / 14.0) -> 0.01 to 4.0
# This is our absolute baseline.

# How to beat 0.0398?
# Let's inject pure features directly into the L2 representation.
# L2 attention output has 64 dimensions.
# What if we explicitly set the last 4 dimensions of the L2 output to just be the L1 space/word features, unmixed?
# No, we can't easily bypass attention.
# But we CAN inject features in the pos_emb!
# We already use 1000/1001 for variance anchor, 28/29 for space/alpha.
# What if we add a "Word Boundary Reset" feature directly into the MLP bias? No, let's keep it simple.

# Let's try changing the Staggering offset slightly.
# 3-way staggering (+0, +6, +12) got 0.0398.
# What if we stagger as (+0, +5, +10)? That's a slightly tighter clump.
# And let's push the L2 decay scale up slightly. 0.05 to 15.0.

new_write_weights = """def write_weights(model: SimpleTransformer) -> None:
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
            
            # --- LAYER 1: Extremely sharp local extraction ---
            # EXACT 0.0398 SCALES
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
            # SLIGHT TWEAK: 0.05 to 15.0
            l2_decay = 0.05 + float(net) * (14.95 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            # Staggered logic: 3-way split (Tighter: +0, +5, +10)
            net_b = (net + 5) % 15
            net_c = (net + 10) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0
                
            # EXACT 0.0398 SCALES
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)"""

new_bottom = """model_shorthand_name = "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_v4"
model_description = "Uses the exact 0.0398 optimal scales, but tightens the 3-way stagger to (+0, +5, +10) and pushes L2 maximum decay to 15.0 to capture extremely long document-level features."

def build_embedder(device: str = 'cuda',
                   d_model: int = 1020, n_heads: int = 10, n_layers: int = 2,
                   d_ff: int = 4000, max_seq_len: int = 64) -> InterpretableEmbedder:
    model = SimpleTransformer(
        vocab_size=len(VOCAB), max_seq_len=max_seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    write_weights(model)
    model.eval()
    return InterpretableEmbedder(model, device=device)"""

content = re.sub(r'def write_weights.*?model\.final_ln\.bias\)', new_write_weights, content, flags=re.DOTALL)
content = re.sub(r'model_shorthand_name = ".*?\n\nif __name__ == "__main__":', new_bottom + '\n\nif __name__ == "__main__":', content, flags=re.DOTALL)

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
