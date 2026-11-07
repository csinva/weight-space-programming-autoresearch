import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# 0.0395! We are so incredibly close to the 0.0405 benchmark!
# We know the following:
# 1. 15 nets over 10 heads (asymmetric mixing) is CRITICAL.
# 2. L1 Decay 10 to 80 is CRITICAL (very sharp word boundaries).
# 3. L2 Decay 0.05 to 12.0 is CRITICAL (long integration).
# 4. Wide L2 variance (0.01 to 4.0) bumped it from 0.0394 to 0.0395.
# Let's push variance even wider: L2 Stdev 0.01 to 6.0
# And let's tweak L1 standard deviation. It's been locked at 0.5 forever.
# What if we grade L1 stdev as well? Fast decays (10.0) maybe need less variance (0.2) 
# and slow extractors (80.0) need more (1.5) to capture complex n-grams?

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
                token_emb[i, 0] = S * 1.0 # Space
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 # Alpha
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 # Identity
                
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
            l1_decay = 10.0 + float(net) * (70.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            # GRADED L1 Variance: 0.2 to 1.5
            std_dev = 0.2 + float(net) * (1.3 / 14.0)
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.05 + float(net) * (11.95 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            # Staggered logic:
            net_b = (net + 7) % 15
            d_start_b = net_b * dim_per_net
            
            for i in range(32):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
                l2_attn.W_v.weight[h_start + 32 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 32 + i, h_start + 32 + i] = S * 1.0
                
            # EXPANDED L2 Variance: 0.01 to 6.0
            std_dev2 = 0.01 + float(net) * (5.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)"""

new_bottom = """model_shorthand_name = "Deep_Ensemble_Staggered_Asymmetric_UltraTune_v2"
model_description = "Builds on the 0.0395 success by grading L1 standard deviations (0.2 to 1.5) and pushing L2 standard deviations even wider (0.01 to 6.0)."

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
