import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# We want to add a suffix detector.
# Specifically, we want to detect when "i", "n", "g" appear in sequence.
# Layer 1: detect bigrams like "in", "ng".
# Layer 2: combine them.
# We will use the spare dimensions. 15 nets * 64 = 960. We have 60 spare dims (960 to 1019).

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
            
        # Suffix detectors (ing, ed, s, ly)
        # We need explicit token embeddings for i, n, g, e, d, s, l, y
        # We'll map them to dim 960+
        ling_start = num_nets * dim_per_net
        
        # 960: i, 961: n, 962: g
        # 963: e, 964: d
        # 965: s
        # 966: l, 967: y
        char_to_dim = {'i': 0, 'n': 1, 'g': 2, 'e': 3, 'd': 4, 's': 5, 'l': 6, 'y': 7, ' ': 8}
        
        for i, c in enumerate(VOCAB):
            if c in char_to_dim:
                token_emb[i, ling_start + char_to_dim[c]] = S * 1.0
                
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
            # Position embeddings for linguistic features
            pos_emb[p, ling_start+10] = S * (p / max_seq_len)
            
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
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0
                
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        # LINGUISTIC DETECTORS (Head 9)
        h_start = 9 * d_head 
        d_start = ling_start
        
        # Layer 1: Local n-gram overlaps. We just use high decay to get recent chars
        l1_attn.W_q.weight[h_start + 0, d_start + 10] = 5.0
        l1_attn.W_k.weight[h_start + 0, d_start + 10] = 50.0 # very sharp
        
        for i in range(10): # copy chars 0-9
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 20 + i, h_start + i] = S * 1.0
            
        # MLP 1: Detect specific combos using non-linearities
        f_start = 15 * 256
        # we have spare capacity in MLP. Let's make an 'ing' detector
        # If 'i', 'n', 'g' are active in recent window...
        # Wait, L1 attention just averages them if they are close.
        # It's an imperfect n-gram detector, more like a continuous bag of recent chars.
        # So if i, n, g are all present recently, their sum is high.
        # Let's map sum of i, n, g to a single dimension.
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 0] = 1.0 # i
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 1] = 1.0 # n
        mlp1.fc1.weight[f_start + 0, d_start + 20 + 2] = 1.0 # g
        mlp1.fc1.bias[f_start + 0] = -2.0 # Threshold for 'ing' bag
        mlp1.fc2.weight[d_start + 30, f_start + 0] = S * 5.0 # Output 'ing' feature
        
        # 'ed' bag
        mlp1.fc1.weight[f_start + 1, d_start + 20 + 3] = 1.0 # e
        mlp1.fc1.weight[f_start + 1, d_start + 20 + 4] = 1.0 # d
        mlp1.fc1.bias[f_start + 1] = -1.5 
        mlp1.fc2.weight[d_start + 31, f_start + 1] = S * 5.0 
        
        # 's' bag
        mlp1.fc1.weight[f_start + 2, d_start + 20 + 5] = 1.0 # s
        mlp1.fc1.bias[f_start + 2] = -0.5
        mlp1.fc2.weight[d_start + 32, f_start + 2] = S * 5.0 
        
        # 'ly' bag
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 6] = 1.0 # l
        mlp1.fc1.weight[f_start + 3, d_start + 20 + 7] = 1.0 # y
        mlp1.fc1.bias[f_start + 3] = -1.5 
        mlp1.fc2.weight[d_start + 33, f_start + 3] = S * 5.0 
        
        # Layer 2: Temporal Integration of morphological features
        l2_attn.W_q.weight[h_start + 0, d_start + 10] = 5.0
        l2_attn.W_k.weight[h_start + 0, d_start + 10] = 2.0 # slow decay
        
        for i in range(4): # 30, 31, 32, 33
            l2_attn.W_v.weight[h_start + i, d_start + 30 + i] = 1.0
            l2_attn.W_o.weight[d_start + 40 + i, h_start + i] = S * 1.0

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)"""

new_bottom = """model_shorthand_name = "Interpretable_Staggered_Morphology"
model_description = "Extends the Ultimate staggered architecture with explicitly wired continuous-bag-of-characters morphological detectors ('ing', 'ed', 's', 'ly') in the spare 60 dimensions."

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
