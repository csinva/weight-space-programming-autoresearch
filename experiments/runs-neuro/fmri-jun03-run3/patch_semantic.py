import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system(f"cp runs-neuro/fmri-jun03-run3/final_model_0421.py {filepath}")

with open(filepath, "r") as f:
    content = f.read()

# Let's add something the program.md suggested:
# "Use the MLP layers as lookup tables mapping character patterns to semantic axes that brain language regions are known to track (e.g. word length, concreteness)."

new_write_weights = """
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
                
        # Semantic mapping based on character patterns:
        # Vowels vs Consonants (rough proxy for syllable structure / word length)
        vowels = 'aeiou'
        for i, c in enumerate(VOCAB):
            if c in vowels:
                token_emb[i, 30] = S * 1.5
            elif c.isalpha():
                token_emb[i, 31] = S * 1.5
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+32] = token_emb[:, 0:32]
            
        token_emb[:, 960:992] = token_emb[:, 0:32]
            
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
            
            l1_decay = 15.0 + float(net) * (65.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(32): # Up to 32 now
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75
                
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            l2_decay = 0.01 + float(net) * (13.99 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.85
                
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        model.blocks[0].mlp.fc1.weight.data[:, 960:992] = 0
        model.blocks[0].mlp.fc2.weight.data[960:992, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:992] = 0
        model.blocks[1].mlp.fc2.weight.data[960:992, :] = 0
        
        model.final_ln.bias.data += 1.18
"""

import re
content = re.sub(r"def write_weights\(model: SimpleTransformer\) -> None:.*?\nmodel_shorthand_name =", new_write_weights + "\n\nmodel_shorthand_name =", content, flags=re.DOTALL)

content = content.replace("Deep_Ensemble_0421_Master", "Semantic_Vowel_Consonant")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Adding explicit orthographic features (vowel vs consonant) to the tokens, which act as proxy for syllable frequency.")

with open(filepath, "w") as f:
    f.write(content)
print("Applied semantic patch")
