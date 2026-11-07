import os
import sys
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace write_weights to compute derivatives with exact alignment
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
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        
        for i, c in enumerate(VOCAB):
            if c == '<pad>' or c == '<unk>':
                continue
            elif len(c) == 1 and c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
            else:
                token_emb[i, 0] = S * 1.0

        # Inject perfectly aligned with d_head
        for h in range(n_heads):
            start_dim = h * d_head
            token_emb[:, start_dim:start_dim+28] = token_emb[:, 0:28]
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 28] = S * (p / max_seq_len)
        model.pos_emb.weight.data.copy_(pos_emb)

        # L1: standard temporal smearing
        l1_attn = model.blocks[0].attn
        for h in range(n_heads):
            d_start = h * d_head
            h_start = h * d_head
            
            # W_v passes through the current character
            for i in range(28):
                l1_attn.W_v.weight.data[h_start + i, d_start + i] = 1.0
                
            for i in range(28):
                # Pass current timestamp with decay
                l1_attn.W_o.weight.data[d_start + i, h_start + i] = S * (0.85 ** h)

        # L2: Standard routing
        l2_attn = model.blocks[1].attn
        for h in range(n_heads):
            d_start = h * d_head
            h_start = h * d_head
            
            for i in range(28):
                l2_attn.W_v.weight.data[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight.data[d_start + i, h_start + i] = S

        # Add the +1.18 bias
        model.final_ln.bias.data += 1.18
"""

content = re.sub(r"def write_weights\(model: SimpleTransformer\) -> None:.*?model_shorthand_name =", new_write_weights + "\nmodel_shorthand_name =", content, flags=re.DOTALL)

content = content.replace("Deep_Ensemble_0421_Master", "Aligned_Orthographic_Tracker")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Perfectly aligns the feature vectors to the attention head boundaries.")

with open(filepath, "w") as f:
    f.write(content)
