import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system(f"cp runs-neuro/fmri-jun03-run3/final_model_0421.py {filepath}")

with open(filepath, "r") as f:
    content = f.read()

import re

new_write_weights = """
def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head
        d_ff = model.blocks[0].mlp.fc1.out_features

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            idx = -1
            if c == ' ': idx = 0
            elif c.isalpha(): idx = 1 + (ord(c) - ord('a'))
            
            if idx != -1:
                for h in range(n_heads):
                    token_emb[i, h * d_head + idx] = 1.0
        
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            for h in range(n_heads):
                pos_emb[p, h * d_head + 28] = p
                pos_emb[p, h * d_head + 29] = 1.0
        
        model.pos_emb.weight.data.copy_(pos_emb)
        
        attn = model.blocks[0].attn
        for h in range(n_heads):
            # exponential decay from 10.0 (very sharp) to 0.01 (very broad)
            decay = 0.01 * (1000.0 ** (h / max(1, n_heads - 1))) 
            
            h_start = h * d_head
            
            attn.W_q.weight[h_start + 0, h_start + 29] = 1.0
            attn.W_k.weight[h_start + 0, h_start + 28] = decay
            
            attn.W_q.weight[h_start + 1, h_start + 28] = 1.0
            attn.W_k.weight[h_start + 1, h_start + 29] = -decay
            
            for i in range(28):
                attn.W_v.weight[h_start + i, h_start + i] = 1.0
                attn.W_o.weight[h_start + i, h_start + i] = 1.0

        mlp = model.blocks[0].mlp
        nn.init.normal_(mlp.fc1.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(mlp.fc2.weight, std=1.0 / math.sqrt(d_ff))
        nn.init.normal_(mlp.fc1.bias, std=0.1)
"""

content = re.sub(r"def write_weights\(model: SimpleTransformer\) -> None:.*?\nmodel_shorthand_name =", new_write_weights + "\n\nmodel_shorthand_name =", content, flags=re.DOTALL)

content = content.replace("Deep_Ensemble_0421_Master", "EMA_Char_Kitchen_Sinks")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "A highly interpretable 1-layer model: 30 heads compute pure Exponential Moving Averages of character frequencies at 30 different half-lives. MLP does random nonlinear projection (Kitchen Sinks).")

content = content.replace("d_model: int = 1020", "d_model: int = 960")
content = content.replace("n_heads: int = 10", "n_heads: int = 30")
content = content.replace("n_layers: int = 2", "n_layers: int = 1")
content = content.replace("d_ff: int = 4000", "d_ff: int = 10000")


with open(filepath, "w") as f:
    f.write(content)
print("Applied EMA Char patch")
