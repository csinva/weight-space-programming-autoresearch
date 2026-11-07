import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add a pre-attention activation in the second block
# Block forward currently:
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))

new_block = """class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, is_layer2: bool = False) -> torch.Tensor:
        h = self.ln1(x)
        if is_layer2:
            # Add mild non-linearity before attention in layer 2
            # to give the staggered integration access to slightly transformed features
            h = F.leaky_relu(h, negative_slope=0.1)
        x = x + self.attn(h)
        x = x + self.mlp(self.ln2(x))
        return x"""

old_block = """class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x"""

content = content.replace(old_block, new_block)

old_transformer_forward = """    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return x"""

new_transformer_forward = """    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        
        x = self.blocks[0](x, is_layer2=False)
        x = self.blocks[1](x, is_layer2=True)
            
        x = self.final_ln(x)
        return x"""

content = content.replace(old_transformer_forward, new_transformer_forward)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Pre_Attn_Act")

with open(filepath, "w") as f:
    f.write(content)
print("Applied pre-attn activation patch")
