import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Create HomeostaticNorm class
homeostatic_class = """class HomeostaticNorm(nn.Module):
    def __init__(self, d_model: int, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = torch.zeros_like(x)
        
        # We need to compute EMA over time for each batch and feature
        ema_mean = torch.zeros(B, D, device=x.device)
        ema_var = torch.ones(B, D, device=x.device)
        
        for t in range(T):
            x_t = x[:, t, :]
            ema_mean = self.alpha * ema_mean + (1 - self.alpha) * x_t
            ema_var = self.alpha * ema_var + (1 - self.alpha) * (x_t - ema_mean)**2
            
            normed = (x_t - ema_mean) / torch.sqrt(ema_var + self.eps)
            out[:, t, :] = normed * self.weight + self.bias
            
        return out

class Block(nn.Module):"""

content = content.replace("class Block(nn.Module):", homeostatic_class)

# Replace LayerNorm with HomeostaticNorm
block_old = """    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)"""

block_new = """    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        # Homeostatic Plasticity: Slow temporal adaptation (alpha=0.9)
        self.ln1 = HomeostaticNorm(d_model, alpha=0.9)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = HomeostaticNorm(d_model, alpha=0.9)
        self.mlp = MLP(d_model, d_ff)"""

content = content.replace(block_old, block_new)

# SimpleTransformer final_ln
st_old = """        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)"""

st_new = """        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = HomeostaticNorm(d_model, alpha=0.9)"""

content = content.replace(st_old, st_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Homeostatic_Temporal_Adaptation"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Homeostatic Plasticity Hypothesis: Replaces standard LayerNorm (which normalizes instantly across features) with a causal Temporal Exponential Moving Average (EMA) Normalization. Tests if biological contrast-adaptation over time (neuronal fatigue/homeostasis) better models the fMRI BOLD signal."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Homeostatic Plasticity.")
