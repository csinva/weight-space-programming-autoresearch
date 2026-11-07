import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Add dynamic decay parameter to CausalSelfAttention
attn_init_old = """    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)"""

attn_init_new = """    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # Liquid Time-Constant: dynamic decay rate predictor
        self.W_tau = nn.Linear(d_model, n_heads, bias=True)
        # Initialize bias to 1.0 (neutral multiplier)
        import torch.nn.init as init
        init.zeros_(self.W_tau.weight)
        init.ones_(self.W_tau.bias)"""
content = content.replace(attn_init_old, attn_init_new)

# Modify forward pass of CausalSelfAttention
attn_fwd_old = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)"""

attn_fwd_new = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        
        # Liquid Time-Constant Hypothesis
        # The time-constant (decay rate) dynamically stretches or shrinks based on the input semantics
        # tau is (B, T, H) -> unsqueeze to (B, H, T, 1) to act as a multiplier on the attention scores
        tau_multiplier = torch.nn.functional.softplus(self.W_tau(x)).transpose(1, 2).unsqueeze(-1)
        
        # Multiply scores (which represent the negative distance -i+j) by the dynamic tau
        scores = scores * tau_multiplier
        
        attn = scores.softmax(dim=-1)"""
content = content.replace(attn_fwd_old, attn_fwd_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Liquid_Time_Constant_Networks"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Liquid Time-Constant Hypothesis: Biological neural networks do not have rigid, fixed decay constants. Liquid Neural Networks adapt their time constants (tau) dynamically based on the input stimulus. I injected a linear predictor that dynamically scales the attention decay rate at each time step based on the current semantic input, allowing the network to stretch or shrink its integration window fluidly."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Liquid Time-Constant.")
