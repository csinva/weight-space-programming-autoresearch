import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# 1. Modify CausalSelfAttention to inject Power-Law decay
attn_old = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)"""

attn_new = """        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        
        # Inject Power-Law (Fractal) Forgetting Curve
        # Biological memory often follows t^(-alpha) instead of e^(-t/tau)
        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.view(1, 1, T, T).float()
        dist = torch.clamp(dist, min=0.0) # Avoid negative log
        
        # We need alpha to vary across heads to provide different timescales.
        # We'll map the heads to exponents ranging from 0.1 to 10.0
        alphas = torch.linspace(0.1, 10.0, H, device=x.device).view(1, H, 1, 1)
        
        # power_law_scores = -alpha * log(1 + dist)
        power_law_scores = -alphas * torch.log(1.0 + dist)
        
        # Apply only to lower triangle
        power_law_scores = power_law_scores.masked_fill(mask.view(1, 1, T, T), 0.0)
        
        # Add to scores
        scores = scores + power_law_scores
        
        attn = scores.softmax(dim=-1)"""

content = content.replace(attn_old, attn_new)

# 2. In write_weights, we must disable the implicit Exponential Decay
# The exponential decay is driven by W_q[..., 29] = 5.0 and W_k[..., 28] = decay
l1_old = "l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0"
l1_new = "l1_attn.W_q.weight[h_start + 0, d_start + 29] = 0.0 # Disabled for Power-Law"
content = content.replace(l1_old, l1_new)

l2_old = "l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0"
l2_new = "l2_attn.W_q.weight[h_start + 0, d_start + 29] = 0.0 # Disabled for Power-Law"
content = content.replace(l2_old, l2_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Power_Law_Fractal_Forgetting"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Power-Law (Fractal) Forgetting Hypothesis: Cognitive psychology (Ebbinghaus) shows biological memory decays as a power-law t^(-alpha), not an exponential e^(-t/tau). Replaced the algebraic exponential decay with an explicit logarithmic attention penalty to induce scale-free fractal temporal integration."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Power-Law Forgetting.")
