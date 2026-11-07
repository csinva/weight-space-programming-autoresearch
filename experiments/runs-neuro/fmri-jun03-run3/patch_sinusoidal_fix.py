import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Fix tensor broadcasting: scores is (B, H, T, T) not (B, T, H, T)
attn_old = """        # Inject Brain-Wave style Resonances (Sinusoidal ripples on top of decay)
        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.unsqueeze(0).unsqueeze(2).expand(B, T, H, T)
        
        # Frequencies spread across heads, mimicking Theta/Alpha/Beta/Gamma bands
        # Normalized relative to the max_seq_len (e.g. 64)
        frequencies = torch.linspace(0.1, 3.14, H, device=x.device).view(1, 1, H, 1)
        
        # Add resonance. amplitude 2.0 to make it noticeable.
        resonance = 2.0 * torch.cos(dist * frequencies)
        
        # Apply only to lower triangle
        resonance = resonance.masked_fill(mask.unsqueeze(0).unsqueeze(2), 0.0)"""

# In the current CausalSelfAttention, q and k are:
# q = self.W_q(x).view(B, T, H, dh).transpose(1, 2) -> (B, H, T, dh)
# k = self.W_k(x).view(B, T, H, dh).transpose(1, 2) -> (B, H, T, dh)
# scores = (q @ k.transpose(-2, -1)) -> (B, H, T, T)

attn_new = """        # Inject Brain-Wave style Resonances (Sinusoidal ripples on top of decay)
        dist = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(T, device=x.device).unsqueeze(0)
        dist = dist.view(1, 1, T, T).float()
        
        # Frequencies spread across heads, mimicking Theta/Alpha/Beta/Gamma bands
        frequencies = torch.linspace(0.1, 3.14, H, device=x.device).view(1, H, 1, 1)
        
        # Add resonance. amplitude 2.0 to make it noticeable.
        resonance = 2.0 * torch.cos(dist * frequencies)
        
        # Apply only to lower triangle
        resonance = resonance.masked_fill(mask.view(1, 1, T, T), 0.0)"""

content = content.replace(attn_old, attn_new)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched sinusoidal resonance (fixed dims).")
