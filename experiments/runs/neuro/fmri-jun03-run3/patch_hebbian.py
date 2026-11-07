import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# We'll inject a Hebbian learning approximation into the forward pass of the MLP.
# "Neurons that fire together, wire together."
# We can't actually update weights across the dataset inside the embedder evaluation cleanly without state,
# but we can implement a fast-weight (dynamic) short-term plasticity mechanism within the sequence.

# Modify MLP class
mlp_old = """class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x"""

mlp_new = """class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # standard pass
        h = self.fc1(x)
        h = torch.nn.functional.relu(h)
        
        # Fast-Weight Hebbian Plasticity (Short-Term Memory)
        # We compute an auto-associative update: Delta W = learning_rate * h(t) @ h(t)^T
        # We apply this cumulatively over the sequence length T.
        B, T, D = h.shape
        out = torch.zeros_like(x)
        
        for b in range(B):
            # Dynamic weight matrix for this sequence
            W_fast = torch.zeros(D, D, device=x.device)
            for t in range(T):
                h_t = h[b, t].unsqueeze(1) # (D, 1)
                
                # Forward pass through base weights + fast weights
                # Output is standard fc2 + fast_weight contribution
                # We project the fast weight through fc2 to match dimensions
                base_out = self.fc2(h_t.squeeze(1))
                
                # Fast weight modulates the hidden state before fc2
                fast_h = W_fast @ h_t
                fast_out = self.fc2(fast_h.squeeze(1))
                
                out[b, t] = base_out + 0.1 * fast_out
                
                # Hebbian update: fire together wire together (decaying)
                W_fast = 0.9 * W_fast + 0.1 * (h_t @ h_t.t())
                
        return out"""

content = content.replace(mlp_old, mlp_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Hebbian_Fast_Weight_Plasticity"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Hebbian Short-Term Plasticity Hypothesis: Implements a fast-weight auto-associative memory (Delta W = h*h^T) within the MLP during the forward pass. Tests if the BOLD signal is driven by dynamic synaptic changes (neurons that fire together, wire together) occurring rapidly during sentence comprehension."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Hebbian Plasticity.")
