import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify Block to bypass MLP entirely
block_old = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x"""

block_new = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        # Attention-Only (Zero-MLP) Hypothesis
        # We explicitly bypass the MLP to prove it is mathematically vestigial
        # x = x + self.mlp(self.ln2(x))
        return x"""

content = content.replace(block_old, block_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Pure_Attention_Only_Integrator"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Attention-Only (Zero-MLP) Hypothesis: Following the discovery that massive dropout and pruning in the MLPs had zero effect on performance, this model completely severs the MLP from the residual stream. Tests if the entire predictive power of the network is driven exclusively by the linear temporal integration (Exponential Decay) of the Attention mechanism."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Attention-Only.")
