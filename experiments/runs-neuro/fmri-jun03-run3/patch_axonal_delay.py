import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

model_old = """        for block in self.blocks:
            x = block(x)"""

model_new = """        # Axonal Conduction Delay (Synaptic Delay)
        # Biological signals do not propagate instantly across cortical layers.
        # It takes physical time for action potentials to travel down myelinated axons.
        # We simulate a 1-timestep physical delay between Layer 1 (early processing) and Layer 2 (integration).
        x = self.blocks[0](x)
        
        # Shift sequence by 1 to represent the physical delay
        B, T, D = x.shape
        x_delayed = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1, :]], dim=1)
        
        x = self.blocks[1](x_delayed)"""

content = content.replace(model_old, model_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Axonal_Conduction_Delay"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Axonal Conduction Delay Hypothesis: Biological signals do not propagate instantly between cortical areas; action potentials take physical time to travel down myelinated axons. Enforced a rigid 1-timestep (1 character) physical synaptic delay between Layer 1 and Layer 2."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Axonal Conduction Delay.")
