import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

model_old = """        for block in self.blocks:
            x = block(x)"""

model_new = """        # Neuromodulatory Arousal (Global Gain Control)
        # The Locus Coeruleus releases noradrenaline in response to unexpected stimuli,
        # globally increasing the gain (responsiveness) of cortical networks.
        # We compute a local 'surprise' metric (temporal derivative of the input sequence)
        # and use it to multiplicatively modulate the hidden states between layers.
        
        x_prev = self.blocks[0](x)
        
        B, T, D = x_prev.shape
        # Compute the temporal derivative (magnitude of change)
        dx = torch.zeros_like(x_prev)
        dx[:, 1:, :] = x_prev[:, 1:, :] - x_prev[:, :-1, :]
        
        # Surprise is the L2 norm of the change at each timestep
        surprise = torch.norm(dx, dim=-1, keepdim=True) # (B, T, 1)
        
        # Normalize surprise to act as a gain multiplier (centered around 1.0)
        # using a simple moving average normalization or just standardization over T
        mean_surprise = surprise.mean(dim=1, keepdim=True) + 1e-5
        gain = surprise / mean_surprise
        
        # Apply global gain control
        x_aroused = x_prev * gain
        
        x = self.blocks[1](x_aroused)"""

content = content.replace(model_old, model_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Neuromodulatory_Arousal_Gain"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Neuromodulatory Arousal (Global Gain Control) Hypothesis: The brain releases noradrenaline (from the locus coeruleus) during unexpected stimuli, globally amplifying cortical responsiveness. Modeled this by computing a running \'surprise\' metric (the temporal derivative of Layer 1 representations) and using it to multiplicatively scale the global activation magnitude before passing to Layer 2."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Neuromodulatory Arousal.")
