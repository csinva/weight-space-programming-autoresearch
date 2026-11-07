import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify InterpretableEmbedder to inject Thermal Noise into the features
# We can inject it at the output of the model before returning.
emb_old = """        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()"""

emb_new = """        hidden_states = self.model(input_ids)
        
        # Stochastic Resonance (Thermal Neural Noise)
        # Biological brains operate at 37C with immense synaptic noise.
        # We inject Gaussian noise into the final hidden states.
        # Because Ridge Regression is linear, adding noise during inference tests 
        # if the system relies on Stochastic Resonance to push signals above threshold or regularize geometry.
        noise_level = 0.5 # Substantial noise relative to the signal
        thermal_noise = torch.randn_like(hidden_states) * noise_level
        hidden_states = hidden_states + thermal_noise
        
        return hidden_states[:, -1, :].cpu()"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Stochastic_Resonance_Thermal_Noise"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Stochastic Resonance (Thermal Noise) Hypothesis: Biological brains operate at 37C with immense ambient synaptic noise. Tested if injecting heavy Gaussian noise (sigma=0.5) into the hidden states during inference improves signal detection and mapping to the BOLD response via Stochastic Resonance."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Stochastic Resonance.")
