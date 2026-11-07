import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify InterpretableEmbedder
emb_old = """        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()"""

emb_new = """        hidden_states = self.model(input_ids)
        
        # Metabolic Energy Constraint (Soft-Thresholding)
        # Brains operate under strict ATP/glucose energy limits.
        # We apply a soft-thresholding operator to the final states, simulating 
        # that only signals strong enough to overcome the metabolic cost are transmitted.
        # S(x) = sign(x) * max(|x| - lambda, 0)
        
        energy_penalty = 0.5 # A significant threshold
        
        final_state = hidden_states[:, -1, :]
        soft_thresholded = torch.sign(final_state) * torch.nn.functional.relu(torch.abs(final_state) - energy_penalty)
        
        return soft_thresholded.cpu()"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Metabolic_Energy_Constraint"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Metabolic Energy Constraint Hypothesis: Biological brains operate under strict ATP/glucose limits (Zipf\'s Law of Least Effort). Applied a Soft-Thresholding operator to the final representations, simulating that only activations strong enough to overcome a baseline metabolic energy cost (lambda=0.5) are propagated, shrinking everything else toward zero."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Metabolic Energy Constraint.")
