import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

model_old = """        for block in self.blocks:
            x = block(x)"""

model_new = """        # Phonological Loop (Auditory Echo Buffer)
        # Baddeley's Working Memory model states we have a phonological loop 
        # that echoes auditory information for ~2 seconds.
        # Assuming ~15 characters per second, 2 seconds is roughly a 30-character delay.
        x = self.blocks[0](x)
        
        B, T, D = x.shape
        echo_delay = 30
        
        if T > echo_delay:
            # Superimpose the echo of the representation from 30 steps ago
            echo_features = torch.cat([torch.zeros(B, echo_delay, D, device=x.device), x[:, :-echo_delay, :]], dim=1)
            # The echo is slightly decayed
            x = x + 0.5 * echo_features
            
        x = self.blocks[1](x)"""

content = content.replace(model_old, model_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Phonological_Loop_Echo"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Phonological Loop (Working Memory) Hypothesis: Baddeley\'s model posits an auditory echo buffer that holds phonetic information for ~2 seconds. Modeled this by superimposing a strict 30-character (approx 2 second) delayed echo of Layer 1 onto itself before processing Layer 2."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Phonological Loop.")
