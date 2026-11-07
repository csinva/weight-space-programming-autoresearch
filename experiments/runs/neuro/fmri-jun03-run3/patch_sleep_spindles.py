import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Add a "Sleep Spindle" offline consolidation mechanism during the forward pass.
# Biological networks periodically consolidate memory via fast sharp-wave ripples (spindles).
# We can simulate this by taking the temporal average of the last N tokens and projecting it
# globally across the sequence as a "consolidated" memory vector.

model_old = """        for block in self.blocks:
            x = block(x)"""

model_new = """        # Sleep Spindle (Memory Consolidation) Hypothesis
        # Brains consolidate short-term sequential memory into a dense, global invariant vector
        # via sharp-wave ripples. We simulate this by extracting the temporal mean of the 
        # sequence at Layer 1 and injecting it as a global context bias into Layer 2.
        
        x = self.blocks[0](x)
        
        B, T, D = x.shape
        # Spindle consolidation: Global temporal average
        spindle_memory = x.mean(dim=1, keepdim=True) # (B, 1, D)
        
        # Inject the consolidated memory as a universal bias across time
        x = x + 0.5 * spindle_memory
        
        x = self.blocks[1](x)"""

content = content.replace(model_old, model_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Sleep_Spindle_Consolidation"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Sleep Spindle (Memory Consolidation) Hypothesis: Tested if the brain utilizes global memory consolidation vectors (like sharp-wave ripples in sleep) during active comprehension. Modeled this by extracting the global temporal mean of Layer 1 and injecting it as a universal context bias into Layer 2, breaking strict causality to provide a holistic semantic summary."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Sleep Spindles.")
