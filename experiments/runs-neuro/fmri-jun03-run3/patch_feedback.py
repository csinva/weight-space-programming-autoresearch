import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

model_old = """        for block in self.blocks:
            x = block(x)"""

model_new = """        # Top-Down Feedback (Recurrent Processing)
        # Pass 1: Standard Bottom-Up sweep
        x1 = self.blocks[0](x)
        x2 = self.blocks[1](x1)
        
        # Pass 2: Top-Down Feedback sweep
        # Higher-level semantic state (x2) modulates the lower-level sensory input (x)
        # This mirrors predictive coding where higher cortical areas send feedback to early sensory areas
        x_modulated = x + 0.5 * x2
        x1_feedback = self.blocks[0](x_modulated)
        x2_feedback = self.blocks[1](x1_feedback)
        
        x = x2_feedback"""

content = content.replace(model_old, model_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Top_Down_Feedback_Loops"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Top-Down Cortical Feedback Hypothesis: Replaces the purely feedforward sweep with a recurrent feedback loop. Layer 2 (higher semantics) sends its output back to modulate the input of Layer 1 (early sensory), simulating top-down predictive coding before taking the final read-out."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Top-Down Feedback.")
