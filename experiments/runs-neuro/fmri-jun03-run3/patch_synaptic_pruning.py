import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# We will inject a pruning step at the very end of write_weights
target = "        model.final_ln.bias.data += 1.18"

pruning_code = """        model.final_ln.bias.data += 1.18
        
        # Synaptic Pruning (Neural Darwinism)
        # Prune 90% of the MLP connections (keep only the top 10% largest magnitudes)
        sparsity_level = 0.90
        for block in model.blocks:
            for fc in [block.mlp.fc1, block.mlp.fc2]:
                weight = fc.weight.data
                # Find the threshold for the 90th percentile
                k = int(weight.numel() * (1.0 - sparsity_level))
                if k > 0:
                    threshold = torch.topk(weight.abs().flatten(), k).values[-1]
                    mask = weight.abs() >= threshold
                    fc.weight.data *= mask.float()"""

content = content.replace(target, pruning_code)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Synaptic_Pruning_Neural_Darwinism"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Synaptic Pruning (Neural Darwinism) Hypothesis: Biological brains undergo massive synaptic pruning during development, resulting in highly sparse connectivity. Applied extreme magnitude pruning (90% sparsity) to all MLP weight matrices to test if the fMRI BOLD signal relies on sparse, isolated sub-networks (Lottery Tickets) rather than dense, fully-connected distributed representations."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Synaptic Pruning.")
