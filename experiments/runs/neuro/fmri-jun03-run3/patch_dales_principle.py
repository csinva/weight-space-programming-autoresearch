import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

target = "        model.final_ln.bias.data += 1.18"

dale_code = """        model.final_ln.bias.data += 1.18
        
        # Dale's Principle (Excitatory / Inhibitory Split)
        # Biological neurons release only one type of neurotransmitter.
        # Thus, a neuron's outgoing synaptic weights must be either all positive (excitatory)
        # or all negative (inhibitory). Standard ANNs mix positive/negative weights freely.
        # We enforce Dale's Principle on the MLPs: 80% Excitatory, 20% Inhibitory.
        for block in model.blocks:
            w2 = block.mlp.fc2.weight.data # shape (d_model, d_ff)
            D_out, D_in = w2.shape
            
            # Make all magnitudes positive
            w2_abs = w2.abs()
            
            # Create signs: 80% +1, 20% -1
            signs = torch.ones(D_in, device=w2.device)
            num_inhib = int(D_in * 0.20)
            signs[:num_inhib] = -1.0
            
            # Shuffle signs across the hidden neurons
            torch.manual_seed(42) # Deterministic
            signs = signs[torch.randperm(D_in)]
            
            # Apply Dale's Principle (each hidden neuron has strictly one sign for all outgoing weights)
            block.mlp.fc2.weight.data = w2_abs * signs.unsqueeze(0)"""

content = content.replace(target, dale_code)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Dales_Principle_Excitatory_Inhibitory"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Dale\'s Principle Hypothesis: Biological neurons release only one type of neurotransmitter. Re-initialized the MLP outgoing weights to strictly obey Dale\'s Principle (80% purely excitatory neurons with only positive weights, 20% purely inhibitory neurons with only negative weights), removing the non-biological mixed-sign weights of standard ANNs."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Dale's Principle.")
