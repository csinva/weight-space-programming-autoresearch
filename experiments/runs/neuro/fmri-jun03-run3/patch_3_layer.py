import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify build_embedder to take n_layers=3
content = content.replace("n_layers: int = 2", "n_layers: int = 3")

# We need to initialize the weights for the 3rd layer in write_weights
# After layer 2 initialization, we add layer 3 initialization
# In write_weights:
#         l2_attn = model.blocks[1].attn
#         mlp2 = model.blocks[1].mlp
#         ... loop ...

new_weights_code = """
        l3_attn = model.blocks[2].attn
        mlp3 = model.blocks[2].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # --- LAYER 3: Ultra-slow deep smoothing ---
            # Even slower decay than layer 2
            l3_decay = 0.001 + float(net) * (2.999 / 14.0)
            
            l3_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l3_attn.W_k.weight[h_start + 0, d_start + 28] = l3_decay
            
            # Direct pass-through, just smoothing it
            for i in range(28):
                l3_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l3_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            std_dev3 = 0.01
            nn.init.normal_(mlp3.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev3)
            nn.init.normal_(mlp3.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev3 * S)
            
        # Zero out exact token dims in layer 3
        model.blocks[2].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[2].mlp.fc2.weight.data[960:988, :] = 0
"""

# Insert right before the final loop over blocks
insert_target = """        # Zero out MLP and Attention for dimensions 960-988 to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0"""

content = content.replace(insert_target, insert_target + new_weights_code)


# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Deep_3Layer_Hierarchical_Smoothing"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Deep Hierarchy Hypothesis: Added a 3rd Transformer Layer that acts as an ultra-slow temporal smoother (decays 0.001 to 3.0), testing if the fMRI signal requires deeper multi-level temporal integration."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched 3-layer model.")
