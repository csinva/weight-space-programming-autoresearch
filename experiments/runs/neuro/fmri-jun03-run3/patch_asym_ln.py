import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Make final layernorm bias asymmetric across dimensions based on network index
old_ln = """        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)
        model.final_ln.bias.data += 1.18
        model.final_ln.bias.data += 1.18"""

new_ln = """        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)
        
        # Base bias
        model.final_ln.bias.data += 2.36
        
        # Add a slope so earlier networks have higher bias than later networks
        for net in range(num_nets):
            start = net * dim_per_net
            # Slope from +0.1 to -0.1
            slope_adj = 0.1 - (float(net) * (0.2 / 14.0))
            model.final_ln.bias.data[start:start+dim_per_net] += slope_adj
"""

content = content.replace(old_ln, new_ln)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Asym_LN_Bias")

with open(filepath, "w") as f:
    f.write(content)
print("Applied asymmetric LN patch")
