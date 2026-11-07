import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Zero exact tokens got 0.0395. The exact tokens ARE doing something since dropping them lowered correlation from 0.0421.
# What if we give the exact tokens a bigger representation window?
# They currently live in 960:988 (28 dimensions).
# This is hard to do without restructuring the whole thing.

# Wait, `zeroing` exact tokens resulted in exactly 0.0395, which is the baseline WITHOUT the 1.18 bias shift.
# This means the +1.18 shift on the exact tokens specifically is responsible for the bump to 0.0421.
# Let's test applying +1.18 ONLY to the exact tokens, and NO bias to the rest of the network.
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data[960:988] += 1.18"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Bias_Exact_Only")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Bias Exact Only patch")
