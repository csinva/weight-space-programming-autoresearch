import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We found that 0.68 on L1/L2 main features got 0.0418, very close to 0.0421.
# What if we give 1.18 to the main features, and give EXACT tokens 1.18 - something?
# Oh wait, we did +1.18 and then += 0.32 on exact tokens (1.50) -> 0.0395
# What about keeping L1/L2 main features at 1.18 and giving exact tokens NO bias (0.0)?
# Wait, we tried taking it to -0.5 and it dropped.
# What if we set exact tokens bias to 0.5?
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[960:988] = 0.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_LN_Exact_0_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied LN Exact 0.5 patch")
