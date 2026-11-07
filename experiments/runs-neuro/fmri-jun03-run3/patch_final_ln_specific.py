import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of blindly +1.18 everywhere, apply +1.18 only on dimensions 900+ (where exact/fast features live) 
# and maybe +1.5 on 0-900 (where long timescale features live).
# Or maybe reverse.
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[:960] += 0.5" # 1.18 + 0.5 = 1.68 on L1/L2 main features
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Bias_Long_1_68")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Final LN bias specific patch")
