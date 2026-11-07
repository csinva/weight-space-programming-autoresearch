import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tested 1.15, 1.20, 1.05, 1.25. None beat 1.18.
# 1.05: 0.0403
# 1.15: 0.0396
# 1.18: 0.0421
# 1.20: 0.0394
# 1.25: 0.0393
# Wait, 1.05 got 0.0403. Let's try 1.10.
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.10"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Bias_1_10")

with open(filepath, "w") as f:
    f.write(content)
print("Applied LN bias 1.10 patch")
