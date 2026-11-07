import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace("Deep_Ensemble_0421_Master", "15_Gram_Size")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Expanding ngram_size to 15 in the eval config to see if the network can use more context.")

content = content.replace("config = EncodingConfig()", "config = EncodingConfig()\n    config.ngram_size = 15")

with open(filepath, "w") as f:
    f.write(content)
print("Applied ngram patch")
