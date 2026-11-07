import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace("max_seq_len: int = 64", "max_seq_len: int = 128")
content = content.replace("pos_emb = torch.zeros(max_seq_len, d_model)", "pos_emb = torch.zeros(max_seq_len, d_model)")
content = content.replace("pos_emb[p, 28] = S * (p / max_seq_len)", "pos_emb[p, 28] = S * (p / max_seq_len)")

content = content.replace("Deep_Ensemble_0421_Master", "Full_Context_Seq_128_Scale")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Increasing max_seq_len to 128 to capture the full 10-gram without truncation.")

with open(filepath, "w") as f:
    f.write(content)
print("Applied seq len patch")
