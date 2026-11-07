import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add multiple frequencies to pos embedding
old_pos = """        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0"""

new_pos = """        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            # Multi-frequency positional embeddings
            pos_emb[p, 24] = S * math.sin(p / 1.0)
            pos_emb[p, 25] = S * math.cos(p / 1.0)
            pos_emb[p, 26] = S * math.sin(p / 4.0)
            pos_emb[p, 27] = S * math.cos(p / 4.0)
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0"""

if old_pos not in content:
    print("Error: Could not find old pos logic to replace.")
    exit(1)

content = content.replace(old_pos, new_pos)

# Need to update pos emb copying for all nets
old_pos_copy = """        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]"""

new_pos_copy = """        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+24:start+30] = pos_emb[:, 24:30]"""
            
content = content.replace(old_pos_copy, new_pos_copy)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Multi_Pos")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Multi-Pos patch")
