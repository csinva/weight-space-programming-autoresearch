import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of block routing just reading strictly from itself (+0) and (+6) and (+12),
# let's try a smoothly decaying read across ALL previous networks.
# We have 64 dimensions. We can read 4 dims from each of the 15 networks!
old_staggered = """            # Staggered logic: 3-way split
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.85"""

new_staggered = """            # Staggered logic: Read from ALL 15 networks
            # 64 dimensions total. Read 4 from each network (15 * 4 = 60), last 4 just from self.
            for n in range(15):
                target_net = (net + n) % 15
                t_start = target_net * dim_per_net
                
                # Attenuate based on distance n
                scale = S * (1.15 - 0.05 * n) # Decreases as we read further away
                
                for j in range(4):
                    idx = n * 4 + j
                    l2_attn.W_v.weight[h_start + idx, t_start + idx] = 1.0
                    l2_attn.W_o.weight[d_start + idx, h_start + idx] = scale
            
            # Remaining 4 back to self
            for j in range(60, 64):
                l2_attn.W_v.weight[h_start + j, d_start + j] = 1.0
                l2_attn.W_o.weight[d_start + j, h_start + j] = S * 1.15"""

content = content.replace(old_staggered, new_staggered)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_All15_Routing")

with open(filepath, "w") as f:
    f.write(content)
print("Applied dense routing patch")
