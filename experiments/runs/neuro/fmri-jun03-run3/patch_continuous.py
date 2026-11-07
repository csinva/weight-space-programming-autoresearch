import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

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

new_staggered = """            # Continuous integration: reads from adjacent networks (+1, +2, etc)
            for i in range(64):
                # Map each dimension i to a gradually shifted network
                # So head dimension i reads from network (net + i//4) % 15
                net_target = (net + (i // 4)) % 15
                d_start_target = net_target * dim_per_net
                
                # Attenuate distant networks
                scale = S * (1.15 - 0.015 * (i // 4))
                
                l2_attn.W_v.weight[h_start + i, d_start_target + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = scale"""

if old_staggered not in content:
    print("Error: Could not find old staggered logic to replace.")
    exit(1)

content = content.replace(old_staggered, new_staggered)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Continuous")

with open(filepath, "w") as f:
    f.write(content)
print("Applied continuous integration patch")
