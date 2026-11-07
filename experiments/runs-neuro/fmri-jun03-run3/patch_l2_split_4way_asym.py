import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

old_logic = """            # Staggered logic: 3-way split
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

new_logic = """            # Staggered logic: 4-way split asymmetric
            net_b = (net + 4) % 15
            net_c = (net + 8) % 15
            net_d = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            d_start_d = net_d * dim_per_net
            
            for i in range(16):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 16 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 16 + i, h_start + 16 + i] = S * 1.05
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 32 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 32 + i, h_start + 32 + i] = S * 0.95
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 48 + i, d_start_d + i] = 1.0
                l2_attn.W_o.weight[d_start + 48 + i, h_start + 48 + i] = S * 0.85"""

content = content.replace(old_logic, new_logic)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_L2_Split_4Way_Asym")

with open(filepath, "w") as f:
    f.write(content)
print("Applied 4-way asymmetric patch")
