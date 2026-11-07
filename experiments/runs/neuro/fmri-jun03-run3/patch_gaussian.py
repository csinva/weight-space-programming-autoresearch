import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# First, modify pos_emb to include p**2
pos_old = """            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]"""
pos_new = """            p_norm = p / max_seq_len
            pos_emb[p, 27] = S * (p_norm ** 2)
            pos_emb[p, 28] = S * p_norm
            pos_emb[p, 29] = S * 1.0
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+27:start+30] = pos_emb[:, 27:30]"""
content = content.replace(pos_old, pos_new)

# Now, modify Layer 1 attention weights
l1_attn_old = """            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay"""
l1_attn_new = """            # Gaussian Decay: -decay * (i-j)^2
            scale_factor = math.sqrt(d_head) * 10.0 # Compensate for softmax temp
            
            l1_attn.W_q.weight[h_start + 0, d_start + 27] = -l1_decay * scale_factor / S
            l1_attn.W_q.weight[h_start + 1, d_start + 28] = 2.0 * l1_decay * scale_factor / S
            l1_attn.W_q.weight[h_start + 2, d_start + 29] = 1.0 / S
            
            l1_attn.W_k.weight[h_start + 0, d_start + 29] = 1.0 / S
            l1_attn.W_k.weight[h_start + 1, d_start + 28] = 1.0 / S
            l1_attn.W_k.weight[h_start + 2, d_start + 27] = -l1_decay * scale_factor / S"""
content = content.replace(l1_attn_old, l1_attn_new)

# Now, modify Layer 2 attention weights
l2_attn_old = """            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay"""
l2_attn_new = """            scale_factor2 = math.sqrt(d_head) * 10.0
            
            l2_attn.W_q.weight[h_start + 0, d_start + 27] = -l2_decay * scale_factor2 / S
            l2_attn.W_q.weight[h_start + 1, d_start + 28] = 2.0 * l2_decay * scale_factor2 / S
            l2_attn.W_q.weight[h_start + 2, d_start + 29] = 1.0 / S
            
            l2_attn.W_k.weight[h_start + 0, d_start + 29] = 1.0 / S
            l2_attn.W_k.weight[h_start + 1, d_start + 28] = 1.0 / S
            l2_attn.W_k.weight[h_start + 2, d_start + 27] = -l2_decay * scale_factor2 / S"""
content = content.replace(l2_attn_old, l2_attn_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Gaussian_Temporal_Window"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Gaussian Window Hypothesis: Replaced the standard Exponential Decay (e^-x) with a Gaussian Temporal Decay (e^-x^2). Achieved by constructing a perfect algebraic polynomial (-i^2 + 2ij - j^2) via q @ k using 3 dimensions."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Gaussian temporal windows.")
