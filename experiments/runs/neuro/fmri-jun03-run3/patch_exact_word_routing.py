import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# We discovered that L1 does exactly nothing except push to 30-57, and L2 reads 0-21 (which are the raw tokens).
# We want to give Ridge regression EXACT access to the raw tokens in the final output embedding.
# The final output is `hidden_states[:, -1, :].cpu()`.
# Right now, `model.final_ln(x)` applies LayerNorm to the final output, which scales the raw tokens and the context based on each other's variance!
# We can bypass final_ln for the raw token dimensions, OR we can just write an explicit "routing token" using the pos_emb to make a completely untouched copy of the token.

# But there is a simpler way. The model has 64 dimensions per net. 
# Dim 63 is completely unused!
# Let's write the raw token identity to dim 63 in L2!
# Or we can write the raw token to dims 58-63!
# Wait, a token is 28 dims. We don't have enough free dims per net to copy the whole token.
# But wait! The `token_emb` writes the token to dims 0-27 for EVERY net!
# `token_emb[:, start:start+28] = token_emb[:, 0:28]`
# So EVERY net has the token in 0-27.
# And L2 writes to 0-21, 22-42, 43-63.
# This OVERWRITES the token in dims 0-21!
# No wonder! L2 is overwriting the exact token representations with the integrated context.
# That means by the end of L2, the pure exact token is mostly GONE from 0-27. It's been added to the integrated context.
# Let's change L2 to NOT overwrite 0-27! Let's have L2 write its integrated context to 28-63 instead!
# Wait, 28 and 29 are pos_emb.
# L1 writes to 30-57. 
# If L2 writes to 0-63, it's colliding with everything!

# What if L2 writes to the MLP instead of residual? It has to write to residual.
# Let's adjust L2 so it writes to dims 30-63, but since it needs 64 dims, we can't fit it.
# Actually, the 3-way staggered split is 22, 21, 21 = 64 dims.
# What if we put the L2 output into 30-63 (which is 34 dims) and compress it to 12, 11, 11?
# No, let's keep L2 writing to 0-63. But we have 1020 total dims.
# What if we expand the model from 1020 to 1020 + 28 = 1048?
# And put the exact token in the last 28 dimensions, completely untouched by any attention or MLP?

expansion = """        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        # Write exactly to the last 28 dimensions
        token_emb[:, 960:988] = token_emb[:, 0:28] * 5.0"""

content = re.sub(
    r'        num_nets = 15\n        dim_per_net = 64\n        ff_per_net = 256\n        \n        C = 10000\.0\n        S = C \/ math\.sqrt\(d_model\)\n\n        token_emb = torch\.zeros\(vocab_size, d_model\)\n        for i, c in enumerate\(VOCAB\):\n            if c == \' \':\n                token_emb\[i, 0\] = S \* 1\.0 \n            elif c\.isalpha\(\):\n                token_emb\[i, 1\] = S \* 1\.0 \n                token_emb\[i, 2 \+ \(ord\(c\) - ord\(\'a\'\)\)\] = S \* 1\.0 \n                \n        for net in range\(num_nets\):\n            start = net \* dim_per_net\n            token_emb\[:, start:start\+28\] = token_emb\[:, 0:28\]',
    expansion,
    content,
    flags=re.DOTALL
)

# And zero out the MLP weights for the last 28 dimensions so they aren't corrupted
zero_mlp = """            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        # Zero out MLP and Attention for dimensions 960-988 to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0
        
        # Bypass LayerNorm for the exact tokens
        # We can't easily bypass LayerNorm without changing the forward pass.
        # But we can set LayerNorm weights to 1 and bias to 0, which is default.
        # Since we initialized LayerNorm to ones, it will scale them by the variance of the whole vector.
        # That's fine, it preserves the information."""

content = content.replace(
    "            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)\n            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)",
    zero_mlp
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Exact_Word_Routing")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) with explicit exact token routing in dims 960-988.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
