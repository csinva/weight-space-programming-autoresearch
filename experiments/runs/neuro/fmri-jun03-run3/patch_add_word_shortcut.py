import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0407 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0407.py")

with open(filepath, "r") as f:
    content = f.read()

# Since scaling L1 superposition worked so well because it injected the sharp local token history into the medium/long context blocks, what if we injected the PURE EXACT token into the very end of the embedding space?
# L2 writes to 0-63.
# Let's map the EXACT RAW token (dims 0-27) directly into dims 64-91 in L2 output!
# Wait, L2 has 64 dimensions per net. But we have 15 nets. 15 * 64 = 960 dimensions.
# Total dimensions `d_model` is 1020!
# So dimensions 960 to 1019 are completely unused right now!
# Let's write the exact raw token to the end of the embedding!

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
        token_emb[:, 960:988] = token_emb[:, 0:28]"""

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
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0"""

content = content.replace(
    "            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)\n            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)",
    zero_mlp
)

content = content.replace("Deep_Ensemble_L1_175_L2_Wide", "Deep_Ensemble_0407_Plus_Raw_Word")
content = content.replace("with L1 output attention projection scaled by 1.75 and L2 bounds widened to 0.01-14.0.", "with exact raw token routing bypassing all layers into dims 960-988.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
