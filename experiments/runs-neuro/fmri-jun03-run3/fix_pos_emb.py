import re

with open("interpretable_transformer.py", "r") as f:
    content = f.read()

old_code = """        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1003] = float(p)**2
            pos_emb[p, 1004] = 1.0"""

new_code = """        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1003] = float(p)**2
            pos_emb[p, 1004] = 1.0
            
            # Balance out the sum to keep LayerNorm mean ≈ 0
            pos_emb[p, 1005] = -float(p)
            pos_emb[p, 1006] = -float(p)**2
            pos_emb[p, 1007] = -1.0"""

content = content.replace(old_code, new_code)
with open("interpretable_transformer.py", "w") as f:
    f.write(content)
