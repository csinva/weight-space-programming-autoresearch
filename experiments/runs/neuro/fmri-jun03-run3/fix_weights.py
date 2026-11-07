import re

with open("interpretable_transformer.py", "r") as f:
    content = f.read()

# Fix W_q
content = content.replace("l1_attn.W_q.weight[1002, k*d_head + 0] = S * 2.0 * beta", "l1_attn.W_q.weight[k*d_head + 0, 1002] = S * 2.0 * beta")
content = content.replace("l1_attn.W_q.weight[1004, k*d_head + 0] = S * (-2.0 * beta * k)", "l1_attn.W_q.weight[k*d_head + 0, 1004] = S * (-2.0 * beta * k)")
content = content.replace("l1_attn.W_q.weight[1004, k*d_head + 1] = S * 1.0", "l1_attn.W_q.weight[k*d_head + 1, 1004] = S * 1.0")

# Fix W_k
content = content.replace("l1_attn.W_k.weight[1002, k*d_head + 0] = S * 1.0", "l1_attn.W_k.weight[k*d_head + 0, 1002] = S * 1.0")
content = content.replace("l1_attn.W_k.weight[1004, k*d_head + 1] = S * 1.0", "l1_attn.W_k.weight[k*d_head + 1, 1004] = S * 1.0")
content = content.replace("l1_attn.W_k.weight[1003, k*d_head + 2] = S * (-beta)", "l1_attn.W_k.weight[k*d_head + 2, 1003] = S * (-beta)")

# Fix W_v and W_o
content = content.replace("l1_attn.W_v.weight[i, k*d_head + i] = S * 1.0", "l1_attn.W_v.weight[k*d_head + i, i] = S * 1.0")
content = content.replace("l1_attn.W_o.weight[k*d_head + i, k*50 + i] = 1.0", "l1_attn.W_o.weight[k*50 + i, k*d_head + i] = 1.0")

with open("interpretable_transformer.py", "w") as f:
    f.write(content)
