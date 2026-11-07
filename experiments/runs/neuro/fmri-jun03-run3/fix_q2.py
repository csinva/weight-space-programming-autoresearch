import re

with open("interpretable_transformer.py", "r") as f:
    content = f.read()

# I need to change:
# l1_attn.W_q.weight[k*d_head + 1, 1004] = S * 1.0
# to:
# l1_attn.W_q.weight[k*d_head + 2, 1004] = S * 1.0
# But wait, what does K have?
# K = S * [m, 1, -beta*m^2]
# K[0] = m
# K[1] = 1
# K[2] = -beta*m^2
# If q = [2*beta*(p-k), 1, 1], then q.K = 2*beta*(p-k)m + 1 - beta*m^2.
# So q[1] = 1 and q[2] = 1. Let's just set q[2] = 1.0.

old_code = "l1_attn.W_q.weight[k*d_head + 1, 1004] = S * 1.0"
new_code = "l1_attn.W_q.weight[k*d_head + 2, 1004] = S * 1.0"

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("interpretable_transformer.py", "w") as f:
        f.write(content)
    print("Fixed q[2]")
else:
    print("Not found!")
