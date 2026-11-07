with open("new_model_single_char_decay.py", "r") as f:
    code = f.read()

code = code.replace("for i in range(d_model):", "for i in range(d_head):")
code = code.replace("l2_attn.W_v.weight[k*d_head + i, i] = S * 1.0", "l2_attn.W_v.weight[k*d_head + i, k*d_head + i] = S * 1.0")
code = code.replace("l2_attn.W_o.weight[i, k*d_head + i] = 1.0 / n_heads # average heads", "l2_attn.W_o.weight[k*d_head + i, k*d_head + i] = 1.0 / n_heads")

with open("new_model_single_char_decay.py", "w") as f:
    f.write(code)
