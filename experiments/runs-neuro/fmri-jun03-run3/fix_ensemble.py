with open("new_model_ensemble_final.py", "r") as f:
    code = f.read()

# For d_model=1020, n_heads=10 -> d_head=102
# So h_start + 0 is fine, but we can't map 1:1 if d_head > dim_per_net.
# Let's decouple h_start from net index explicitly.

code = code.replace("h_start = net * d_head", "h_start = (net % n_heads) * d_head")

with open("new_model_ensemble_final.py", "w") as f:
    f.write(code)
