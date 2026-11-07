import os
import random
import sys
import time

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

def run_trial(iteration):
    # Base params
    ln_bias = random.uniform(1.0, 1.4)
    l1_decay_start = random.uniform(5.0, 20.0)
    l1_decay_range = random.uniform(40.0, 80.0)
    l2_decay_start = random.uniform(0.001, 0.05)
    l2_decay_range = random.uniform(5.0, 20.0)
    stagger_1 = random.choice([4, 5, 6, 7])
    stagger_2 = random.choice([10, 11, 12, 13])
    
    with open("runs-neuro/fmri-jun03-run3/final_model_0421.py", "r") as f:
        content = f.read()

    # Modify LN bias
    content = content.replace("model.final_ln.bias.data += 1.18", f"model.final_ln.bias.data += {ln_bias / 2.0:.4f}")

    # Modify L1 decay
    content = content.replace("l1_decay = 15.0 + float(net) * (65.0 / 14.0)", f"l1_decay = {l1_decay_start:.4f} + float(net) * ({l1_decay_range:.4f} / 14.0)")

    # Modify L2 decay
    content = content.replace("l2_decay = 0.01 + float(net) * (13.99 / 14.0)", f"l2_decay = {l2_decay_start:.4f} + float(net) * ({l2_decay_range:.4f} / 14.0)")

    # Modify stagger
    content = content.replace("net_b = (net + 6) % 15", f"net_b = (net + {stagger_1}) % 15")
    content = content.replace("net_c = (net + 12) % 15", f"net_c = (net + {stagger_2}) % 15")

    name = f"Deep_Ensemble_Rand_{iteration}"
    content = content.replace("Deep_Ensemble_0421_Master", name)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"\n--- Running {name} ---")
    print(f"LN Bias: {ln_bias:.4f}, L1: {l1_decay_start:.2f}-{l1_decay_start+l1_decay_range:.2f}, L2: {l2_decay_start:.4f}-{l2_decay_start+l2_decay_range:.2f}, Stagger: {stagger_1}, {stagger_2}")
    
    os.system("uv run python runs-neuro/fmri-jun03-run3/interpretable_transformer.py")

for i in range(5):
    run_trial(i)

