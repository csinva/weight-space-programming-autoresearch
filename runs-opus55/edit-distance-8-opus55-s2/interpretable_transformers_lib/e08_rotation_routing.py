"""
e08_rotation_routing snapshot (FAILED, 16.5%): identical to interpretable_transformers_lib/e09_rotation_routing_inert17.py
apart from the shorthand name/description and this one line in write_weights():

    e08:  pos[p, M(i)] = 1.0                 # inert: "matches everything", so z = 0
    e09:  pos[p, M(i)] = N_LAYERS + 1.0      # inert: m >= 1 for all N_LAYERS sweeps, so z = 0

Cause of the failure: cosine match values can be negative at inert positions, so m fell below 1 and woke them.
"""
