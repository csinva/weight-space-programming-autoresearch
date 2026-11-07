"""Phase 9: push the directional frontier. BASE = right-main d160/w6 + sym-2nd d80/w5
+ topic50 + identK65 + cs1.5 (0.0567). Full directional decomposition + multi-scale right."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# replace symmetric 2nd view with a LARGE LEFT view (right-main + big-left = fwd+bwd at scale)
for d in [80, 100, 120, 150]:
    for w in [5, 6, 8]:
        add(f"bigleft2_d{d}_w{w}", LSA2_DIRECTION="left", LSA2_DIM=d, LSA2_WINDOW=w)

# right-main + sym-2nd + LARGE LEFT 3rd view (keep sym, add big left)
for d3 in [80, 100, 120]:
    for w3 in [6, 8]:
        add(f"left3big_d{d3}_w{w3}", LSA3_DIM=d3, LSA3_DIRECTION="left", LSA3_WINDOW=w3)

# multi-scale RIGHT: main right-w6 + 2nd right-w3 (local) + 3rd right-w12 (broad)
for d2 in [60, 80]:
    add(f"multiR_2w3_d{d2}", LSA2_DIRECTION="right", LSA2_WINDOW=3, LSA2_DIM=d2,
        LSA3_DIM=60, LSA3_DIRECTION="right", LSA3_WINDOW=12)
add("multiR_full", LSA2_DIRECTION="right", LSA2_WINDOW=3, LSA2_DIM=80,
    LSA3_DIM=60, LSA3_DIRECTION="right", LSA3_WINDOW=12, LSA_DIM=160)

# bigger right-main with a left 2nd (let main carry more, left complements)
for d in [180, 200]:
    add(f"mR{d}_left2_d100", LSA_DIM=d, LSA2_DIRECTION="left", LSA2_DIM=100, LSA2_WINDOW=6)

# right-main + sym-2nd + left-3rd + ortho (combine the two best new families cleanly)
add("R_sym2_left3_ortho", LSA3_DIM=80, LSA3_DIRECTION="left", LSA3_WINDOW=6,
    ORTHO_DIM=50, ORTHO_SCALE=1.0)
add("R_left2_100_ortho", LSA2_DIRECTION="left", LSA2_DIM=100, LSA2_WINDOW=6,
    ORTHO_DIM=50, ORTHO_SCALE=1.0)

# SPPMI re-tune for the directional views (denoise the right view more/less)
for sp in [6.0, 8.0, 14.0, 20.0]:
    add(f"sppmi{int(sp)}", SPPMI_SHIFT=sp)

add("ref_base_p9")
