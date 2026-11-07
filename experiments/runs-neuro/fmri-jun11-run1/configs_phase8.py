"""Phase 8: exploit the right-context axis. BASE = right-main d160/w6 + sym-2nd d80/w5
+ topic50 + identK65 + cs1.5 (0.0567). Explore directional decomposition + re-tests."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# 2nd view as LEFT-context (pure fwd/bwd decomposition: right-main + left-2nd)
for d in [60, 80, 100]:
    for w in [5, 8, 12]:
        add(f"left2_d{d}_w{w}", LSA2_DIRECTION="left", LSA2_DIM=d, LSA2_WINDOW=w)

# 2nd view as RIGHT at a different window (multi-scale right): main right-w6 + right-2nd
for d in [60, 80]:
    for w in [3, 4, 10, 12]:
        add(f"right2_d{d}_w{w}", LSA2_DIRECTION="right", LSA2_DIM=d, LSA2_WINDOW=w)

# add a 3rd LEFT view on top of right-main + sym-2nd (all-directions)
for d3 in [40, 60]:
    for dr in ["left", "right"]:
        add(f"v3_{dr}_d{d3}", LSA3_DIM=d3, LSA3_DIRECTION=dr, LSA3_WINDOW=6)

# bigger right-main (with the better 2nd views, main may want more dim)
for d in [180, 200, 220]:
    add(f"mainbig_d{d}", LSA_DIM=d)

# re-test previously-rejected families ON THE RIGHT-CONTEXT BASE
for od in [40, 60, 80]:
    add(f"ortho{od}", ORTHO_DIM=od, ORTHO_SCALE=1.0)
add("morph_on", USE_MORPH=True)
add("recency03", RECENCY_LAMBDA=0.3)   # recency-weighted bag on right base
add("hash_tail", HASH_DIM=100)

# finer main dim x window with cs1.5 base
for d in [155, 160, 165]:
    for w in [5, 6, 7]:
        add(f"m_d{d}_w{w}", LSA_DIM=d, LSA_WINDOW=w)

# combine best-guess: right-main + left-2nd + sym-3rd
add("combo_R_L2", LSA2_DIRECTION="left", LSA2_DIM=80, LSA2_WINDOW=8)
add("combo_R_L2_sym3", LSA2_DIRECTION="left", LSA2_DIM=80, LSA2_WINDOW=8,
    LSA3_DIM=60, LSA3_DIRECTION="both", LSA3_WINDOW=5)
add("ref_base_p8")
