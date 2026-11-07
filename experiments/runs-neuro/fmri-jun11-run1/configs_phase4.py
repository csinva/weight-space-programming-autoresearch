"""Phase 4: the right-context LSA is the dominant lever. Push its dim, test a
right-directional MAIN view, and try leaner models. BASE = right-LSA2 d80/w5 (0.0541)."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# push right-LSA2 dim higher (still climbing at 80), window 5 and 6
for d in [90, 100, 120, 150, 200]:
    for w in [5, 6]:
        add(f"r_d{d}_w{w}", LSA2_DIM=d, LSA2_WINDOW=w)

# make the MAIN LSA right-directional too (forward-predictive primary view)
for d in [150, 200, 250]:
    add(f"mainR_d{d}", LSA_DIRECTION="right", LSA_DIM=d)
add("mainR_lsa2both", LSA_DIRECTION="right", LSA2_DIRECTION="both")
add("mainR_lsa2left", LSA_DIRECTION="right", LSA2_DIRECTION="left", LSA2_DIM=80, LSA2_WINDOW=5)

# best right-LSA2 + clean ortho (d60 scale1) at the optimal LSA2 settings
for d in [100, 120]:
    add(f"r_d{d}_w5_ortho60", LSA2_DIM=d, LSA2_WINDOW=5, ORTHO_DIM=60, ORTHO_SCALE=1.0)

# LEANER models: strong right-LSA2, drop topic and/or categories (stack overfit in P3)
add("lean_notopic_rd120", LSA2_DIM=120, LSA2_WINDOW=5, TOPIC_DIM=0)
add("lean_noident_rd120", LSA2_DIM=120, LSA2_WINDOW=5, IDENT_TOPK=0)
add("lean_rd150_topic40", LSA2_DIM=150, LSA2_WINDOW=5, TOPIC_DIM=40)
add("lean_main175_rd120", LSA_DIM=175, LSA2_DIM=120, LSA2_WINDOW=5)

# right-LSA2 big + sppmi re-tune (denoise the larger view)
for sp in [8.0, 12.0, 15.0]:
    add(f"r_d120_sppmi{int(sp)}", LSA2_DIM=120, LSA2_WINDOW=5, SPPMI_SHIFT=sp)

add("ref_base_p4")
