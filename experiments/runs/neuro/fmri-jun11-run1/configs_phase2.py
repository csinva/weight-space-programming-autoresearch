"""Phase 2: new feature families (orthographic, directional LSA) + combos."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# orthographic (char-trigram word-form) family: dim x scale
for d in [20, 30, 50, 80, 120]:
    for sc in [1.0, 2.0, 4.0]:
        add(f"ortho_d{d}_s{int(sc)}", ORTHO_DIM=d, ORTHO_SCALE=sc)

# directional second LSA view: left / right context, dim x window
for direction in ["left", "right"]:
    for d in [40, 60]:
        for w in [8, 12]:
            add(f"lsa2{direction}_d{d}_w{w}", LSA2_DIRECTION=direction, LSA2_DIM=d, LSA2_WINDOW=w)

# orthographic combined with the current best stack knobs
for d in [30, 50, 80]:
    add(f"ortho{d}_best", ORTHO_DIM=d, ORTHO_SCALE=2.0)

# directional LSA2 + orthographic combos
add("dirL_ortho50", LSA2_DIRECTION="left", ORTHO_DIM=50, ORTHO_SCALE=2.0)
add("dirR_ortho50", LSA2_DIRECTION="right", ORTHO_DIM=50, ORTHO_SCALE=2.0)

# add orthographic on top of small tweaks to identity/topic (interaction checks)
add("ortho50_k65", ORTHO_DIM=50, ORTHO_SCALE=2.0, IDENT_TOPK=65)
add("ortho50_topic60", ORTHO_DIM=50, ORTHO_SCALE=2.0, TOPIC_DIM=60)
add("ortho50_sppmi8", ORTHO_DIM=50, ORTHO_SCALE=2.0, SPPMI_SHIFT=8.0)

# larger orthographic with stronger downweight (avoid overfit)
for d in [40, 60]:
    for sc in [0.5, 1.5]:
        add(f"ortho_d{d}_s{sc}", ORTHO_DIM=d, ORTHO_SCALE=sc)

# --- Phase 1 found CAT_SCALE=2 best (0.0519); BASE now uses 2.0. Fine cat-scale + combos: ---
for cs in [1.0, 1.5, 2.5, 3.5]:
    add(f"catscale_fine{cs}", CAT_SCALE=cs)
# combine the cat-scale win with the other near-best knobs
add("cs2_lsa175", LSA_DIM=175)
add("cs2_topic60", TOPIC_DIM=60)
add("cs2_identscale3", IDENT_SCALE=3.0)
add("cs2_lsa175_topic60", LSA_DIM=175, TOPIC_DIM=60)
add("cs2_lsa175_identscale3", LSA_DIM=175, IDENT_SCALE=3.0)
add("cs2_topic60_identscale3", TOPIC_DIM=60, IDENT_SCALE=3.0)
add("cs2_all", LSA_DIM=175, TOPIC_DIM=60, IDENT_SCALE=3.0)
# cat-scale win + new families
add("cs2_ortho50", ORTHO_DIM=50, ORTHO_SCALE=2.0)
add("cs2_ortho30", ORTHO_DIM=30, ORTHO_SCALE=2.0)
add("cs2_dirL", LSA2_DIRECTION="left")
# reference (now CAT_SCALE=2 base)
add("ref_cs2_p2")
