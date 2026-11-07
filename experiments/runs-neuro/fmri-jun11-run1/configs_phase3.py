"""Phase 3: optimize the directional-right LSA2 winner (0.0537) and combine winners.
BASE now = CAT_SCALE2 + LSA2(right,d60,w8). Sweep around it + ortho + 3rd view + stack."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# fine-optimize the right-context LSA2: dim x window
for d in [40, 50, 60, 70, 80]:
    for w in [4, 5, 6, 7, 9, 10]:
        add(f"r_d{d}_w{w}", LSA2_DIM=d, LSA2_WINDOW=w)

# right-LSA2(best) + orthographic (clean combo, ortho scale 1 / 1.5)
for od in [40, 60, 80]:
    for osc in [1.0, 1.5]:
        add(f"r60w8_ortho{od}_s{osc}", ORTHO_DIM=od, ORTHO_SCALE=osc)

# add a THIRD (left-context) LSA view on top of right-LSA2 (fwd+bwd predictive)
for d3 in [30, 40, 60]:
    for w3 in [6, 8]:
        add(f"r60w8_left3_d{d3}_w{w3}", LSA3_DIM=d3, LSA3_WINDOW=w3, LSA3_DIRECTION="left")

# combine winners: right-LSA2 + ortho + identscale3 + topic tweaks
add("stack_ortho80s1_is3", ORTHO_DIM=80, ORTHO_SCALE=1.0, IDENT_SCALE=3.0)
add("stack_ortho60s1.5_is3", ORTHO_DIM=60, ORTHO_SCALE=1.5, IDENT_SCALE=3.0)
add("stack_left3_ortho80", LSA3_DIM=40, LSA3_DIRECTION="left", ORTHO_DIM=80, ORTHO_SCALE=1.0)
add("stack_left3_ortho_is3", LSA3_DIM=40, LSA3_DIRECTION="left", ORTHO_DIM=80,
    ORTHO_SCALE=1.0, IDENT_SCALE=3.0)
add("stack_full", LSA2_DIM=60, LSA2_WINDOW=8, LSA3_DIM=40, LSA3_DIRECTION="left",
    ORTHO_DIM=80, ORTHO_SCALE=1.0, IDENT_SCALE=3.0, TOPIC_DIM=60)

# reference (current base = right-LSA2 d60 w8)
add("ref_base_p3")
