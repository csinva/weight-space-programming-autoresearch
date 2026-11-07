"""Final test: does MAX feature richness at MAX data (ntr93) help? (pieces overfit at low data)."""
CONFIGS = []
NT = 93


def add(name, **kw):
    kw.setdefault("NUM_TRAIN", NT)
    CONFIGS.append(dict(name=name, **kw))


# multi-directional high-dim at max data: right-main + left-2nd + symmetric-3rd
add("multidir93", LSA_DIM=160, LSA2_DIRECTION="left", LSA2_DIM=100, LSA2_WINDOW=6,
    LSA3_DIM=100, LSA3_DIRECTION="both", LSA3_WINDOW=5)
# bigger right-main + bigger symmetric 2nd + more topic
add("big93", LSA_DIM=220, LSA2_DIM=140, LSA2_WINDOW=5, TOPIC_DIM=120)
# kitchen sink: more of everything
add("sink93", LSA_DIM=200, LSA2_DIM=120, LSA2_WINDOW=5, LSA3_DIM=80, LSA3_DIRECTION="left",
    TOPIC_DIM=120, IDENT_TOPK=150, ORTHO_DIM=80, ORTHO_SCALE=1.0)
# raw-cooc as main at max data (Huth-style) with full stack
add("rawmain93", RAW_COOC_N=985, NUM_TRAIN=93)
# moderate richness bump that might be the sweet spot at high data
add("mod93", LSA_DIM=180, LSA2_DIM=100, LSA2_WINDOW=5, TOPIC_DIM=90, IDENT_TOPK=100)
# control: current best config at ntr93 (=0.0899 expected)
add("ctrl93")
