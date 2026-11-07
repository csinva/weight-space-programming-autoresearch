"""Fast WordNet signal at ntr32/64 (cheaper than ntr93). Does hand-built taxonomy help?"""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=0, WORDNET_SCALE=1.5, NUM_TRAIN=32)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


# quick: WordNet vs no-WordNet at ntr32 and ntr64
add("ref32", NUM_TRAIN=32)
add("wn30_32", WORDNET_MINW=30, NUM_TRAIN=32)
add("wn15_32", WORDNET_MINW=15, NUM_TRAIN=32)
add("ref64", NUM_TRAIN=64)
add("wn30_64", WORDNET_MINW=30, NUM_TRAIN=64)
add("wn15_64", WORDNET_MINW=15, NUM_TRAIN=64)
# WN semantics only (no ortho) at ntr64 — does taxonomy generalize?
add("wn15_noortho_64", WORDNET_MINW=15, ORTHO_DIM=0, NUM_TRAIN=64)
add("wn30_smallortho_64", WORDNET_MINW=30, ORTHO_DIM=200, NUM_TRAIN=64)
