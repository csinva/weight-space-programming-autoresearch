"""Lean WordNet @ntr93 (smaller ortho -> fewer features -> faster). Headline legit runs."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=200, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=30, WORDNET_SCALE=1.5, NUM_TRAIN=93)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


add("wn30_o200_ntr93", WORDNET_MINW=30, ORTHO_DIM=200)
add("wn50_o200_ntr93", WORDNET_MINW=50, ORTHO_DIM=200)
add("wn20_o200_ntr93", WORDNET_MINW=20, ORTHO_DIM=200)
add("wn30_o0_ntr93", WORDNET_MINW=30, ORTHO_DIM=0)  # WN semantics only (no ortho) — leanest
