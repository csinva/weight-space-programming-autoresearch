"""Decisive WordNet comparison at ntr64, fewer parallel jobs (more CPU each)."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=0, WORDNET_SCALE=1.5, NUM_TRAIN=64)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


add("ref64", NUM_TRAIN=64)                       # ortho600+cats, no WN
add("wn30_64", WORDNET_MINW=30, NUM_TRAIN=64)    # + WordNet (314 dims)
add("wn50_64", WORDNET_MINW=50, NUM_TRAIN=64)    # + WordNet (smaller, ~150 dims)
add("wn30_o200_64", WORDNET_MINW=30, ORTHO_DIM=200, NUM_TRAIN=64)  # smaller ortho + WN
