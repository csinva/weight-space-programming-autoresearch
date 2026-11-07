"""Final WordNet tuning at ntr93 to maximize beyond 0.0831 (wn50+ortho200)."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=200, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=50, WORDNET_SCALE=1.5, NUM_TRAIN=93)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


add("wn50_o400", ORTHO_DIM=400)
add("wn50_o600", ORTHO_DIM=600)
add("wn40_o300", WORDNET_MINW=40, ORTHO_DIM=300)
add("wn50_sc2.5_o200", WORDNET_SCALE=2.5)
add("wn60_o200", WORDNET_MINW=60)
add("wn50_o200_catsc3", CAT_SCALE=3.0)
