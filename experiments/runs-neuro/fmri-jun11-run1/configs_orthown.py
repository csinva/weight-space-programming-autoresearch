"""Push orthographic capacity WITH WordNet at ntr93 (WN semantics support more ortho)."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=50, WORDNET_SCALE=1.5, NUM_TRAIN=93)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


add("wn50_o800", ORTHO_DIM=800)
add("wn50_o1000", ORTHO_DIM=1000)
add("wn50_o600_wnsc1", WORDNET_SCALE=1.0)
add("wn30_o600", WORDNET_MINW=30, ORTHO_DIM=600)
