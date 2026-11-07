"""FAST WordNet enrichment at ntr64 with small ortho (200) for speed. No corpus stats."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=200, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=30, WORDNET_SCALE=1.5, WORDNET_LEX=True, WORDNET_NSENSES=2,
             NUM_TRAIN=64)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


add("wn30_lex")
add("wn30_nolex", WORDNET_LEX=False)
add("wn15_lex", WORDNET_MINW=15)
add("wn50_lex", WORDNET_MINW=50)
add("wn30_1sense", WORDNET_NSENSES=1)
add("wn30_3sense", WORDNET_NSENSES=3)
add("wn30_sc2.5", WORDNET_SCALE=2.5)
add("wn30_sc1", WORDNET_SCALE=1.0)
