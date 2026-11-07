"""WordNet enrichment tuning at ntr64 (fast): lexnames, granularity, senses, ortho.
Find best WN config, then confirm at ntr93. No corpus statistics."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=50, WORDNET_SCALE=1.5, WORDNET_LEX=True, WORDNET_NSENSES=2,
             NUM_TRAIN=64)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


add("ref_lex_64")                                   # wn50+lex, ortho600
add("nolex_64", WORDNET_LEX=False)                  # hypernyms only (no lexnames)
add("wn30_lex_64", WORDNET_MINW=30)                 # finer granularity + lex
add("wn20_lex_64", WORDNET_MINW=20)
add("wn30_1sense_64", WORDNET_MINW=30, WORDNET_NSENSES=1)
add("wn30_3sense_64", WORDNET_MINW=30, WORDNET_NSENSES=3)
add("wn30_sc2.5_64", WORDNET_MINW=30, WORDNET_SCALE=2.5)
add("wn30_o400_64", WORDNET_MINW=30, ORTHO_DIM=400)
