"""WordNet semantics at MAX data (ntr93) — the legit semantic lever. WN50@ntr64=0.0762."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=30, WORDNET_SCALE=1.5, NUM_TRAIN=93)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


# WordNet granularity at ntr93 (min_words -> dim)
for mw in [50, 30, 20, 12]:
    add(f"wn{mw}_ntr93", WORDNET_MINW=mw)
# WordNet scale tuning at ntr93
add("wn30_sc1_ntr93", WORDNET_SCALE=1.0)
add("wn30_sc2.5_ntr93", WORDNET_SCALE=2.5)
# WN + smaller ortho (let WN carry semantics, ortho just identity)
add("wn20_o300_ntr93", WORDNET_MINW=20, ORTHO_DIM=300)
add("wn20_o800_ntr93", WORDNET_MINW=20, ORTHO_DIM=800)
