"""WordNet hypernym semantic features (hand-built taxonomy, NO corpus statistics).
Comprehensive version of hand-coded categories. Base = legit ortho600 + cats + morph."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5,
             WORDNET_MINW=0, WORDNET_SCALE=1.5, NUM_TRAIN=93)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


# WordNet feature granularity (min_words/synset -> dim): smaller = more dims
for mw in [50, 30, 15, 8]:
    add(f"wn{mw}_ntr93", WORDNET_MINW=mw)
# WordNet at lower data (does it scale or overfit?)
add("wn30_ntr32", WORDNET_MINW=30, NUM_TRAIN=32)
add("wn30_ntr64", WORDNET_MINW=30, NUM_TRAIN=64)
# WordNet scale tuning
add("wn30_sc0.7_ntr93", WORDNET_MINW=30, WORDNET_SCALE=0.7)
add("wn30_sc3_ntr93", WORDNET_MINW=30, WORDNET_SCALE=3.0)
# WordNet WITHOUT the coarse 57 cats (replace cats with WordNet)
add("wn30_nocat_ntr93", WORDNET_MINW=30, CAT_SCALE=0.0)
# WordNet without orthographic (semantics only) — does WN generalize?
add("wn15_noortho_ntr93", WORDNET_MINW=15, ORTHO_DIM=0)
# reference (no WordNet)
add("ref_noWN_ntr93")
