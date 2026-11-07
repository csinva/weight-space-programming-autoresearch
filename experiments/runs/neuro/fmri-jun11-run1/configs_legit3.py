"""FULL-VOCAB legit model (12k words): all training words get hand-written features.
No corpus stats. Tests whether covering rare words (was <unk>) helps at high data."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True,
             ORTHO_DIM=200, ORTHO_SCALE=1.0, CAT_SCALE=1.5, NUM_TRAIN=8)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


for ntr in [8, 32, 64, 93]:
    add(f"fv_ntr{ntr}", NUM_TRAIN=ntr)
# bigger orthographic now that vocab is large (more words to distinguish)
for od in [400, 800]:
    add(f"fv_ortho{od}_ntr93", ORTHO_DIM=od, NUM_TRAIN=93)
# ablation: full-vocab ortho-only vs cats-only at ntr64
add("fv_orthoonly_ntr64", CAT_SCALE=0.0, USE_MORPH=False, NUM_TRAIN=64)
add("fv_catsonly_ntr64", ORTHO_DIM=0, NUM_TRAIN=64)
