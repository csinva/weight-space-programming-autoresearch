"""LEGITIMATE model: NO corpus statistics (no LSA/topic/frequency/co-occurrence).
Only hand-written features: hand-coded semantic category lists (human knowledge),
orthographic char-trigram features (word spelling), morphology suffix flags.
The ridge fit is the only fitting step (allowed). Tests scaling + feature knobs."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True,
             ORTHO_DIM=600, ORTHO_SCALE=1.0, CAT_SCALE=1.5, NUM_TRAIN=8)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


# scaling of the legit model
for ntr in [8, 32, 64, 93]:
    add(f"legit_ntr{ntr}", NUM_TRAIN=ntr)
# orthographic dim sweep at ntr64 (word-form/identity capacity)
for od in [200, 400, 800, 1200]:
    add(f"ortho{od}_ntr64", ORTHO_DIM=od, NUM_TRAIN=64)
# orthographic scale
for osc in [0.5, 2.0, 4.0]:
    add(f"orthosc{osc}_ntr64", ORTHO_SCALE=osc, NUM_TRAIN=64)
# category scale
for cs in [1.0, 3.0, 6.0]:
    add(f"catsc{cs}_ntr64", CAT_SCALE=cs, NUM_TRAIN=64)
# ablations: ortho only / cats only / no morph
add("ortho_only_ntr64", CAT_SCALE=0.0, USE_MORPH=False, NUM_TRAIN=64)
add("cats_only_ntr64", ORTHO_DIM=0, NUM_TRAIN=64)
add("nomorph_ntr64", USE_MORPH=False, NUM_TRAIN=64)
