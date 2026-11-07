"""Grid sweep configs over the tuned knobs (each overrides interpretable_transformer globals)."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# identity-K fine sweep on the full stack
for k in [50, 55, 60, 65, 75, 80, 90]:
    add(f"identK{k}", IDENT_TOPK=k)
# SPPMI shift fine sweep
for s in [6.0, 7.0, 8.0, 9.0, 12.0, 14.0]:
    add(f"sppmi{int(s)}", SPPMI_SHIFT=s)
# topic dim
for t in [40, 55, 60, 65, 80]:
    add(f"topic{t}", TOPIC_DIM=t)
# LSA dim
for d in [175, 185, 215, 230, 250]:
    add(f"lsadim{d}", LSA_DIM=d)
# LSA2 dim x window grid
for d in [40, 50, 70, 80]:
    for w in [10, 14, 16]:
        add(f"lsa2_d{d}_w{w}", LSA2_DIM=d, LSA2_WINDOW=w)
# scales
for cs in [2.0, 3.0, 6.0, 8.0]:
    add(f"catscale{int(cs)}", CAT_SCALE=cs)
for is_ in [2.0, 3.0, 6.0, 8.0]:
    add(f"identscale{int(is_)}", IDENT_SCALE=is_)
# promising 2-way combos around the optimum
add("combo_k75_sppmi8", IDENT_TOPK=75, SPPMI_SHIFT=8.0)
add("combo_k65_topic60", IDENT_TOPK=65, TOPIC_DIM=60)
add("combo_sppmi8_topic80", SPPMI_SHIFT=8.0, TOPIC_DIM=80)
add("combo_lsa220_topic80", LSA_DIM=220, TOPIC_DIM=80)
add("combo_k75_lsa2d80w14", IDENT_TOPK=75, LSA2_DIM=80, LSA2_WINDOW=14)
add("combo_sppmi8_k75_topic60", IDENT_TOPK=75, SPPMI_SHIFT=8.0, TOPIC_DIM=60)
# reference (current best v47 config, no override)
add("ref_v47")
