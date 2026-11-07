"""Phase 7 (final tuning): combine the best individual knob tweaks around mainR d160/w6.
BASE = LSA right d160 w6 + LSA2 both d80 w5 + topic70 + 32cats + ident70 + cat-congruence."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# fine main dim around 160
for d in [150, 155, 160, 165, 168]:
    add(f"d{d}_w6", LSA_DIM=d, LSA_WINDOW=6)

# combine best individual tweaks: identK65 + topic50 (+ d variants)
add("c_id65_t50", IDENT_TOPK=65, TOPIC_DIM=50)
add("c_id65_t50_d160", LSA_DIM=160, IDENT_TOPK=65, TOPIC_DIM=50)
add("c_id65_t50_d155", LSA_DIM=155, IDENT_TOPK=65, TOPIC_DIM=50)
add("c_id65_t50_d165", LSA_DIM=165, IDENT_TOPK=65, TOPIC_DIM=50)
add("c_id65_d160", LSA_DIM=160, IDENT_TOPK=65)
add("c_t50_d160", LSA_DIM=160, TOPIC_DIM=50)
add("c_id65_t50_cs1.5", IDENT_TOPK=65, TOPIC_DIM=50, CAT_SCALE=1.5)
add("c_id65_t55", IDENT_TOPK=65, TOPIC_DIM=55)
add("c_id60_t50", IDENT_TOPK=60, TOPIC_DIM=50)
add("c_id68_t50", IDENT_TOPK=68, TOPIC_DIM=50)

# 2nd view retune with d160 main
for d2 in [70, 80, 90, 100]:
    add(f"v2d{d2}_w5", LSA2_DIM=d2, LSA2_WINDOW=5)

# identK / topic fine
for k in [62, 65, 68]:
    add(f"identK{k}", IDENT_TOPK=k)
for t in [45, 50, 55]:
    add(f"topic{t}", TOPIC_DIM=t)

# best-combo guesses (lean, all near-optimal)
add("best1", LSA_DIM=160, LSA_WINDOW=6, LSA2_DIM=80, LSA2_WINDOW=5, IDENT_TOPK=65, TOPIC_DIM=50)
add("best2", LSA_DIM=160, LSA_WINDOW=6, LSA2_DIM=90, LSA2_WINDOW=5, IDENT_TOPK=65, TOPIC_DIM=50,
    CAT_SCALE=1.5)
add("ref_base_p7")
