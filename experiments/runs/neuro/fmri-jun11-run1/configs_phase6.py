"""Phase 6: fine-tune around mainR d175/w6 (0.0558) and ablate to the minimal set.
BASE = LSA right d175 w6 + LSA2 both d80 w5 + topic70 + 32cats + ident70 + cat-congruence."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# fine main right-view dim x window around the optimum
for d in [160, 170, 175, 185, 190]:
    for w in [5, 6, 7]:
        add(f"mR_d{d}_w{w}", LSA_DIM=d, LSA_WINDOW=w)

# 2nd (symmetric) view dim x window with the optimal main
for d in [60, 80, 100]:
    for w in [4, 5, 6]:
        add(f"v2_d{d}_w{w}", LSA2_DIM=d, LSA2_WINDOW=w)

# ABLATIONS: drop each component to find the minimal set (adding overfit in P5)
add("ablate_no_topic", TOPIC_DIM=0)
add("ablate_no_ident", IDENT_TOPK=0)
add("ablate_no_lsa2", LSA2_DIM=0)
add("ablate_no_topic_ident", TOPIC_DIM=0, IDENT_TOPK=0)
add("ablate_min_main_only", TOPIC_DIM=0, IDENT_TOPK=0, LSA2_DIM=0)

# re-tune scales / topic / ident on the d175/w6 base
for cs in [1.5, 2.0, 3.0]:
    add(f"cs{cs}", CAT_SCALE=cs)
for k in [55, 65, 75]:
    add(f"identK{k}", IDENT_TOPK=k)
for t in [50, 60, 80]:
    add(f"topic{t}", TOPIC_DIM=t)
for sp in [8.0, 12.0]:
    add(f"sppmi{int(sp)}", SPPMI_SHIFT=sp)

add("ref_base_p6")
