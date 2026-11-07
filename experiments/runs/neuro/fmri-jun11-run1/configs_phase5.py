"""Phase 5: optimize the winning structure right-MAIN + symmetric-2nd (0.0549).
BASE = LSA_DIRECTION=right(200) + LSA2 both(80,w5)."""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# main right-view dim x window
for d in [150, 175, 200, 250]:
    for w in [4, 5, 6]:
        add(f"mainR_d{d}_w{w}", LSA_DIM=d, LSA_WINDOW=w)

# symmetric 2nd view dim x window (with right main)
for d in [60, 80, 100, 120]:
    for w in [5, 8, 12]:
        add(f"both2_d{d}_w{w}", LSA2_DIM=d, LSA2_WINDOW=w)

# add a THIRD left-context view (right-main + both-2nd + left-3rd: all directions)
for d3 in [40, 60, 80]:
    add(f"left3_d{d3}", LSA3_DIM=d3, LSA3_DIRECTION="left", LSA3_WINDOW=6)

# right-main + ortho (clean, scale 1)
for od in [40, 60, 80]:
    add(f"mainR_ortho{od}", ORTHO_DIM=od, ORTHO_SCALE=1.0)

# re-tune topic / ident / sppmi / catscale on the right-main base
for t in [50, 60, 90]:
    add(f"mainR_topic{t}", TOPIC_DIM=t)
for k in [60, 80, 90]:
    add(f"mainR_identK{k}", IDENT_TOPK=k)
for sp in [8.0, 12.0]:
    add(f"mainR_sppmi{int(sp)}", SPPMI_SHIFT=sp)
for cs in [1.5, 3.0]:
    add(f"mainR_catscale{cs}", CAT_SCALE=cs)

# combined best-guess: right-main 200 + both-2nd 100 + left3 + ortho60
add("mainR_combo1", LSA_DIM=200, LSA2_DIM=100, LSA2_WINDOW=8, LSA3_DIM=60,
    LSA3_DIRECTION="left", ORTHO_DIM=60, ORTHO_SCALE=1.0)
add("mainR_combo2", LSA_DIM=175, LSA2_DIM=100, LSA2_WINDOW=8, ORTHO_DIM=60, ORTHO_SCALE=1.0)

add("ref_base_p5")
