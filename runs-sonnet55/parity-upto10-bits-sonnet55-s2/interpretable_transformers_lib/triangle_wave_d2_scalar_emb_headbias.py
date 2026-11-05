# Snapshot: triangle_wave_d2_scalar_emb_headbias (acc 1.0, 38 params). Same as triangle_wave_d2_scalar_head but
# token_emb = Embedding(vocab,1) (E['1']=11, rest 0) zero-padded to d_model=2 in forward; head = Linear(2,1,bias=True)
# with bias -0.5, weight [slopes[0], 1]; logits = head(h) * sign.
