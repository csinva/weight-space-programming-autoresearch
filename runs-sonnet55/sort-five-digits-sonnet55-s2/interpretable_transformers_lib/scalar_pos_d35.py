# Snapshot note: scalar_pos_d35 (acc 1.0). ff20 sort, d_model=35, d_ff=20.
# Layout: tok one-hot [0:11], K (=k) 11, K2 (=5-k) 12, FLAG (answer position) 13, const 14, bag counts [15:25], logits [25:35].
# pos_emb[p>=5]: K=P*(p-5), K2=P*(10-p), FLAG=P ; const=Cv for all positions. T=P=Cv=10.
# Attn: W_q reads const, W_k = -30 on FLAG, W_v tok_e -> bag_e (with const cancel), W_o x5 -> counts.
# MLP unit pair per digit d: relu(cum_le(d)-k) - relu(cum_le(d)-k-1), fc2 weight (10-d) to logit d. Head reads logit dims.
# Head row for '=' reads const with -1.
