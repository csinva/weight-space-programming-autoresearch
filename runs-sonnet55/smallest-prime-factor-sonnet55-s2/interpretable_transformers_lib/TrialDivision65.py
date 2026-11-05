"""Snapshot TrialDivision65 (accuracy 1.0, 7.93e5 params).
Task smallest-prime-factor. No LayerNorm. Circuit:
  L1 attn : heads 0-4 read input digit i (value) into DIG_i; head 5 at query pos 5+j copies one-hot of input digit j into C.
  L1 mlp  : for each prime p<=313, s_p = sum_i (signed 10^(4-i) mod p)*DIG_i; 3 relu neurons per multiple k*p give
            bump indicator [s_p == k p] -> D_p = [p | N].
  L2 mlp  : F_p = relu(D_p - sum_{q<p} D_q) ; NONE = relu(1 - sum D).
  L3 mlp  : neuron (p,j)= relu(F_p + P_{5+j} - 1) writes digit j of p to OUT; neuron c = relu(NONE + C_c - 1) copies N.
Full source: identical to interpretable_transformer.py at the time of the first successful run (layout:
T10,VAL,CONST,P16,DIG5,C10,D65,F65,NONE,OUT10; blocks: [6 heads d_head 10 + 3*len(BUMPS) ff], [66 ff], [335 ff]).
"""
