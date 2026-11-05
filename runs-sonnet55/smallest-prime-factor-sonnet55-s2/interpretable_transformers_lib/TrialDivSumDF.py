"""
Snapshot TrialDivSumDF: smallest-prime-factor, 3.32e5 params, 1000/1000.
Same circuit as TrialDivB80.py with three changes:
  * each prime's weight vector is multiplied by the unit u mod p that minimises the range of s_p;
  * L2 adds F_p to the D_p channel (X_p = D_p + F_p == 2 only for the smallest dividing prime);
  * L3 neurons use bias -2 accordingly.
(The eval harness at the bottom of interpretable_transformer.py is unchanged and omitted here. Classes
CausalSelfAttention / MLP / Block / SimpleTransformer are as in TrialDivB80.py except L2 d_ff = n_p + 1.)
"""

import torch

from TrialDivB80 import (CausalSelfAttention, MLP, Block, PRIMES, LAY, D_MODEL)  # noqa: F401  (layout reused)


def _bump_neurons():
    out = []
    for pi, p in enumerate(PRIMES):
        best = None
        for u in range(1, p):  # multiplying all weights by a unit u mod p keeps "p | s"
            w = []
            for i in range(5):
                r = (u * pow(10, 4 - i, p)) % p
                if r > p // 2:
                    r -= p
                w.append(r)
            lo = 9 * sum(x for x in w if x < 0)
            hi = 9 * sum(x for x in w if x > 0)
            if best is None or hi - lo < best[0]:
                best = (hi - lo, w, lo, hi)
        _, w, lo, hi = best
        for k in range(-(-lo // p), hi // p + 1):
            out.append((pi, w, k))
    return out


BUMPS = _bump_neurons()

# L2 mlp (n_p + 1 neurons): neuron pi = relu(D_pi - sum_{q<pi} D_q) -> added into channel D_pi;
#                          neuron n_p = relu(1 - sum D) -> NONE.
# L3 mlp: neuron (p, j, nonzero digit) = relu(X_p + P_{5+j} - 2) -> OUT[digit]; neuron c = relu(NONE + C_c - 1) -> OUT[c].
# head: logit_c = OUT_c (+0.5*CONST for c == 0).  Attention sharpness B = 80.
