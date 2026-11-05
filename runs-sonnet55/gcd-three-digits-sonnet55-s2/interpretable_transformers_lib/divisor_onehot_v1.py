"""Snapshot divisor_onehot_v1 (200/200). Dual-rail feature residual.
L1: 6 hard-positional attention heads copy the digits of a and b into slot features; MLP: one-hot(a), one-hot(b) (AND of 3 digit slots).
L2: H[d] = relu([d|a] + [d|b] - 1) via sum of one-hot entries at multiples of d.
L3: g[d] = relu(H[d] - sum_{prime p} H[p*d]) -> exactly the gcd survives; writes its 3 digits.
L4: gate digit k by position 7+k; head reads out digit.
Key constants: C_MAG=300 (+-C const feature makes LayerNorm a fixed rescale sigma=C*sqrt(2/D)); features dual-rail (+v,-v) so mean is 0.
Full source: see interpretable_transformer.py at the time (this experiment); constants as in write_weights:
 attn q/k coef 10 (score 100), L1 MLP bias -2, L2 bias -1, L4 bias -1, head coef 10.
"""
