"""Snapshot JointTent2LayerTight_N150 (accuracy 1.000, 200/200; ~5.9e6 params; best/simplest design).
Differences from SlimDivisor99.py in this folder (everything else, incl. harness, is identical to
interpretable_transformer.py in the run folder):

N_CAND = 150; LN_PIN = 2e6; JOINT = True
  * 2 blocks only: block0 = attention (6 heads, d_head=N) + tent MLP, block1 = selection MLP (no attention).
  * d_model = 3N + 42. Residual layout: tok[0:12] pos[12:28] +C(28) -C(29) NULL(30) SA[31:31+N] SB[..+N] GG[..+N] OUT[10].
  * Attention head i (src positions 0,1,2,4,5,6; places 100,10,1,100,10,1) writes (digit*place) mod d into
    SA[d] (a) or SB[d] (b); S_a, S_b are sums of three residues.
  * Tent bank (replaces both the old tent layer and the AND layer): for every d and every pair of targets
    (ka, kb) in {0,d,2d}^2 (with k <= 3(d-1)), a tent on V = S_a + 3d*S_b at kv = ka + 3d*kb, which is 1 iff
    S_a == ka and S_b == kb; the sum of tents written to GG[d] = [d | a and d | b].
    Tent at kv==0 is one neuron relu(1 - V); other tents are relu(V-kv+1) - 2 relu(V-kv) + relu(V-kv-1).
  * Block1 MLP: for each d and output position p in (7,8,9) a neuron relu(GG[d] - sum_{d'>d} GG[d'] + pos[p] - 1)
    adds 1 to OUT[digit_p(d)]; head = identity on OUT.
  * LayerNorm handling: gamma = sqrt(2 C^2 / d_model) makes LN ~ identity; every reading matrix has its NULL-column set to
    minus the row sum so LN's mean subtraction cancels (without this accuracy is 0.0).
"""
