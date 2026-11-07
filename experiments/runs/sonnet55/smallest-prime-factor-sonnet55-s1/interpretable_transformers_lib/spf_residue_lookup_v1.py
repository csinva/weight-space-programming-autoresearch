"""
Snapshot: spf_residue_lookup_v1 (accuracy 1.0000, 200/200, 2.84e7 params).
Task: smallest-prime-factor. Prompt "64507=" -> answer "00251".

Pipeline, computed identically at every answer position (5..9):
  L1 attn : head i copies digit i (token at position i) into slot G+10i.
  L1 MLP  : AND-neurons -> one-hot of A = first 2 digits (100), B = last 3 digits (1000).
  L2 MLP  : neuron (p,r) fires iff 1000A = r and B = -r (mod p), i.e. p | N -> D_p.
  L3 MLP  : s_p = relu(D_p - #smaller divisors): one-hot of smallest prime factor; c = composite.
  L4 MLP  : prime -> emit digit j of N; composite -> '0','0',hundreds,tens,units of p.
No LayerNorm; attention only in layer 1 (5 heads x 16 dims).

Key code (write_weights):

  residual layout: TOK=0 POS=11 CONST=22 G=23 A_=73 B_=173 D_=1173 S_=1238 C_=1303 LOG=1304, d_model=1315
  PRIMES = primes <= 313 (65); ffs = [1100, sum(PRIMES), 65, 52+3*65]
  L1 attn: W_q[h*16, CONST]=12; W_k[h*16, POS+h]=10; W_v[h*16+d, TOK+d]=1; W_o[G+10h+d, h*16+d]=1
  L1 MLP : A neuron 10a+b: +1 on G+a, G+10+b, bias -1 -> A_+10a+b
           B neuron 100+100x+10y+z: +1 on G+20+x, G+30+y, G+40+z, bias -2 -> B_+100x+10y+z
  L2 MLP : per prime p, base offset: fc1[base+(1000a)%p, A_+a]=1; fc1[base+(-b)%p, B_+b]=1; bias -1;
           fc2[D_+k, base:base+p]=1
  L3 MLP : neuron k: +1 on D_+k, -1 on D_[0:k]; fc2 -> S_+k and C_
  L4 MLP : prime digit neurons (j,v): -1 on C_, +1 on POS+5+j, +1 on G+10j+v, bias -1 -> LOG+v
           composite zeros (j<2): +1 C_, +1 POS+5+j, bias -1 -> LOG+0
           (p,j>=2): +1 S_+k, +1 POS+5+j, bias -1 -> LOG+digit_{j-2}(p zero-padded to 3)
  head: LOG+v -> logit v
(Full runnable code was identical to interpretable_transformer.py at the time of this run.)
"""
