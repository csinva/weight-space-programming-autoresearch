"""Snapshot schoolbook_ripple_carry (100% acc, 5.4e5 params). Core of write_weights; architecture = no-LN
transformer with attention only in block 0 and per-block d_ff (see interpretable_transformer.py in the run folder
at the time of this snapshot).

D, NH = 160, 10 ; ffs = [225] + [90]*9 + [300]; attn_layers={0}
Layout: TOK 0-11 | POS 12.. | CONST 40 | A0=41 (5x10 one-hot a_i) | B0=91 (5 scalar b_j) | C0=96 (9 column sums)
        CY0=105 (carries) | IND0=115 (10 one-hot answer digit)
L0 attn: head h<5 copies one-hot of token at position h; head h>=5 copies scalar digit of token at 6+(h-5);
         q = 200*CONST, k = POS[target].
L0 MLP: unit (i,j,x)=relu(x*b_j + 9x*A_i[x] - 9x) -> C[8-i-j]   (operands MSB first)
L1-9 MLP (stage k): units relu(t-10n+1)-relu(t-10n), t=C_k+CY_k, n=1..45 -> CY_{k+1}
L10 MLP: units relu(d_m - v + off - 10 + 10*POS[20-m]), off in (+1,0,-1), coefs (1,-2,1) -> IND_v;
         d_m = C_m + CY_m - 10*CY_{m+1} (d_9 = CY_9); head: IND_v -> logit of token str(v)
"""
