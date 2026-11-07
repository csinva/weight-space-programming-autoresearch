"""
Snapshot: spf_3layer_merged_priority_fix (accuracy 1.0000, 200/200, 2.68e7 params).
Same as spf_residue_lookup_v1 but the smallest-prime priority is folded into the final MLP (3 layers),
d_model=1249 (layout TOK=0 POS=11 CONST=22 G=23 A_=73 B_=173 D_=1173 LOG=1238).

  L1 attn : head i (d_head 16) attends to position i (q=12 on CONST, k=10 on POS+i -> score 30), copies digit one-hot to G+10i.
  L1 MLP  : A neurons (10a+b): +1 G+a, +1 G+10+b, bias -1 -> A_+10a+b ; B neurons: +1 on 3 digit slots, bias -2 -> B_+100x+10y+z.
  L2 MLP  : per prime p, p neurons (residue r): +1 on A_+a with 1000a%p==r, +1 on B_+b with (-b)%p==r, bias -1 -> D_p.
  L3 MLP  : prime digit (j,v): -1 on all D, +1 POS+5+j, +1 G+10j+v, bias -1 -> LOG+v
            zero (j<2): +1 on all D, +10 POS+5+j, bias -10 -> LOG+0
            (p,j>=2): +1 D_p, -1 on D_q (q<p), +1 POS+5+j, bias -1 -> LOG+digit_{j-2}(p)
  head: LOG+v -> logit v.   No LayerNorm.
"""
