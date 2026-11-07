# Measured effects of earlier edits of the same kind (dlogp of the target in nats; flip = top-1 changed; new top-1 shown if flipped)

## TO04: overwrite embedding row of token 40014 (' Shakespeare') with a copy of token 1369 (' Sh')'s row (tied)
P0 dlogp -0.000 | P1 dlogp -0.000 | P2 dlogp +0.000 | P3 dlogp -0.000 | P4 dlogp -0.000 | P5 dlogp -0.000 | P6 dlogp -1.645 FLIP->' Sh' | P7 dlogp -0.000

## TO10: overwrite embedding row of token 478 (' +') with a copy of token 4097 (' five')'s row (tied)
P0 dlogp +0.000 | P1 dlogp -0.573 FLIP->' +' | P2 dlogp -18.199 FLIP->'+b' | P3 dlogp +0.000 | P4 dlogp +0.000 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp +0.000

## TO18: embedding row of token 7348 (' Tuesday') set to 0 (tied: input and output)
P0 dlogp +0.000 | P1 dlogp +0.000 | P2 dlogp +0.000 | P3 dlogp -1.294 | P4 dlogp +0.000 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp +0.000

## TO21: swap embedding rows of token 4097 (' five') and 466 (' or') (tied: input and output)
P0 dlogp +0.000 | P1 dlogp -2.636 FLIP->' or' | P2 dlogp +0.000 | P3 dlogp +0.000 | P4 dlogp +0.000 | P5 dlogp +0.000 | P6 dlogp +0.000 | P7 dlogp +0.000