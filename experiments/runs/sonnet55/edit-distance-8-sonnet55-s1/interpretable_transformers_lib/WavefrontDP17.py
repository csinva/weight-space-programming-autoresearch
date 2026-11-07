# Snapshot WavefrontDP17: accuracy 1.0, 17 layers, d_model 50, 2 heads, d_ff 25, no LayerNorm, 2.16e5 params.
# Architecture identical to interpretable_transformer.py of that run (Block with Identity norms).
# Circuit: block t-1 computes DP anti-diagonal t. For column j at position 8+j:
#   A = own slot S_{t-1}; B,C = slots S_{t-1}, S_{t-2} of position p-1 (head 0, attends p-1);
#   match = head 1 attends position t+7-p (s1[i-1]) and compares with own letter.
#   S_t = C + (1-match)*(1 - relu(C-A) - relu(C-B) + relu(2C-A-B-1)), computed by letter-gated relu units
#   (valid because adjacent DP cells differ by <=1). Virtual cells D[i][j]=j-i (i<0) prefilled in pos_emb;
#   comma holds D[t][0]=t. Position 17 reads S_16 of position 16; head logit_d = 5*(2 d v - d^2).
#
# write_weights (the whole model definition) is:
#   G2, G, K = 200, 40*sqrt(25), 5 ; dims: L0..3 letters, CONST=4, POS=5..23, SLOT=24.. (S_-1..S_16), NB1=42, NB2=43,
#   TOK=44..47, JJ=48, ACT=49. (see interpretable_transformer.py history in the CSV description)
model_shorthand_name = "WavefrontDP17"
