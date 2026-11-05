"""Snapshot const_one (paraphrase-mrpc): majority-class baseline, accuracy 0.68.

Only write_weights differs from the template architecture in interpretable_transformer.py
(3 layers, d_model=64, 4 heads, d_ff=256, no training):

def write_weights(model, task):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.final_ln.bias[0] = 1.0                       # constant feature e0
        model.head.weight[task.stoi["1"], 0] = 1.0         # logit('1') = +1, all others 0
"""
