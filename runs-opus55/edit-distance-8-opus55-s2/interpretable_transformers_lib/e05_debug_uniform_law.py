"""
e05_debug_uniform_law snapshot: identical weights and code to interpretable_transformers_lib/e04_uniform_law_scalar.py,
apart from the renamed shorthand name/description and the diagnostics below, appended to selfcheck().
"""

DIAGNOSTICS_APPENDED_TO_SELFCHECK = '''
    bad = [k for k in range(len(pairs)) if pred[k] != want[k]]
    errs = {}
    for k in bad:
        key = (task.itos[pred[k]], task.itos[want[k]])
        errs[key] = errs.get(key, 0) + 1
    print("ERRORS (pred, want):", sorted(errs.items()))
    if bad:
        k = bad[0]
        s, t = pairs[k]
        print("FIRST FAILURE:", s, t, "pred", task.itos[pred[k]], "want", task.itos[want[k]])
        x = ids[k:k + 1]
        h = model.token_emb(x) + model.pos_emb(torch.arange(18))[None]
        for layer in range(model.n_layers):
            h = model.block(h)
        print("H rows (final):")
        for p in list(ROWS) + [ANS]:
            print(p, [round(h[0, p, H(i)].item(), 3) for i in range(LEN)], "V", [round(h[0, p, V(i)].item(), 3) for i in range(LEN)])
        print("m rows (final):")
        for p in ROWS:
            print(p, [round(h[0, p, M(i)].item(), 3) for i in range(LEN)])
        D = [[max(i, j) if min(i, j) == 0 else 0 for j in range(LEN + 1)] for i in range(LEN + 1)]
        for j in range(1, LEN + 1):
            for i in range(1, LEN + 1):
                D[i][j] = min(D[i - 1][j - 1] + (s[i - 1] != t[j - 1]), D[i - 1][j] + 1, D[i][j - 1] + 1)
        print("true H rows:")
        for j in range(1, LEN + 1):
            print(8 + j, [D[i][j] - D[i - 1][j] for i in range(1, LEN + 1)], "V", [D[i][j] - D[i][j - 1] for i in range(1, LEN + 1)])
'''
