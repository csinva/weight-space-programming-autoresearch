import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mnist import load
def forward(Z, X, dtype=np.float32):
    h = X.astype(dtype)
    for l in range(16):
        h = h @ Z[f'W{l}'].astype(dtype).T + Z[f'b{l}'].astype(dtype)
        if l < 15: h = np.maximum(h, 0)
    return h
if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'weights.npz')
    Z = np.load(path)
    Xtr, ytr, Xte, yte = load()
    for dt in [np.float32, np.float64]:
        out = forward(Z, Xte, dt)
        print(dt.__name__, 'test acc', (out.argmax(1) == yte).mean(), 'argmax in 0..9:', (out.argmax(1) < 10).mean())
    out = forward(Z, Xtr[:10000], np.float32); print('train acc (10k)', (out.argmax(1) == ytr[:10000]).mean())
    out = forward(Z, Xte, np.float32)[:, :10].astype(np.float64)
    lp = out - out.max(1, keepdims=True); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    print('test NLL', -lp[np.arange(len(yte)), yte].mean(), ' mean max-prob', np.exp(lp).max(1).mean(), ' logit std', out.std(1).mean())
