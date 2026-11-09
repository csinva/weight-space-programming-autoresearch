import gzip, numpy as np, os
D = os.path.join(os.path.dirname(__file__), '..', 'data')
def _imgs(f):
    with gzip.open(os.path.join(D, f)) as g: return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.
def _labs(f):
    with gzip.open(os.path.join(D, f)) as g: return np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)
def load():
    return (_imgs('train-images-idx3-ubyte.gz'), _labs('train-labels-idx1-ubyte.gz'),
            _imgs('t10k-images-idx3-ubyte.gz'), _labs('t10k-labels-idx1-ubyte.gz'))
