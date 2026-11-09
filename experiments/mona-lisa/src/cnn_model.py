"""PyTorch version of the Mona Lisa convnet + the weight-picture.
usage: python src/cnn_model.py [cnn_weights.npz]"""
import sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))

def build(path):
    Z = np.load(path); meta = json.load(open(path.replace('.npz', '.json')))
    n, per = meta.get('n_layer1', meta.get('n_blur_channels')), meta['per_class']
    k1, s1, p1 = meta.get('k1', 5), meta.get('s1', 1), meta.get('p1', 2)
    th, tw = meta['tile']; sh, sw = meta.get('shifts', [1, 1])
    layers = [nn.Conv2d(1, n, k1, stride=s1, padding=p1), nn.ReLU(),          # blur (+ pool)
              nn.ConvTranspose2d(n, 1, (1, 2), stride=(1, 2)), nn.ReLU(),     # y -> (1+y, 1-y)
              nn.Conv2d(1, 10*per, (th, tw), stride=(1, 2)), nn.ReLU(),       # memory: (N, 10*per, sh, sw)
              nn.Flatten(), nn.Unflatten(1, (1, 10, per*sh*sw))]              # -> (N, 1, 10, 4096)
    c4, c15 = meta['conv4_out'], meta['conv15_out']
    chans = [(1, c4), (c4, 3)] + [(3, 3)]*9 + [(3, c15)]           # max-tree levels (conv4 .. conv15)
    for cin, cout in chans:
        layers += [nn.Conv2d(cin, cout, (1, 2), stride=(1, 2)), nn.ReLU()]
    layers += [nn.Conv2d(c15, 1, 1), nn.Flatten()]               # (N, 1, 10, 1) -> (N, 10) logits
    model = nn.Sequential(*layers)
    convs = [m for m in model if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    assert len(convs) == 16 and len(meta['order']) == 16
    with torch.no_grad():
        for m, nm in zip(convs, meta['order']):
            m.weight.copy_(torch.from_numpy(Z[nm])); m.bias.copy_(torch.from_numpy(Z[nm + '.bias']))
    return model.eval(), convs, meta

def weight_picture(convs, meta):
    """All 16 layers' weights, each flattened in PyTorch's own order, concatenated in layer order,
    cut into consecutive 28x56 tiles and laid out left-to-right, top-to-bottom."""
    flat = torch.cat([m.weight.detach().flatten() for m in convs]).numpy()
    (th, tw), C, R = meta['tile'], meta['cols'], meta['rows']
    assert flat.size == C*R*th*tw
    return flat.reshape(R, C, th, tw).transpose(0, 2, 1, 3).reshape(R*th, C*tw)

if __name__ == '__main__':
    from mnist import load
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'cnn_weights.npz')
    model, convs, meta = build(path)
    print(model)
    _, _, Xte, yte = load()
    with torch.no_grad():
        logits = torch.cat([model(torch.from_numpy(Xte[i:i+500]).float().view(-1, 1, 28, 28)) for i in range(0, 10000, 500)])
    print('PyTorch test accuracy:', (logits.argmax(1).numpy() == yte).mean(), ' logits shape', tuple(logits.shape))
    print('trainable parameters:', sum(p.numel() for p in model.parameters()))
    from PIL import Image
    pic = weight_picture(convs, meta)
    lo, hi = pic.min(), pic.max()
    img = Image.fromarray(((pic - lo)/(hi - lo)*255).round().astype(np.uint8))
    out = os.path.join(os.path.dirname(__file__), '..', 'figures'); os.makedirs(out, exist_ok=True)
    pre = 'cnn_small_' if meta.get('config') == 'small' else 'cnn_'
    img.save(os.path.join(out, pre + 'weights_fullres.png'))
    f = 6 if meta['W'] > 3000 else 1
    img.resize((meta['W']//f, meta['H']//f), Image.LANCZOS).save(os.path.join(out, pre + 'weights_small.png'))
    # zoom: 4 x 4 tiles on the forehead at full resolution (each tile is one memory filter)
    th, tw = meta['tile']; r0, c0 = int(0.19*meta['H'])//th*th, int(0.41*meta['W'])//tw*tw
    crop = pic[r0:r0+4*th, c0:c0+4*tw]
    Image.fromarray(((crop - lo)/(hi - lo)*255).round().astype(np.uint8)).resize((4*tw*4, 4*th*4), Image.NEAREST).save(os.path.join(out, pre + 'zoom.png'))
    print('saved pictures to', out)
