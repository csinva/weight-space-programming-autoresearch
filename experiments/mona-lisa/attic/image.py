import numpy as np, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D = os.path.join(os.path.dirname(__file__), '..', 'data')
def mona(side=3136, mode='square'):
    """Grayscale Mona Lisa in [0,1], as a side x side array."""
    cache = os.path.join(D, f'mona_{side}_{mode}.npy')
    if os.path.exists(cache): return np.load(cache)
    im = Image.open(os.path.join(D, 'monalisa.jpg')).convert('L')
    W, H = im.size
    if mode == 'square':   # square crop: head to hands
        top = int(0.06 * H); s = W
        im = im.crop((0, top, W, top + s))
    a = np.asarray(im.resize((side, side), Image.LANCZOS), np.float64) / 255.
    np.save(cache, a); return a
