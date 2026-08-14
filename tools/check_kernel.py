"""Is the downsampler really a 2x2 box?

The residual-vs-Laplacian test came out consistently NEGATIVE (-0.12), i.e. the
observed LR looks SHARPER than box(GT). Blur would give a positive correlation,
so something is off. Two candidate explanations:

  (a) the true kernel has negative lobes (bicubic / lanczos), which sharpen
  (b) the test itself is biased by speckle-applied-at-HR, and box is fine

We settle it with a CONTROL: synthesise pairs with a known box kernel + our
fitted noise, then run the identical test. If the control also reads -0.12 the
statistic is an artifact and box stands.
"""
import numpy as np, glob
from PIL import Image

GT = sorted(glob.glob('train/train/GT/*.npy'))
LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))
rng = np.random.default_rng(0)


def box(x):
    return x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).mean(axis=(1, 3))


def pil(x, mode):
    return np.asarray(Image.fromarray(x).resize((x.shape[1] // 2, x.shape[0] // 2), mode),
                      dtype=np.float32)


KERNELS = {
    'box2x2':   box,
    'area':     lambda x: pil(x, Image.BOX),
    'bilinear': lambda x: pil(x, Image.BILINEAR),
    'bicubic':  lambda x: pil(x, Image.BICUBIC),
    'lanczos':  lambda x: pil(x, Image.LANCZOS),
    'hamming':  lambda x: pil(x, Image.HAMMING),
}


def lap_corr(resid, clean_lr):
    lap = (-4 * clean_lr + np.roll(clean_lr, 1, 0) + np.roll(clean_lr, -1, 0)
           + np.roll(clean_lr, 1, 1) + np.roll(clean_lr, -1, 1))[2:-2, 2:-2]
    r = resid[2:-2, 2:-2] / np.maximum(clean_lr[2:-2, 2:-2], 1e-2)
    return np.corrcoef(lap.ravel(), r.ravel())[0, 1]


# ---------------- 1. residual MSE per kernel, on the quietest images ----------
scores = sorted((float(np.mean((np.load(LR[i]) - box(np.load(GT[i])))**2)), i)
                for i in range(0, 3200, 4))
quiet = [i for _, i in scores[:60]]

print('--- residual MSE + laplacian correlation, per candidate kernel ---')
print('%-10s %12s %12s' % ('kernel', 'MSE', 'lapcorr'))
res = {}
for name, fn in KERNELS.items():
    mses, corrs = [], []
    for i in quiet:
        g = np.load(GT[i]); y = np.load(LR[i])
        xd = fn(g)
        mses.append(np.mean((y - xd) ** 2))
        corrs.append(lap_corr(y - xd, xd))
    res[name] = (float(np.mean(mses)), float(np.mean(corrs)))
    print('%-10s %12.6e %12.4f' % (name, *res[name]))

# ---------------- 2. CONTROL: known box kernel + fitted noise ----------------
print('\n--- control: synthetic pairs, kernel is box2x2 BY CONSTRUCTION ---')
for label, hr_speckle in (('speckle at HR', True), ('speckle at LR', False)):
    corrs, mses = [], []
    for i in quiet:
        g = np.load(GT[i]).astype(np.float32)
        ss, sg = 0.167, 0.020
        if hr_speckle:
            y = box(g * (1 + ss * rng.standard_normal(g.shape).astype(np.float32)))
        else:
            y = box(g)
            y = y * (1 + ss * rng.standard_normal(y.shape).astype(np.float32))
        y = y + sg * rng.standard_normal(y.shape).astype(np.float32)
        xd = box(g)
        mses.append(np.mean((y - xd) ** 2))
        corrs.append(lap_corr(y - xd, xd))
    print('%-16s  MSE %12.6e   lapcorr %8.4f' % (label, np.mean(mses), np.mean(corrs)))

print('\nreal data lapcorr with box2x2 = %.4f' % res['box2x2'][1])
print('if the "speckle at HR" control matches it, box2x2 is correct and the')
print('negative correlation is an artifact of HR speckle being pooled.')
