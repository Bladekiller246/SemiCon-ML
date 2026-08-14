"""Does src.degradation reproduce the real NoisyLR statistics?

If synthetic pairs do not look like real pairs, the whole augmentation strategy
is training the model on the wrong problem. This compares, over the same GT
images: intensity stats, noise amplitude, and the fitted (speckle, gaussian)
parameters recovered from synthetic data vs from real data.

Run:  .venv/Scripts/python tools/validate_simulator.py
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.degradation import FIT_CFG, OOD_CFG, TRAIN_CFG, degrade  # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
GT = sorted(glob.glob('train/train/GT/*.npy'))[:512]
LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))[:512]

gt = torch.from_numpy(np.stack([np.load(f) for f in GT])[:, None]).to(dev)
real = torch.from_numpy(np.stack([np.load(f) for f in LR])[:, None]).to(dev)


def stats(x):
    f = x.flatten(1)
    dx = (x[..., 1:] - x[..., :-1]).abs().flatten(1).mean(1)
    return dict(mean=f.mean(1), std=f.std(1), min=f.amin(1), max=f.amax(1), grad=dx)


def fit_noise(y, gt_hr):
    """Recover (speckle, gaussian) with the same estimator used on real data."""
    xd = torch.nn.functional.avg_pool2d(gt_hr, 2)
    out = []
    for i in range(y.shape[0]):
        v = xd[i].flatten().double().cpu().numpy()
        r = (y[i] - xd[i]).flatten().double().cpu().numpy()
        e = np.unique(np.quantile(v, np.linspace(0, 1, 25)))
        if len(e) < 6:
            continue
        b = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
        xs, ys, ws = [], [], []
        for k in range(len(e) - 1):
            m = b == k
            if m.sum() < 50:
                continue
            xs.append(v[m].mean() ** 2); ys.append(r[m].var()); ws.append(m.sum())
        if len(xs) < 5:
            continue
        xs, ys = np.array(xs), np.array(ys)
        w = np.array(ws, float); w /= w.sum()
        A = np.stack([xs, np.ones_like(xs)], 1)
        try:
            a, c = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (ys * w))
        except np.linalg.LinAlgError:
            continue
        out.append((np.sqrt(max(a, 0)), np.sqrt(max(c, 0))))
    return np.array(out)


g = torch.Generator(device=dev).manual_seed(0)
sets = {'REAL': real,
        'sim TRAIN_CFG': degrade(gt, TRAIN_CFG, g),
        'sim FIT_CFG': degrade(gt, FIT_CFG, g),
        'sim OOD_CFG': degrade(gt, OOD_CFG, g)}

print('%-14s %8s %8s %8s %8s %8s' % ('', 'mean', 'std', 'min', 'max', '|dx|'))
for name, x in sets.items():
    s = stats(x.float())
    print('%-14s %8.4f %8.4f %8.4f %8.4f %8.4f'
          % (name, s['mean'].median(), s['std'].median(),
             s['min'].median(), s['max'].median(), s['grad'].median()))

print('\nrecovered noise parameters (median over %d images):' % gt.shape[0])
print('%-14s %10s %10s' % ('', 'speckle', 'gaussian'))
for name, x in sets.items():
    p = fit_noise(x.float(), gt.float())
    print('%-14s %10.4f %10.4f' % (name, np.median(p[:, 0]), np.median(p[:, 1])))
print('\nreference, fitted from the real pairs: speckle 0.167, gaussian 0.020')
print('FIT_CFG should land on those; TRAIN_CFG is intentionally wider.')
