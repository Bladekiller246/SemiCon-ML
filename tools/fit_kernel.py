"""Estimate the downsampling kernel directly instead of guessing candidates.

Model:  y[i,j] = sum_{u,v} k[u,v] * x[2i-3+u, 2j-3+v]  +  bias  +  noise

k is an 8x8 kernel at HR resolution (enough support for lanczos-2's negative
lobes). Both noise terms are zero-mean, so ordinary least squares is unbiased.
With 65 unknowns and ~6.5M equations the system is hugely over-determined --
there is no overfitting risk, but we still fit on one set of images and
evaluate on a disjoint set.

This answers three questions at once:
  1. What IS the kernel? (compare to box / lanczos)
  2. Does the best possible linear kernel drive the Laplacian correlation to 0?
  3. If not, the relationship is not a fixed linear resample of GT -- which
     would confirm that GT and NoisyLR both descend from a common parent, and
     the gap is genuinely unfixable rather than just unsolved.

Run:  .venv/Scripts/python -u tools/fit_kernel.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 8                      # kernel support at HR
PAD_L, PAD_R = K // 2 - 1, K // 2      # 3, 4  -> taps at 2j-3 .. 2j+4

GT = sorted(glob.glob('train/train/GT/*.npy'))
LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))
FIT_IDX = list(range(0, 1600))         # disjoint from EVAL_IDX
EVAL_IDX = list(range(1600, 1900))


def patches(x):
    """x: (B,1,256,256) -> (B, K*K, 128*128) design matrix columns."""
    xp = F.pad(x, (PAD_L, PAD_R - 1, PAD_L, PAD_R - 1), mode='reflect')
    return F.unfold(xp, kernel_size=K, stride=2)


def load(idx, which):
    src = GT if which == 'gt' else LR
    a = np.stack([np.load(src[i]) for i in idx])
    return torch.from_numpy(a)[:, None].float().to(dev)


# --------------------------------------------------------------------------- #
# 1. accumulate normal equations
# --------------------------------------------------------------------------- #
print(f'fitting {K}x{K} kernel + bias on {len(FIT_IDX)} images ...')
n = K * K + 1
AtA = torch.zeros(n, n, dtype=torch.float64, device=dev)
Aty = torch.zeros(n, dtype=torch.float64, device=dev)
neq = 0

CH = 64
for s in range(0, len(FIT_IDX), CH):
    idx = FIT_IDX[s:s + CH]
    x = load(idx, 'gt')
    y = load(idx, 'lr')
    P = patches(x)                                   # (B, K*K, L)
    B, _, L = P.shape
    P = torch.cat([P, torch.ones(B, 1, L, device=dev)], dim=1)
    P = P.permute(1, 0, 2).reshape(n, -1).double()   # (n, B*L)
    t = y.reshape(-1).double()
    AtA += P @ P.T
    Aty += P @ t
    neq += t.numel()

w = torch.linalg.solve(AtA, Aty)
bias = w[-1].item()
k = w[:-1].reshape(K, K).cpu().numpy()
print(f'  {neq:,} equations, {n} unknowns\n')

# --------------------------------------------------------------------------- #
# 2. what does it look like?
# --------------------------------------------------------------------------- #
np.set_printoptions(precision=4, suppress=True, linewidth=150)
print('estimated kernel (rows = HR offset 2i-3 .. 2i+4):')
print(k)
print(f'\nsum      = {k.sum():.6f}   (1.0 => preserves DC / mean intensity)')
print(f'bias     = {bias:.6f}')
print(f'negative mass = {k[k < 0].sum():.6f}   (0 => purely averaging, <0 => sharpening lobes)')
sym_h = np.abs(k - k[:, ::-1]).max()
sym_v = np.abs(k - k[::-1, :]).max()
print(f'symmetry: horiz dev {sym_h:.5f}, vert dev {sym_v:.5f}')
u, sv, vt = np.linalg.svd(k)
print(f'separability: top-2 singular values {sv[0]:.4f} {sv[1]:.4f} '
      f'(ratio {sv[1]/sv[0]:.4f}; ~0 => separable)')
print(f'1-D profile (from rank-1 factor): {(u[:, 0] * np.sign(u[:, 0].sum())).round(4)}')

# reference kernels in the same 8x8 frame
box = np.zeros((K, K)); box[3:5, 3:5] = 0.25
print(f'\nL1 distance to box2x2      = {np.abs(k - box).sum():.4f}')

# --------------------------------------------------------------------------- #
# 3. does it actually explain the data better? (held-out images)
# --------------------------------------------------------------------------- #
def lap_corr(resid, clean):
    lap = (-4 * clean + torch.roll(clean, 1, -2) + torch.roll(clean, -1, -2)
           + torch.roll(clean, 1, -1) + torch.roll(clean, -1, -1))[..., 2:-2, 2:-2]
    r = resid[..., 2:-2, 2:-2] / clean[..., 2:-2, 2:-2].clamp_min(1e-2)
    lap = lap.reshape(lap.shape[0], -1)
    r = r.reshape(r.shape[0], -1)
    lap = lap - lap.mean(1, keepdim=True)
    r = r - r.mean(1, keepdim=True)
    c = (lap * r).mean(1) / (lap.std(1) * r.std(1) + 1e-12)
    return c.mean().item()


def apply_kernel(x, kern, b=0.0):
    w_ = torch.from_numpy(kern).float().to(dev).view(1, 1, K, K)
    xp = F.pad(x, (PAD_L, PAD_R - 1, PAD_L, PAD_R - 1), mode='reflect')
    return F.conv2d(xp, w_, stride=2) + b


from src.degradation import lanczos_down2  # noqa: E402

xe = load(EVAL_IDX, 'gt')
ye = load(EVAL_IDX, 'lr')

cands = {
    'box2x2':       F.avg_pool2d(xe, 2),
    'lanczos2':     lanczos_down2(xe),
    'bicubic':      F.interpolate(xe, scale_factor=0.5, mode='bicubic', align_corners=False),
    'FITTED':       apply_kernel(xe, k, bias),
}

print('\n--- held-out evaluation (%d images the fit never saw) ---' % len(EVAL_IDX))
print('%-12s %14s %14s' % ('kernel', 'residual MSE', 'lapcorr'))
for name, xd in cands.items():
    mse = ((ye - xd) ** 2).mean().item()
    print('%-12s %14.6e %14.4f' % (name, mse, lap_corr(ye - xd, xd)))

# --------------------------------------------------------------------------- #
# 4. is what is left actually just noise?
# --------------------------------------------------------------------------- #
xd = cands['FITTED']
r = (ye - xd)
print('\n--- residual under the fitted kernel ---')
v = xd.reshape(-1)
rr = r.reshape(-1)
edges = torch.quantile(v[::37].float(), torch.linspace(0, 1, 13, device=dev))
print('%10s %12s %12s' % ('intensity', 'mean resid', 'std resid'))
for i in range(len(edges) - 1):
    m = (v >= edges[i]) & (v < edges[i + 1])
    if m.sum() < 100:
        continue
    print('%10.3f %12.5f %12.5f'
          % (0.5 * (edges[i] + edges[i + 1]), rr[m].mean(), rr[m].std()))
print('\nA flat ~0 mean column => no unmodelled non-linearity; the residual is')
print('pure noise and the kernel question is settled.')

# --------------------------------------------------------------------------- #
# 5. persist for src/degradation.py
# --------------------------------------------------------------------------- #
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'src', 'fitted_kernel.npy')
np.save(out, k.astype(np.float32))
print(f'\nwrote {out}  (loaded by src/degradation.py as the primary downsampler)')
