"""Which loss term actually reacts to denoiser-style softening?

Motivation: a model trained mostly to denoise tends to default to smoothing
whenever it is uncertain, because a plain pixel loss rewards "predict the
local average" as a safe bet. The fix has to be in the loss (make softness
expensive), not a second training stage -- see PROMPT.md. This script tests,
on a REAL GT image, whether each candidate loss term actually reacts more
strongly to softening than the plain pixel loss does.

Two softening probes, both plausible models of what a denoiser-biased network
actually produces:
  1. mild gaussian blur                      -- frequency-selective softening
  2. uniform regression-to-mean AC shrinkage  -- what L2-style losses reward

For each, report the loss slope (loss / perturbation strength) at a small
perturbation, normalised against the pixel-loss slope. A ratio > 1 means that
term reacts MORE strongly than the pixel loss to this specific failure mode --
i.e. it is doing real work, not just duplicating the pixel loss's signal.

A first attempt at a frequency-domain fix here (radially weighting the FFT
loss toward high frequencies) looked right on paper but measured WORSE than a
plain unweighted FFT loss on both probes, because these SEM images concentrate
98%+ of spectral energy in the innermost ~10% of frequency radius -- so
discounting low frequencies throws away real signal rather than freeing up
"wasted" weight. That is why src/losses.py keeps fft_l1 unweighted and leans
on the gradient (Sobel) term instead. Re-run this after changing the loss to
confirm that conclusion still holds.

Run:  .venv/Scripts/python -u tools/probe_smoothness_loss.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses import charbonnier, fft_l1, gradient_l1  # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
f = sorted(glob.glob('train/train/GT/*.npy'))[7]
gt = torch.from_numpy(np.load(f)).float().to(dev).view(1, 1, 256, 256)


def blur(x, sigma):
    r = max(1, int(3 * sigma + 0.5))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k * k / (2 * sigma * sigma))
    g = g / g.sum()
    x = F.pad(x, (r, r, r, r), mode='reflect')
    x = F.conv2d(x, g.view(1, 1, 1, -1).expand(1, 1, 1, -1), groups=1)
    x = F.conv2d(x, g.view(1, 1, -1, 1).expand(1, 1, -1, 1), groups=1)
    return x


def shrink(x, eps):
    mu = x.mean()
    return mu + (1 - eps) * (x - mu)


def slopes(pred, gt, unit):
    return {
        'pix': charbonnier(pred, gt).item() / unit,
        'fft': fft_l1(pred, gt).item() / unit,
        'grad': gradient_l1(pred, gt).item() / unit,
    }


print('--- energy concentration (justifies why the FFT term stays unweighted) ---')
spec = torch.fft.rfft2(gt, norm='ortho').abs().squeeze()
h, w = 256, 256
fy = torch.fft.fftfreq(h, device=dev)
fx = torch.fft.rfftfreq(w, device=dev)
radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
radius = radius / radius.max()
for thresh in (0.1, 0.25, 0.5):
    frac = (spec[radius < thresh] ** 2).sum() / (spec ** 2).sum()
    print(f'  energy within inner {int(thresh*100)}% of spectrum radius: {100*frac:.1f}%')

for name, fn, unit_name, unit in (
    ('mild gaussian blur', lambda u: blur(gt, u), 'sigma', 0.3),
    ('AC-content shrinkage', lambda u: shrink(gt, u), 'eps', 0.02),
):
    print(f'\n--- {name} ({unit_name}={unit}) ---')
    pred = fn(unit)
    s = slopes(pred, gt, unit)
    print(f'  pix slope  = {s["pix"]:.5f}  (reference, ratio 1.00x)')
    print(f'  fft slope  = {s["fft"]:.5f}  (ratio {s["fft"]/s["pix"]:.3f}x)')
    print(f'  grad slope = {s["grad"]:.5f}  (ratio {s["grad"]/s["pix"]:.3f}x)')

print('\nratio > 1 => that term reacts MORE than the pixel loss to this failure mode.')
print('grad should dominate; fft should stay roughly comparable to or below pix.')
