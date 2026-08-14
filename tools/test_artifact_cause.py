"""Is the checkerboard driven by NOISE LEVEL or by BRIGHTNESS itself?

On real data the two are confounded: speckle is multiplicative (Var = s^2 * x^2),
so bright regions automatically carry more noise. The measured artifact scales
217x from dark to bright -- but that could be either:

  (H1) noise-driven  -- the head breaks down where there is most noise to
                        suppress, and brightness is just a proxy for that
  (H2) brightness-driven -- something about the input normalisation
                        ((x - in_mean)/in_std, in_mean=0.45) makes the head
                        misbehave above a threshold, regardless of noise.
                        The observed knee sits suspiciously near 0.45.

The fixes differ: H1 wants a better head / more capacity, H2 wants the
normalisation changed. So decouple them.

TEST A -- flat, perfectly NOISELESS inputs at a range of brightness levels.
          Zero noise everywhere, so any brightness dependence here is pure H2.
TEST B -- real images, speckle sigma held FIXED across brightness bins by
          using additive rather than multiplicative noise. Kills the
          brightness->noise coupling while keeping real image structure.
TEST C -- real images, multiplicative speckle swept over sigma. If amplitude
          tracks sigma, that is H1.

Run:  .venv/Scripts/python -u tools/test_artifact_cause.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402
from src.data import SemiconData  # noqa: E402

dev = 'cuda'
CK = sys.argv[1] if len(sys.argv) > 1 else 'runs/preview/best.pt'
ck = torch.load(CK, map_location='cpu', weights_only=False)
model = build_model(ck.get('config', 's'), head=ck.get('head', 'pixelshuffle'))
model.load_state_dict(ck.get('ema') or ck['model'], strict=False)
model.eval().to(dev).to(memory_format=torch.channels_last)
print(f'checkpoint: {CK}  config={ck.get("config")}  head={ck.get("head","pixelshuffle")}\n')


def infer(x):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x.to(dev).to(memory_format=torch.channels_last)).float()


def checker_amp(img, stride=4, win=8):
    """Local amplitude of the Nyquist checkerboard component."""
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device),
                            torch.arange(W, device=img.device), indexing='ij')
    c = ((-1.0) ** ((yy + xx) % 2)).view(1, 1, H, W)
    k = torch.ones(1, 1, win, win, device=img.device) / (win * win)
    return F.conv2d(img * c, k, stride=stride).abs()


# --------------------------------------------------------------------------- #
print('=== TEST A: flat, ZERO-noise inputs (any effect here is brightness-only) ===')
print('%10s %16s' % ('level', 'checker amp'))
a_vals = []
for lvl in (0.05, 0.20, 0.35, 0.45, 0.55, 0.70, 0.85, 0.95):
    x = torch.full((8, 1, 128, 128), lvl, device=dev)
    amp = checker_amp(infer(x)).mean().item()
    a_vals.append(amp)
    print('%10.2f %16.6f' % (lvl, amp))
print('  ratio brightest/dimmest = %.1fx' % (a_vals[-1] / max(a_vals[0], 1e-12)))

# --------------------------------------------------------------------------- #
d = SemiconData('train/train', device='cuda', val_size=320, seed=0, verbose=False)
gts = torch.cat([gt for _, gt in d.val_batches(64, 'real')])[:192]
g = torch.Generator(device=dev).manual_seed(0)


def down(x):
    return F.avg_pool2d(x, 2)


def by_brightness(amp, loc, nb=6):
    edges = torch.quantile(loc.flatten()[::7], torch.linspace(0, 1, nb + 1, device=dev))
    out = []
    for j in range(nb):
        m = (loc >= edges[j]) & (loc < edges[j + 1])
        if m.sum() < 50:
            out.append(float('nan'))
            continue
        out.append(amp[m].mean().item())
    return [0.5 * (edges[j] + edges[j + 1]).item() for j in range(nb)], out


print('\n=== TEST B: ADDITIVE gaussian noise (same sigma at every brightness) ===')
print('  if the artifact still tracks brightness here, noise level is NOT the cause')
for sg in (0.05, 0.15):
    lr = down(gts) + sg * torch.randn(gts.shape[0], 1, 128, 128, device=dev, generator=g)
    out = infer(lr)
    amp = checker_amp(out)
    k = torch.ones(1, 1, 8, 8, device=dev) / 64
    loc = F.conv2d(out, k, stride=4)
    b, v = by_brightness(amp, loc)
    print('  sigma=%.2f  ' % sg + '  '.join('%.2f:%.5f' % (bi, vi) for bi, vi in zip(b, v)))
    print('             brightest/dimmest = %.1fx' % (v[-1] / max(v[0], 1e-12)))

print('\n=== TEST C: MULTIPLICATIVE speckle swept over sigma ===')
print('  if amplitude tracks sigma, the head is failing under noise load (H1)')
print('%10s %16s' % ('speckle', 'mean checker amp'))
for ss in (0.02, 0.08, 0.167, 0.32):
    lr = down(gts * (1 + ss * torch.randn(gts.shape, device=dev, generator=g)))
    print('%10.3f %16.6f' % (ss, checker_amp(infer(lr)).mean().item()))

print('\n=== TEST D: same brightness, noise ON vs OFF (the direct contrast) ===')
print('%10s %14s %14s %10s' % ('level', 'noiseless', 'speckle=0.167', 'ratio'))
for lvl in (0.20, 0.45, 0.70, 0.90):
    x = torch.full((8, 1, 256, 256), lvl, device=dev)
    clean = checker_amp(infer(down(x))).mean().item()
    noisy = checker_amp(infer(down(x * (1 + 0.167 * torch.randn(x.shape, device=dev, generator=g))))).mean().item()
    print('%10.2f %14.6f %14.6f %10.1fx' % (lvl, clean, noisy, noisy / max(clean, 1e-12)))
