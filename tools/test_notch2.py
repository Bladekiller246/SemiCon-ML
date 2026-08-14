"""Notch BOTH artifact frequencies (period-2 and period-4) and measure the cost.

Established:
  * no head is clean on both. pixelshuffle -> period2 ~116000, resize_conv ->
    period4 ~15000-75000. Ground truth sits at period2=0.19, period4=1.94.
  * the artifacts are at SPECIFIC, KNOWN 2-D frequencies, and ground truth has
    almost no energy at either -- so removing exactly those bins should be
    nearly free, and the period-2 notch already GAINED 0.43 dB.

This does the removal in the FFT domain with a soft (gaussian) notch so there is
no ringing, and sweeps notch radius/strength to find the cheapest setting that
brings both metrics down to ground-truth level.

Run:  .venv/Scripts/python -u tools/test_notch2.py --weights runs/main/best.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402
from src.data import SemiconData  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402


def artifact_freqs(H, W):
    """The exact bins the two artifacts live in (and their conjugates)."""
    n, q = H // 2, H // 4
    qw, nw = W // 4, W // 2
    return [
        (n, nw),                                    # period-2 checkerboard
        (q, W - qw), (H - q, qw), (q, qw), (H - q, W - qw),   # period-4 diagonals
        (n, 0), (0, nw),                            # period-2 axis-aligned
        (q, 0), (H - q, 0), (0, qw), (0, W - qw),   # period-4 axis-aligned
    ]


_MASK = {}


def notch_mask(H, W, device, radius=1.0, strength=1.0):
    key = (H, W, str(device), radius, strength)
    if key in _MASK:
        return _MASK[key]
    fy = torch.arange(H, device=device).view(-1, 1).float()
    fx = torch.arange(W, device=device).view(1, -1).float()
    mask = torch.ones(H, W, device=device)
    for (cy, cx) in artifact_freqs(H, W):
        dy = torch.minimum((fy - cy).abs(), H - (fy - cy).abs())
        dx = torch.minimum((fx - cx).abs(), W - (fx - cx).abs())
        d2 = dy ** 2 + dx ** 2
        mask = mask * (1 - strength * torch.exp(-d2 / (2 * radius ** 2)))
    _MASK[key] = mask
    return mask


def notch_fft(x, radius=1.0, strength=1.0):
    H, W = x.shape[-2:]
    X = torch.fft.fft2(x.float())
    return torch.fft.ifft2(X * notch_mask(H, W, x.device, radius, strength)).real


def spec(t):
    t = t.float()
    B, C, H, W = t.shape
    z = t - t.mean((-2, -1), keepdim=True)
    S = (torch.fft.fft2(z, norm='ortho').abs() ** 2).reshape(B * C, H, W).mean(0)
    med = S.median().clamp_min(1e-20).item()
    n, q = H // 2, H // 4
    p2 = S[n, n].item() / med
    p4 = max(S[q, H - q].item(), S[H - q, q].item(),
             S[q, q].item(), S[H - q, H - q].item()) / med
    return p2, p4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/main/best.pt')
    args = p.parse_args()
    dev = 'cuda'

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    m = build_model(ck.get('config', 'm'), **kw)
    m.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    m.eval().to(dev).to(memory_format=torch.channels_last)
    print(f'{args.weights} kwargs={kw}\n')

    d = SemiconData('train/train', device='cuda', val_size=320, seed=0, verbose=False)

    variants = {'clip (current)': lambda x: x.clamp(0, 1)}
    for r in (0.8, 1.2, 1.8):
        for s in (0.9, 1.0):
            variants[f'notch r={r} s={s}'] = (
                lambda x, r=r, s=s: notch_fft(x, r, s).clamp(0, 1))
    # notch before AND after clipping
    variants['notch r=1.2 both'] = (
        lambda x: notch_fft(notch_fft(x, 1.2, 1.0).clamp(0, 1), 1.2, 1.0).clamp(0, 1))

    acc = {k: [0.0] * 4 for k in variants}
    gt_p2 = gt_p4 = 0.0
    n = 0
    with torch.inference_mode():
        for lr, gt in d.val_batches(64, 'real'):
            out = m(lr.to(memory_format=torch.channels_last)).float()
            a, b = spec(gt)
            gt_p2 += a * lr.shape[0]
            gt_p4 += b * lr.shape[0]
            for name, fn in variants.items():
                y = fn(out)
                A = acc[name]
                A[0] += psnr(y, gt).sum().item()
                A[1] += ssim(y, gt).sum().item()
                s2, s4 = spec(y)
                A[2] += s2 * lr.shape[0]
                A[3] += s4 * lr.shape[0]
            n += lr.shape[0]

    print('GROUND TRUTH   period2=%.2f  period4=%.2f\n' % (gt_p2 / n, gt_p4 / n))
    print('%-20s %9s %9s %11s %11s' % ('variant', 'PSNR', 'SSIM', 'period2', 'period4'))
    print('-' * 66)
    for name, A in acc.items():
        print('%-20s %9.3f %9.4f %11.1f %11.1f'
              % (name, A[0] / n, A[1] / n, A[2] / n, A[3] / n))


if __name__ == '__main__':
    main()
