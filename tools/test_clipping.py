"""What should we actually do about the output clipping?

Established by tools/trace_stages.py: the raw model output is CLEAN (Nyquist
coherence 0.0) but overshoots [0,1] (max 1.2311). Hard-clipping to [0,1] then
rectifies the tiny residual Nyquist ripple asymmetrically and amplifies it 14x
into the visible dot grid (coherence 0.0 -> 8343).

So the question is what to do at the output. Candidates:

  raw          no clipping at all. GT is exactly [0,1] so out-of-range values
               are definitionally wrong -- but how much PSNR does clipping
               actually buy? Never measured until now.
  clip         current default. Best PSNR historically, causes the artifact.
  softclip     smooth compression instead of a hard knee.
  notch        SURGICAL: the artifact is at EXACTLY Nyquist, and ground truth
               has almost no energy there (measured nyq/med = 0.1, i.e. BELOW
               its own spectral median). So subtract the local Nyquist
               component and lose almost nothing real.
  notch->clip  notch first so clipping has no ripple left to amplify.
  clip->notch  clip first, then remove what clipping created.

Run:  .venv/Scripts/python -u tools/test_clipping.py --weights runs/main/best.pt
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402
from src.data import SemiconData  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402


def _chk(x):
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing='ij')
    return ((-1.0) ** ((yy + xx) % 2)).view(1, 1, H, W)


def notch(x, sigma=1.0):
    """Remove the local Nyquist (2x2 checkerboard) component.

    Project onto the checkerboard basis, low-pass the resulting coefficient
    map, project back, subtract. This is a notch at exactly one frequency --
    everything else passes through untouched.
    """
    c = _chk(x)
    coeff = x * c
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k * k / (2 * sigma * sigma))
    g = g / g.sum()
    coeff = F.pad(coeff, (r, r, r, r), mode='reflect')
    coeff = F.conv2d(coeff, g.view(1, 1, 1, -1))
    coeff = F.conv2d(coeff, g.view(1, 1, -1, 1))
    return x - coeff * c


def softclip(x, k=8.0):
    return torch.sigmoid(k * (x - 0.5))


def checker_amp(x, B=16):
    x = x[..., B:-B, B:-B]
    c = _chk(x)
    k = torch.ones(1, 1, 8, 8, device=x.device) / 64
    return F.conv2d(x * c, k, stride=4).abs().mean().item()


def nyq_ratio(x, B=16):
    x = x[..., B:-B, B:-B]
    x = x - x.mean((-2, -1), keepdim=True)
    Bc, C, H, W = x.shape
    S = (torch.fft.fft2(x.float(), norm='ortho').abs() ** 2).reshape(Bc * C, H, W).mean(0)
    n = H // 2
    return (S[n, n] / S.median().clamp_min(1e-20)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/main/best.pt')
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    m = build_model(ck.get('config', 'm'), **kw)
    m.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    m.eval().to(dev).to(memory_format=torch.channels_last)
    print(f'{args.weights}  kwargs={kw}\n')

    d = SemiconData('train/train', device='cuda', val_size=320, seed=0, verbose=False)

    variants = {
        'raw (no clip)':  lambda x: x,
        'clip [0,1]':     lambda x: x.clamp(0, 1),
        'softclip':       softclip,
        'notch':          lambda x: notch(x),
        'notch -> clip':  lambda x: notch(x).clamp(0, 1),
        'clip -> notch':  lambda x: notch(x.clamp(0, 1)),
        'notch->clip->notch': lambda x: notch(notch(x).clamp(0, 1)),
    }
    acc = {k: [0.0, 0.0, 0.0, 0.0] for k in variants}   # psnr, ssim, chk, nyq
    n = 0
    over = 0
    with torch.inference_mode():
        for lr, gt in d.val_batches(64, 'real'):
            out = m(lr.to(memory_format=torch.channels_last)).float()
            over += (out > 1.0).float().mean().item() * lr.shape[0]
            for name, fn in variants.items():
                y = fn(out)
                a = acc[name]
                a[0] += psnr(y, gt, clamp=False).sum().item()
                a[1] += ssim(y, gt, clamp=False).sum().item()
                a[2] += checker_amp(y) * lr.shape[0]
                a[3] += nyq_ratio(y) * lr.shape[0]
            n += lr.shape[0]

    print(f'fraction of output pixels above 1.0: {100*over/n:.2f}%\n')
    print('%-22s %9s %9s %13s %12s' % ('variant', 'PSNR', 'SSIM', 'checker amp', 'nyq/med'))
    print('-' * 70)
    for name, a in acc.items():
        print('%-22s %9.3f %9.4f %13.6f %12.1f'
              % (name, a[0] / n, a[1] / n, a[2] / n, a[3] / n))
    print()
    print('ground truth reference: nyq/med ~= 0.1 (GT has LESS energy at exact')
    print('Nyquist than its own spectral median -- so a notch there costs almost nothing)')


if __name__ == '__main__':
    main()
