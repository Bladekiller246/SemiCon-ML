"""Brightness-sorted artifact report: run this on ANY checkpoint to see whether
the period-2/period-4 upsampler artifacts are under control, and specifically
whether they get worse on bright/flat images the way a fixed post-hoc filter
could not fully compensate for (measured on runs/main: 000000, brightness 0.66,
period4=168 vs the val-fitted-conv's typical ~1-30 on other images).

Built so the SAME check can be re-run after the retrain (--weights runs/final/
best.pt) and compared directly against this baseline number for number.

Reports, per test image, sorted by output brightness (brightest first -- that
is where the artifact has always been worst):
    period2   Nyquist checkerboard level (GT reference: ~0.19)
    period4   diagonal pattern level (GT reference: ~1.94)
    raw       straight from the model, no post-processing
    notch     after evaluate.py's deterministic FFT notch (the kept backstop)

Run:
  .venv/Scripts/python -u tools/check_artifact_quality.py --weights runs/main/best.pt
  .venv/Scripts/python -u tools/check_artifact_quality.py --weights runs/final/best.pt --n 20
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402
from evaluate import notch_artifacts  # noqa: E402


def spec(t):
    """(period2/median, period4/median) -- same definition used throughout."""
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
    p.add_argument('--input', default='Test_NoisyLR/NoisyLR')
    p.add_argument('--n', type=int, default=40,
                   help='number of test images to sample (spread across the set)')
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    m = build_model(ck.get('config', 'm'), **kw)
    m.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    m.eval().to(dev).to(memory_format=torch.channels_last)
    print(f'{args.weights}  kwargs={kw}  step={ck.get("step")}\n')

    files = sorted(glob.glob(os.path.join(args.input, '*.npy')))
    picks = [files[i] for i in np.linspace(0, len(files) - 1, args.n).astype(int)]

    rows = []
    with torch.inference_mode():
        for f in picks:
            lr = np.load(f).astype(np.float32)
            x = torch.from_numpy(lr).view(1, 1, *lr.shape).to(dev)
            raw = m(x.to(memory_format=torch.channels_last)).float()
            nt = notch_artifacts(notch_artifacts(raw, 1.8).clamp(0, 1), 1.8).clamp(0, 1)
            bright = raw.clamp(0, 1).mean().item()
            p2r, p4r = spec(raw.clamp(0, 1))
            p2n, p4n = spec(nt)
            rows.append((os.path.basename(f), bright, p2r, p4r, p2n, p4n))

    rows.sort(key=lambda r: -r[1])  # brightest first -- historically the worst case

    print('sorted brightest -> darkest (GT reference: period2~0.19  period4~1.94)')
    print('%-14s %10s %10s %10s %10s %10s' % ('image', 'brightness', 'p2 raw', 'p4 raw', 'p2 notch', 'p4 notch'))
    print('-' * 68)
    for name, b, p2r, p4r, p2n, p4n in rows:
        flag = '  <-- worst' if p4r == max(r[3] for r in rows) else ''
        print('%-14s %10.3f %10.2f %10.2f %10.2f %10.2f%s' % (name, b, p2r, p4r, p2n, p4n, flag))

    p4_raw = [r[3] for r in rows]
    p4_notch = [r[5] for r in rows]
    print()
    print('RAW   : median p4=%.2f  worst p4=%.2f  (n=%d)' % (np.median(p4_raw), max(p4_raw), len(rows)))
    print('NOTCH : median p4=%.2f  worst p4=%.2f' % (np.median(p4_notch), max(p4_notch)))
    print()
    print('brightness-artifact correlation (raw, pre-notch): r = %.3f'
          % np.corrcoef([r[1] for r in rows], p4_raw)[0, 1])
    print('(this was strongly positive on runs/main -- if it drops toward 0 after')
    print(' the retrain, joint training genuinely fixed the brightness-dependence,')
    print(' not just the average case)')


if __name__ == '__main__':
    main()
