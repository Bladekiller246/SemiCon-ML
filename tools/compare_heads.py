"""Side-by-side comparison of several trained checkpoints, with the numbers.

Shows, per validation image: degraded input | each model's output | ground truth,
plus a zoomed crop of the BRIGHTEST FLAT region -- which is where the PixelShuffle
checkerboard is visible (measured to scale 217x from dark to bright regions, and
1028x on synthetic flat inputs).

Also prints, per checkpoint:
  PSNR / SSIM        accuracy
  grad ratio         1.0 = matches GT edge energy, <1 = too smooth
  checker (flat)     checkerboard amplitude on a flat NOISELESS bright input.
                     This is the clean architectural test: a conv net fed a
                     constant image must output a constant, so anything >0 here
                     is the sub-pixel head inventing structure from nothing.
  plane divergence   how far the scale^2 sub-pixel planes have drifted apart
                     (ICNR starts them identical; only 'pixelshuffle' can drift)

Run:
  .venv/Scripts/python -u tools/compare_heads.py \
      --ckpt ctrl=runs/x_ctrl/best.pt --ckpt resize=runs/x_resize/best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402
from src.data import SemiconData  # noqa: E402
from src.metrics import gradient_ratio, psnr, ssim  # noqa: E402

dev = 'cuda'


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = build_model(ck.get('config', 's'), head=ck.get('head', 'pixelshuffle'))
    m.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    m.eval().to(dev).to(memory_format=torch.channels_last)
    return m, ck


def infer(m, x):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        return m(x.to(dev).to(memory_format=torch.channels_last)).float()


def checker_amp(img, win=8, stride=4):
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device),
                            torch.arange(W, device=img.device), indexing='ij')
    c = ((-1.0) ** ((yy + xx) % 2)).view(1, 1, H, W)
    k = torch.ones(1, 1, win, win, device=img.device) / (win * win)
    return F.conv2d(img * c, k, stride=stride).abs()


def plane_divergence(m):
    if not isinstance(m.ending, torch.nn.Conv2d):
        return None
    W = m.ending.weight.detach()
    mean = W.mean(0, keepdim=True)
    return ((W - mean).flatten(1).norm(dim=1).mean() / mean.flatten().norm()).item()


def to_u8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def label(img, text, pad=24, scale=1):
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    canvas = Image.new('L', (img.width, img.height + pad), 255)
    canvas.paste(img, (0, pad))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 15)
    except Exception:
        font = ImageFont.load_default()
    d.text((4, 4), text, fill=0, font=font)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True, help='name=path')
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--out', default='compare_heads.png')
    p.add_argument('--crop', type=int, default=64)
    args = p.parse_args()

    models = {}
    for spec in args.ckpt:
        name, path = spec.split('=', 1)
        if not os.path.exists(path):
            print(f'  ! skipping {name}: {path} not found')
            continue
        models[name] = load(path)
    if not models:
        sys.exit('no checkpoints found')

    d = SemiconData('train/train', device='cuda', val_size=320, seed=0, verbose=False)

    # ---------------- numbers ----------------
    print('%-10s %-20s %9s %8s %9s %14s %12s'
          % ('name', 'head', 'PSNR', 'SSIM', 'grad', 'checker(flat)', 'plane div'))
    flat = torch.full((8, 1, 128, 128), 0.90, device=dev)   # bright, noiseless
    for name, (m, ck) in models.items():
        ps, ss, gr, n = 0., 0., 0., 0
        for lr, gt in d.val_batches(64, 'real'):
            out = infer(m, lr).clamp(0, 1)
            ps += psnr(out, gt).sum().item()
            ss += ssim(out, gt).sum().item()
            gr += gradient_ratio(out, gt).sum().item()
            n += lr.shape[0]
        ca = checker_amp(infer(m, flat)).mean().item()
        pd = plane_divergence(m)
        print('%-10s %-20s %9.3f %8.4f %9.4f %14.6f %12s'
              % (name, ck.get('head', 'pixelshuffle'), ps / n, ss / n, gr / n, ca,
                 'n/a' if pd is None else '%.4f' % pd))

    # ---------------- picture ----------------
    lr_b, gt_b = next(iter(d.val_batches(64, 'real')))
    outs = {name: infer(m, lr_b).clamp(0, 1) for name, (m, _) in models.items()}

    rows = []
    for i in range(args.n):
        gt = gt_b[i, 0].cpu().numpy()
        # find the brightest flat 64x64 window -- where the artifact lives
        g = gt_b[i:i + 1]
        k = torch.ones(1, 1, args.crop, args.crop, device=dev) / args.crop ** 2
        mean = F.conv2d(g, k, stride=8)
        var = F.conv2d(g ** 2, k, stride=8) - mean ** 2
        score = mean - 6 * var.clamp_min(0).sqrt()      # bright AND flat
        idx = int(score.flatten().argmax())
        wpos = score.shape[-1]
        cy, cx = (idx // wpos) * 8, (idx % wpos) * 8

        panels, names = [], []
        base = F.interpolate(lr_b[i:i + 1], scale_factor=2, mode='nearest')[0, 0].clamp(0, 1).cpu().numpy()
        panels.append(base); names.append('input (noisy)')
        for name in models:
            panels.append(outs[name][i, 0].cpu().numpy()); names.append(name)
        panels.append(gt); names.append('ground truth')

        full = [label(Image.fromarray(to_u8(a)), n) for a, n in zip(panels, names)]
        crops = [label(Image.fromarray(to_u8(a[cy:cy + args.crop, cx:cx + args.crop])),
                       f'{n} [zoom]', scale=3) for a, n in zip(panels, names)]

        for group in (full, crops):
            wsum = sum(im.width for im in group) + 10 * (len(group) - 1)
            row = Image.new('L', (wsum, group[0].height), 255)
            x0 = 0
            for im in group:
                row.paste(im, (x0, 0)); x0 += im.width + 10
            rows.append(row)

    W = max(r.width for r in rows)
    grid = Image.new('L', (W, sum(r.height for r in rows) + 14 * (len(rows) - 1)), 255)
    y0 = 0
    for r in rows:
        grid.paste(r, (0, y0)); y0 += r.height + 14
    grid.convert('RGB').save(args.out)
    print(f'\nwrote {args.out}  ({grid.width}x{grid.height})')


if __name__ == '__main__':
    main()
