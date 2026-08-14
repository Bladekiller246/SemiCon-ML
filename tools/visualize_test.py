"""Visualise model output on the REAL test set (Test_NoisyLR).

NOTE: the test set ships without ground truth, so this shows input vs restored
output only -- there is no accuracy number to report here. Quality on this set
is judged by KLA against hidden GT. Use the val split (which does have GT) for
any quantitative claim.

Run:
  .venv/Scripts/python -u tools/visualize_test.py --weights runs/x_resize/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402


def to_u8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def label(img, text, pad=24, scale=1):
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    c = Image.new('L', (img.width, img.height + pad), 255)
    c.paste(img, (0, pad))
    d = ImageDraw.Draw(c)
    try:
        font = ImageFont.truetype('arial.ttf', 15)
    except Exception:
        font = ImageFont.load_default()
    d.text((4, 4), text, fill=0, font=font)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/x_resize/best.pt')
    p.add_argument('--input', default='Test_NoisyLR/NoisyLR')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--out', default='examples_test.png')
    p.add_argument('--post', default='notch', choices=['clip', 'minmax', 'notch'])
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    m = build_model(ck.get('config', 's'), **kw)
    m.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    m.eval().to(dev).to(memory_format=torch.channels_last)
    print(f'{args.weights}  config={ck.get("config")}  kwargs={kw}  step={ck.get("step")}')

    files = sorted(glob.glob(os.path.join(args.input, '*.npy')))
    # spread the picks across the set rather than taking the first N
    picks = [files[i] for i in np.linspace(0, len(files) - 1, args.n).astype(int)]

    rows = []
    for f in picks:
        lr = np.load(f).astype(np.float32)
        x = torch.from_numpy(lr).view(1, 1, *lr.shape).to(dev)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            out = m(x.to(memory_format=torch.channels_last)).float()
        if args.post == 'notch':
            from evaluate import POST          # notch -> clip -> notch
            out = POST['notch'](out)
        elif args.post == 'clip':
            out = out.clamp(0, 1)
        else:
            out = (out - out.amin((-2, -1), keepdim=True)) / \
                  (out.amax((-2, -1), keepdim=True)
                   - out.amin((-2, -1), keepdim=True)).clamp_min(1e-6)
        out = out.cpu().numpy()[0, 0]

        up = F.interpolate(torch.from_numpy(lr).view(1, 1, *lr.shape),
                           scale_factor=2, mode='nearest').numpy()[0, 0]
        base = os.path.basename(f)
        panels = [(np.clip(up, 0, 1), f'input {base} (128->256 nearest, clipped)'),
                  (out, 'model output 256x256')]

        # zoom on the brightest flat window -- historically where artifacts show
        t = torch.from_numpy(out).view(1, 1, 256, 256)
        k = torch.ones(1, 1, 48, 48) / 48 ** 2
        mu = F.conv2d(t, k, stride=8)
        var = F.conv2d(t ** 2, k, stride=8) - mu ** 2
        idx = int((mu - 6 * var.clamp_min(0).sqrt()).flatten().argmax())
        cy, cx = (idx // mu.shape[-1]) * 8, (idx % mu.shape[-1]) * 8
        panels += [(np.clip(up, 0, 1)[cy:cy + 48, cx:cx + 48], 'input [zoom]'),
                   (out[cy:cy + 48, cx:cx + 48], 'output [zoom]')]

        imgs = [label(Image.fromarray(to_u8(a)), n, scale=1 if i < 2 else 4)
                for i, (a, n) in enumerate(panels)]
        w = sum(im.width for im in imgs) + 10 * (len(imgs) - 1)
        h = max(im.height for im in imgs)
        row = Image.new('L', (w, h), 255)
        x0 = 0
        for im in imgs:
            row.paste(im, (x0, 0)); x0 += im.width + 10
        rows.append(row)

    W = max(r.width for r in rows)
    grid = Image.new('L', (W, sum(r.height for r in rows) + 14 * (len(rows) - 1)), 255)
    y0 = 0
    for r in rows:
        grid.paste(r, (0, y0)); y0 += r.height + 14
    grid.convert('RGB').save(args.out)
    print(f'wrote {args.out} ({grid.width}x{grid.height})')


if __name__ == '__main__':
    main()
