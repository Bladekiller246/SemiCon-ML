"""Side-by-side comparison grid: degraded input | model output | ground truth.

Uses the same held-out validation split as train.py/tools/baseline.py
(torch.randperm(3200, seed=0)[:320]), so these are images the model never
trained on.

Run:  .venv/Scripts/python -u tools/visualize_examples.py --weights runs/preview/best.pt
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


def to_u8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def label(img, text, pad=28):
    w, h = img.size
    canvas = Image.new('L', (w, h + pad), 255)
    canvas.paste(img, (0, pad))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((6, 4), text, fill=0, font=font)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/preview/best.pt')
    p.add_argument('--config', default=None)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--out', default='examples_preview.png')
    p.add_argument('--post', default='clip', choices=['clip', 'minmax', 'robust'])
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    GT = sorted(glob.glob('train/train/GT/*.npy'))
    LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))
    perm = torch.randperm(len(GT), generator=torch.Generator().manual_seed(0))
    val_idx = perm[:320].tolist()[:args.n]

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    cfg = args.config or ck.get('config', 'm')
    model = build_model(cfg)
    sd = ck.get('ema') or ck.get('model') or ck
    model.load_state_dict(sd, strict=False)
    model.eval().to(dev).to(memory_format=torch.channels_last)
    print(f'loaded {args.weights}  config={cfg}  step={ck.get("step", "?")}')

    post = {'clip': lambda x: x.clamp(0, 1),
            'minmax': lambda x: (x - x.amin((-2, -1), keepdim=True)) /
                                (x.amax((-2, -1), keepdim=True) - x.amin((-2, -1), keepdim=True)).clamp_min(1e-6)
            }[args.post]

    rows = []
    for i in val_idx:
        gt = np.load(GT[i]).astype(np.float32)
        lr = np.load(LR[i]).astype(np.float32)

        x = torch.from_numpy(lr).view(1, 1, 128, 128).to(dev).to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            out = model(x).float()
        out = post(out).cpu().numpy()[0, 0]

        lr_disp = F.interpolate(torch.from_numpy(lr).view(1, 1, 128, 128),
                                scale_factor=2, mode='nearest').numpy()[0, 0]
        lr_disp = np.clip(lr_disp, 0, 1)   # NoisyLR legitimately exceeds [0,1]; clip only for display

        imgs = [Image.fromarray(to_u8(lr_disp)), Image.fromarray(to_u8(out)), Image.fromarray(to_u8(gt))]
        names = [f'input NoisyLR #{i:04d} (128->256 nearest, clipped for display)',
                 'model output', 'ground truth']
        labeled = [label(im, n) for im, n in zip(imgs, names)]
        row = Image.new('L', (sum(im.width for im in labeled) + 12 * (len(labeled) - 1), labeled[0].height), 255)
        x0 = 0
        for im in labeled:
            row.paste(im, (x0, 0))
            x0 += im.width + 12
        rows.append(row)

    grid = Image.new('L', (rows[0].width, sum(r.height for r in rows) + 12 * (len(rows) - 1)), 255)
    y0 = 0
    for r in rows:
        grid.paste(r, (0, y0))
        y0 += r.height + 12
    grid = grid.convert('RGB')
    grid.save(args.out)
    print(f'wrote {args.out}  ({grid.width}x{grid.height})')


if __name__ == '__main__':
    main()
