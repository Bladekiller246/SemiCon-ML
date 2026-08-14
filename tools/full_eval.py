"""PSNR / SSIM / LPIPS on the held-out validation split, plus a labelled
input -> output -> ground-truth comparison grid, for external review.

WHAT "TEST SPLIT" MEANS HERE. Test_NoisyLR/ (the 400-image challenge test
set) ships with NO ground truth -- KLA holds that back to score submissions,
so no PSNR/SSIM/LPIPS can be computed against it from this side. The numbers
below are instead on the 320-image HELD-OUT VALIDATION split carved out of
train/train by src/data.py's source-aware group split (same seed=0 used
throughout training/eval, leakage-checked to 0) -- the closest available
stand-in, and the same split every other script in this repo (train.py,
evaluate.py's dev tools) reports against.

The comparison grid uses VAL images specifically (not train images), so nothing
shown was seen during training -- verified against the same split logic
src/data.py uses.

Metrics:
  PSNR / SSIM   matched to skimage's implementation (src/metrics.py); higher
                is better; this is what the training loop selects checkpoints on.
  LPIPS         a learned perceptual distance (AlexNet features, Zhang et al.
                2018) -- unlike PSNR/SSIM it correlates with HUMAN judgements
                of similarity, and specifically penalises the kind of
                regression-to-mean blur that raises PSNR while looking worse.
                Lower is better. Needs `pip install lpips torchvision` (not a
                dependency of evaluate.py itself, which stays minimal on
                purpose -- this is a dev/reporting tool only).

Run:
  .venv/Scripts/python -u tools/full_eval.py --weights runs/final/best.pt
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
from evaluate import POST  # noqa: E402
from nafnet_sr import build_model  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402


# --------------------------------------------------------------------------- #
def val_split(n_imgs, group_size=4, val_size=320, seed=0):
    """Reproduces src/data.py's SemiconData split exactly (same seed/logic),
    but returns the actual INDICES rather than tensors, so images can be
    labelled by filename. Two calls with the same seed always agree -- this
    is what SemiconData itself does internally, just not exposed."""
    n_groups = (n_imgs + group_size - 1) // group_size
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_groups, generator=g).tolist()
    val_groups = set(perm[:max(1, round(val_size / group_size))])
    val_idx = [i for i in range(n_imgs) if (i // group_size) in val_groups]
    return val_idx


def to_u8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def label(img, lines, pad=44):
    c = Image.new('L', (img.width, img.height + pad), 255)
    c.paste(img, (0, pad))
    d = ImageDraw.Draw(c)
    try:
        font = ImageFont.truetype('arial.ttf', 13)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        d.text((5, 3 + 14 * i), line, fill=0, font=font)
    return c


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/final/best.pt')
    p.add_argument('--data', default='train/train')
    p.add_argument('--post', default='notch', choices=list(POST),
                   help="matches evaluate.py's default output handling")
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--grid-n', type=int, default=6,
                   help='how many VAL images to render in the comparison grid')
    p.add_argument('--out-prefix', default='eval_report')
    args = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------- model ----------------
    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    model = build_model(ck.get('config', 'm'), **kw)
    missing, unexpected = model.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    model.eval().to(dev).to(memory_format=torch.channels_last)
    print(f'[model] {args.weights}  config={ck.get("config")}  step={ck.get("step")}  '
          f'kwargs={kw}  post={args.post}')
    if missing or unexpected:
        print(f'[warn] missing={len(missing)} unexpected={len(unexpected)} state_dict keys')

    # ---------------- LPIPS ----------------
    try:
        import lpips
        lp = lpips.LPIPS(net='alex').to(dev).eval()
        for p_ in lp.parameters():
            p_.requires_grad_(False)
        have_lpips = True
    except ImportError:
        print('[warn] `lpips` not installed (pip install lpips torchvision) -- '
              'skipping LPIPS, reporting PSNR/SSIM only')
        have_lpips = False

    def lpips_batch(pred, target):
        # lpips expects 3-channel, range [-1,1]
        p3 = pred.repeat(1, 3, 1, 1) * 2 - 1
        t3 = target.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            return lp(p3, t3).flatten()

    # ---------------- data / split ----------------
    GT = sorted(glob.glob(os.path.join(args.data, 'GT', '*.npy')))
    LR = sorted(glob.glob(os.path.join(args.data, 'NoisyLR', '*.npy')))
    assert len(GT) == len(LR) and len(GT) > 0, 'GT/NoisyLR mismatch or empty'
    vidx = val_split(len(GT))
    print(f'[data] {len(GT)} total pairs, {len(vidx)} held out for validation '
          f'(source-aware split, seed=0, same as training)\n')

    # ---------------- full-split metrics ----------------
    tot_psnr = tot_ssim = tot_lpips = 0.0
    per_image = []  # (idx, psnr, ssim, lpips)
    with torch.inference_mode():
        for i in range(0, len(vidx), args.batch):
            chunk = vidx[i:i + args.batch]
            gt = torch.from_numpy(np.stack(
                [np.load(GT[j]).astype(np.float32) for j in chunk]))[:, None].to(dev)
            lr = torch.from_numpy(np.stack(
                [np.load(LR[j]).astype(np.float32) for j in chunk]))[:, None].to(dev)

            raw = model(lr.to(memory_format=torch.channels_last)).float()
            out = POST[args.post](raw)

            pv = psnr(out, gt)
            sv = ssim(out, gt)
            lv = lpips_batch(out, gt) if have_lpips else torch.zeros_like(pv)

            tot_psnr += pv.sum().item()
            tot_ssim += sv.sum().item()
            tot_lpips += lv.sum().item()
            for k, j in enumerate(chunk):
                per_image.append((j, pv[k].item(), sv[k].item(), lv[k].item()))
            print(f'\r  scored {min(i + args.batch, len(vidx))}/{len(vidx)}',
                  end='', flush=True)
    n = len(vidx)
    print()

    ps = np.array([r[1] for r in per_image])
    ss = np.array([r[2] for r in per_image])
    ls = np.array([r[3] for r in per_image])

    print(f'\n=== {n}-image held-out validation split ===')
    print(f'  weights: {args.weights}   post-process: {args.post}\n')
    print(f'  {"metric":8s} {"mean":>8s} {"std":>8s} {"min":>8s} {"median":>8s} {"max":>8s}')
    print(f'  {"PSNR":8s} {ps.mean():8.3f} {ps.std():8.3f} {ps.min():8.3f} '
          f'{np.median(ps):8.3f} {ps.max():8.3f}   (dB, higher better)')
    print(f'  {"SSIM":8s} {ss.mean():8.4f} {ss.std():8.4f} {ss.min():8.4f} '
          f'{np.median(ss):8.4f} {ss.max():8.4f}   (higher better)')
    if have_lpips:
        print(f'  {"LPIPS":8s} {ls.mean():8.4f} {ls.std():8.4f} {ls.min():8.4f} '
              f'{np.median(ls):8.4f} {ls.max():8.4f}   (AlexNet, lower better)')

    with open(f'{args.out_prefix}_metrics.csv', 'w') as f:
        f.write('image,psnr_db,ssim,lpips\n')
        for j, pv, sv, lv in sorted(per_image):
            f.write(f'{os.path.basename(GT[j])},{pv:.4f},{sv:.4f},{lv:.4f}\n')
    print(f'\n[wrote] {args.out_prefix}_metrics.csv  (per-image, all {n} val images)')

    # ---------------- visual grid: spread across the score distribution ----------------
    order = np.argsort(ps)  # worst -> best PSNR
    pick_pos = np.linspace(0, n - 1, args.grid_n).astype(int)
    picks = [per_image[order[p]] for p in pick_pos]

    rows = []
    with torch.inference_mode():
        for j, pv, sv, lv in picks:
            gt_a = np.load(GT[j]).astype(np.float32)
            lr_a = np.load(LR[j]).astype(np.float32)
            x = torch.from_numpy(lr_a)[None, None].to(dev)
            out = POST[args.post](model(x.to(memory_format=torch.channels_last)).float())
            out_a = out.cpu().numpy()[0, 0]
            lr_disp = np.clip(F.interpolate(
                torch.from_numpy(lr_a)[None, None], scale_factor=2,
                mode='nearest').numpy()[0, 0], 0, 1)

            stem = os.path.basename(GT[j])
            lpips_s = f'{lv:.4f}' if have_lpips else 'n/a'
            tiles = [
                label(Image.fromarray(to_u8(lr_disp)),
                      [f'{stem}  [VAL, unseen]', 'degraded input (128->256)'], pad=58),
                label(Image.fromarray(to_u8(out_a)),
                      ['restored output', f'PSNR {pv:.2f} dB   SSIM {sv:.4f}',
                       f'LPIPS {lpips_s}'], pad=58),
                label(Image.fromarray(to_u8(gt_a)),
                      ['ground truth', '', ''], pad=58),
            ]
            w = sum(t.width for t in tiles) + 10 * (len(tiles) - 1)
            row = Image.new('L', (w, tiles[0].height), 255)
            ox = 0
            for t in tiles:
                row.paste(t, (ox, 0))
                ox += t.width + 10
            rows.append(row)

    sheet = Image.new('L', (max(r.width for r in rows),
                            sum(r.height for r in rows) + 10 * (len(rows) - 1)), 255)
    oy = 0
    for r in rows:
        sheet.paste(r, (0, oy))
        oy += r.height + 10
    out_png = f'{args.out_prefix}_grid.png'
    sheet.save(out_png)
    print(f'[wrote] {out_png}  ({args.grid_n} images spanning worst->best PSNR in val)')


if __name__ == '__main__':
    main()
