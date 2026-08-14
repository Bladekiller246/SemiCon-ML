"""No-model baselines on the held-out validation split.

Every later claim is measured against these. If a trained model does not
clearly beat plain bicubic upsampling, something is wrong with the training,
not with the metric.

Uses the SAME val split as train.py (torch.randperm(3200, seed=0)[:320]) so the
numbers are directly comparable to the val_psnr_* columns in metrics.csv.

Run:  .venv/Scripts/python -u tools/baseline.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import POSTPROC, psnr, ssim  # noqa: E402

dev = 'cpu'          # deliberately CPU: leaves the GPU free for training
GT = sorted(glob.glob('train/train/GT/*.npy'))
LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))

# identical split to SemiconData
perm = torch.randperm(len(GT), generator=torch.Generator().manual_seed(0))
val_idx = perm[:320].tolist()

gt = torch.from_numpy(np.stack([np.load(GT[i]) for i in val_idx])[:, None]).float()
lr = torch.from_numpy(np.stack([np.load(LR[i]) for i in val_idx])[:, None]).float()
print(f'val split: {gt.shape[0]} images\n')

METHODS = {
    'nearest':  lambda x: F.interpolate(x, scale_factor=2, mode='nearest'),
    'bilinear': lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False),
    'bicubic':  lambda x: F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False),
}

print('%-10s %-8s %9s %9s' % ('upsample', 'post', 'PSNR', 'SSIM'))
print('-' * 40)
best = None
for mname, fn in METHODS.items():
    up = fn(lr)
    for pname, pp in POSTPROC.items():
        out = pp(up)
        p = psnr(out, gt).mean().item()
        s = ssim(out, gt).mean().item()
        print('%-10s %-8s %9.3f %9.4f' % (mname, pname, p, s))
        if best is None or p > best[0]:
            best = (p, s, mname, pname)
    print()

print(f'BEST BASELINE: {best[0]:.3f} dB / {best[1]:.4f} SSIM  ({best[2]} + {best[3]})')
print('\nThe model must beat this by a wide margin. Note how much the choice of')
print('post-processing alone moves PSNR -- that is the same knob evaluate.py')
print('exposes as --post, and it is worth free dB on the model output too.')
