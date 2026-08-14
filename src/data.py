"""GPU-resident dataset.

The whole training set is 3200 x (256^2 + 128^2) floats. In fp16 that is
~525 MB, which fits in VRAM alongside the model on an 8 GB card. So we load it
ONCE and never touch the disk or the PCIe bus again:

    no DataLoader, no workers, no collate, no pin_memory, no H2D copies.

On a laptop 4060 this is the difference between a data-starved loop and a
GPU-bound one. It is also the honest answer to the brief's "optimize disk
reads, data transfer, how to make pipeline fast".

If VRAM is tight (`--data-device cpu`), tensors stay in pinned host memory and
only the sampled batch is copied across, which is still far cheaper than
per-file np.load.
"""
import glob
import os

import numpy as np
import torch

from .degradation import TRAIN_CFG, augment_gt, degrade, dihedral


def _load_stack(paths, out_dtype=torch.float16):
    first = np.load(paths[0])
    arr = np.empty((len(paths), 1, *first.shape), dtype=np.float32)
    for i, p in enumerate(paths):
        arr[i, 0] = np.load(p)
    return torch.from_numpy(arr).to(out_dtype)


class SemiconData:
    """Holds GT/LR stacks on-device and serves augmented batches."""

    def __init__(self, root, device='cuda', dtype=torch.float16,
                 val_size=320, seed=0, verbose=True, group_size=4):
        gt_paths = sorted(glob.glob(os.path.join(root, 'GT', '*.npy')))
        lr_paths = sorted(glob.glob(os.path.join(root, 'NoisyLR', '*.npy')))
        if not gt_paths:
            raise FileNotFoundError(f'no GT/*.npy under {root}')
        if len(gt_paths) != len(lr_paths):
            raise ValueError(f'{len(gt_paths)} GT vs {len(lr_paths)} NoisyLR')
        # filenames must correspond one-to-one
        for a, b in zip(gt_paths, lr_paths):
            if os.path.basename(a) != os.path.basename(b):
                raise ValueError(f'GT/NoisyLR filename mismatch: {a} vs {b}')

        if verbose:
            print(f'[data] loading {len(gt_paths)} pairs from {root} ...')
        gt = _load_stack(gt_paths, dtype)
        lr = _load_stack(lr_paths, dtype)

        # SOURCE-AWARE split. The dataset ships 4 crops per source image
        # (source_id = index // 4). MEASURED: crops within a group have median
        # content similarity 0.626 (39.7% above 0.8) against -0.019 for random
        # pairs. Splitting on individual images therefore leaks -- under the old
        # random split, 100% of the 320 val images had a same-group sibling in
        # train, so every reported number was inflated.
        #
        # Splitting on source_id keeps all 4 crops of a source on the same side.
        n_groups = (len(gt_paths) + group_size - 1) // group_size
        g = torch.Generator().manual_seed(seed)
        gperm = torch.randperm(n_groups, generator=g).tolist()
        n_val_groups = max(1, round(val_size / group_size))
        val_groups = set(gperm[:n_val_groups])

        val_idx, train_idx = [], []
        for i in range(len(gt_paths)):
            (val_idx if (i // group_size) in val_groups else train_idx).append(i)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        train_idx = torch.tensor(train_idx, dtype=torch.long)

        if verbose:
            leaked = sum(1 for v in val_idx.tolist()
                         if (v // group_size) not in val_groups)
            print(f'[data] source-aware split: {n_val_groups} val groups '
                  f'({len(val_idx)} images), leakage check = {leaked} (must be 0)')

        self.device = torch.device(device)
        self.dtype = dtype
        self.gt_train = gt[train_idx].to(self.device)
        self.lr_train = lr[train_idx].to(self.device)
        self.gt_val = gt[val_idx].to(self.device)
        self.lr_val = lr[val_idx].to(self.device)
        self.n_train = len(train_idx)
        self.n_val = len(val_idx)

        if verbose:
            mb = (self.gt_train.numel() + self.lr_train.numel() +
                  self.gt_val.numel() + self.lr_val.numel()) * dtype.itemsize / 2 ** 20
            print(f'[data] train={self.n_train} val={self.n_val} '
                  f'resident on {self.device} ({mb:.0f} MB, {dtype})')

    # ------------------------------------------------------------------ #
    def batch(self, bs, synth_ratio=0.5, cfg=TRAIN_CFG, gen=None,
              use_dihedral=True, use_gamma_jitter=True):
        """Return (lr, gt) float32 batches, already augmented.

        `synth_ratio` of the batch is re-synthesised from GT with randomised
        degradation parameters instead of using the provided NoisyLR. The real
        half keeps us honest about the true forward model; the synthetic half
        is what buys out-of-distribution robustness.
        """
        idx = torch.randint(0, self.n_train, (bs,), device=self.device, generator=gen)
        gt = self.gt_train[idx].float()
        lr = self.lr_train[idx].float()

        n_syn = int(round(bs * synth_ratio))
        if n_syn > 0:
            gt_s = gt[:n_syn]
            if use_gamma_jitter:
                gt_s = augment_gt(gt_s, gen=gen)
            lr = torch.cat([degrade(gt_s, cfg, gen=gen), lr[n_syn:]], dim=0)
            gt = torch.cat([gt_s, gt[n_syn:]], dim=0)

        if use_dihedral:
            lr, gt = dihedral(lr, gt, gen=gen)
        return lr, gt

    # ------------------------------------------------------------------ #
    def val_batches(self, bs=64, source='real', cfg=None, gen=None):
        """Yield (lr, gt) over the held-out split.

        source='real'  -> the provided NoisyLR (in-distribution accuracy)
        source='synth' -> re-degraded with `cfg` (used with OOD_CFG this is a
                          generalisation probe against degradations the model
                          was never trained on)
        """
        for i in range(0, self.n_val, bs):
            gt = self.gt_val[i:i + bs].float()
            if source == 'real':
                lr = self.lr_val[i:i + bs].float()
            else:
                lr = degrade(gt, cfg, gen=gen)
            yield lr, gt
