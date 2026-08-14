"""Train NAFNet-SR on the SemiCon / KLA image restoration task.

Typical use
-----------
  # 1. sanity check the whole loop in ~1 minute
  python train.py --iters 200 --val-every 100 --name smoke

  # 2. measure steps/sec and pick a batch size that fits
  python train.py --bench

  # 3. the real run
  python train.py --name main --iters 200000 --batch 32 --config m

Everything about the run (args, git-less config dump, per-eval metrics) is
written to runs/<name>/ so results are reproducible without archaeology.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nafnet_sr import build_model, count_params, load_pretrained_body
from src.data import SemiconData
from src.degradation import FIT_CFG, OOD_CFG, TRAIN_CFG
from src.losses import RestorationLoss
from src.metrics import POSTPROC, evaluate_batches, evaluate_sharpness


# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of weights.

    Consistently worth ~0.05-0.15 dB on restoration tasks for the price of one
    extra copy of the weights and a lerp per step. We evaluate and ship the EMA
    weights, not the raw ones.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone()
                        for k, v in model.state_dict().items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model, step):
        # ramp the decay in so early steps are not dominated by the init
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), 1.0 - d)

    def state_dict(self):
        out = {k: v.clone() for k, v in self.shadow.items()}
        out.update({k: v.clone() for k, v in self.buffers.items()})
        return out

    class _Swap:
        def __init__(self, ema, model):
            self.ema, self.model = ema, model

        def __enter__(self):
            self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.state_dict(), strict=False)

        def __exit__(self, *a):
            self.model.load_state_dict(self.backup, strict=False)

    def swapped(self, model):
        return EMA._Swap(self, model)


def resolve_warmup(warmup, iters, frac=0.02, cap=2000, floor=50):
    """Warmup as a FRACTION of the run, not a fixed step count.

    A fixed 2000 was tuned for a 100k run (2%). Reused unchanged on a 4k
    experiment it becomes 50% of training: the lr peaks halfway through and
    then immediately anneals to 1e-6, so the model never trains at a useful
    rate and every short run understates the architecture. That invalidated a
    whole ablation sweep before this was caught.

    -1 (default) => auto: 2% of iters, clamped to [50, 2000]. For iters=100000
    this reproduces the original 2000 exactly, so the tuned long-run behaviour
    is unchanged; short runs now get a proportionate warmup instead.
    """
    if warmup is not None and warmup >= 0:
        return warmup
    return int(max(floor, min(cap, round(frac * iters))))


def lr_at(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='train/train', help='dir containing GT/ and NoisyLR/')
    p.add_argument('--name', default='main')
    p.add_argument('--out', default='runs')
    p.add_argument('--config', default='m', choices=['s', 'm', 'l'])
    p.add_argument('--pretrained', default=None,
                   help='official NAFNet .pth to initialise the body from')

    # Defaults chosen from tools/bench_sweep.py on an 8 GB RTX 4060 Laptop:
    # config m + batch 16 peaks at 4.19 GiB and runs 4.09 it/s (~6.8 h / 100k).
    # batch 32 peaks near 8 GiB, spills into WDDM shared memory and collapses
    # to a crawl -- raise this only on a bigger card.
    p.add_argument('--iters', type=int, default=100000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--warmup', type=int, default=-1,
                   help='-1 = auto (2%% of --iters, clamped to [50, 2000]). '
                        'A fixed value that is large relative to a short run '
                        'silently wastes most of it on warmup.')
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--ema', type=float, default=0.999)

    p.add_argument('--synth-ratio', type=float, default=0.5,
                   help='fraction of each batch re-synthesised from GT')
    p.add_argument('--w-fft', type=float, default=0.05)
    p.add_argument('--w-grad', type=float, default=0.15)
    p.add_argument('--w-ssim', type=float, default=0.10)
    p.add_argument('--ms-ssim', action='store_true',
                   help='use MS-SSIM instead of single-scale SSIM in the loss')
    p.add_argument('--w-artifact', type=float, default=0.0,
                   help='targeted penalty on excess power at the two known '
                        'upsampler-artifact frequencies. Broadband losses alone '
                        'measured insufficient over 100k iters -- see '
                        'src/losses.py::artifact_freq_loss.')
    p.add_argument('--use-refine', action='store_true',
                   help='add a learnable dirac-initialised conv after the '
                        'residual, giving the network a place to fix the '
                        'artifact (only useful paired with --w-artifact > 0)')
    p.add_argument('--w-range', type=float, default=0.0,
                   help='penalty on predictions outside [0,1]. Overshoot is what '
                        'lets output clipping amplify the residual Nyquist ripple '
                        '~35x into a visible dot grid.')
    p.add_argument('--head', default='resize_conv',
                   choices=['pixelshuffle', 'resize_conv', 'pixelshuffle_smooth'])
    p.add_argument('--residual', default='bicubic',
                   choices=['bicubic', 'none', 'gated'],
                   help='global skip from the NOISY input. It carries input '
                        'noise straight to the output, making "pass through" a '
                        'cheap local minimum.')
    p.add_argument('--up-mode', default='pixelshuffle',
                   choices=['pixelshuffle', 'pixelshuffle_icnr',
                            'pixelshuffle_smooth', 'resize', 'resize_nearest'],
                   help="internal decoder upsamplers. 'pixelshuffle' is NAFNet's "
                        'original and measurably injects checkerboard structure; '
                        "'resize' upsamples then convolves (Odena et al.).")
    p.add_argument('--input-transform', default='affine', choices=['affine', 'log'],
                   help="log = variance-stabilising transform; turns this task's "
                        'multiplicative speckle into approximately additive noise.')

    p.add_argument('--val-every', type=int, default=2000)
    p.add_argument('--val-size', type=int, default=320)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--data-device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--amp', default='bf16', choices=['bf16', 'fp16', 'off'])
    p.add_argument('--compile', action='store_true', help='try torch.compile')
    p.add_argument('--resume', default=None)
    p.add_argument('--bench', action='store_true', help='time 100 steps and exit')
    args = p.parse_args()
    args.warmup = resolve_warmup(args.warmup, args.iters)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('WARNING: no CUDA device found, this will be unusably slow')
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    if not args.bench:
        # BUG THAT ACTUALLY HAPPENED: --name defaults to 'main', so running
        # --bench without an explicit --name silently overwrote
        # runs/main/args.json with the benchmark's throwaway args -- while
        # leaving best.pt/last.pt (and their internally-saved args/model_kwargs)
        # completely correct. The checkpoint's own metadata is the trustworthy
        # source of truth if this ever happens again; args.json is a convenience
        # mirror, not the record. Guarding here so --bench can no longer corrupt it.
        json.dump(vars(args), open(os.path.join(run_dir, 'args.json'), 'w'), indent=2)

    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'off': None}[args.amp]
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 'fp16'))

    # ---------------- data ----------------
    data = SemiconData(args.data, device=args.data_device, val_size=args.val_size,
                       seed=args.seed)
    gen = torch.Generator(device=args.data_device).manual_seed(args.seed + 1)

    # ---------------- model ----------------
    model_kwargs = dict(head=args.head, residual=args.residual,
                        input_transform=args.input_transform, up_mode=args.up_mode,
                        use_refine=args.use_refine)
    model = build_model(args.config, **model_kwargs).to(device).to(memory_format=torch.channels_last)
    if args.pretrained:
        load_pretrained_body(model, args.pretrained)
    print(f'[model] config={args.config} head={args.head} up_mode={args.up_mode} '
          f'residual={args.residual} input_transform={args.input_transform} '
          f'params={count_params(model)/1e6:.2f}M')

    ema = EMA(model, args.ema) if args.ema > 0 else None
    criterion = RestorationLoss(w_fft=args.w_fft, w_grad=args.w_grad,
                                w_ssim=args.w_ssim, w_range=args.w_range,
                                w_artifact=args.w_artifact,
                                ms_ssim=args.ms_ssim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9),
                            weight_decay=0.0)

    start = 0
    best = -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start = ck['step'] + 1
        best = ck.get('best', best)
        if ema and 'ema' in ck:
            ema.shadow = {k: v.to(device).float() for k, v in ck['ema'].items()
                          if v.dtype.is_floating_point}
        print(f'[resume] from {args.resume} at step {start}')

    fwd = model
    if args.compile:
        try:
            fwd = torch.compile(model)
            print('[compile] torch.compile enabled')
        except Exception as e:                              # Triton is flaky on Windows
            print(f'[compile] unavailable, running eager: {e}')

    # ---------------- bench ----------------
    if args.bench:
        for warm in range(10):
            lr_b, gt_b = data.batch(args.batch, args.synth_ratio, gen=gen)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, _ = criterion(fwd(lr_b.to(memory_format=torch.channels_last)), gt_b)
            loss.backward()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            lr_b, gt_b = data.batch(args.batch, args.synth_ratio, gen=gen)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, _ = criterion(fwd(lr_b.to(memory_format=torch.channels_last)), gt_b)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f'[bench] {100/dt:.2f} it/s  ({dt*10:.1f} ms/it)  batch={args.batch}')
        print(f'[bench] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
        print(f'[bench] 100k iters would take {100000*dt/100/3600:.2f} h')
        return

    # ---------------- log ----------------
    csv_path = os.path.join(run_dir, 'metrics.csv')
    new = not os.path.exists(csv_path)
    csv_f = open(csv_path, 'a', newline='')
    csv_w = csv.writer(csv_f)
    if new:
        csv_w.writerow(['step', 'lr', 'loss',
                        'val_psnr_clip', 'val_ssim_clip',
                        'val_psnr_minmax', 'val_ssim_minmax',
                        'val_psnr_robust', 'val_ssim_robust',
                        'ood_psnr', 'ood_ssim', 'grad_ratio', 'hf_ratio', 'it_per_s'])

    print(f'[train] {args.iters} iters, batch {args.batch}, '
          f'warmup {args.warmup} ({100*args.warmup/max(1,args.iters):.1f}%), '
          f'synth_ratio {args.synth_ratio}, amp {args.amp}')
    model.train()
    run_loss = torch.zeros((), device=device)
    t0 = tlog = time.perf_counter()

    for step in range(start, args.iters):
        cur_lr = lr_at(step, args.iters, args.lr, args.min_lr, args.warmup)
        for gp in opt.param_groups:
            gp['lr'] = cur_lr

        lr_b, gt_b = data.batch(args.batch, args.synth_ratio, TRAIN_CFG, gen=gen)
        lr_b = lr_b.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        gt_b = gt_b.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            out = fwd(lr_b)
            loss, parts = criterion(out, gt_b)

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

        if ema:
            ema.update(model, step)
        # accumulate on-device: calling .item() every step would force a
        # host sync per iteration and throttle the loop
        run_loss = run_loss + loss.detach()

        if (step + 1) % args.log_every == 0:
            if device == 'cuda':
                torch.cuda.synchronize()
            run_loss = float(run_loss)
            ips = args.log_every / (time.perf_counter() - tlog)
            tlog = time.perf_counter()
            eta = (args.iters - step - 1) / ips / 3600
            print(f'  step {step+1:>7}/{args.iters}  lr {cur_lr:.2e}  '
                  f'loss {run_loss/args.log_every:.5f}  {ips:.1f} it/s  eta {eta:.2f}h')
            run_loss = 0.0

        # ---------------- validation ----------------
        if (step + 1) % args.val_every == 0 or step + 1 == args.iters:
            ctx = ema.swapped(model) if ema else torch.no_grad()
            row = {}
            with ctx:
                for pp_name, pp in POSTPROC.items():
                    ps, ss = evaluate_batches(
                        model, data.val_batches(64, 'real'), postproc=pp,
                        amp_dtype=amp_dtype, device=device)
                    row[f'val_psnr_{pp_name}'] = ps
                    row[f'val_ssim_{pp_name}'] = ss
                ood_ps, ood_ss = evaluate_batches(
                    model, data.val_batches(64, 'synth', OOD_CFG, gen=gen),
                    postproc=POSTPROC['minmax'], amp_dtype=amp_dtype, device=device)
                # gradient ratio: 1.0 = output carries as much edge energy as
                # GT, <1 = too smooth. PSNR alone will not reveal this.
                g_ratio, hf_ratio = evaluate_sharpness(
                    model, data.val_batches(64, 'real'), postproc=POSTPROC['clip'],
                    amp_dtype=amp_dtype, device=device)

            ips = (step + 1 - start) / (time.perf_counter() - t0)
            print(f'  [val {step+1}]  clip {row["val_psnr_clip"]:.3f}dB/{row["val_ssim_clip"]:.4f}   '
                  f'minmax {row["val_psnr_minmax"]:.3f}dB/{row["val_ssim_minmax"]:.4f}   '
                  f'robust {row["val_psnr_robust"]:.3f}dB/{row["val_ssim_robust"]:.4f}   '
                  f'| OOD {ood_ps:.3f}dB/{ood_ss:.4f} | grad {g_ratio:.3f} hf {hf_ratio:.2f}')
            csv_w.writerow([step + 1, f'{cur_lr:.3e}', f'{loss.item():.5f}'] +
                           [f'{row[f"val_{m}_{k}"]:.4f}' for k in POSTPROC for m in ('psnr', 'ssim')] +
                           [f'{ood_ps:.4f}', f'{ood_ss:.4f}',
                            f'{g_ratio:.4f}', f'{hf_ratio:.4f}', f'{ips:.2f}'])
            csv_f.flush()

            score = max(row['val_psnr_clip'], row['val_psnr_minmax'], row['val_psnr_robust'])
            ck = {'step': step, 'model': model.state_dict(), 'opt': opt.state_dict(),
                  'best': max(best, score), 'args': vars(args),
                  'config': args.config, 'head': args.head,
                  'model_kwargs': model_kwargs, 'metrics': row}
            if ema:
                ck['ema'] = ema.state_dict()
            torch.save(ck, os.path.join(run_dir, 'last.pt'))
            if score > best:
                best = score
                torch.save(ck, os.path.join(run_dir, 'best.pt'))
                print(f'  [val] new best {best:.3f} dB -> best.pt')

    csv_f.close()
    print(f'[done] best val PSNR {best:.3f} dB   weights in {run_dir}')


if __name__ == '__main__':
    main()
