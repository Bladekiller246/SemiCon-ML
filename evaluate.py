"""Standalone inference: degraded 128x128 .npy in -> restored 256x256 .npy out.

This is the file KLA runs as-is on the benchmark machine, so it is deliberately
defensive:

  * accepts both positional and flagged paths
        python evaluate.py <input_dir> <output_dir>
        python evaluate.py --input_dir <in> --output_dir <out>
    (also --input-dir / --output-dir, and --input/--output)
  * defaults --weights to weights/best.pt *relative to this file*, so it does
    not matter what the working directory is
  * imports only torch + numpy, plus nafnet_sr.py which sits next to this file
  * falls back to CPU if there is no GPU rather than crashing
  * never writes outside output_dir, and preserves input basenames

Run `python evaluate.py --self-test` to verify the whole path on random data
without any real inputs.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from nafnet_sr import build_model  # noqa: E402

IMAGE_EXTS = ('.npy', '.png', '.tif', '.tiff', '.bmp')


# --------------------------------------------------------------------------- #
def load_image(path):
    """-> float32 HxW array. .npy is the challenge format; the image formats are
    a defensive fallback in case the test set ships differently."""
    if path.lower().endswith('.npy'):
        a = np.load(path)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path))
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype == np.uint16:
            a = a.astype(np.float32) / 65535.0
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:                      # collapse an accidental colour axis
        a = a.mean(axis=2) if a.shape[2] <= 4 else a.mean(axis=0)
    if a.ndim != 2:
        raise ValueError(f'{path}: expected 2-D image, got shape {a.shape}')
    return a


def save_image(path, arr):
    if path.lower().endswith('.npy'):
        np.save(path, arr.astype(np.float32))
    else:
        from PIL import Image
        Image.fromarray((np.clip(arr, 0, 1) * 65535).astype(np.uint16)).save(path)


def minmax(x, eps=1e-6):
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(eps)


def robust_minmax(x, q=0.001, eps=1e-6):
    flat = x.flatten(1)
    lo = torch.quantile(flat, q, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 1 - q, dim=1).view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(eps)).clamp(0, 1)


def _checker_basis(x):
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing='ij')
    return ((-1.0) ** ((yy + xx) % 2)).view(1, 1, H, W)


def notch(x, sigma=1.0):
    """Remove the local Nyquist (2x2 checkerboard) component.

    WHY THIS EXISTS. Traced stage by stage (tools/trace_stages.py): the raw
    model output is clean (Nyquist coherence 0.0), but it overshoots [0,1]
    (1.10% of pixels above 1.0). Hard-clipping then rectifies the tiny residual
    Nyquist ripple asymmetrically and amplifies it ~14x into a visible dot
    grid -- coherence 0.0 -> 744.

    A notch at exactly Nyquist is nearly free: ground truth has LESS energy at
    that frequency than its own spectral median (nyq/med ~ 0.1), so almost
    nothing real lives there. Measured on the val split, it removes the artifact
    AND gains accuracy rather than trading against it:

        clip [0,1]            25.823 dB / 0.6440   nyq 744.3   <- was the default
        clip -> notch         26.065 dB / 0.6488   nyq 0.0
        notch -> clip -> notch 26.252 dB / 0.6528  nyq 0.0     <- best on both

    Projects onto the checkerboard basis, low-passes that coefficient map,
    projects back, subtracts. One frequency touched; everything else passes.
    """
    c = _checker_basis(x)
    coeff = x * c
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k * k / (2 * sigma * sigma))
    g = g / g.sum()
    coeff = torch.nn.functional.pad(coeff, (r, r, r, r), mode='reflect')
    coeff = torch.nn.functional.conv2d(coeff, g.view(1, 1, 1, -1))
    coeff = torch.nn.functional.conv2d(coeff, g.view(1, 1, -1, 1))
    return x - coeff * c


_ARTIFACT_MASK = {}


def _artifact_mask(H, W, device, radius=1.8):
    """Soft (gaussian) notch at the exact frequencies the two artifacts occupy.

    Both artifacts are resampling-phase artifacts of the upsampler and sit at
    fixed, known 2-D frequencies -- period-2 (Nyquist checkerboard, produced by
    PixelShuffle heads) and period-4 (produced by the bilinear resize_conv head).
    Ground truth has almost no energy at either (period2 0.19, period4 1.94
    relative to its own spectral median), so removing those bins costs almost
    nothing real while the model puts 880x / 14667x there.
    """
    key = (H, W, str(device), radius)
    if key in _ARTIFACT_MASK:
        return _ARTIFACT_MASK[key]
    n, q, nw, qw = H // 2, H // 4, W // 2, W // 4
    centres = [(n, nw), (q, W - qw), (H - q, qw), (q, qw), (H - q, W - qw),
               (n, 0), (0, nw), (q, 0), (H - q, 0), (0, qw), (0, W - qw)]
    fy = torch.arange(H, device=device).view(-1, 1).float()
    fx = torch.arange(W, device=device).view(1, -1).float()
    mask = torch.ones(H, W, device=device)
    for cy, cx in centres:
        dy = torch.minimum((fy - cy).abs(), H - (fy - cy).abs())
        dx = torch.minimum((fx - cx).abs(), W - (fx - cx).abs())
        mask = mask * (1 - torch.exp(-(dy ** 2 + dx ** 2) / (2 * radius ** 2)))
    _ARTIFACT_MASK[key] = mask
    return mask


def notch_artifacts(x, radius=1.8):
    """Remove both upsampler artifacts. MEASURED on the val split:

        clip (was default)  25.823 dB / 0.6440   period2 880.6  period4 14667
        notch r=1.8         27.170 dB / 0.7133   period2   0.89 period4     0.36
                            (ground truth: period2 0.19, period4 1.94)

    +1.35 dB AND both artifacts driven below ground-truth level. Applied twice,
    around the clamp, because clipping re-creates a little of the ripple it
    rectifies. Radius 1.8 is deliberate: larger radii keep raising PSNR (r=6.0
    reaches 27.456) but do it by blurring -- gradient ratio falls 0.900 -> 0.833
    -- and the artifacts are already below GT level at 1.8, so anything beyond
    trades real detail for a metric that has stopped measuring the artifact.
    """
    H, W = x.shape[-2:]
    m = _artifact_mask(H, W, x.device, radius)
    return torch.fft.ifft2(torch.fft.fft2(x.float()) * m).real


POST = {
    'minmax': minmax,
    'robust': robust_minmax,
    'clip': lambda x: x.clamp(0, 1),
    'nyquist_only': lambda x: notch(notch(x).clamp(0, 1)),
    # default: removes BOTH upsampler artifacts. +1.35 dB over plain clip.
    'notch': lambda x: notch_artifacts(notch_artifacts(x).clamp(0, 1)).clamp(0, 1),
}


# --------------------------------------------------------------------------- #
def load_model(weights, device, config=None):
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    cfg = config or ck.get('config', 'm')
    # older checkpoints predate model_kwargs; fall back to the head field
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    model = build_model(cfg, **kw)
    # prefer the EMA weights -- those are what validation selected on
    sd = ck.get('ema') or ck.get('model') or ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys, e.g. {missing[:3]}')
    model.eval().to(device).to(memory_format=torch.channels_last)
    return model, cfg


@torch.inference_mode()
def run_batch(model, x, amp_dtype, self_ensemble=False):
    """x: (B,1,H,W) on device."""
    def fwd(t):
        with torch.autocast('cuda', dtype=amp_dtype,
                            enabled=amp_dtype is not None and t.is_cuda):
            return model(t.to(memory_format=torch.channels_last)).float()

    if not self_ensemble:
        return fwd(x)

    # 8x dihedral self-ensemble: ~8x the cost for a few hundredths of a dB.
    # Off by default because inference time is explicitly scored.
    acc = None
    for k in range(4):
        for flip in (False, True):
            t = torch.rot90(x, k, dims=(-2, -1))
            if flip:
                t = torch.flip(t, dims=(-1,))
            y = fwd(t)
            if flip:
                y = torch.flip(y, dims=(-1,))
            y = torch.rot90(y, -k, dims=(-2, -1))
            acc = y if acc is None else acc + y
    return acc / 8.0


def main():
    p = argparse.ArgumentParser(description='NAFNet-SR inference for the KLA restoration task')
    p.add_argument('input_dir', nargs='?', default=None)
    p.add_argument('output_dir', nargs='?', default=None)
    p.add_argument('--input_dir', '--input-dir', '--input', dest='input_kw', default=None)
    p.add_argument('--output_dir', '--output-dir', '--output', dest='output_kw', default=None)
    p.add_argument('--weights', default=os.path.join(_HERE, 'weights', 'best.pt'))
    p.add_argument('--config', default=None, choices=[None, 's', 'm', 'l'])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', default='fp16', choices=['fp16', 'bf16', 'off'])
    p.add_argument('--post', default='notch', choices=list(POST),
                   help="output handling. Default 'notch' = notch->clip->notch, "
                        'measured best on BOTH accuracy (26.252 dB vs 25.823 for '
                        "plain clip) and artifact suppression (Nyquist coherence "
                        '0.0 vs 744). See the notch() docstring.')
    p.add_argument('--self-ensemble', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    in_dir = args.input_kw or args.input_dir
    out_dir = args.output_kw or args.output_dir

    device = torch.device(args.device)
    amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'off': None}[args.amp]
    if device.type != 'cuda':
        amp_dtype = None

    # MEASURED: cudnn.benchmark=True costs ~8.5 s of autotune on the first batch
    # and buys nothing in steady state here (0.280 s/batch either way). Over a
    # 400-image run that is 22.8 ms/image instead of 4.4 -- a 5x penalty on the
    # exact number KLA benchmarks. Autotuning only pays off across thousands of
    # steps, which is training, not inference. Leave this False.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------- self test ----------------
    if args.self_test:
        model, cfg = load_model(args.weights, device) if os.path.exists(args.weights) \
            else (build_model('m').eval().to(device), 'm (untrained)')
        x = torch.rand(4, 1, 128, 128, device=device)
        y = POST[args.post](run_batch(model, x, amp_dtype))
        print(f'[self-test] config={cfg} {tuple(x.shape)} -> {tuple(y.shape)}  '
              f'range [{y.min():.3f}, {y.max():.3f}]  OK')
        return

    if not in_dir or not out_dir:
        p.error('need input and output directories:\n'
                '  python evaluate.py <input_dir> <output_dir>')
    if not os.path.isdir(in_dir):
        p.error(f'input dir does not exist: {in_dir}')
    if not os.path.exists(args.weights):
        p.error(f'weights not found: {args.weights}  (pass --weights)')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(in_dir, '*'))
                   if f.lower().endswith(IMAGE_EXTS))
    if not files:
        p.error(f'no {"/".join(IMAGE_EXTS)} files in {in_dir}')

    model, cfg = load_model(args.weights, device, args.config)
    print(f'[eval] {len(files)} images | config={cfg} | device={device} | '
          f'amp={args.amp} | post={args.post} | self_ensemble={args.self_ensemble}')

    post = POST[args.post]
    t_start = time.perf_counter()
    t_infer = 0.0

    for i in range(0, len(files), args.batch_size):
        chunk = files[i:i + args.batch_size]
        arrs = [load_image(f) for f in chunk]
        shapes = {a.shape for a in arrs}
        # group-by-shape is unnecessary for this dataset (all 128x128) but keeps
        # the script correct if the real test set mixes 128 and 256 inputs
        for shape in sorted(shapes):
            sel = [j for j, a in enumerate(arrs) if a.shape == shape]
            x = torch.from_numpy(np.stack([arrs[j] for j in sel])[:, None]).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = post(run_batch(model, x, amp_dtype, args.self_ensemble))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_infer += time.perf_counter() - t0

            y = y.cpu().numpy()[:, 0]
            for k, j in enumerate(sel):
                base = os.path.basename(chunk[j])
                stem = os.path.splitext(base)[0]
                save_image(os.path.join(out_dir, stem + '.npy'), y[k])

        print(f'\r  {min(i + args.batch_size, len(files))}/{len(files)}', end='', flush=True)

    total = time.perf_counter() - t_start
    print(f'\n[eval] wrote {len(files)} images to {out_dir}')
    print(f'[eval] model time  {t_infer:.3f} s  ({1000*t_infer/len(files):.2f} ms/image)')
    print(f'[eval] end-to-end  {total:.3f} s  ({1000*total/len(files):.2f} ms/image, incl. I/O)')


if __name__ == '__main__':
    main()
