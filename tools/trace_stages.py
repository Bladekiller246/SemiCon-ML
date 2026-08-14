"""Stage-by-stage probe: WHERE does the checkerboard first appear?

The model is not a denoise->deblur->upscale cascade (that design was rejected --
the degradations are applied in random order, so there is no fixed order to
invert). It is one joint UNet pass. But it has distinct architectural stages,
and we can measure every one:

    intro -> enc0..enc3 (+downs) -> middle -> up0/dec0..up3/dec3 -> head -> out

At each stage we report:
  rel_checker : checkerboard energy as a FRACTION of that tensor's own std.
                Raw amplitude is useless here because feature magnitudes vary
                by orders of magnitude between stages; the fraction is what
                tells us whether the checkerboard is being CREATED or merely
                carried along.
  nyq/med     : power at exactly the Nyquist corner over the spectral median.
                Separates a coherent periodic grid (high) from ordinary
                broadband texture (~1).

Run:
  .venv/Scripts/python -u tools/trace_stages.py --weights runs/main/best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model  # noqa: E402


def rel_checker(x):
    """Checkerboard energy / tensor std. Scale-invariant."""
    x = x.float()
    B, C, H, W = x.shape
    if H < 8 or W < 8:
        return float('nan')
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing='ij')
    c = ((-1.0) ** ((yy + xx) % 2)).view(1, 1, H, W)
    k = torch.ones(1, 1, 8, 8, device=x.device) / 64
    amp = F.conv2d((x * c).reshape(B * C, 1, H, W), k, stride=4).abs().mean()
    return (amp / x.std().clamp_min(1e-8)).item()


def nyq_ratio(x):
    x = x.float()
    x = x - x.mean((-2, -1), keepdim=True)
    B, C, H, W = x.shape
    if H < 8:
        return float('nan')
    S = (torch.fft.fft2(x, norm='ortho').abs() ** 2).reshape(B * C, H, W).mean(0)
    n = H // 2
    return (S[n, n] / S.median().clamp_min(1e-20)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/main/best.pt')
    p.add_argument('--image', default='Test_NoisyLR/NoisyLR/000000.npy',
                   help='default is a test image MEASURED to overshoot (max 1.231) '
                        'and show 15.6x clip amplification')
    p.add_argument('--out', default='trace_stages.png')
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kw = ck.get('model_kwargs') or {'head': ck.get('head', 'pixelshuffle')}
    m = build_model(ck.get('config', 'm'), **kw)
    m.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    m.eval().to(dev)
    print(f'{args.weights}  config={ck.get("config")}  kwargs={kw}\n')

    lr = np.load(args.image).astype(np.float32)
    x = torch.from_numpy(lr).view(1, 1, *lr.shape).to(dev)

    # ---- register hooks on every stage, in execution order ----
    trace = []          # (name, tensor)
    hooks = []

    def hook(name):
        def fn(mod, inp, out):
            trace.append((name, out.detach()))
        return fn

    hooks.append(m.intro.register_forward_hook(hook('intro')))
    for i, (e, d) in enumerate(zip(m.encoders, m.downs)):
        hooks.append(e.register_forward_hook(hook(f'enc{i}')))
        hooks.append(d.register_forward_hook(hook(f'down{i}')))
    hooks.append(m.middle_blks.register_forward_hook(hook('middle')))
    for i, (u, d) in enumerate(zip(m.ups, m.decoders)):
        hooks.append(u.register_forward_hook(hook(f'up{i}')))
        hooks.append(d.register_forward_hook(hook(f'dec{i}')))
    if isinstance(m.post, nn.Conv2d):
        hooks.append(m.post.register_forward_hook(hook('head_post')))

    with torch.inference_mode():
        out = m(x).float()
    for h in hooks:
        h.remove()
    trace.append(('OUTPUT raw', out))
    trace.append(('OUTPUT clipped', out.clamp(0, 1)))

    # ---- report ----
    print('%-14s %10s %12s %14s %10s' % ('stage', 'shape', 'rel_checker', 'nyq/med', 'std'))
    print('-' * 66)
    prev = None
    flagged = None
    for name, t in trace:
        rc, nr = rel_checker(t), nyq_ratio(t)
        res = f'{t.shape[1]}c@{t.shape[-1]}'
        mark = ''
        if prev is not None and not np.isnan(rc) and not np.isnan(prev):
            if rc > prev * 3 and rc > 0.02:
                mark = '  <== JUMPS HERE'
                if flagged is None:
                    flagged = name
        print('%-14s %10s %12.4f %14.1f %10.4f%s' % (name, res, rc, nr, t.float().std(), mark))
        if not np.isnan(rc):
            prev = rc

    print()
    print(f'raw output range: [{out.min():.4f}, {out.max():.4f}]  '
          f'(values outside [0,1] are what clipping then amplifies)')
    if flagged:
        print(f'first large jump in relative checkerboard content: {flagged}')

    # ---- picture: mean over channels at each stage, upsampled to a common size ----
    panels = []
    for name, t in trace:
        v = t.float().mean(1)[0]                       # (H,W) mean over channels
        v = (v - v.min()) / (v.max() - v.min()).clamp_min(1e-8)
        img = Image.fromarray((v.cpu().numpy() * 255).astype(np.uint8))
        img = img.resize((192, 192), Image.NEAREST)
        canvas = Image.new('L', (192, 214), 255)
        canvas.paste(img, (0, 22))
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('arial.ttf', 13)
        except Exception:
            font = ImageFont.load_default()
        d.text((3, 4), f'{name} {t.shape[1]}c@{t.shape[-1]}', fill=0, font=font)
        panels.append(canvas)

    cols = 6
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new('L', (cols * 196, rows * 218), 255)
    for i, pn in enumerate(panels):
        grid.paste(pn, ((i % cols) * 196, (i // cols) * 218))
    grid.convert('RGB').save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
