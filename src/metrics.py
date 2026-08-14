"""PSNR / SSIM on GPU, matched to the reference implementations.

SSIM follows skimage.metrics.structural_similarity(gaussian_weights=True,
sigma=1.5, use_sample_covariance=False, data_range=1.0), i.e. the Wang et al.
formulation with an 11x11 gaussian window -- the same thing MATLAB's ssim and
every image-restoration paper report. Getting this right matters: a boxcar
window reads ~0.01-0.02 higher and would make our numbers non-comparable to
whatever KLA scores us with.
"""
import torch
import torch.nn.functional as F

_WIN_CACHE = {}


def _window(sigma, device, dtype, channels=1):
    key = (sigma, device, dtype, channels)
    if key in _WIN_CACHE:
        return _WIN_CACHE[key]
    r = int(3.5 * sigma + 0.5)              # skimage: win_size = 2*ceil(3.5*s)+1
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    k = k / k.sum()
    w = torch.outer(k, k).view(1, 1, k.numel(), k.numel()).expand(channels, 1, -1, -1).contiguous()
    _WIN_CACHE[key] = w
    return w


def psnr(pred, target, data_range=1.0, clamp=True):
    """Per-image PSNR in dB. pred/target: (B,C,H,W). Returns (B,)."""
    if clamp:
        pred = pred.clamp(0, data_range)
    mse = ((pred.float() - target.float()) ** 2).flatten(1).mean(1)
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))


def ssim(pred, target, data_range=1.0, sigma=1.5, clamp=True):
    """Per-image SSIM. pred/target: (B,C,H,W). Returns (B,)."""
    if clamp:
        pred = pred.clamp(0, data_range)
    pred = pred.float()
    target = target.float()
    c = pred.shape[1]
    w = _window(sigma, pred.device, pred.dtype, c)
    pad = w.shape[-1] // 2

    # skimage crops the border rather than padding; do the same by using
    # 'valid' convolution.
    mu1 = F.conv2d(pred, w, groups=c)
    mu2 = F.conv2d(target, w, groups=c)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    s11 = F.conv2d(pred * pred, w, groups=c) - mu1s
    s22 = F.conv2d(target * target, w, groups=c) - mu2s
    s12 = F.conv2d(pred * target, w, groups=c) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s11 + s22 + c2))
    return m.flatten(1).mean(1)


_SOBEL_M = {}


def _sobel(device, dtype):
    key = (device, dtype)
    if key not in _SOBEL_M:
        kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                          device=device, dtype=dtype)
        _SOBEL_M[key] = torch.stack([kx, kx.t()]).unsqueeze(1)
    return _SOBEL_M[key]


def gradient_ratio(pred, target):
    """mean|grad(pred)| / mean|grad(target)|, per image. Returns (B,).

    THE metric for "is the output too smooth". 1.0 means the prediction carries
    exactly as much edge energy as the ground truth; <1 means it is softer.
    PSNR will not tell you this -- a slightly blurred prediction can score well
    on PSNR precisely because blurring is the safe, error-minimising choice.
    """
    pred, target = pred.float(), target.float()
    c = pred.shape[1]
    k = _sobel(pred.device, pred.dtype).repeat(c, 1, 1, 1)
    pg = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode='reflect'), k, groups=c)
    tg = F.conv2d(F.pad(target, (1, 1, 1, 1), mode='reflect'), k, groups=c)
    return pg.abs().flatten(1).mean(1) / tg.abs().flatten(1).mean(1).clamp_min(1e-8)


def hf_energy_ratio(pred, target, cutoff=0.25):
    """Spectral energy above `cutoff` (fraction of Nyquist radius) in pred vs
    target. Complements gradient_ratio: same question in the frequency domain."""
    pred, target = pred.float(), target.float()
    P = torch.fft.rfft2(pred, norm='ortho').abs() ** 2
    T = torch.fft.rfft2(target, norm='ortho').abs() ** 2
    h, w = pred.shape[-2:]
    fy = torch.fft.fftfreq(h, device=pred.device)
    fx = torch.fft.rfftfreq(w, device=pred.device)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    m = (r / r.max()) > cutoff
    return P[..., m].flatten(1).sum(1) / T[..., m].flatten(1).sum(1).clamp_min(1e-12)


@torch.no_grad()
def evaluate_batches(model, batches, postproc=None, amp_dtype=torch.bfloat16,
                     device='cuda'):
    """Run model over an iterable of (lr, gt) and return mean PSNR/SSIM."""
    model.eval()
    ps, ss, n = 0.0, 0.0, 0
    use_amp = amp_dtype is not None and torch.device(device).type == 'cuda'
    for lr, gt in batches:
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype or torch.bfloat16, enabled=use_amp):
            out = model(lr.to(memory_format=torch.channels_last))
        out = out.float()
        if postproc is not None:
            out = postproc(out)
        ps += psnr(out, gt).sum().item()
        ss += ssim(out, gt).sum().item()
        n += lr.shape[0]
    model.train()
    return ps / n, ss / n


@torch.no_grad()
def evaluate_sharpness(model, batches, postproc=None, amp_dtype=torch.bfloat16,
                       device='cuda'):
    """Mean gradient_ratio / hf_energy_ratio over an iterable of (lr, gt)."""
    model.eval()
    gr, hf, n = 0.0, 0.0, 0
    use_amp = amp_dtype is not None and torch.device(device).type == 'cuda'
    for lr, gt in batches:
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype or torch.bfloat16, enabled=use_amp):
            out = model(lr.to(memory_format=torch.channels_last))
        out = out.float()
        if postproc is not None:
            out = postproc(out)
        gr += gradient_ratio(out, gt).sum().item()
        hf += hf_energy_ratio(out, gt).sum().item()
        n += lr.shape[0]
    model.train()
    return gr / n, hf / n


# --------------------------------------------------------------------------- #
# Output post-processing
# --------------------------------------------------------------------------- #
def minmax_norm(x, eps=1e-6):
    """Stretch each image to exactly [0,1].

    Justified by measurement, not by hope: all 3200 provided GT images have
    min exactly 0.0 and max exactly 1.0, and the brief states "GT is always
    [0,1]". Matching that normalisation exactly is free accuracy -- but it is
    also the one post-process that a single outlier pixel can wreck, so
    `robust_minmax_norm` exists and train.py reports all three variants so the
    choice is made on validation numbers rather than on this argument.
    """
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(eps)


def robust_minmax_norm(x, q=0.001, eps=1e-6):
    """Percentile version: stretch on the q / 1-q quantiles, then clip."""
    flat = x.flatten(1)
    lo = torch.quantile(flat, q, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 1 - q, dim=1).view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(eps)).clamp(0, 1)


def clip_norm(x):
    return x.clamp(0, 1)


POSTPROC = {'clip': clip_norm, 'minmax': minmax_norm, 'robust': robust_minmax_norm}
