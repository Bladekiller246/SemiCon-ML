"""Composite restoration loss.

    L = w_pix  * Charbonnier(pred, gt)
      + w_fft  * L1(rFFT(pred), rFFT(gt))
      + w_grad * L1(Sobel(pred), Sobel(gt))
      + w_ssim * (1 - SSIM(pred, gt))

Why each term (with MEASURED sensitivity, not assumed):

  Charbonnier  smooth-L1 in disguise; L1-family losses beat L2 for restoration
               because L2 regresses to the conditional mean and comes out
               blurry -- exactly the failure mode that destroys the fine
               structural detail this task is measured on.

  FFT L1       plain, UNweighted L1 over the complex spectrum. A radial
               high-frequency weighting was tried and measured against two
               synthetic softening probes (mild gaussian blur, and uniform
               regression-to-mean shrinkage of AC content) applied to a real
               GT image. It made the term LESS sensitive to both, not more:
               these SEM images concentrate 98.5% of spectral energy inside
               the inner 10% of the frequency radius, so discounting
               low-frequency bins throws away real error signal that a mild
               blur still perturbs there -- it does not free up "wasted"
               weight the way it would for images with a flatter spectrum.
               Measured slope ratio to the pixel loss, blur sigma=0.3: flat
               0.209x, radially-weighted 0.128x. Kept flat.

  Gradient L1  the term that actually does the work. Sobel gradients shrink
               wherever an image is blurred or regressed toward its local
               mean, by construction, and this measures that directly.
               Measured slope ratio to the pixel loss: 0.99x for mild blur,
               1.91x for 2% AC-content shrinkage -- both several times
               stronger than either FFT variant (0.13-0.32x). This is the
               real fix for denoiser-induced softness, not the FFT term.

  SSIM         SSIM is a reported metric, so optimising a differentiable form
               of it directly is the obvious move.

See tools/probe_smoothness_loss.py for the measurements above -- re-run it
before trusting these numbers on a different dataset; the "which term reacts
to softening" answer depends on how a given image set's energy is distributed
across frequencies, which is not something to assume twice.

NOTHING adversarial. Hallucinated detail on a semiconductor inspection image is
worse than no detail -- a GAN that invents a plausible edge where a real defect
was is actively harmful, so the whole GAN family is off the table by design.
The gradient and FFT terms push the model to reconstruct real high-frequency
content it has evidence for (from the input/receptive field); they do not
reward inventing detail that isn't supported, the way an adversarial loss would.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import ssim as ssim_metric


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def fft_l1(pred, target):
    """L1 between complex spectra, computed in fp32 (fft under autocast is
    unreliable in low precision). Deliberately unweighted -- see module
    docstring for why a radial high-frequency weighting was tried and
    measured to make this LESS sensitive to softening on this dataset."""
    with torch.autocast('cuda', enabled=False):
        p = torch.fft.rfft2(pred.float(), norm='ortho')
        t = torch.fft.rfft2(target.float(), norm='ortho')
        return (torch.view_as_real(p) - torch.view_as_real(t)).abs().mean()


_SOBEL_CACHE = {}


def _sobel_kernel(device, dtype):
    key = (device, dtype)
    if key not in _SOBEL_CACHE:
        kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                          device=device, dtype=dtype)
        _SOBEL_CACHE[key] = torch.stack([kx, kx.t()]).unsqueeze(1)  # (2,1,3,3)
    return _SOBEL_CACHE[key]


def gradient_l1(pred, target):
    """L1 between Sobel gradients. A blurred or regression-to-mean-shrunk
    region has smaller gradient magnitude than a sharp one by construction,
    so this is a direct, interpretable, and (measured) strong tax on
    smoothness -- cheap, one 3x3 depthwise conv."""
    with torch.autocast('cuda', enabled=False):
        pred, target = pred.float(), target.float()
        c = pred.shape[1]
        k = _sobel_kernel(pred.device, pred.dtype).repeat(c, 1, 1, 1)
        pg = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode='reflect'), k, groups=c)
        tg = F.conv2d(F.pad(target, (1, 1, 1, 1), mode='reflect'), k, groups=c)
        return (pg - tg).abs().mean()


_ARTIFACT_BIN_CACHE = {}


def _artifact_bins(H, W, device):
    """Exact 2-D frequency bins the two known upsampler artifacts occupy:
    period-2 (Nyquist checkerboard, PixelShuffle heads) and period-4 (diagonal
    pattern, bilinear resize_conv heads) -- plus their axis-aligned and
    conjugate-symmetric counterparts. See nafnet_sr.py's head docstrings for
    how each was measured and localised (tools/trace_stages.py)."""
    key = (H, W, str(device))
    if key not in _ARTIFACT_BIN_CACHE:
        n, q, nw, qw = H // 2, H // 4, W // 2, W // 4
        centres = [(n, nw), (q, W - qw), (H - q, qw), (q, qw), (H - q, W - qw),
                   (n, 0), (0, nw), (q, 0), (H - q, 0), (0, qw), (0, W - qw)]
        iy = torch.tensor([c[0] for c in centres], device=device)
        ix = torch.tensor([c[1] for c in centres], device=device)
        _ARTIFACT_BIN_CACHE[key] = (iy, ix)
    return _ARTIFACT_BIN_CACHE[key]


def artifact_freq_loss(pred, target):
    """One-sided penalty on EXCESS power at the two known upsampler-artifact
    frequencies, relative to how much power ground truth actually has there.

    WHY THIS TERM EXISTS: the broadband losses above (pixel/SSIM/gradient/FFT)
    were already active for a full 100k-iteration run and did not stop the
    period-4 pattern from reaching 14667x the ground-truth level at that exact
    frequency. One bin out of thousands of spectral samples contributes almost
    nothing to a broadband loss even when it is enormous relative to what
    should be there -- so give the network gradient signal AT that bin
    specifically, instead of hoping it falls out of everything else.

    Normalised by each image's own spectral median (detached) so the ratio is
    directly comparable to the period2/period4 diagnostics used throughout
    (1.0 = ground-truth level). log1p-compressed because the measured ratios
    span 3-5 orders of magnitude (GT ~2, broken models ~14000-116000) -- an
    uncompressed penalty of that size early in training would swamp every
    other loss term and could destabilise optimisation before the artifact has
    a chance to shrink gradually.

    One-sided (relu) rather than symmetric: only SUPPRESSING excess artifact
    power is wanted, not forcing pred to add power at these bins if it already
    has less than target.
    """
    with torch.autocast('cuda', enabled=False):
        H, W = pred.shape[-2:]
        iy, ix = _artifact_bins(H, W, pred.device)
        Pm = torch.fft.fft2(pred.float()).abs()
        Tm = torch.fft.fft2(target.float()).abs()
        med = Pm.flatten(2).median(dim=2).values.clamp_min(1e-8).detach()  # (B,C)
        p = torch.log1p(Pm[..., iy, ix] / med.unsqueeze(-1))
        t = torch.log1p(Tm[..., iy, ix] / med.unsqueeze(-1))
        return F.relu(p - t).mean()


def range_penalty(pred, lo=0.0, hi=1.0):
    """Penalise predictions outside [0,1].

    Ground truth is EXACTLY [0,1] on all 3200 images, so anything outside is
    definitionally wrong -- but the pixel loss only sees it as symmetric error
    and tolerates a ripple whose mean is correct. The overshoot matters far more
    than its size suggests: the final output carries a small Nyquist-coherent
    ripple, and clipping it to [0,1] AMPLIFIES that ripple ~35x into a visible
    dot grid. Measured per test image:

        000000  raw max 1.231  ->  clip amplifies artifact 15.6x
        000266  raw max 1.282  ->  clip amplifies artifact 21x
        000133  raw max 0.966  ->  NO overshoot, NO amplification (1.0x)
        000399  raw max 1.007  ->  NO overshoot, NO amplification (1.0x)

    Squared hinge, so it is zero inside the range and gives smooth gradient
    pressure outside it (unlike a clamp, which has zero gradient there and so
    cannot pull values back).
    """
    return (F.relu(pred - hi) ** 2 + F.relu(lo - pred) ** 2).mean()


_MSSSIM_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ms_ssim_loss(pred, target, scales=_MSSSIM_W, data_range=1.0):
    """1 - MS-SSIM. Multi-scale variant of the SSIM term.

    Single-scale SSIM only sees structure at one resolution (an 11x11 gaussian
    window). MS-SSIM evaluates the same statistic across a pyramid, so errors in
    coarse structure are penalised as well as fine. SSIM is the metric we are
    measurably behind on, and it is the metric KLA scores, so optimise the
    multi-scale form of it directly rather than a one-scale proxy.

    Uses the standard Wang et al. weights: contrast/structure at every level,
    luminance only at the coarsest.
    """
    from .metrics import _window
    x, y = pred.float(), target.float()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mcs = []
    for i, _ in enumerate(scales):
        if min(x.shape[-2:]) < 16:
            scales = scales[:i]
            break
        c = x.shape[1]
        w = _window(1.5, x.device, x.dtype, c)
        mu1, mu2 = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
        m1s, m2s, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s11 = F.conv2d(x * x, w, groups=c) - m1s
        s22 = F.conv2d(y * y, w, groups=c) - m2s
        s12 = F.conv2d(x * y, w, groups=c) - m12
        cs = ((2 * s12 + c2) / (s11 + s22 + c2)).clamp_min(1e-6)
        mcs.append(cs.flatten(1).mean(1))
        if i < len(scales) - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    lum = ((2 * m12 + c1) / (m1s + m2s + c1)).clamp_min(1e-6).flatten(1).mean(1)
    out = lum ** scales[-1]
    for cs, wgt in zip(mcs, scales):
        out = out * cs ** wgt
    return (1.0 - out).mean()


class RestorationLoss(nn.Module):
    def __init__(self, w_pix=1.0, w_fft=0.05, w_grad=0.15, w_ssim=0.10,
                 w_range=0.0, w_artifact=0.0, ms_ssim=False, eps=1e-3):
        super().__init__()
        self.w_pix, self.w_fft, self.w_grad, self.w_ssim = w_pix, w_fft, w_grad, w_ssim
        self.w_range = w_range
        self.w_artifact = w_artifact
        self.ms_ssim = ms_ssim
        self.eps = eps

    def forward(self, pred, target):
        parts = {}
        total = pred.new_zeros(())

        if self.w_pix:
            parts['pix'] = charbonnier(pred, target, self.eps)
            total = total + self.w_pix * parts['pix']
        if self.w_fft:
            parts['fft'] = fft_l1(pred, target)
            total = total + self.w_fft * parts['fft']
        if self.w_grad:
            parts['grad'] = gradient_l1(pred, target)
            total = total + self.w_grad * parts['grad']
        if self.w_ssim:
            # do not clamp: clamping kills the gradient wherever the model
            # overshoots, which is precisely where it needs correcting
            parts['ssim'] = (ms_ssim_loss(pred, target) if self.ms_ssim
                             else 1.0 - ssim_metric(pred, target, clamp=False).mean())
            total = total + self.w_ssim * parts['ssim']
        if self.w_range:
            parts['range'] = range_penalty(pred)
            total = total + self.w_range * parts['range']
        if self.w_artifact:
            parts['artifact'] = artifact_freq_loss(pred, target)
            total = total + self.w_artifact * parts['artifact']

        parts['total'] = total
        return total, parts
