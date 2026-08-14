"""GPU forward-model simulator: clean GT (256) -> degraded NoisyLR (128).

Everything here is fitted from the 3200 provided pairs, not guessed.
See tools/analyze_degradation.py for the fit; results:

  downsampler   2x2 box average          (MSE 1.35e-3 vs 1.79e-3 for the best
                                          gaussian+stride variant, 2.02e-3 for
                                          naive subsampling)
  noise model   Var(residual) = ss^2 * x^2 + sg^2 ,  median R^2 = 0.975
                -> variance scales with x^2, i.e. MULTIPLICATIVE speckle.
                   NOT Poisson/shot noise (that would give Var ∝ x).
  speckle ss    p05=0.131  med=0.167  p95=0.212   present in 99.7% of images
  gaussian sg   p05=0.000  med=0.020  p95=0.074   ABSENT in 29% of images
                -> this is the "random subset" the brief describes
  ordering      speckle-applied-at-HR fits marginally better than at-LR
                (R^2 0.9760 vs 0.9727, wins on 67% of images)
  residual      kurtosis 3.56, skew +0.34 -> slightly heavier-tailed and
                right-skewed than gaussian; a gamma multiplier reproduces this
                better, so we mix it in.

  kernel        SOLVED by direct least-squares estimation rather than by trying
                candidates -- see tools/fit_kernel.py. Fitting an 8x8 HR kernel
                over 26.2M equations gives, on held-out images:

                    kernel      residual MSE     laplacian corr
                    box2x2       7.707e-3            -0.103
                    lanczos2     7.946e-3            -0.089
                    bicubic      7.648e-3            +0.093
                    FITTED       7.611e-3            +0.004

                The fitted kernel wins on BOTH criteria at once, which no
                hand-picked candidate did. It is separable (2nd/1st singular
                value = 0.009), symmetric, sums to 1.000 (DC preserving), and
                carries -0.43 of negative mass -- i.e. it SHARPENS, which is
                why every purely-averaging candidate left a negative Laplacian
                correlation. Its 1-D profile is

                    [-0.004, +0.022, -0.091, +0.562, +0.576, -0.084, +0.022, -0.003]

                Alternating signs decaying out from two large central taps is
                the signature of a resample composed with a deconvolution --
                consistent with GT and NoisyLR descending from a common
                higher-resolution original, with this kernel being the
                effective GT->NoisyLR operator. Either way it is now measured,
                and the residual under it is pure noise: its mean is flat at
                ~0 across every intensity bin (so no unmodelled nonlinearity)
                while its std rises with intensity exactly as speckle predicts.

The fitted kernel is stored in fitted_kernel.npy and used as the primary
downsampler. The others are kept at lower weight purely for OOD robustness --
the test set comes from different sources and may well have been resampled
differently.

TRAIN_CFG deliberately widens every range past the fitted p05/p95. The test set
contains out-of-distribution samples; a model that has only ever seen the exact
fitted noise band is the one that falls over there.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
@dataclass
class DegradeCfg:
    speckle_min: float = 0.04
    speckle_max: float = 0.32
    gauss_min: float = 0.0
    gauss_max: float = 0.13
    p_gauss: float = 0.72          # fraction of images that get gaussian at all
    # sigma_g = min + (max-min) * u**gauss_pow, u ~ U(0,1). The real gaussian
    # distribution is right-skewed (p25 0.007, med 0.020, p75 0.038, p95 0.074),
    # not uniform. pow=2 reproduces it almost exactly -- u**2 has median 0.25
    # and p75 0.5625, giving 0.0185 / 0.0416 against the real 0.020 / 0.038.
    # Sampling uniformly instead recovered 0.030, a 50% overshoot.
    gauss_pow: float = 1.5         # mild skew for training (coverage still wanted)
    p_speckle_hr: float = 0.67     # apply speckle before downsampling
    p_gauss_hr: float = 0.35       # apply gaussian before downsampling
    # Noise applied at HR is attenuated by the downsampling kernel (box pooling
    # averages 4 independent samples -> variance/4 -> sigma/2). All sigmas above
    # are quoted in LR space, because that is where they were fitted, so noise
    # applied at HR is pre-scaled by 1/attenuation to land on the same effective
    # LR level. Without this every speckle-at-HR sample was degraded at HALF
    # strength. The attenuation is MEASURED per kernel (see _attenuation) rather
    # than hardcoded to the box factor, since the fitted kernel sharpens and so
    # attenuates noise much less than box does.
    # ALWAYS gamma. Speckle is physically multi-look: the multiplier is
    # Gamma(L, 1/L) with L the number of looks, giving mean 1 and std 1/sqrt(L).
    # Our fitted sigma range 0.131-0.212 corresponds to L = 22-58. A gaussian
    # multiplier has the right variance but the wrong shape -- the measured
    # residual skew is +0.34 and kurtosis 3.56, both of which gamma reproduces
    # and gaussian does not. Independently confirmed by a second analysis of
    # this dataset that identified multi-look gamma speckle directly.
    p_gamma_speckle: float = 1.0
    # Extra source blur before downsampling. This is the secondary lever
    # against denoiser-induced softness -- the primary fix is in the loss
    # (see src/losses.py: radially-weighted FFT + Sobel gradient terms), which
    # taxes smoothness in the objective itself. This augmentation makes sure
    # the model also gets PRACTICE at the "sharpen, don't smooth" regime,
    # since if every training pair is noisy-but-sharp, the model has no
    # examples where the correct answer is to reconstruct lost edges rather
    # than average out noise. Raised from 0.15/0.9 after that gap was flagged.
    p_blur: float = 0.30
    blur_sigma_max: float = 1.3
    # Downsampler mix (must sum to 1). `fitted` is the measured GT->NoisyLR
    # operator and carries most of the mass; the rest are kept so the model does
    # not memorise one resampling kernel, since the OOD test images come from
    # different sources and may have been resampled differently.
    p_fitted: float = 0.55
    p_box: float = 0.15
    p_bicubic: float = 0.12
    p_lanczos: float = 0.10
    p_gauss_stride: float = 0.08


#: widened ranges used for training
TRAIN_CFG = DegradeCfg()

#: exactly the fitted in-distribution band -- for the "matched" validation split
FIT_CFG = DegradeCfg(
    speckle_min=0.131, speckle_max=0.212,
    gauss_min=0.0, gauss_max=0.074, p_gauss=0.71, gauss_pow=2.0,
    p_gamma_speckle=0.0, p_blur=0.0,
    p_fitted=1.0, p_box=0.0, p_bicubic=0.0, p_lanczos=0.0, p_gauss_stride=0.0,
)

#: deliberately outside anything seen in training -- the OOD robustness probe
OOD_CFG = DegradeCfg(
    speckle_min=0.30, speckle_max=0.45,
    gauss_min=0.10, gauss_max=0.20, p_gauss=1.0, gauss_pow=1.0,
    p_gamma_speckle=0.5, p_blur=0.6, blur_sigma_max=1.4,
    p_fitted=0.0, p_box=0.2, p_bicubic=0.3, p_lanczos=0.2, p_gauss_stride=0.3,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rand(b, lo, hi, device, gen):
    return torch.rand(b, 1, 1, 1, device=device, generator=gen) * (hi - lo) + lo


def _bern(b, p, device, gen):
    return torch.rand(b, 1, 1, 1, device=device, generator=gen) < p


def _gaussian_kernel1d(sigma, device, dtype):
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(x, sigma):
    """Separable gaussian blur, reflect-padded."""
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def _speckle(x, sigma, use_gamma, gen):
    """Multiplicative noise with mean 1 and std `sigma`.

    gaussian branch: m = 1 + sigma*N(0,1)
    gamma branch:    m ~ Gamma(k, 1/k) with k = 1/sigma^2  -> mean 1, std sigma,
                     right-skewed, which matches the measured skew of +0.34.
    """
    g = 1.0 + sigma * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    if bool(use_gamma.any()):
        k = (1.0 / sigma.clamp_min(1e-3) ** 2).clamp(1.0, 1e5).float()
        gam = torch._standard_gamma(k.expand_as(x).contiguous()) / k
        g = torch.where(use_gamma, gam.to(x.dtype), g)
    return x * g


_LANCZOS_CACHE = {}


def _lanczos_taps(a=2, scale=2, device='cpu', dtype=torch.float32):
    """Separable Lanczos-`a` taps for integer decimation by `scale`."""
    key = (a, scale, device, dtype)
    if key in _LANCZOS_CACHE:
        return _LANCZOS_CACHE[key]
    n = 2 * a * scale
    offs = torch.arange(n, device=device, dtype=dtype) - (n / 2 - 0.5)
    t = offs / scale
    pi = torch.pi
    w = torch.where(
        t.abs() < 1e-8,
        torch.ones_like(t),
        torch.sinc(t) * torch.sinc(t / a) * (t.abs() < a).to(dtype),
    )
    w = w / w.sum()
    _LANCZOS_CACHE[key] = w
    return w


def lanczos_down2(x, a=2):
    """x: (B,C,H,W) -> (B,C,H/2,W/2) with a Lanczos-2 kernel (negative lobes,
    so it is mildly sharpening -- unlike box, which is purely averaging)."""
    w = _lanczos_taps(a, 2, x.device, x.dtype)
    n = w.numel()
    c = x.shape[1]
    lo, hi = n // 2 - 1, n // 2
    x = F.pad(x, (lo, hi, lo, hi), mode='reflect')
    x = F.conv2d(x, w.view(1, 1, 1, -1).expand(c, 1, 1, -1), stride=(1, 2), groups=c)
    x = F.conv2d(x, w.view(1, 1, -1, 1).expand(c, 1, -1, 1), stride=(2, 1), groups=c)
    return x


_FITTED = None


def fitted_down2(x):
    """The measured GT->NoisyLR operator (tools/fit_kernel.py)."""
    global _FITTED
    if _FITTED is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fitted_kernel.npy')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} missing -- run: python -u tools/fit_kernel.py')
        _FITTED = torch.from_numpy(np.load(p)).float()
    k = _FITTED.to(x.device, x.dtype)
    n = k.shape[-1]
    lo, hi = n // 2 - 1, n // 2 - 1
    c = x.shape[1]
    xp = F.pad(x, (lo, hi, lo, hi), mode='reflect')
    return F.conv2d(xp, k.view(1, 1, n, n).expand(c, 1, n, n), stride=2, groups=c)


def _pick_downsampler(cfg, gen, device):
    """-> (name, fn). One kernel per batch: cheap, and over thousands of steps
    the mix is the same as choosing per-sample."""
    r = torch.rand((), device=device, generator=gen).item()
    for name, p, fn in (
        ('fitted', cfg.p_fitted, fitted_down2),
        ('box', cfg.p_box, lambda t: F.avg_pool2d(t, 2)),
        ('bicubic', cfg.p_bicubic,
         lambda t: F.interpolate(t, scale_factor=0.5, mode='bicubic', align_corners=False)),
        ('lanczos', cfg.p_lanczos, lanczos_down2),
    ):
        if r < p:
            return name, fn
        r -= p
    sigma = 0.5 + 0.6 * torch.rand((), device=device, generator=gen).item()
    return f'gauss{sigma:.2f}', lambda t: gaussian_blur(t, sigma)[:, :, ::2, ::2]


_ATT_CACHE = {}


def _attenuation(name, fn, device):
    """How much this downsampler shrinks the std of white noise.

    Measured once per kernel instead of assuming the box factor of 0.5: the
    fitted kernel sharpens, so it attenuates noise far less, and using 0.5 for
    it would over-inject HR noise by ~30%.
    """
    key = (name, str(device))
    if key not in _ATT_CACHE:
        z = torch.randn(8, 1, 128, 128, device=device)
        _ATT_CACHE[key] = float(fn(z).std())
    return _ATT_CACHE[key]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def degrade(gt, cfg=TRAIN_CFG, gen=None):
    """gt: (B,1,256,256) float in [0,1] on GPU  ->  (B,1,128,128) degraded.

    Degradations are applied in a per-sample randomised order around the
    downsample, matching the brief's "do not read into the order of it".
    Output is intentionally NOT clipped: the real NoisyLR exceeds [0,1]
    (observed range roughly -0.10 .. 1.73) and that overshoot is a genuine cue
    about how much speckle is present.
    """
    b = gt.shape[0]
    device = gt.device

    ss = _rand(b, cfg.speckle_min, cfg.speckle_max, device, gen).to(gt.dtype)
    u = torch.rand(b, 1, 1, 1, device=device, generator=gen) ** cfg.gauss_pow
    sg = (cfg.gauss_min + (cfg.gauss_max - cfg.gauss_min) * u).to(gt.dtype)
    sg = sg * _bern(b, cfg.p_gauss, device, gen).to(gt.dtype)

    speckle_hr = _bern(b, cfg.p_speckle_hr, device, gen)
    gauss_hr = _bern(b, cfg.p_gauss_hr, device, gen)
    use_gamma = _bern(b, cfg.p_gamma_speckle, device, gen)

    x = gt

    # optional extra source blur (simulates a differently-focused instrument)
    if cfg.p_blur > 0 and torch.rand((), device=device, generator=gen).item() < cfg.p_blur:
        sigma = 0.3 + (cfg.blur_sigma_max - 0.3) * torch.rand((), device=device, generator=gen).item()
        x = gaussian_blur(x, sigma)

    # Pick the kernel first: HR noise has to be pre-scaled by this kernel's own
    # noise attenuation so it lands at the intended LR level after downsampling.
    ds_name, ds_fn = _pick_downsampler(cfg, gen, device)
    k = 1.0 / _attenuation(ds_name, ds_fn, device)

    # ---- HR stage (sigmas scaled up so downsampling brings them to target) ---
    x = torch.where(speckle_hr, _speckle(x, ss * k, use_gamma, gen), x)
    n = torch.randn(x.shape, device=device, dtype=x.dtype, generator=gen)
    x = torch.where(gauss_hr, x + (sg * k) * n, x)

    # ---- downsample ----
    x = ds_fn(x)

    # ---- LR stage (whatever was not applied at HR) ----
    x = torch.where(~speckle_hr, _speckle(x, ss, use_gamma, gen), x)
    n = torch.randn(x.shape, device=device, dtype=x.dtype, generator=gen)
    x = torch.where(~gauss_hr, x + sg * n, x)

    return x


def augment_gt(gt, gen=None, p_gamma=0.35):
    """Photometric jitter on the CLEAN image, applied before degradation.

    gt**gamma maps [0,1] -> [0,1] with 0 and 1 exactly fixed, so the output is
    still exactly min-max normalised -- which every one of the 3200 provided GT
    images is (verified: min==0.0 and max==1.0 on all 3200). Preserving that
    invariant is the whole reason we use gamma here rather than brightness or
    contrast shifts, which would break it.
    """
    if torch.rand((), device=gt.device, generator=gen).item() >= p_gamma:
        return gt
    b = gt.shape[0]
    g = _rand(b, 0.7, 1.45, gt.device, gen).to(gt.dtype)
    return gt.clamp_min(0).pow(g)


def dihedral(lr, hr, gen=None):
    """Same random element of the 8-fold dihedral group applied to both."""
    k = int(torch.randint(0, 4, (), device=lr.device, generator=gen).item())
    flip = torch.rand((), device=lr.device, generator=gen).item() < 0.5
    if k:
        lr = torch.rot90(lr, k, dims=(-2, -1))
        hr = torch.rot90(hr, k, dims=(-2, -1))
    if flip:
        lr = torch.flip(lr, dims=(-1,))
        hr = torch.flip(hr, dims=(-1,))
    return lr, hr
