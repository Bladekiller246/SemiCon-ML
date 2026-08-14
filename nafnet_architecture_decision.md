# Architecture Decision: NAFNet

## Problem Recap

Restoring semiconductor SEM (scanning electron microscope) imagery degraded by three
combined corruptions applied **in no particular order, on random subsets** of the dataset:

1. Gaussian blur
2. Speckle (multiplicative, signal-dependent) noise
3. Downsampling (fixed x2 factor)

Some training images have only one degradation applied, others two or more — the model must
be robust to an unknown combination and order of degradations per image, not a fixed pipeline.

Hard requirements for the solution:
- Pretrained weights may be used as an initialization point, with fine-tuning on the provided
  dataset — not required to train fully from scratch.
- Must reuse an existing open-source architecture rather than design a new one.

## What We Learned From the Training Data

- **Domain confirmed as semiconductor SEM imagery** — likely a via/contact/pillar structure.
  Fine line/wire features visible in the images are likely real, measurable structural
  detail (critical dimensions, edges, defects) rather than incidental texture. This makes
  **hallucinated detail actively harmful**, not just undesirable — ruling out adversarial
  (GAN) sharpening approaches.
- **Images are grayscale (single channel)**, not RGB — every candidate architecture needs its
  input/output stem adapted regardless of choice.
- **Noise is severe** in sampled examples (std comparable in magnitude to the underlying
  structural signal), to the point of making content hard to identify by eye. Noise appeared
  to scale with local pixel intensity — consistent with a shot-noise / speckle model rather
  than fixed-variance additive noise, matching how SEM images are physically formed (noise is
  tied to detected-electron counts per pixel).
- **Downsampling is a fixed x2 factor** (e.g. 512→256) — never variable — simplifying the
  resolution-change requirement to something a lightweight upsampling head (or a simple
  pre-upsample of the input) can handle.
- **Degradation order and count are randomized per image** — ruling out any fixed-order
  cascade (e.g. denoise → deblur → SR) in favor of a single joint model trained on
  synthetically randomized degradation combinations.

## Why NAFNet

**1. Efficiency is a hard requirement, and NAFNet is the strongest fit for it.**
NAFNet ("Nonlinear Activation Free Network") is explicitly built to show that neither
self-attention nor nonlinear activations are necessary for SOTA restoration performance. It
is a pure convolutional encoder-decoder, cheaper per image than any attention-based
alternative — a direct, measurable advantage for inference throughput scoring.

**2. Its pretrained weights are trained on real, signal-dependent sensor noise (SIDD), not
synthetic additive Gaussian noise.** This is architecturally and empirically closer to our
speckle/shot-noise problem than degradation models built around additive Gaussian corruption,
which is what most classical super-resolution pretrained weights (e.g. BSRGAN, Real-ESRGAN)
assume by default.

**3. It satisfies both hard requirements directly.** NAFNet has official open-source
pretrained checkpoints (SIDD for denoising, GoPro for deblurring) that can be used as an
initialization point, with full fine-tuning afterward on our synthetic degradation data. The
architecture itself is reused unmodified aside from standard input/output adaptation (RGB→
grayscale stem, small upsampling head for the x2 factor) — not a bespoke design.

**4. It is pixel-fidelity oriented, not adversarial.** Trained with L1/PSNR-style objectives
rather than a GAN loss, which fits the requirement that real structural features not be
hallucinated or artistically embellished.

## Why Other Models Were Removed From Consideration

- **BSRGAN / Real-ESRGAN (GAN-based SR):** closest published match to a combined
  blur+noise+downsample degradation pipeline, and easiest to get running quickly — but built
  around additive Gaussian noise (a swap-out, not a natural fit for speckle), and their GAN
  variants risk hallucinating detail on real structural features, which is disqualifying
  given the semiconductor context. Their PSNR-oriented, non-GAN variant was considered
  passable but still less efficient and less noise-model-appropriate than NAFNet.
- **SwinIR:** pretrained checkpoints are split by task (denoise-only, deblur-only, SR-only),
  each trained assuming a single isolated degradation. This does not compose well against our
  randomized, combined, order-independent degradation setup without significant joint
  fine-tuning — removed in favor of architectures with an existing "combined restoration"
  training philosophy.
- **Diffusion-based methods (DiffBIR, StableSR, DDRM/DPS):** strong generalization and
  quality potential, but multi-step sampling directly conflicts with the efficiency/throughput
  requirement. Considered too high-risk for the compute/time budget available.
- **Building a custom architecture:** explicitly excluded by the hard requirement to reuse an
  existing open-source architecture.

## Backup: Restormer

**Restormer** is the designated fallback if NAFNet underperforms. It satisfies the same two
hard requirements (open-source architecture, official pretrained weights spanning denoising /
deblurring / deraining tasks) and is architecturally close to NAFNet's use case, but trades
some efficiency for a larger effective receptive field via channel-wise self-attention.

**When to escalate to Restormer:** if NAFNet's more local convolutional receptive field
struggles to separate fine structural detail (thin wires/lines) from heavy speckle noise —
plausible given how severe the noise appeared in sampled training images. Restormer's global
context per pixel may recover structure more reliably in these cases, at a moderate (not
extreme) efficiency cost relative to NAFNet — still meaningfully cheaper than
attention-heavy or diffusion-based alternatives.

**Recommended validation step:** train both NAFNet and Restormer on a small subset of the
data first and compare denoising quality specifically on the heaviest-noise training examples
before committing fully to one architecture.
