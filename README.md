# SemiCon-ML

A NAFNet-derived image restoration model built for the **SemiCon AI Hackathon (KLA Problem Statement 01)**.

The task: recover a clean **256×256** image from a **128×128** input that has been degraded by three effects applied jointly, in random combination and order — **multiplicative speckle noise**, **additive Gaussian noise**, and **2× downsampling**. The model does denoising and super-resolution in a single forward pass.

## Result

| metric | value |
|---|---|
| PSNR | **29.33 dB** |
| SSIM | **0.776** |
| LPIPS | **0.267** |

(320-image held-out validation split, `notch` post-processing, no source-image leakage between train/val — see [Data](#data--degradation-model) below.)

## Architecture

`nafnet_sr.py` — a NAFNet body (config `m`, width 32, **29.16M params**), verified byte-identical to the official [NAFNet](https://github.com/megvii-research/NAFNet) implementation, with the input/output ends adapted for joint denoise + 2× super-resolution:

- **Input transform: `log`** — a variance-stabilising log transform, since the dominant noise (speckle) is multiplicative rather than additive; this makes it behave approximately additively before the network sees it.
- **Upsampling head: `resize_conv`** — resize-then-convolve at the final SR head, chosen over `pixelshuffle` because it avoids a period-2 (Nyquist) checkerboard artifact that pixelshuffle heads develop during training.
- **Internal decoder upsampling: `pixelshuffle`** (NAFNet's original, kept as-is) — the same resize-then-convolve trick was tested at the *internal* decoder stages too and measured **9 dB worse** (low-pass filtering destroys high-frequency content across 4 decoder stages); it's only safe at the single final head.
- **Global residual: `bicubic`** — a bicubic-upsampled skip connection from the noisy input, so the network only has to learn the correction, not full reconstruction from scratch.
- **Post-processing: deterministic FFT notch filter** (`evaluate.py`, `--post notch`, default) — suppresses residual period-2/period-4 spectral artifacts at their known exact frequency bins. Costs nothing at inference and works on any checkpoint.

Full architecture rationale and every option that was tried and rejected (with numbers) is documented in [CHECKPOINT.md](CHECKPOINT.md).

## Loss

```
L = 1.0   · Charbonnier (pixel)
  + 0.10  · L1 on FFT magnitude
  + 0.15  · L1 on Sobel gradient        (fixes denoiser-typical blur)
  + SSIM / MS-SSIM term (optional, --ms-ssim)
  + optional artifact-frequency penalty (--w-artifact, off in the shipped run)
```

Full loss definitions in `src/losses.py`.

## Data & degradation model

The forward degradation model (GT → NoisyLR) was **fitted from the provided 3,200 training pairs**, not assumed:

- **Downsampling kernel**: solved by direct least-squares over 26.2M equations (`tools/fit_kernel.py`) rather than picked from candidates — the fitted 8×8 kernel beats box/lanczos/bicubic on both residual MSE and Laplacian correlation.
- **Noise model**: `Var(residual) = σs²·x² + σg²` (median R² = 0.975) — confirms multiplicative speckle (present in 99.7% of images) plus additive Gaussian noise (absent in 29% of images, matching the brief's "random subset" description). A gamma-distributed multiplier is used to match the measured heavier-tailed, right-skewed residual (kurtosis 3.56, skew +0.34) more accurately than a Gaussian multiplier would.

This fitted simulator (`src/degradation.py`) is used to synthesize additional training pairs on the fly (`--synth-ratio`), on top of the provided dataset.

**Train/val split is source-aware** (`src/data.py`): the dataset ships 4 crops per source image, and a naive random split leaked 100% of validation images into the same source group as a training image. The split is done on `source_id = index // group_size` instead — this is verified with a leakage check on every data load and must always read 0.

## Usage

```bash
# Train (recommended settings)
python train.py --name final --config m --head resize_conv \
    --residual bicubic --input-transform log \
    --iters 60000 --batch 16

# Run inference / generate submission outputs
python evaluate.py <input_dir> <output_dir> --weights runs/final/best.pt
```

See `python train.py --help` and `python evaluate.py --help` for the full set of flags (loss weights, head/residual/upsample-mode ablation switches, self-ensembling, etc).

## Repository layout

```
nafnet_sr.py        model definition (NAFNet body + adapted SR ends)
train.py            training entry point
evaluate.py         inference / submission script (notch post-processing)
src/
  data.py           source-aware GPU-resident dataset
  degradation.py    fitted forward-model noise/downsampling simulator
  losses.py         composite training loss
  metrics.py        PSNR/SSIM (skimage-matched) + post-processing
tools/               diagnostic & ablation scripts (kernel fitting, artifact
                     quality checks, head/upsample comparisons, benchmarks)
runs/                training run checkpoints and per-run metrics
CHECKPOINT.md        detailed rollback reference: exact settings, every
                     experiment that was tried and its measured result
```
