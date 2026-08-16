# SemiCon-ML — NAFNet-SR

A NAFNet-derived image restoration model built for the **SemiCon AI Hackathon (KLA Problem Statement 01)**.

**The task:** recover a clean **256×256** image from a **128×128** input degraded by three effects applied jointly, in random combination and order — **multiplicative speckle noise**, **additive Gaussian noise**, and **2× downsampling**. The model performs denoising and super-resolution in a single forward pass.

> **For teammates building the deck:** every number below is measured and reproducible — no estimates. Section 12 maps these directly onto a suggested 9-slide structure. Figures ready to drop into slides are listed in Section 11.

---

## 0. Quick start — clone and run inference

**Requirements:** Python 3.10+ and [Git LFS](https://git-lfs.com). A CUDA GPU is optional — the script falls back to CPU automatically.

```bash
# 1. Install Git LFS first, or the model weights arrive as a text pointer, not a model
git lfs install

# 2. Clone (LFS files download automatically once LFS is installed)
git clone https://github.com/Bladekiller246/SemiCon-ML.git
cd SemiCon-ML

# 3. If you cloned BEFORE installing Git LFS, fetch the real weights now
git lfs pull

# 4. Install dependencies
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# NOTE: requirements.txt pins torch 2.13.0+cu126 (CUDA 12.6). The "+cu126"
# builds are NOT on PyPI, so the PyTorch index must be added:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# CPU-only machine? Install plain torch instead, then the rest:
#   pip install torch numpy pillow
# Inference falls back to CPU automatically.

# 5. Verify the model loads (no input data needed)
python evaluate.py --self-test

# 6. Run inference
python evaluate.py <input_dir> <output_dir>
```

Only `torch`, `numpy` and `pillow` are needed for **inference**. The remaining
packages (`lpips`, `torchvision`, `scipy`, `tqdm`) are used solely by the
metric/reporting tools in `tools/`.

### Confirming it worked

`--self-test` must print **`config=m`**. If it prints `config=m (untrained)` the weights did not load — run `git lfs pull` and check the file size:

```bash
du -h weights/best.pt       # must be ~446M, not ~130 bytes (an LFS pointer)
```

Expected self-test output (the printed range varies — the self-test feeds random data):

```
[self-test] config=m (4, 1, 128, 128) -> (4, 1, 256, 256)  range [0.000, 0.999]  OK
```

### Input / output contract

| | format |
|---|---|
| **Input** | `.npy` float32, `H×W` grayscale (128×128 for this task). `.png/.tif/.bmp` also accepted as a fallback. |
| **Output** | `.npy` float32, `2H×2W`, values in `[0, 1]`, **same basename as the input** |

`evaluate.py` needs no manual edits — it resolves `weights/best.pt` relative to the script (not the working directory), accepts either positional or flagged paths, and never writes outside `output_dir`.

```bash
python evaluate.py Test_NoisyLR/NoisyLR outputs_final          # positional
python evaluate.py --input_dir <in> --output_dir <out>         # flagged
```

Pre-generated outputs for the provided test set are committed in **`outputs_final/`** (400 images).

---

## 1. Headline results

Measured on the **320-image held-out validation split** (source-aware, leakage-verified), `notch` post-processing, `data_range=1.0`.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×2 *(no-model floor)* | 22.639 | 0.4933 | 0.4631 |
| **NAFNet-SR (ours)** | **29.333** | **0.7764** | **0.2669** |
| *Perfect denoise + bicubic up (ceiling)* | *32.220* | *0.8661* | — |

**+6.69 dB over the no-model baseline.** The ceiling is what a model would score by removing *all* noise and doing nothing clever about resolution — we capture **62% of the total available gain** over baseline.

**Model:** NAFNet config `m`, width 32, **29.16M parameters**, checkpoint `weights/best.pt` (step 35,999).

### Inference speed (scored metric)

| | ms / image |
|---|---|
| Model time | **12.64** |
| End-to-end incl. I/O | **14.30** |

400 images, batch 64, fp16, `channels_last`, RTX 4060 Laptop (8 GB). Self-ensembling is available (`--self-ensemble`) but **off by default** — 8× the cost for a few hundredths of a dB, and throughput is scored.

---

## 2. Architecture

`nafnet_sr.py` — a NAFNet body verified byte-identical to the official [NAFNet](https://github.com/megvii-research/NAFNet) implementation, with input/output ends adapted for joint denoise + 2× SR.

| Component | Choice | Why (measured) |
|---|---|---|
| Input transform | `log` | Variance-stabilising. Dominant noise is *multiplicative* speckle; log makes it behave approximately additively before the network sees it. |
| Final SR head | `resize_conv` | **+0.59 dB and 6000× less checkerboard** vs `pixelshuffle`. |
| Internal decoder upsampling | `pixelshuffle` | Resize-conv tested here too and measured **9 dB worse** (25.189 → 16.098 dB). See below. |
| Global residual | `bicubic` | Network learns only the correction, not full reconstruction. |
| Post-processing | FFT `notch` | Removes period-2/period-4 artifacts at known exact bins. **+1.35 dB, free at inference.** |

### The one counter-intuitive finding worth a slide

Resize-then-convolve **fixes** checkerboard artifacts at the final head, but is **catastrophic** at the internal decoder stages:

| `up_mode` | PSNR | SSIM |
|---|---|---|
| `pixelshuffle` (kept) | **25.189** | **0.6114** |
| `resize` | 16.098 | 0.3769 |

**Why:** PixelShuffle is information-*preserving* — a pure reshape trading channels for space. Bilinear interpolation is a **low-pass filter**; applied at all four decoder stages it progressively destroys the exact high-frequency content the task exists to reconstruct. The sub-pixel freedom that lets PixelShuffle emit a checkerboard is the *same* mechanism it uses to represent fine detail.

---

## 3. Loss function

```
L = 1.00 · Charbonnier (pixel)          smooth L1; L1-family avoids the blur
                                        that L2's regression-to-mean produces
  + 0.10 · L1 on FFT spectrum           global frequency-content match
  + 0.15 · L1 on Sobel gradient         direct, measured tax on blurring
  + 0.00 · SSIM                         (disabled in shipped run — see §7)
```

**Deliberately no adversarial/GAN term.** Hallucinated detail on an inspection image is worse than no detail — a GAN that invents a plausible edge where a real defect was is actively harmful. The gradient and FFT terms push the model to *recover* detail it has evidence for; they do not reward *inventing* it.

**Measured term sensitivity** (slope ratio vs pixel loss, on synthetic softening probes):

| term | mild blur | 2% AC shrinkage |
|---|---|---|
| Gradient L1 | 0.99× | **1.91×** |
| FFT L1 (flat) | 0.21× | 0.32× |
| FFT L1 (radially weighted) | 0.13× | — |

The gradient term does the real anti-blur work. Radial HF weighting was tried and made the FFT term **less** sensitive — these images concentrate 98.5% of spectral energy in the inner 10% of the frequency radius, so discounting low frequencies throws away real error signal.

---

## 4. Data & fitted degradation model

The forward model (GT → NoisyLR) was **fitted from the 3,200 provided pairs**, not assumed.

- **Downsampling kernel** — solved by direct least-squares over **26.2M equations** (`tools/fit_kernel.py`). The fitted 8×8 kernel beats box / lanczos / bicubic on both residual MSE and Laplacian correlation.
- **Noise model** — `Var(residual) = σs²·x² + σg²`, median **R² = 0.975**. Variance scales with x², confirming **multiplicative speckle** (not Poisson shot noise, which would give Var ∝ x).
  - speckle σs: p05 0.131, median 0.167, p95 0.212 — present in **99.7%** of images
  - gaussian σg: p05 0.000, median 0.020, p95 0.074 — **absent in 29%** of images, matching the brief's "random subset"
  - Gamma-distributed multiplier used to match measured right-skew (kurtosis 3.56, skew +0.34)

This simulator (`src/degradation.py`) synthesizes additional pairs on the fly — `--synth-ratio 0.5` means **half of every training batch** is freshly degraded, giving effectively unlimited augmentation.

### Train/val split — a bug worth showing

The dataset ships **4 crops per source image**. A naive random split leaked **100% of validation images** into the same source group as a training image, inflating every reported number.

Fixed by splitting on `source_id = index // 4`. A leakage check runs on every data load and **must read 0**:

```
[data] source-aware split: 80 val groups (320 images), leakage check = 0 (must be 0)
```

Crops within a group have median content similarity **0.626** (39.7% above 0.8) versus **−0.019** for random pairs — the leak was real, not theoretical.

---

## 5. The artifact investigation

The model developed a visible periodic dot grid. Traced end-to-end (`tools/trace_stages.py`):

1. Raw model output is **clean** (Nyquist coherence 0.0) but overshoots [0,1] — 1.10% of pixels above 1.0
2. Hard-clipping to [0,1] **rectifies** the tiny residual ripple asymmetrically
3. This **amplifies** it 15–35× into a visible grid — coherence 0.0 → 744

| post-process | PSNR | SSIM | period2 | period4 |
|---|---|---|---|---|
| `clip` | 25.565 | 0.6111 | 1043.42 | 25356.61 |
| **`notch`** (shipped) | **27.322** | **0.7028** | **0.88** | **0.35** |
| *ground truth* | — | — | *0.15* | *2.25* |

*(measured on `runs/main`, where the artifact was strongest)*

The notch drives both artifacts **below ground-truth level** while *gaining* accuracy — it isn't a trade.

---

## 6. Robustness / generalisation

### Out-of-distribution noise types (Kaggle Extra_DataSet — natural photos, never seen in training)

Mean over 2 sample images per noise type — a spot check, not a full benchmark:

| noise type | PSNR | SSIM |
|---|---|---|
| Speckle | **32.53** | 0.876 |
| Gaussian | 31.38 | 0.841 |
| Salt & pepper | 27.99 | 0.724 |

**Speckle and Gaussian hold up strongly** even on a total domain shift (natural photos vs SEM). **Salt & pepper is the known weak point** — sparse impulse noise is a statistically different problem from the continuous, spatially-correlated noise this model was trained on.

### JPEG compression — verified absent, not assumed

We tested whether KLA's `.npy` test files carry baked-in JPEG block artifacts, with controls:

| dataset | period-8 blockiness |
|---|---|
| Known-clean `.npy` | 0.9969 |
| GT → JPEG q=90 → back | 1.0777 |
| GT → JPEG q=70 → back | 1.3367 |
| **KLA `Test_NoisyLR`** | **0.9984** ← clean |

Their data has never passed through a lossy codec. The detector is sensitive enough to catch even quality-90 compression, so this is a positive verification rather than an absence of evidence.

---

## 7. Negative results — things we tested and rejected

**This is strong material for the deck: it shows the choices were measured, not assumed.**

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| 1 | Train longer (100k iters) | grad ratio flat 2.49 → 2.23, val PSNR **declined** after 36k | Rejected |
| 2 | `w_fft=0.05` for sharpness | period4 artifacts **25,357×** GT level; worse LPIPS | Rejected |
| 3 | Add MS-SSIM to loss | −0.58 dB, −0.016 SSIM, **worse LPIPS**, hallucinated lattice texture | Rejected |
| 4 | Width 48 (2.24× params) | +0.087 dB at 7k iters; **doesn't improve sharpness** (grad 0.707 vs 0.710); locked to batch 8 | Rejected |
| 5 | Radial HF weighting on FFT loss | Made the term *less* sensitive to blur | Rejected |
| 6 | `resize` internal upsampling | **−9 dB** | Rejected |

### The honest limitation

Output is **softer than ground truth** — gradient ratio **0.712** against GT's 1.000. This affects roughly a third of images, concentrated in high-texture content.

Four independent attempts to fix it all failed (rows 1–4 above). Error maps (`results/figures/diff_maps.png`) show why: on structured content the error is a thin outline along edges (sub-pixel boundary placement, benign), but on pure-texture content it is uniform grain across the whole frame — **that texture is destroyed by the degradation and is not recoverable**. When MS-SSIM tried to synthesize it, it invented lattice patterns and scored worse on every metric.

**This is an information limit, not a tuning failure.**

---

## 8. Key ablations

All at config `s`, 6000 iters, matched settings.

| Run | Residual | Input transform | PSNR | SSIM |
|---|---|---|---|---|
| `a_base` | bicubic | affine | 25.053 | 0.6081 |
| `a_nores` | none | affine | 25.133 | 0.6085 |
| `a_gated` | gated | affine | 25.044 | 0.6084 |
| `a_log` | bicubic | log | 25.197 | 0.6115 |
| `a_both` | none | log | 25.189 | 0.6114 |

Loss-term isolation (at `w_fft=0.1`):

| Run | Terms | PSNR | grad ratio |
|---|---|---|---|
| `iso_grad` | gradient only | **28.961** | 0.683 |
| `iso_ssim` | MS-SSIM only | 28.138 | **0.787** |
| `c_all` | both | 28.250 | 0.744 |
| `c_low` | both, low SSIM weight | 28.574 | 0.720 |

---

## 9. Usage

```bash
# Inference — this is what the graders run
python evaluate.py <input_dir> <output_dir>
# defaults to weights/best.pt relative to the script; no flags needed

# Verify the whole path on random data, no inputs required
python evaluate.py --self-test

# Training (shipped configuration)
python train.py --name final --config m --head resize_conv \
    --residual bicubic --input-transform log --up-mode pixelshuffle \
    --w-fft 0.1 --w-grad 0.15 --w-ssim 0.0 \
    --iters 60000 --batch 16

# Full metrics + comparison grid (needs: pip install lpips torchvision)
python tools/full_eval.py --weights weights/best.pt
```

`evaluate.py` is deliberately defensive: accepts positional or flagged paths, falls back to CPU without a GPU, resolves weights relative to the script (not CWD), and depends only on `torch` + `numpy` + `nafnet_sr.py`.

---

## 10. Repository layout

```
README.md           this file
evaluate.py         inference / submission script (notch post-processing)
train.py            training entry point
nafnet_sr.py        model definition (NAFNet body + adapted SR ends)
requirements.txt    pip freeze of the training environment
weights/best.pt     shipped checkpoint (config m, step 35999, 29.469 dB)
outputs_final/      restored outputs for the 400 provided test images

src/
  data.py           source-aware GPU-resident dataset
  degradation.py    fitted forward-model simulator
  losses.py         composite training loss
  metrics.py        PSNR/SSIM (skimage-matched) + post-processing
tools/              diagnostics: kernel fitting, full_eval, artifact tracing
scripts/            ablation sweep shell scripts

results/
  figures/          key result figures (referenced throughout this README)
  diagnostics/      exploratory analysis images from the investigation
  metrics/          per-image PSNR/SSIM/LPIPS CSVs
docs/
  CHECKPOINT.md     detailed rollback reference: every experiment + result
  PROMPT.md         working notes, baselines, ceiling analysis
  reference/        problem statement and provided reference PDFs
```

**Performance note:** the entire dataset lives on the GPU (500 MB fp16). No DataLoader, no workers, no host↔device copies during training — the honest answer to the brief's "optimize disk reads / data transfer".

---

## 11. Figures available for slides

| File | Shows |
|---|---|
| `results/figures/eval_final_grid.png` | input → output → GT, 6 images spanning worst→best PSNR |
| `results/figures/diff_maps.png` | per-pixel error maps; edge vs texture failure modes |
| `results/figures/kaggle_noise_grid.png` | OOD robustness across noise types |
| `results/metrics/eval_final_metrics.csv` | per-image PSNR/SSIM/LPIPS, all 320 images |
| `results/figures/probe_w32_vs_w48.png` | width-48 capacity probe (negative result) |
| `results/figures/compare_heads.png` | head comparison |
| `results/figures/trace_stages.png` | artifact traced through the pipeline |

---

## 12. Suggested 9-slide structure

1. **Problem** — 128×128 degraded → 256×256 clean; speckle + Gaussian + downsampling, joint denoise/SR (§ intro)
2. **Approach** — NAFNet body, log input transform, resize-conv head, bicubic residual (§2)
3. **Fitted degradation model** — 26.2M-equation kernel solve, R²=0.975 noise fit; *we measured the forward model rather than guessing it* (§4)
4. **Training rigour** — source-aware split, the 100% leakage bug we found and fixed (§4)
5. **Results** — headline table + floor/ceiling context, 62% of available gain (§1)
6. **Visual results** — `results/figures/eval_final_grid.png` (§11)
7. **The artifact investigation** — traced clip→amplification chain, notch fix, +1.35 dB free (§5)
8. **What we rejected and why** — the negative-results table; measured, not assumed (§7)
9. **Robustness + limitations** — OOD noise types, JPEG verification, the honest softness limit (§6, §7)

**Framing that plays well:** the strongest story here is *measurement discipline* — a fitted forward model, a leakage bug caught and fixed, six rejected hypotheses with numbers, and a stated limitation with evidence for why it's an information limit rather than a tuning failure.
