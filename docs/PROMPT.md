# SemiCon AI Hackathon — PS01: AI-Based Restoration of Degraded Images

**Read this file first. It is the complete context for the project — you do not
need to re-read the PDFs, the screenshots, or re-analyse the dataset.**
Everything below marked *measured* was computed from the actual data, not
assumed. Scripts that produced those numbers are in `tools/`.

---

## 1. The task

Learn the inverse of a degradation `f: x → y` where `f` applies **speckle noise
(multiplicative), additive Gaussian noise, and 2× downsampling, in no
particular order, to random subsets of images**.

- Input: degraded 128×128 grayscale `.npy`
- Output: restored 256×256 grayscale `.npy` matching the clean ground truth
- Single joint model. **Not** a fixed denoise→deblur→SR cascade — the brief says
  "do not read into the order of it".

### Scored on three axes, not just leaderboard PSNR
1. **Accuracy + generalisation** — the real test set contains out-of-distribution
   samples from image sources not in training.
2. **Throughput** — end-to-end inference time, benchmarked on an H100.
3. **Training hygiene** — data / model / loss / compute are each explicitly
   weighted. Ablations and documented decisions score points here.

### Hard rules (from the webinar)
- Reusing an open-source architecture is **preferred** over inventing one.
- Pretrained weights as init + fine-tuning is **explicitly allowed**.
- External datasets are allowed.
- No model-size limit, but size costs throughput.

---

## 2. What the data actually is (*measured*)

| | |
|---|---|
| Train pairs | 3200, `train/train/GT/*.npy` + `train/train/NoisyLR/*.npy` |
| GT | 256×256 float32, **exactly `[0,1]`** |
| NoisyLR | 128×128 float32, range ≈ **−0.10 … 1.73** (overshoot is real) |
| Test | 400 × 128×128 in `Test_NoisyLR/NoisyLR/` |
| Total size | ~1 GB float32 → **~525 MB in fp16, fits in VRAM** |

**Every one of the 3200 GT images has `min == 0.0` and `max == 1.0` exactly.**
GT is per-image min–max normalised. This is exploitable (see §5).

### The fitted forward model
Regressing residual variance against intensity gives
`Var(y − D(x)) = σs²·x² + σg²` with **median R² = 0.975**:

| Component | Finding |
|---|---|
| Downsampler | box/area minimise residual MSE — but see the caveat below |
| Speckle σs | p05 **0.131**, med **0.167**, p95 **0.212** — present in **99.7%** of images |
| Gaussian σg | p05 **0.000**, med **0.020**, p95 **0.074** — **absent in 29%** of images ← this is the "random subset" |
| Ordering | speckle-at-HR fits slightly better than at-LR (R² 0.9760 vs 0.9727, wins on 67%) |
| Residual shape | kurtosis 3.56, skew +0.34 — slightly heavy-tailed and right-skewed |

**Two corrections to earlier assumptions, both important:**
- **Variance scales with x², not x.** So this is genuine *multiplicative
  speckle*, **not** Poisson/shot noise. Do not model it as shot noise.
- **There is no separate blur degradation.** The three degradations are speckle,
  Gaussian, downsample. Blur only enters implicitly via the downsampler.

### The kernel — SOLVED by estimating it instead of guessing
Hand-picked candidates all left a **−0.10 correlation between the residual and
the Laplacian of the clean LR**, i.e. the real NoisyLR is *sharper* than
box(GT). A control (synthesise with a known box kernel, run the identical test)
reads 0.000, so the effect was real. The fix was to stop testing candidates and
**solve for the kernel directly** — least squares over an 8×8 HR kernel,
26.2M equations, 65 unknowns (`tools/fit_kernel.py`). Held-out results:

| kernel | residual MSE | Laplacian corr |
|---|---|---|
| box2x2 | 7.707e-3 | −0.103 |
| lanczos2 | 7.946e-3 | −0.089 |
| bicubic | 7.648e-3 | +0.093 |
| **FITTED** | **7.611e-3** | **+0.004** |

The fitted kernel wins on **both** criteria at once, which no candidate did. It
is separable (σ₂/σ₁ = 0.009), symmetric, sums to 1.0000, bias ≈ 0, and carries
**−0.43 negative mass — it sharpens**, which is exactly why every averaging
kernel left a negative correlation. 1-D profile:

```
[-0.004, +0.022, -0.091, +0.562, +0.576, -0.084, +0.022, -0.003]
```

Alternating signs decaying out from two big central taps is the signature of a
resample composed with a deconvolution — consistent with GT and NoisyLR both
descending from a common higher-res original, this being the effective GT→LR
operator. **The residual under it is pure noise**: mean flat at ~0 across every
intensity bin (no unmodelled nonlinearity) while std rises with intensity
exactly as speckle predicts.

Stored in `src/fitted_kernel.npy`, used as the primary downsampler (55% of
synthetic samples). Box/bicubic/lanczos/gaussian are kept at lower weight purely
for OOD robustness — the test images come from other sources and may have been
resampled differently. Regenerate with `python -u tools/fit_kernel.py`.

### Two bugs worth knowing about (fixed, don't reintroduce)
**1. HR noise attenuation.** Noise applied before downsampling is attenuated by
the kernel. All sigmas in `DegradeCfg` are quoted in *LR space* (that is where
they were fitted), so HR noise is pre-scaled by `1/attenuation`. Without it
every speckle-at-HR sample was degraded at half strength. The attenuation is
**measured per kernel**, not hardcoded — box is 0.501 (gain 2.00) but the
fitted kernel sharpens and is only 0.658 (gain 1.52), so assuming the box factor
over-injected noise by ~32%.

**2. σ_g sampled uniformly.** The real gaussian is right-skewed (p25 0.007,
med 0.020, p75 0.038, p95 0.074). Uniform sampling recovered 0.030 — a 50%
overshoot. `gauss_pow` fixes it: `σ = max·u^pow`, and `u²` has median 0.25 /
p75 0.5625 → 0.0185 / 0.0416 against the real 0.020 / 0.038.

`tools/validate_simulator.py` now recovers **speckle 0.172 / gaussian 0.018**
against the real **0.168 / 0.020**. **Re-run it after any change to
`degradation.py`** — both bugs were invisible until it was run.

### Distribution shift check
The provided 400 test images are **statistically in-distribution** with training
(mean/std/min/max/gradient-magnitude all match within noise; test tails are
marginally wider). The OOD samples the brief warns about are in the *hidden*
test set KLA releases later — which is exactly why §4's synthetic augmentation
matters.

---

## 2b. THE RUN TO LAUNCH (start here)

```bash
.venv/Scripts/python -u train.py --name main --config m --head resize_conv \
    --iters 100000 --batch 16 > main.log 2>&1
```
≈7 h on the 4060. Everything below is already the default; the flags are
written out so the run is self-documenting. Watch `grad` in the val lines: it
should approach 1.0 from BELOW as denoising improves. Then:
```bash
.venv/Scripts/python -u evaluate.py Test_NoisyLR/NoisyLR outputs --weights runs/main/best.pt
```

### Two traps already hit; do not re-enter them
1. **`--warmup` must scale with `--iters`.** It is now `-1` = auto (2% of
   iters, clamped [50, 2000]), which reproduces the tuned 2000 at 100k. The old
   fixed 2000 on a 4k run meant the lr peaked at the halfway point then annealed
   to 1e-6 — half the run was warmup, and it invalidated a whole ablation sweep.
   The startup banner now prints `warmup N (X%)`; if X is large, stop and fix it.
2. **Never measure the checkerboard on clamped output.** Clamping to [0,1]
   rectifies the model's overshoot (a flat 0.90 input produces output spanning
   [0.64, 1.16]) and that rectification reads as a checkerboard. It inflated a
   real 6000x improvement into a fake 662x one. Measure unclamped, fp32, and
   crop ≥32 px of border — deep-UNet reflect padding distorts the edges.

---

## 3. Architecture: NAFNet-SR (`nafnet_sr.py`)

NAFNet (Chen et al., ECCV 2022) — single-stage UNet, `LN → 1×1 → 3×3 dw →
SimpleGate → SCA → 1×1` blocks, **no nonlinear activations**, element-wise skip
fusion, stride-2 conv down, PixelShuffle up. 40.30 dB on SIDD at 65 GMACs,
beating Restormer with half the compute. Its efficiency is why it wins here,
given throughput is scored.

**Our modification** — the body stays byte-identical so official checkpoints
load; only the ends change:

| Change | Why |
|---|---|
| Body runs at **128×128** (LR), `PixelShuffle(2)` head at the end | 4× cheaper than upsampling the input first. This is the single biggest throughput decision. |
| Global residual = **bicubic ×2 upsample of input** | Network only learns the correction. |
| 1-channel stem/head | Data is grayscale. |
| ICNR init on the sub-pixel conv | Avoids checkerboard from the start. |
| Fixed input affine `(x − 0.45)/0.25`, **no per-image renormalisation of input** | The input's min/max are *noise-driven* and unstable — renormalising the input would inject noise into the conditioning. The overshoot past 1.0 is a genuine cue about how much speckle is present; keep it. |

Configs in `CONFIGS`: `s` (fast, for ablations), **`m` = official width32
topology, the default**, `l` (width64, only if VRAM and time allow).

**TLC is deliberately not used.** TLC exists to fix train-on-patches /
test-on-full-images mismatch. Our images are 128×128 end to end — we train on
whole images. No mismatch, so no TLC, and no TLC cost at inference.

### The output head: use `resize_conv` (now the default). MEASURED.
The original `pixelshuffle` head develops a **severe checkerboard during
training**. ICNR init starts the `scale²=4` sub-pixel planes identical
(divergence 0.0000, artifact 0.0) but training pulls them **61% apart**, and
PixelShuffle then interleaves four different constants into a 2×2 tile — so
even a perfectly *constant* input yields a periodic pattern. A conv net fed a
constant image must output a constant; this head cannot.

Controlled tests (`tools/test_artifact_cause.py`) show it is **brightness-driven,
not noise-driven** — which was the opposite of the first hypothesis:

| test | result |
|---|---|
| flat, **zero-noise** input | still **1028×** brightness scaling |
| additive noise, σ constant across brightness | still **114–128×** |
| speckle σ swept 0.02→0.32 (16× range) | artifact moves **7%** |
| noiseless vs noisy at same brightness | ratio **~1.0** |

Head comparison, 4k iters, config `s` (unclamped, interior-only):

| | PSNR | SSIM | grad ratio | checker (flat) |
|---|---|---|---|---|
| pixelshuffle | 24.270 | 0.5837 | 0.732 | 0.061129 |
| **resize_conv** | **24.855** | **0.5962** | 1.162 | **0.000010** |
| pixelshuffle_smooth | 24.634 | 0.5860 | 0.804 | — |

`resize_conv` on real inputs measures **0.000547** against ground truth's own
**0.001469** — i.e. below the halftone texture already present in the source
images. **No loss function can fix this**; it is a structural degree of freedom
in the head, present at zero noise and zero input variation.

---

## 4. The generalisation play: synthetic re-degradation (`src/degradation.py`)

This is the main differentiator. We have 3200 clean GT images and a *fitted*
forward model, so we can synthesise **unlimited** fresh training pairs with
randomised degradation parameters.

`TRAIN_CFG` deliberately **widens every fitted range**: speckle 0.04–0.32 (fitted
band 0.13–0.21), Gaussian 0–0.13, plus randomised ordering, a gamma-distributed
speckle branch (matches the measured +0.34 skew better than Gaussian), an extra
source blur pass (`p_blur=0.30`, up from an initial 0.15 — see §7, this is the
secondary lever against denoiser-induced softness, giving the model practice at
sharpening rather than only ever practicing suppressing noise), and a kernel mix
(55% fitted / 15% box / 12% bicubic / 10% lanczos / 8% gaussian+stride). A model
that has only ever seen the exact fitted noise band is the one that falls over
on the OOD test set.

`--synth-ratio 0.5` mixes half real pairs, half synthetic per batch. Real pairs
keep us anchored to the true forward model; synthetic pairs buy robustness.

**Photometric augmentation uses `gt ** γ`, γ ∈ [0.7, 1.45].** This is chosen
specifically because `x**γ` maps `[0,1] → [0,1]` with 0 and 1 *exactly* fixed,
so the augmented GT is still exactly min–max normalised. Brightness/contrast
shifts would break that invariant. Same reason we use **whole images and dihedral
flips, never random crops** — a crop's min/max would no longer be 0 and 1.

Also defined: `FIT_CFG` (the exact fitted band, for matched validation) and
`OOD_CFG` (deliberately outside anything trained on — the robustness probe
reported every validation).

---

## 5. Output normalisation — measured, and it is NOT the free win it looks like

Since GT is *provably* always exactly `[0,1]`, min–max stretching the output to
`[0,1]` looks like free accuracy. **On measurement it is the opposite.**
No-model baselines on the val split (`tools/baseline.py`):

| upsample | clip | minmax | robust |
|---|---|---|---|
| nearest | 21.342 | 16.866 | 18.109 |
| bilinear | **24.542** | 19.476 | 20.721 |
| bicubic | 22.795 | 18.068 | 19.422 |

min–max costs **~5 dB**, because the signal being stretched contains noise
outliers (the input reaches ±0.1 past its true range): stretching to the
outliers compresses the real signal into a narrow band. `robust` (0.1%
quantiles) recovers some of that but still loses to plain `clip`.

This only flips if the *model's* output is clean and outlier-free, which is
plausible but unproven. So `evaluate.py --post` **defaults to `clip`**, and
`train.py` logs all three at every validation. **Pick the winner from
`runs/<name>/metrics.csv`. Do not assume min–max wins because GT is
normalised.**

### Baseline to beat, and the actual ceiling (`tools/baseline.py`)
| | PSNR | SSIM |
|---|---|---|
| no-model bilinear + clip | 24.542 | 0.5870 |
| best short run so far (4k iters, crippled warmup) | 24.855 | 0.5962 |
| **perfect denoise + bicubic up (CEILING)** | **32.220** | **0.8661** |

Note bilinear beats bicubic as a baseline — bicubic's negative lobes sharpen
the noise.

**The ceiling is the number that matters.** It is what a model achieves by
removing *all* noise and doing nothing clever about resolution. The 4k-iter
model sits **7.4 dB below it** and removes only **22.7%** of the input noise
(measured as residual std inside flat GT regions: input 0.06008 → output
0.04641, vs 0.00664 for perfect denoising).

So the headline problem is **not** sharpness or loss design — it is that the
core denoising is barely happening yet, because the runs were far too short and
half of each was warmup. Re-measure this gap after the 100k run before
concluding anything about loss weights.

**Corollary — `grad ratio > 1` is NOT good news while denoising is incomplete.**
Leftover noise inflates gradient magnitude. `resize_conv` reads 1.162 not
because it recovers extra edge detail but because it still carries noise. Read
`grad` together with the noise-removal figure, never alone.

---

## 6. Compute optimisation (`src/data.py`)

The whole dataset is ~525 MB in fp16, so it is **loaded once into VRAM and never
touched again**: no `DataLoader`, no workers, no collate, no `pin_memory`, no
per-step H2D copies, no `np.load` in the training loop. Augmentation and
synthetic degradation both run on GPU. The loop is GPU-bound, not data-starved.

Also on: `bf16` autocast, `channels_last`, `cudnn.benchmark`, TF32, EMA of
weights, gradient clipping. `--compile` attempts `torch.compile` and falls back
to eager (Triton is unreliable on Windows).

`--data-device cpu` keeps tensors in host memory if VRAM gets tight.

---

## 7. Loss (`src/losses.py`) — and the denoiser-induced-blur fix

```
L = Charbonnier(pred, gt) + 0.05·L1(rFFT(pred), rFFT(gt))
                          + 0.15·L1(Sobel(pred), Sobel(gt))
                          + 0.10·(1 − SSIM)
```
- **Charbonnier** — L1-family; L2 regresses to the conditional mean and comes out
  blurry, destroying exactly the fine structural detail this task is scored on.
- **FFT L1, deliberately unweighted** — the brief explicitly asks about
  frequency-domain losses. A *radial high-frequency weighting* was tried first
  (theory: flat L1 is dominated by low-frequency energy, so weight up the
  high-frequency bins that smoothing wipes out) and **measured to make things
  worse**: these SEM images concentrate 98.5% of spectral energy inside the
  inner 10% of frequency radius, so discounting low frequencies throws away
  real error signal rather than freeing up "wasted" weight. Slope ratio to the
  pixel loss under mild blur: flat 0.209×, radially-weighted 0.128×. Kept flat.
- **Gradient L1 (Sobel), new — this is the actual fix** — measures whether
  edge/gradient magnitude is preserved. Both a denoiser's regression-to-mean
  and a straightforward blur reduce gradient magnitude by construction, so
  this taxes softness directly. Measured slope ratio to the pixel loss: **0.99×
  for mild blur, 1.91× for 2% regression-to-mean shrinkage** — several times
  stronger than the FFT term (0.13–0.32×) on both probes. Cheap: one 3×3
  depthwise conv.
- **SSIM** — it is a reported metric, so optimise it directly.

**Why this matters and what it fixes:** a model trained mostly to denoise
learns "when uncertain, predict the local average" — that minimizes pixel loss
even though it destroys structure. A plain pixel loss barely penalizes this;
the gradient term makes softness expensive directly, at training time, in the
one joint model — not via a second denoise→deblur stage, which would risk
sharpening whatever errors the first stage made (hallucinated detail on
semiconductor imagery is worse than a soft image).

**Verify with `tools/probe_smoothness_loss.py`** before trusting these numbers
on different data — which loss term reacts to softening depends on how that
dataset's energy is distributed across frequencies, and that is not something
to assume twice.

**No GAN / adversarial loss, ever.** On semiconductor inspection imagery a
hallucinated edge where a real defect was is worse than a soft one. This rules
out BSRGAN / Real-ESRGAN GAN variants regardless of their sharpness.

Weights `w_fft` / `w_grad` / `w_ssim` are `--w-fft` / `--w-grad` / `--w-ssim`.
`w_grad` is set from the measured sensitivity above; the others are still
*untuned guesses* — ablate them.

---

## 8. Repo layout

```
nafnet_sr.py                    model, self-contained (torch only) + pretrained loader
evaluate.py                     SUBMISSION: standalone inference, runs as-is
train.py                        SUBMISSION: training entry point
src/degradation.py              fitted GPU forward-model simulator
src/data.py                     GPU-resident dataset + augmentation
src/losses.py                   Charbonnier + FFT + Sobel gradient + SSIM
tools/probe_smoothness_loss.py  measures which loss term reacts to softening --
                                RUN AFTER ANY CHANGE TO losses.py
tools/test_artifact_cause.py    decouples brightness vs noise as artifact cause
tools/compare_heads.py          side-by-side heads + metrics + zoom crops
tools/visualize_examples.py     input | output | GT grid for one checkpoint
src/metrics.py                  PSNR / SSIM (skimage-matched) + post-processing
tools/analyze_degradation.py    the noise fit behind §2
tools/refine_degradation_fit.py HR-vs-LR ordering, GT normalisation, train/test shift
tools/check_kernel.py           candidate-kernel comparison + the box control
tools/fit_kernel.py             SOLVES the kernel by least squares; writes
                                src/fitted_kernel.npy
tools/validate_simulator.py     simulator vs real statistics -- RUN AFTER ANY
                                CHANGE TO degradation.py
src/fitted_kernel.npy           the measured 8x8 GT->NoisyLR operator
tools/baseline.py               no-model baselines on the val split
tools/bench_sweep.py            VRAM / throughput sweep
runs/<name>/                    args.json, metrics.csv, best.pt, last.pt
```

### Environment
Project-local venv, **torch 2.13.0+cu126** (plain `pip install torch` on Windows
silently gives a CPU-only build — use the `--index-url` below):
```bash
.venv/Scripts/python -m pip install numpy pillow \
    torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### Commands
```bash
.venv/Scripts/python evaluate.py --self-test              # verify the path, no data needed
.venv/Scripts/python -u tools/validate_simulator.py       # simulator fidelity
.venv/Scripts/python -u tools/baseline.py                 # the number to beat
.venv/Scripts/python -u tools/bench_sweep.py              # it/s + VRAM per config
.venv/Scripts/python -u train.py --name smoke --iters 300 --val-every 150
.venv/Scripts/python -u train.py --name main              # defaults: m, batch 16, 100k
.venv/Scripts/python -u evaluate.py Test_NoisyLR/NoisyLR outputs --weights runs/main/best.pt
```
Always pass `-u` and redirect to a log (`> run.log 2>&1`) — piping to `tail`
buffers everything until the process exits, so you see nothing while it runs.

---

## 9. Hardware reality (*measured on the actual 4060*)

Training machine is an **RTX 4060 Laptop, 8 GB VRAM** — not the H100 used for
benchmarking. `tools/bench_sweep.py`:

| config | batch | params | it/s | peak VRAM | 100k iters |
|---|---|---|---|---|---|
| s | 8 | 13.3M | 13.47 | 1.17 GiB | 2.1 h |
| s | 16 | 13.3M | 8.02 | 2.29 GiB | 3.5 h |
| s | 24 | 13.3M | 5.45 | 3.39 GiB | 5.1 h |
| s | 32 | 13.3M | 4.20 | 4.45 GiB | 6.6 h |
| **m** | **16** | **29.2M** | **4.09** | **4.19 GiB** | **6.8 h** |
| m | 8 | 29.2M | 7.41 | 2.28 GiB | 3.7 h |

**Defaults are now `--config m --batch 16 --iters 100000`** (≈7 h, an overnight
run). Do not raise the batch on this card: **batch 32 with config m peaks near
8 GiB, spills into WDDM shared memory and collapses to a crawl** — it looks like
a hang, not an OOM. Throughput per image plateaus around batch 16–24 anyway.

### Inference is not the constraint — quality is
| config | batch | ms/image | img/s |
|---|---|---|---|
| s | 32 | 2.09 | 479 |
| m | 32 | 4.01 | 250 |

4 ms/image on a *laptop* 4060, and the brief's own framing was "10 minutes bad,
10 seconds useful". There is enormous headroom, so **pick the model for quality,
not for speed.** Config `l` is only worth trying if ablations show `m`
underfitting — it will not fit training on 8 GB at a useful batch size.

### The inference gotcha that actually cost 4×
`cudnn.benchmark=True` spends **~8.5 s autotuning on the first batch** and buys
nothing in steady state (0.280 vs 0.279 s/batch). Over a 400-image run that is
22.8 ms/image instead of 4.4. `evaluate.py` sets it **False** and end-to-end went
**48.45 → 11.98 ms/image**. Autotuning only amortises over thousands of steps —
i.e. training. `train.py` keeps it True; `evaluate.py` must not.

---

## 10. Work queue

**Done and verified end-to-end on the real data**
- [x] Degradation model fitted (R² 0.975) and the simulator validated against it
      (`tools/validate_simulator.py`: recovers speckle 0.170 vs real 0.168)
- [x] NAFNet-SR architecture, GPU data pipeline, loss, metrics, train loop
- [x] Baselines measured: **24.542 dB / 0.5870 SSIM** (bilinear + clip)
- [x] Bench sweep → defaults set to `--config m --batch 16`
- [x] Smoke run: 300 iters, loss falls 0.170 → 0.115, val 23.9 dB, 4.1 it/s,
      checkpointing + OOD probe + 3-way post-process logging all working
- [x] `evaluate.py` on the real 400-image test set: correct shapes, filenames
      and range, **11.98 ms/image end-to-end**
- [x] Denoiser-induced-softness fix: added a Sobel gradient loss term and
      measured it against a first-attempt radial FFT weighting, which
      underperformed and was reverted. Raised blur augmentation probability.
      See §7 and `tools/probe_smoothness_loss.py`.
- [x] **Found and fixed a checkerboard artifact in the PixelShuffle head**
      (§3). `resize_conv` is now the default: +0.59 dB, +0.0125 SSIM, and the
      artifact drops from 0.061129 to 0.000010 on a flat input.
- [x] **Found and fixed non-scaling `--warmup`** which invalidated the first
      ablation sweep (§2b). Auto-scales now.
- [x] Established the task CEILING at 32.220 dB (§5) -- the single most
      important number for judging whether a run is actually good.

**INVALID -- do not cite:** the `runs/x_*` sweep was run with warmup=2000 on
4000 iters, i.e. 50% warmup. The head comparison within it is still meaningful
(all variants were equally handicapped) but the absolute numbers are not, and
the `w_grad` arm never completed. Re-run the loss ablations after the 100k run.

**Next, in order**
1. **Ablations** — short runs (~20k iters) for the ones that matter:
   `--synth-ratio 0 / 0.5 / 0.8`, `--w-fft`/`--w-grad`/`--w-ssim` on/off,
   pretrained-vs-scratch, config `s` vs `m`, `p_blur` 0.15 vs 0.30.
   *Keep the table — it is the Slide 5/6 evidence and the training-hygiene score.*
4. Full run with the winning settings.
5. Choose `--post` from `metrics.csv`.
6. Measure inference time properly (batched, fp16, `channels_last`). Decide on
   `--self-ensemble` — it is ~8× the cost for a few hundredths of a dB, and
   throughput is scored, so the default is off.
7. Submission: README, `requirements.txt` (`pip freeze`), weights (Git LFS or a
   drive link), restored test outputs, 8–9 slide PDF named
   `TeamName_KLA_PS01.pdf`.

### Open questions / honest caveats
- **Pretrained value is unproven.** Official NAFNet checkpoints are RGB natural
  images (SIDD/GoPro) and our head differs; `load_pretrained_body` adapts the
  stem by averaging RGB weights and skips `ending`. Expect a small convergence
  speedup at best. Run the ablation rather than assuming — and note the width32
  checkpoints are Google Drive links in `NAFNet-main/NAFNet-main/docs/`, so they
  need a manual download.
- **`w_fft` and `w_ssim` are still guesses; `w_grad` is set from measured
  sensitivity but not yet ablated end-to-end.** Ablate all three.
- The kernel is solved for the *training* distribution. Nothing guarantees the
  OOD test images were resampled with the same operator — that is exactly why
  the mix keeps 45% non-fitted kernels rather than going all-in on the fitted
  one now that we have it.
- Config `l` (116M) is no longer ruled out by throughput — inference is 4 ms/img
  and `l` would be ~16 ms, still trivial. It is ruled out by *training* VRAM on
  8 GB. Worth revisiting only if ablations show `m` underfitting.
- **Restormer remains the documented fallback** if NAFNet's local receptive
  field can't separate fine structure from heavy speckle — but check the OOD
  validation column before escalating, not vibes.
