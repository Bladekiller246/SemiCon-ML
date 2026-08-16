# CHECKPOINT — 2026-08-14, before the clean retrain

**Purpose of this file:** a rollback reference. If a future change breaks
something, this is what "known good" looked like — exact commands, exact
numbers, exact known issues, at the point right before the real training run
started. Nothing in this document requires re-deriving anything; every number
here was measured, not assumed (measurement scripts are named throughout).

**Training status as of writing: NOTHING HAS BEEN TRAINED with the new
settings yet.** `runs/main/` is the only completed full run, and it predates
everything in §3–§6 below. Treat its numbers as the pre-fix baseline, not
the current expected result.

---

## 1. The one thing that must never regress: source-aware split

**`src/data.py`** splits on `source_id = index // group_size` (default
`group_size=4`), not on individual image index.

**Why this exists:** the dataset ships 4 crops per source image. A plain
random split put 100% of validation images in the same group as a training
image (content similarity 0.626 median within-group vs −0.019 random-pair —
measured, not assumed). Every number reported before this fix (`runs/main`'s
25.823 dB included) was inflated by leakage.

**Verify it's still working**, any time, in 2 seconds:
```
.venv/Scripts/python -u -c "
import sys; sys.path.insert(0,'.')
from src.data import SemiconData
d = SemiconData('train/train', device='cuda', val_size=320, seed=0)
"
```
Must print `leakage check = 0 (must be 0)`. If it ever doesn't, stop — every
downstream number becomes untrustworthy again.

---

## 2. A real trap that already happened once — read before running `--bench`

`train.py --name` defaults to `**main**`. Running `--bench` **without** an
explicit `--name` silently overwrote `runs/main/args.json` with the
benchmark's throwaway settings, while leaving `best.pt`/`last.pt` completely
correct (checkpoints save their own `args`/`model_kwargs` internally,
independent of the standalone `args.json` file).

**Fixed in `train.py`**: `args.json` is no longer written at all when
`--bench` is passed. If `args.json` for a run and the checkpoint's *internal*
`ck['args']` ever disagree, **the checkpoint's internal record is the truth**:
```
.venv/Scripts/python -u -c "
import torch
ck = torch.load('runs/<name>/best.pt', map_location='cpu', weights_only=False)
print(ck['args'])          # ground truth
print(ck['model_kwargs'])  # what build_model() needs to reconstruct it
"
```

---

## 3. Current architecture (`nafnet_sr.py`, 449 lines)

NAFNet body **verified byte-identical** to the official implementation
(`tools/` — LayerNorm2d forward/backward diff 1.3e-15, NAFBlock output diff
0.000e+00 against the real `NAFNet_arch.py`). Only the ends are adapted for
this task.

### `build_model(config, **kwargs)` — every option and its status

| kwarg | choices | **default** | status |
|---|---|---|---|
| `config` | s / m / l | m | `m` = official width32 topology, 29.16M params |
| `head` | pixelshuffle / **resize_conv** / pixelshuffle_smooth | resize_conv | see §5 |
| `residual` | **bicubic** / none / gated | bicubic | see §5 |
| `input_transform` | affine / **log** | log (train.py) / affine (build_model direct) | log = variance-stabilising, matches multiplicative speckle |
| `up_mode` | **pixelshuffle** / pixelshuffle_icnr / pixelshuffle_smooth / resize / resize_nearest | pixelshuffle | **`resize`/`resize_nearest` are PROVEN HARMFUL — see §4** |
| `use_refine` | bool | False | new, untrained end-to-end — see §6 |

**Note the CLI/library default mismatch**: `build_model()` directly defaults
to `input_transform='affine'`, but `train.py`'s CLI defaults to `affine` too
— you must pass `--input-transform log` explicitly. This is intentional
(library stays conservative; the recommended training command spells
everything out).

---

## 4. DO NOT re-attempt: replacing internal decoder PixelShuffle

`up_mode='resize'` (bilinear upsample-then-conv, applied to the **internal**
decoder stages, not just the final head) was tried and measured:

| | PSNR | SSIM |
|---|---|---|
| `up_mode=pixelshuffle` (a_both) | 25.189 dB | 0.6114 |
| `up_mode=resize` (up_resize) | **16.098 dB** | 0.3769 |

**9 dB worse.** PixelShuffle is information-preserving (lossless
reshape); bilinear is a low-pass filter. Applying it at all 4 internal
decoder stages progressively destroys the high-frequency content the task
needs reconstructed. Confirmed by feature stats: output std collapsed to
0.134 vs ground truth's 0.236, floor stuck at 0.200 instead of 0.0 (washed
out). Full reasoning is in the `UpsampleConv` docstring in `nafnet_sr.py` —
**read it before ever touching `up_mode` again.**

The **final SR head** is a different story — `resize_conv` there is correct
and proven (§5).

---

## 5. The artifact story — diagnosis, what worked, what didn't

Two separate artifacts, different causes, different fixes:

| | period-2 (Nyquist) | period-4 (diagonal) |
|---|---|---|
| caused by | `pixelshuffle` **head** — sub-pixel planes diverge during training (0% divergence at init → 55%+ after 6k iters) | `resize_conv` **head** — bilinear-then-conv phase geometry, present **at random init**, before any training |
| head that has it | pixelshuffle | resize_conv |
| head that avoids it | resize_conv | (none tested is fully clean — see below) |

**Ground truth reference values** (what "clean" looks like):
`period2/median ≈ 0.19`, `period4/median ≈ 1.94`.

### Confirmed NOT caused by speckle noise (two independent proofs)
1. Fed the trained model a **perfectly constant, zero-variance input** — no
   randomness at all. Output still showed period4 up to **161,560,969×** GT
   level, scaling with brightness. Nothing random went in; the model
   manufactures the pattern.
2. Classical despeckle filters (Lee, bilateral — designed specifically to
   distinguish noise from signal) left the artifact at 300–650× GT level.
   Filters built around the noise/signal distinction don't treat this as
   noise, because it isn't. (`tools/test_clipping.py`-style tests; see
   conversation history for the exact numbers if needed.)

### Confirmed NOT caused by gamma augmentation (`augment_gt`, `src/degradation.py`)
Raised as a hypothesis since gamma augmentation *is* a real intensity
transform in the pipeline (§ "color grading" discussion) and artifact
severity correlates with brightness. Ruled out by the same two proofs above,
both of which bypass the training-augmentation pipeline entirely:
- The pattern exists at **random init, before any training step** — no
  augmentation has been applied to anything yet.
- The constant-input test never touches `augment_gt` at all — it's a
  synthetic probe fed straight to the model, not a training pair.
- Structurally: `x**gamma` is a pointwise value transform with no spatial
  structure of its own; it cannot inject a specific exact spatial frequency
  (period-2 / period-4) into learned weights.

**One unproven, low-priority nuance worth a glance when reviewing the new
run's results**: gamma augmentation could still have a *secondary* influence
on how well-calibrated the trained network's brightness-dependent response
is (not on whether the artifact exists, only on how evenly its severity is
suppressed across brightness levels), since it changes what brightness range
the model sees most during training. If `tools/check_artifact_quality.py`
run on `runs/final` still shows a strong brightness-correlation in residual
artifact after the new fixes, this is worth a second look — otherwise treat
it as noise.

### What actually works, ranked

1. **Deterministic FFT notch (`evaluate.py::notch_artifacts`, `--post notch`,
   already the default)** — soft gaussian notch at the exact known bins
   (period-2 and period-4, both axis and diagonal). On a **40-image spread**
   of the real test set (`tools/check_artifact_quality.py`):
   - raw: median p4 = 2,843, worst = 2,332,263
   - **after notch: median p4 = 0.00, worst = 3.16** (GT is 1.94)
   - Robust across brightness — does NOT have the brightness-dependent
     weakness a small linear filter has (see next point).
   - **This is the safety net. It costs nothing, needs no training, and
     works on any checkpoint. Keep it regardless of what else changes.**

2. **Post-hoc learned 5×5 conv**, fitted (Adam, 400 steps) on a *frozen*
   model's output, with `artifact_freq_loss` as part of the objective:
   - Beat the notch on PSNR/SSIM (28.361 dB / 0.7605 vs notch's 27.147 /
     0.6972, held-out half of val) but is a **single fixed filter** — has
     real per-image variance correlated with brightness (worst case 000000,
     brightness 0.66, left at period4=168 even after this filter — vs the
     notch's 0.03 on the same image). A fixed-attenuation filter leaves
     residual proportional to input amplitude; the notch removes the bin
     near-completely regardless of amplitude. **This is why the notch stays
     as backstop even if joint training helps.**

3. **NOT tried yet, this is what `--use-refine --w-artifact` is for**: the
   same idea (learnable conv + targeted frequency loss) but trained **jointly**
   from scratch, not patched onto a frozen model. Hypothesis: the network's
   earlier layers can learn not to produce the brightness-scaled artifact in
   the first place, rather than compensating after the fact. **Unverified.**
   Sanity-checked only (dirac init = exact identity at step 0, confirmed
   0.000e+00 diff; gradient flows; one-sided loss confirmed correct on
   synthetic tests) — never run through a real training loop to convergence.

---

## 6. Loss (`src/losses.py`, 255 lines)

```
L = w_pix(1.0) · Charbonnier
  + w_fft(0.05) · L1(FFT)                    [unweighted — radial weighting tried, made it WORSE]
  + w_grad(0.15) · L1(Sobel)                 [the term that actually fixes denoiser-blur]
  + w_ssim(0.10) · (1-SSIM or 1-MS-SSIM if --ms-ssim)
  + w_range(0.0, off by default) · squared-hinge outside [0,1]
  + w_artifact(0.0, off by default) · one-sided log-compressed penalty at the
                                       exact artifact frequency bins
```

- **MS-SSIM** (`--ms-ssim`) exists because we're measurably behind a
  reference implementation on SSIM specifically (0.7333 vs our leaked-split
  0.7133) — optimise the multi-scale form of the metric we're behind on
  directly. Verified: 0.000000 on identical images, reacts correctly to
  noise.
- **Gamma speckle is now ALWAYS on** (`p_gamma_speckle=1.0` in
  `src/degradation.py`, was 30%) — physically correct multi-look model,
  matches measured residual skew (+0.34) and kurtosis (3.56) that Gaussian
  multiplicative noise doesn't reproduce.
- **`artifact_freq_loss`**: normalised by each image's own spectral median
  (so 1.0 = GT level, directly comparable to the period2/4 diagnostics
  everywhere else), log1p-compressed (raw ratios span 2 → 116,000×, would
  destabilise training uncompressed), one-sided (only penalises EXCESS over
  what GT has, never forces pred to add power it doesn't need).

---

## 7. `evaluate.py` — what KLA actually runs

- `--post` default is **`notch`** (`notch_artifacts` twice, around the
  clamp). Do not change this default without re-running
  `tools/check_artifact_quality.py` on whatever checkpoint you're about to
  ship.
- `cudnn.benchmark = False` here specifically (opposite of `train.py`) —
  autotuning costs ~8.5s on the first batch for zero steady-state gain at
  inference batch sizes; measured 5× penalty on a 400-image run if left on.
- Handles variable input sizes correctly (tested 64→300px, all correct 2×
  output shapes) — not hardcoded to 128→256.
- `load_model()` reconstructs via `ck['model_kwargs']`, falling back to
  `ck.get('head')` for older checkpoints that predate the kwargs dict.

---

## 8. The recommended training command (NOT YET RUN)

```bash
.venv/Scripts/python -u train.py --name final --config m --head resize_conv \
    --residual bicubic --input-transform log --ms-ssim \
    --use-refine --w-artifact 0.1 \
    --iters 60000 --batch 16 > final.log 2>&1
```

- **60,000 iters**, not 100,000 — `runs/main` plateaued hard after step
  60,000 (25.796 → 25.802 dB over the last 40k steps, i.e. nothing). Measured
  throughput with all current loss terms: **3.28 it/s → ~5.1 hours.**
- `--w-artifact 0.1` is an untuned starting guess.
- A checkpoint (`runs/final/`) already exists from a **paused, incomplete**
  earlier attempt at step 1,000/60,000 — no checkpoint was saved yet at that
  point (`args.json` + empty `metrics.csv` only), so resuming vs restarting
  is equivalent. Confirm `runs/final/last.pt` does not exist before assuming
  a resume is possible.

**After training, immediately run** (this is the rollback-relevant check —
confirms the retrain didn't silently reintroduce the leaked-split problem or
regress the artifact):
```bash
.venv/Scripts/python -u -c "
import sys; sys.path.insert(0,'.')
from src.data import SemiconData
SemiconData('train/train', device='cuda', val_size=320, seed=0)
"   # must print leakage check = 0

.venv/Scripts/python -u tools/check_artifact_quality.py --weights runs/final/best.pt --n 40
# compare median/worst p4 against this file's §5 baseline (runs/main: median
# 2843, worst 2.3M raw; 0.00/3.16 after notch)
```

---

## 9. File manifest (2,081 lines across the core library)

```
nafnet_sr.py (449)      model — verified identical body, adapted ends (§3-4)
train.py (384)          training entry point, all CLI flags documented in --help
evaluate.py (322)       SUBMISSION script — notch default, cudnn.benchmark=False
src/losses.py (255)     composite loss incl. MS-SSIM + artifact_freq_loss (§6)
src/degradation.py (355) GPU forward-model simulator, gamma speckle always-on
src/metrics.py (179)    PSNR/SSIM (skimage-matched) + post-processing
src/data.py (137)       source-aware GPU-resident dataset (§1 — the critical fix)

tools/                  17 scripts, every number in this file traces to one:
  check_artifact_quality.py  the §5/§8 diagnostic — RUN AFTER EVERY RETRAIN
  fit_kernel.py               solves the downsampling kernel by least squares
  validate_simulator.py       RUN AFTER ANY CHANGE TO degradation.py
  trace_stages.py             stage-by-stage artifact localisation
  test_notch2.py, test_clipping.py, test_artifact_cause.py  artifact diagnosis
  compare_heads.py, visualize_test.py, visualize_examples.py  visual checks
  bench_sweep.py, baseline.py  throughput / no-model reference numbers
```

## 10. Existing checkpoints (disk state as of writing)

| run | config | notes |
|---|---|---|
| `runs/main/` | m, head=resize_conv, residual=**none**, no ms-ssim/artifact/refine | **only completed 100k run.** Leaked split. 25.823 dB (untrustworthy number, see §1). Weights are fine to inspect/probe; do not cite its PSNR. |
| `runs/final/` | (new settings) | paused at step 1,000, **no checkpoint saved**, safe to restart |
| `runs/a_base` `a_nores` `a_gated` `a_log` `a_both` | s config, 6k iters, ablation sweep | leaked split, `a_both` picked as source of `residual=none`+`log` recommendation — **superseded by §5's finding that residual=bicubic is the current pick** |
| `runs/x_ctrl` `x_resize` `x_smooth` | s config, 4k iters, head comparison | leaked split, but the head-vs-head *relative* comparison (not absolute numbers) is still the basis for resize_conv being the head default |
| `runs/up_resize` | the −9dB disaster (§4) | kept as a documented negative result, never load for anything else |
| `runs/preview`, `runs/f_range`, `runs/f_icnr` | early/interrupted experiments | safe to delete if disk space is needed |
