"""Estimate the forward degradation model f: GT(256) -> NoisyLR(128).

Model hypothesis (from problem statement):
    y = D(x) + D(x)*ns + ng ,  ns~N(0,ss^2)  (speckle, multiplicative)
                               ng~N(0,sg^2)  (additive gaussian)
    D = some 2x downsampling operator

Strategy:
  1. Find which D best explains the data (minimise residual variance).
  2. With best D, per-image regress Var(residual | intensity bin) against intensity^2:
         Var(r) = ss^2 * xd^2 + sg^2   ->  slope=ss^2, intercept=sg^2
  3. Report distribution of (ss, sg) across images -> reveals random subsets.
  4. Spatial autocorrelation of residual -> is noise white at LR (added after D)?
"""
import numpy as np, glob, os, json

GT = sorted(glob.glob('train/train/GT/*.npy'))
LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))
assert len(GT) == len(LR)

def box(x):          # 2x2 average pool
    return x.reshape(x.shape[0]//2, 2, x.shape[1]//2, 2).mean(axis=(1, 3))

def subsample(x):    # take every other pixel (top-left)
    return x[0::2, 0::2]

def gauss_blur(x, sigma):
    r = max(1, int(3*sigma))
    k = np.exp(-(np.arange(-r, r+1)**2)/(2*sigma*sigma)); k /= k.sum()
    p = np.pad(x, r, mode='reflect')
    t = np.apply_along_axis(lambda m: np.convolve(m, k, 'valid'), 1, p)
    return np.apply_along_axis(lambda m: np.convolve(m, k, 'valid'), 0, t)

def blur_sub(x, sigma):
    return subsample(gauss_blur(x, sigma))

KERNELS = {
    'box2x2':        box,
    'subsample':     subsample,
    'blur0.5+sub':   lambda x: blur_sub(x, 0.5),
    'blur0.8+sub':   lambda x: blur_sub(x, 0.8),
    'blur1.0+sub':   lambda x: blur_sub(x, 1.0),
}

# ---------- STEP 1: which downsampler? ----------
# Use the *lowest-noise* images so the kernel signal isn't drowned by noise.
# First pass: rank images by residual magnitude under box filter.
print('scanning for low-noise images...')
scores = []
for i in range(0, len(GT), 4):                       # sample 800 images
    g = np.load(GT[i]); y = np.load(LR[i])
    scores.append((float(np.mean((y - box(g))**2)), i))
scores.sort()
quiet = [i for _, i in scores[:40]]
print('quietest residual MSE: %.3e   loudest sampled: %.3e' % (scores[0][0], scores[-1][0]))

print('\n--- STEP 1: downsampling kernel fit (on 40 quietest images) ---')
kern_res = {}
for name, fn in KERNELS.items():
    errs = []
    for i in quiet:
        g = np.load(GT[i]); y = np.load(LR[i])
        errs.append(np.mean((y - fn(g))**2))
    kern_res[name] = float(np.mean(errs))
    print('  %-14s mean residual MSE = %.6e' % (name, kern_res[name]))
best_kernel = min(kern_res, key=kern_res.get)
print('  => best: %s' % best_kernel)
D = KERNELS[best_kernel]

# ---------- STEP 2: per-image noise parameter estimation ----------
print('\n--- STEP 2: per-image (speckle, gaussian) estimation ---')
N = 600                                              # subsample for speed
idx = np.linspace(0, len(GT)-1, N).astype(int)
ss_list, sg_list, r2_list = [], [], []
for i in idx:
    g = np.load(GT[i]); y = np.load(LR[i])
    xd = D(g)
    r = (y - xd).ravel()
    v = xd.ravel()
    # bin by intensity, compute residual variance per bin
    nb = 24
    edges = np.quantile(v, np.linspace(0, 1, nb+1))
    edges = np.unique(edges)
    if len(edges) < 6:
        continue
    b = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges)-2)
    xs, ys, ws = [], [], []
    for k in range(len(edges)-1):
        m = b == k
        c = int(m.sum())
        if c < 50:
            continue
        xs.append(np.mean(v[m])**2)
        ys.append(np.var(r[m]))
        ws.append(c)
    if len(xs) < 5:
        continue
    xs = np.array(xs); ys = np.array(ys); ws = np.array(ws, float)
    # weighted least squares: ys = a*xs + c
    A = np.stack([xs, np.ones_like(xs)], 1)
    W = ws / ws.sum()
    ATA = A.T @ (A * W[:, None]); ATb = A.T @ (ys * W)
    try:
        a, c = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        continue
    pred = A @ np.array([a, c])
    ss_res = np.sum(W*(ys-pred)**2); ss_tot = np.sum(W*(ys-np.sum(W*ys))**2)
    ss_list.append(np.sqrt(max(a, 0))); sg_list.append(np.sqrt(max(c, 0)))
    r2_list.append(1 - ss_res/ss_tot if ss_tot > 0 else 0)

ss = np.array(ss_list); sg = np.array(sg_list); r2 = np.array(r2_list)
print('  fitted %d images, median regression R^2 = %.3f' % (len(ss), np.median(r2)))
def q(name, a):
    print('  %-9s min=%.4f  p05=%.4f  p25=%.4f  med=%.4f  p75=%.4f  p95=%.4f  max=%.4f'
          % (name, a.min(), *np.quantile(a, [.05, .25, .5, .75, .95]), a.max()))
q('speckle', ss); q('gaussian', sg)
print('  frac speckle<0.01 : %.3f' % np.mean(ss < 0.01))
print('  frac gauss  <0.01 : %.3f' % np.mean(sg < 0.01))
print('  frac BOTH   <0.01 : %.3f' % np.mean((ss < 0.01) & (sg < 0.01)))
print('  corr(speckle,gauss) = %.3f' % np.corrcoef(ss, sg)[0, 1])

# ---------- STEP 3: residual whiteness ----------
print('\n--- STEP 3: residual spatial autocorrelation (lag-1) ---')
ac_h, ac_v = [], []
for i in idx[::6]:
    g = np.load(GT[i]); y = np.load(LR[i])
    r = y - D(g)
    r = r - r.mean()
    d = r.std()
    if d < 1e-6:
        continue
    ac_h.append(np.mean(r[:, :-1]*r[:, 1:])/d**2)
    ac_v.append(np.mean(r[:-1, :]*r[1:, :])/d**2)
print('  median lag-1 horiz = %.4f   vert = %.4f' % (np.median(ac_h), np.median(ac_v)))
print('  (near 0 => white noise applied AFTER downsampling)')

# ---------- STEP 4: GT / LR global stats ----------
print('\n--- STEP 4: intensity stats ---')
gmin, gmax, gmean, lmin, lmax = [], [], [], [], []
for i in idx[::3]:
    g = np.load(GT[i]); y = np.load(LR[i])
    gmin.append(g.min()); gmax.append(g.max()); gmean.append(g.mean())
    lmin.append(y.min()); lmax.append(y.max())
print('  GT  min: med=%.4f  max: med=%.4f  mean: med=%.4f' % (np.median(gmin), np.median(gmax), np.median(gmean)))
print('  GT  frac images with max==1.0 : %.3f' % np.mean(np.array(gmax) > 0.999))
print('  GT  frac images with min==0.0 : %.3f' % np.mean(np.array(gmin) < 0.001))
print('  LR  min: p05=%.4f med=%.4f   max: med=%.4f p95=%.4f'
      % (np.quantile(lmin, .05), np.median(lmin), np.median(lmax), np.quantile(lmax, .95)))

json.dump({'kernel': best_kernel, 'kernel_scores': kern_res,
           'speckle_q': np.quantile(ss, [.05, .25, .5, .75, .95]).tolist(),
           'gauss_q': np.quantile(sg, [.05, .25, .5, .75, .95]).tolist()},
          open('degradation_fit.json', 'w'), indent=2)
print('\nwrote degradation_fit.json')
