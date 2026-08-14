import numpy as np, glob

GT = sorted(glob.glob('train/train/GT/*.npy'))
LR = sorted(glob.glob('train/train/NoisyLR/*.npy'))
TE = sorted(glob.glob('Test_NoisyLR/NoisyLR/*.npy'))

def box(x): return x.reshape(x.shape[0]//2,2,x.shape[1]//2,2).mean(axis=(1,3))

# ---- 1. Is GT exactly per-image min-max normalised? (all 3200) ----
print('--- 1. GT normalisation, all 3200 ---')
mn = np.empty(len(GT)); mx = np.empty(len(GT))
for i,f in enumerate(GT):
    g = np.load(f); mn[i]=g.min(); mx[i]=g.max()
print('  min: exactly 0.0 in %d/%d   max: exactly 1.0 in %d/%d'
      % ((mn==0).sum(), len(GT), (mx==1).sum(), len(GT)))
print('  min range [%.2e, %.2e]   max range [%.6f, %.6f]' % (mn.min(), mn.max(), mx.min(), mx.max()))

# ---- 2. speckle applied at HR (before pooling) or at LR (after)? ----
# HR model: Var(r) = ss^2 * mean(x_i^2)/4 + sg^2      (predictor P_hr)
# LR model: Var(r) = ss^2 * (mean x_i)^2   + sg^2      (predictor P_lr)
print('\n--- 2. speckle before vs after downsampling ---')
def fit(pred, resid):
    nb=24
    e=np.unique(np.quantile(pred,np.linspace(0,1,nb+1)))
    if len(e)<6: return None
    b=np.clip(np.digitize(pred,e[1:-1]),0,len(e)-2)
    xs,ys,ws=[],[],[]
    for k in range(len(e)-1):
        m=b==k; c=int(m.sum())
        if c<50: continue
        xs.append(pred[m].mean()); ys.append(resid[m].var()); ws.append(c)
    if len(xs)<5: return None
    xs=np.array(xs);ys=np.array(ys);w=np.array(ws,float);w/=w.sum()
    A=np.stack([xs,np.ones_like(xs)],1)
    try: a,c=np.linalg.solve(A.T@(A*w[:,None]), A.T@(ys*w))
    except np.linalg.LinAlgError: return None
    p=A@np.array([a,c]); m=(w*ys).sum()
    return 1-(w*(ys-p)**2).sum()/max((w*(ys-m)**2).sum(),1e-30), a, c

r2hr,r2lr=[],[]
idx=np.linspace(0,len(GT)-1,300).astype(int)
for i in idx:
    g=np.load(GT[i]); y=np.load(LR[i])
    xd=box(g)
    x2 = (g**2).reshape(128,2,128,2).mean(axis=(1,3))     # mean of x_i^2
    r=(y-xd).ravel()
    a=fit((x2/4).ravel(), r); b=fit((xd**2).ravel(), r)
    if a and b: r2hr.append(a[0]); r2lr.append(b[0])
print('  median R^2  speckle-at-HR = %.4f   speckle-at-LR = %.4f' % (np.median(r2hr), np.median(r2lr)))
print('  LR model wins on %.1f%% of images' % (100*np.mean(np.array(r2lr)>np.array(r2hr))))

# ---- 3. residual normality ----
print('\n--- 3. normalised residual shape ---')
ku,sk=[],[]
for i in idx[::4]:
    g=np.load(GT[i]); y=np.load(LR[i]); xd=box(g)
    r=(y-xd)/np.maximum(xd,1e-3)          # divide out speckle scaling
    m=xd>0.15                              # only where scaling is meaningful
    if m.sum()<500: continue
    v=r[m]; v=(v-v.mean())/v.std()
    ku.append((v**4).mean()); sk.append((v**3).mean())
print('  median kurtosis = %.3f (gaussian=3.0), skew = %.3f (gaussian=0)' % (np.median(ku), np.median(sk)))

# ---- 4. TRAIN vs TEST NoisyLR distribution ----
print('\n--- 4. train NoisyLR vs test NoisyLR (distribution shift?) ---')
def stats(files):
    o=[]
    for f in files:
        a=np.load(f)
        # local-gradient proxy for noise level, and global stats
        o.append([a.mean(), a.std(), a.min(), a.max(),
                  np.abs(np.diff(a,axis=1)).mean()])
    return np.array(o)
tr = stats(LR[::8]); te = stats(TE)
names=['mean','std','min','max','|dx|']
for j,n in enumerate(names):
    print('  %-5s train med=%8.4f p05=%8.4f p95=%8.4f  |  test med=%8.4f p05=%8.4f p95=%8.4f'
          % (n, np.median(tr[:,j]), np.quantile(tr[:,j],.05), np.quantile(tr[:,j],.95),
                np.median(te[:,j]), np.quantile(te[:,j],.05), np.quantile(te[:,j],.95)))
