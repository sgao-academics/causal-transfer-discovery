# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""
CausalGPT: Prior-regularized NOTEARS.
Loss = MSE + L1 + DAG + lambda_prior * ||W - W_sparse||^2
where W_sparse = thresholded source-domain W (keep |W| > tau).
"""
import numpy as np, time, os, pandas as pd, json
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

TCGA_DIR = DATA_DIR
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)
d = 500

# ========== 1. Data ==========
print(f'Loading top-{d} genes...')
other = []
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    try: other.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
    except: continue
other = pd.concat(other,axis=0)
mads = np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads = np.nan_to_num(mads,nan=0)
top_genes = list(other.columns[np.argsort(mads)[-d:]])
tcga_X = other[top_genes].values.astype(np.float64)
tcga_X = np.nan_to_num(tcga_X,nan=0)
tcga_X = (tcga_X-tcga_X.mean(0))/(tcga_X.std(0)+1e-12)

brca = pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common = [g for g in top_genes if g in brca.columns]
d_actual = len(common)
brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X,nan=0)
brca_X = (brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)
tcga_ci = [list(top_genes).index(g) for g in common]; tcga_X = tcga_X[:,tcga_ci]
d = d_actual
print(f'  TCGA(-BRCA): {tcga_X.shape}  BRCA: {brca_X.shape}')

# ========== 2. Get source W_prior (cached) ==========
CACHE = os.path.join(OUT, 'W_src_d500.npy')
if os.path.exists(CACHE):
    print('Loading cached W_source...')
    W_src = np.load(CACHE)
else:
    print('Running NOTEARS on TCGA(-BRCA)...')
    from _eval_transfer_d500 import notears  # reuse if available
    # Inline NOTEARS for standalone
    n,d_=tcga_X.shape; cov=tcga_X.T@tcga_X/n
    w_vec=np.random.randn(d_*d_)*0.1; rho,alpha=0.1,0.0; best_w=w_vec.copy()
    for it in range(25):
        def lag(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
            return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+0.01*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T
            return (cov@(W-np.eye(d_))+0.01*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':120})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if it%10==0: print(f'  src iter {it} h={hv:.2e}')
        best_w=w_vec.copy()
    W_src=best_w.reshape(d_,d_); np.save(CACHE,W_src)
nz_src=int((np.abs(W_src)>0.3).sum())
print(f'  W_source: {nz_src} edges at tau=0.3')

# ========== 3. Sparse prior: keep only strong edges ==========
tau_prior = 0.3
W_prior = W_src.copy()
mask = np.abs(W_prior) < tau_prior
W_prior[mask] = 0.0
np.fill_diagonal(W_prior, 0)
nz_prior = int((np.abs(W_prior)>0).sum())
print(f'  W_prior (sparse, tau={tau_prior}): {nz_prior} non-zero entries')

# ========== 4. Prior-regularized NOTEARS ==========
def notears_prior(X, W_prior, lambda_prior=1.0, lam=0.01, max_outer=30):
    n,d_=X.shape; cov=X.T@X/n
    w_vec=np.random.randn(d_*d_)*0.1; rho,alpha=0.1,0.0
    best_w,best_h=w_vec.copy(),1e10; steps=[]
    for it in range(max_outer):
        def lag(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
            prior_loss = lambda_prior * np.sum((W - W_prior)**2) / (d_*d_)
            return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv+prior_loss
        def lag_grad(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T
            prior_grad = 2*lambda_prior*(W-W_prior)/(d_*d_)
            return (cov@(W-np.eye(d_))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()+prior_grad.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':100})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w=abs(hv),w_vec.copy()
        steps.append({'iter':it,'h':float(hv)})
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if it%10==0: print(f'    iter {it:2d} h={hv:.2e}',flush=True)
    return best_w.reshape(d_,d_),best_h,steps

# ========== 5. Experiment: sweep lambda_prior ==========
results = {}
for lam_p in [0.01, 0.1, 0.5, 1.0, 2.0]:
    print(f'\n--- lambda_prior = {lam_p} ---')
    t0=time.time()
    W_p,h_p,steps = notears_prior(brca_X, W_prior, lambda_prior=lam_p, max_outer=25)
    dt=time.time()-t0
    nz03=int((np.abs(W_p)>0.3).sum()); nz01=int((np.abs(W_p)>0.1).sum())
    
    # Overlap with source
    e_p=set(); e_src=set()
    for i in range(d):
        for j in range(d):
            if i!=j and abs(W_p[i,j])>0.3: e_p.add((i,j))
            if i!=j and abs(W_src[i,j])>0.3: e_src.add((i,j))
    overlap=len(e_p&e_src)
    
    # Pearson r with source
    from scipy.stats import pearsonr
    r_val=pearsonr(W_p.flatten(),W_src.flatten())[0]
    
    results[lam_p]={
        'tau03':nz03,'tau01':nz01,'h':float(h_p),'overlap':overlap,
        'r':r_val,'time':dt,'iters':len(steps)
    }
    print(f'  tau0.3={nz03} tau0.1={nz01} overlap={overlap}/{nz_src} r={r_val:.4f} {dt:.0f}s')

# ========== 6. Summary ==========
print(f'\n{"="*75}')
print(f'PRIOR REGULARIZATION RESULTS (d={d})')
print(f'{"="*75}')
print(f'  {"lambda":>8s} {"tau=0.3":>8s} {"tau=0.1":>8s} {"overlap":>8s} {"%src":>6s} {"r":>8s} {"time":>6s} {"iters":>6s}')
print(f'  {"-"*75}')
for lam_p in sorted(results.keys()):
    r=results[lam_p]
    print(f'  {lam_p:8.2f} {r["tau03"]:8d} {r["tau01"]:8d} {r["overlap"]:8d} {r["overlap"]/nz_src*100:5.1f}% '
          f'{r["r"]:8.4f} {r["time"]:6.0f}s {r["iters"]:6d}')

# Best lambda by overlap
best_lam = max(results.keys(), key=lambda k: results[k]['overlap'])
best = results[best_lam]
print(f'\n  Best lambda={best_lam:.2f}: {best["tau03"]} edges, '
      f'overlap={best["overlap"]}/{nz_src} ({best["overlap"]/nz_src*100:.0f}%), r={best["r"]:.4f}')

json.dump({str(k):v for k,v in results.items()},
          open(os.path.join(OUT,'prior_sweep_d500.json'),'w'),indent=2)
