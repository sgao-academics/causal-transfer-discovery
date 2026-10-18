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
CausalGPT: Cross-dataset causal knowledge transfer.
Train NOTEARS on TCGA(-BRCA) → warm-start BRCA fine-tuning.
"""
import numpy as np, time, os, pandas as pd, json
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

ROOT = os.path.dirname(os.path.abspath(__file__))
TCGA_DIR = DATA_DIR
DATA = os.path.join(ROOT,'data')
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)

# ========== 1. Select top-d genes by MAD (from non-BRCA cancers only) ==========
d = 200
print(f'Selecting top-{d} genes from non-BRCA TCGA...')
other = []
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    other.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
other = pd.concat(other,axis=0)
mads = np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads = np.nan_to_num(mads,nan=0)
top_genes = list(other.columns[np.argsort(mads)[-d:]])
print(f'  Non-BRCA: {other.shape} → {d} genes')

# ========== 2. TCGA(-BRCA) training data ==========
tcga_X = other[top_genes].values.astype(np.float64)
tcga_X = np.nan_to_num(tcga_X,nan=0)
tcga_X = (tcga_X-tcga_X.mean(0))/(tcga_X.std(0)+1e-12)
print(f'  TCGA(-BRCA): {tcga_X.shape}')

# ========== 3. BRCA held-out ==========
brca = pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common = [g for g in top_genes if g in brca.columns]
d_actual = len(common)
brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X,nan=0)
brca_X = (brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)
# Remap tcga to same gene order
tcga_common_idx = [list(top_genes).index(g) for g in common]
tcga_X = tcga_X[:, tcga_common_idx]
print(f'  BRCA: {brca_X.shape} (shared {d_actual} genes)')
d = d_actual

# ========== 4. NOTEARS ==========
def notears(X, init_W=None, lam=0.01, max_outer=40, verbose=True):
    n,d_ = X.shape; cov = X.T@X/n
    w_vec = init_W.flatten() if init_W is not None else np.random.randn(d_*d_)*0.1
    rho,alpha = 0.1,0.0; best_w,best_h = w_vec.copy(),1e10
    steps = []
    for it in range(max_outer):
        def lag(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
            return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T
            return (cov@(W-np.eye(d_))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res = minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':100,'ftol':1e-10})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w = abs(hv),w_vec.copy()
        steps.append({'iter':it,'h':float(hv),'rho':rho,'alpha':alpha})
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if verbose and it%10==0: print(f'    iter {it:2d}/{max_outer} h={hv:.2e}',flush=True)
    return best_w.reshape(d_,d_), best_h, steps

# ========== 5. Experiment A: NOTEARS on TCGA(-BRCA) ==========
print('\n(A) NOTEARS on TCGA(-BRCA) [source domain]...')
t0=time.time()
W_tcga, h_tcga, steps_tcga = notears(tcga_X, max_outer=40)
dt_tcga = time.time()-t0
nz_tcga = int((np.abs(W_tcga)>0.3).sum())
print(f'  Edges={nz_tcga} h={h_tcga:.2e} {dt_tcga:.0f}s')

# ========== 6. Experiment B: NOTEARS on BRCA from scratch ==========
print('\n(B) NOTEARS on BRCA from scratch [baseline]...')
t0=time.time()
W_scratch, h_scratch, steps_s = notears(brca_X, max_outer=40)
dt_scratch = time.time()-t0
nz_scratch = int((np.abs(W_scratch)>0.3).sum())
print(f'  Edges={nz_scratch} h={h_scratch:.2e} {dt_scratch:.0f}s')

# ========== 7. Experiment C: NOTEARS on BRCA warm-started with W_tcga ==========
print('\n(C) NOTEARS on BRCA warm-started with TCGA(-BRCA) W [transfer]...')
t0=time.time()
W_transfer, h_transfer, steps_t = notears(brca_X, init_W=W_tcga, max_outer=40)
dt_transfer = time.time()-t0
nz_transfer = int((np.abs(W_transfer)>0.3).sum())
print(f'  Edges={nz_transfer} h={h_transfer:.2e} {dt_transfer:.0f}s')

# ========== 8. RESULTS ==========
print(f'\n{"="*65}')
print(f'RESULTS: Causal Knowledge Transfer (d={d})')
print(f'{"="*65}')
print(f'  {"Method":30s} {"tau=0.3":>8s} {"h(W)":>10s} {"time":>6s} {"iters":>6s}')
print(f'  {"-"*65}')
print(f'  {"TCGA(-BRCA) source":30s} {nz_tcga:8d} {h_tcga:10.2e} {dt_tcga:6.0f}s {len(steps_tcga):6d}')
print(f'  {"BRCA scratch":30s} {nz_scratch:8d} {h_scratch:10.2e} {dt_scratch:6.0f}s {len(steps_s):6d}')
print(f'  {"BRCA transfer (TCGA warm)":30s} {nz_transfer:8d} {h_transfer:10.2e} {dt_transfer:6.0f}s {len(steps_t):6d}')
delta = nz_transfer - nz_scratch
print(f'  {"Delta (transfer - scratch)":30s} {delta:+8d}')

# Edge overlap
e_tcga = set(); e_scratch = set(); e_transfer = set()
for i in range(d):
    for j in range(d):
        if i!=j and abs(W_tcga[i,j])>0.3: e_tcga.add((i,j))
        if i!=j and abs(W_scratch[i,j])>0.3: e_scratch.add((i,j))
        if i!=j and abs(W_transfer[i,j])>0.3: e_transfer.add((i,j))
print(f'\n  Edge overlap (tau=0.3):')
print(f'    TCGA ∩ scratch: {len(e_tcga & e_scratch)}')
print(f'    TCGA ∩ transfer: {len(e_tcga & e_transfer)}')
print(f'    scratch ∩ transfer: {len(e_scratch & e_transfer)}')
print(f'    All 3 shared: {len(e_tcga & e_scratch & e_transfer)}')

# Edge weight correlation
flat_tcga = W_tcga.flatten(); flat_scratch = W_scratch.flatten()
flat_transfer = W_transfer.flatten()
from scipy.stats import pearsonr, spearmanr
r_ts = pearsonr(flat_tcga, flat_scratch)[0]
r_tt = pearsonr(flat_tcga, flat_transfer)[0]
rho_ts = spearmanr(flat_tcga, flat_scratch)[0]
rho_tt = spearmanr(flat_tcga, flat_transfer)[0]
print(f'\n  Weight correlation with TCGA source:')
print(f'    Scratch Pearson r={r_ts:.4f} Spearman rho={rho_ts:.4f}')
print(f'    Transfer Pearson r={r_tt:.4f} Spearman rho={rho_tt:.4f}')

if dt_scratch>0:
    print(f'\n  Convergence speedup: {dt_scratch/dt_transfer:.1f}x')
    print(f'  NOTEARS iterations: scratch={len(steps_s)} transfer={len(steps_t)}')

# Save
json.dump({
    'd':d,
    'tcga_source':{'edges':nz_tcga,'h':float(h_tcga),'time':dt_tcga,'iters':len(steps_tcga)},
    'brca_scratch':{'edges':nz_scratch,'h':float(h_scratch),'time':dt_scratch,'iters':len(steps_s)},
    'brca_transfer':{'edges':nz_transfer,'h':float(h_transfer),'time':dt_transfer,'iters':len(steps_t)},
    'delta':delta,'r_scratch':r_ts,'r_transfer':r_tt,'rho_scratch':rho_ts,'rho_transfer':rho_tt,
},open(os.path.join(OUT,'transfer.json'),'w'),indent=2)
print(f'\nSaved: {os.path.join(OUT,"transfer.json")}')
