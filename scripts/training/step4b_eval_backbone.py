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
Evaluate CausalGPT backbone on BRCA held-out:
  (A) NOTEARS from scratch
  (B) NOTEARS warm-started with W = UV^T (fine-tuned)
  Compare edge counts and convergence.
"""
import os, sys, json, time, numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import median_abs_deviation

ROOT = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(ROOT,'checkpoints','causal-gpt-v1-d500')
DATA = os.path.join(ROOT,'data')
TCGA_DIR = DATA_DIR
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)

# ========== 1. LOAD BRCA DATA (d=500, same genes as backbone) ==========
print('Loading BRCA...')
gene_idx = np.load(os.path.join(CK,'gene_idx.npy'))
brca = pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
# Get gene names from gene_list.txt (full 16078)
with open(os.path.join(DATA,'gene_list.txt')) as f:
    all_genes = f.read().strip().split('\n')
top_genes = [all_genes[i] for i in gene_idx]
common = [g for g in top_genes if g in brca.columns]
print(f'  Genes: {len(common)}/{len(top_genes)} match BRCA columns')
brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X,nan=0)
brca_X = (brca_X - brca_X.mean(0))/(brca_X.std(0)+1e-12)
d = len(common)
print(f'  BRCA: {brca_X.shape[0]} samples x {d} genes')

# ========== 2. LOAD BACKBONE ==========
U0 = np.load(os.path.join(CK,'U0.npy'))
V0 = np.load(os.path.join(CK,'V0.npy'))
# Map backbone gene order to BRCA gene order
bb_idx = [list(top_genes).index(g) for g in common]
U_sub = U0[bb_idx]; V_sub = V0[bb_idx]
W_bb = U_sub @ V_sub.T
max_w_bb = float(np.abs(W_bb).max())
print(f'  Backbone W: max|W|={max_w_bb:.4f}, tau0.1 edges={(np.abs(W_bb)>0.1).sum()}')

# ========== 3. NOTEARS L-BFGS-B ==========
def notears(X, init_W=None, lam=0.01, max_outer=50, seed=42):
    n,d = X.shape; cov = X.T@X/n
    if init_W is None:
        np.random.seed(seed); w_vec = np.random.randn(d*d)*0.1
    else:
        w_vec = init_W.flatten()
    rho,alpha = 0.1,0.0; best_w,best_h = w_vec.copy(),1e10
    for it in range(max_outer):
        def lag(w):
            W=w.reshape(d,d); hv=np.trace(expm(W*W))-d
            return 0.5*np.trace((np.eye(d)-W).T@cov@(np.eye(d)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W=w.reshape(d,d); hv=np.trace(expm(W*W))-d
            dh=2*W*expm(W*W).T
            return (cov@(W-np.eye(d))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res = minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,
                      options={'maxiter':200,'ftol':1e-12,'gtol':1e-12})
        w_vec=res.x; W=w_vec.reshape(d,d); hv=np.trace(expm(W*W))-d
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w = abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0 and abs(hv)>1e-10: rho=min(rho*2,1e10)
    return best_w.reshape(d,d), best_h

# (A) Scratch
print('\n(A) NOTEARS from scratch...')
t0=time.time()
W_s,h_s = notears(brca_X, init_W=None, max_outer=50)
nz_s03 = int(np.sum(np.abs(W_s)>0.3)); nz_s01 = int(np.sum(np.abs(W_s)>0.1))
t_s = time.time()-t0
print(f'  tau0.3={nz_s03} tau0.1={nz_s01} h={h_s:.2e} {t_s:.0f}s')

# (B) Fine-tuned (warm-started with backbone)
print('\n(B) CausalGPT fine-tuned (W=UV^T warm-start)...')
t0=time.time()
W_ft,h_ft = notears(brca_X, init_W=W_bb, max_outer=50)
nz_ft03 = int(np.sum(np.abs(W_ft)>0.3)); nz_ft01 = int(np.sum(np.abs(W_ft)>0.1))
t_ft = time.time()-t0
print(f'  tau0.3={nz_ft03} tau0.1={nz_ft01} h={h_ft:.2e} {t_ft:.0f}s')

# ========== 4. RESULTS ==========
print(f'\n{"="*55}')
print(f'RESULTS (BRCA held-out, d={d})')
print(f'{"="*55}')
print(f'  {"Method":25s} {"tau=0.3":>8s} {"tau=0.1":>8s} {"h(W)":>10s} {"time":>6s}')
print(f'  {"-"*55}')
print(f'  {"Scratch":25s} {nz_s03:8d} {nz_s01:8d} {h_s:10.2e} {t_s:6.0f}s')
print(f'  {"Fine-tuned":25s} {nz_ft03:8d} {nz_ft01:8d} {h_ft:10.2e} {t_ft:6.0f}s')
delta03 = nz_ft03-nz_s03; delta01 = nz_ft01-nz_s01
print(f'  {"Delta":25s} {delta03:+8d} {delta01:+8d}')
if t_s>0 and t_ft>0: print(f'  {"Speedup":25s} {t_s/t_ft:8.1f}x')

# Edge overlap
e_s = set(); e_ft = set()
for i in range(d):
    for j in range(d):
        if i!=j and abs(W_s[i,j])>0.3: e_s.add((i,j))
        if i!=j and abs(W_ft[i,j])>0.3: e_ft.add((i,j))
sh = len(e_s&e_ft)
print(f'  Shared edges: {sh} | Scratch-only: {len(e_s-e_ft)} | FT-only: {len(e_ft-e_s)}')

# Save
json.dump({
    'd':d, 'scratch':{'tau03':nz_s03,'tau01':nz_s01,'h':float(h_s),'time':t_s},
    'finetuned':{'tau03':nz_ft03,'tau01':nz_ft01,'h':float(h_ft),'time':t_ft},
    'delta_tau03':delta03,'delta_tau01':delta01,'shared':sh,
    'backbone_max_w':max_w_bb,
},open(os.path.join(OUT,'eval_bone.json'),'w'),indent=2)
print(f'\nSaved: {os.path.join(OUT,"eval_bone.json")}')
