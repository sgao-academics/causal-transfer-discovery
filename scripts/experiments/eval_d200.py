# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""Fast eval: d=200 NOTEARS scratch vs backbone warm-start."""
import numpy as np, time, os, pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT,'data')
TCGA = DATA_DIR
CK = os.path.join(ROOT,'checkpoints','causal-gpt-v1-d500')

# Load backbone, pick top-200 genes by |W|
U0 = np.load(os.path.join(CK,'U0.npy'))
V0 = np.load(os.path.join(CK,'V0.npy'))
W_bb_full = U0 @ V0.T
row_w = np.abs(W_bb_full).sum(axis=0)
bb_top200 = np.argsort(-row_w)[:200]

gene_idx = np.load(os.path.join(CK,'gene_idx.npy'))
with open(os.path.join(DATA,'gene_list.txt')) as f:
    all_genes = f.read().strip().split('\n')
top_genes = [all_genes[i] for i in gene_idx]
sel = [top_genes[i] for i in bb_top200]

brca = pd.read_csv(os.path.join(TCGA,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common = [g for g in sel if g in brca.columns]
d = len(common)
brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X,nan=0)
brca_X = (brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)
print(f'BRCA: {brca_X.shape[0]}x{d}')

bb_map = [list(top_genes).index(g) for g in common]
W_bb = U0[bb_map] @ V0[bb_map].T
print(f'Backbone max|W|={np.abs(W_bb).max():.4f}')

def notears(X,init_W=None,lam=0.01,max_outer=30):
    n,d=X.shape; cov=X.T@X/n
    w_vec = init_W.flatten() if init_W is not None else (np.random.seed(42),np.random.randn(d*d)*0.1)[1]
    rho,alpha=0.1,0.0; best_w,best_h=w_vec.copy(),1e10
    for it in range(max_outer):
        def lag(w):
            W=w.reshape(d,d); hv=np.trace(expm(W*W))-d
            return 0.5*np.trace((np.eye(d)-W).T@cov@(np.eye(d)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W=w.reshape(d,d); hv=np.trace(expm(W*W))-d; dh=2*W*expm(W*W).T
            return (cov@(W-np.eye(d))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':100,'ftol':1e-10})
        w_vec=res.x; W=w_vec.reshape(d,d); hv=np.trace(expm(W*W))-d
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w=abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if it%5==0: print(f'  outer {it:2d}/{max_outer} h={hv:.2e}',flush=True)
    return best_w.reshape(d,d),best_h

print('\n(A) Scratch...')
t0=time.time(); Ws,hs=notears(brca_X,max_outer=30)
nz_s=int((np.abs(Ws)>0.3).sum()); dt_s=time.time()-t0
print(f'  Edges={nz_s} h={hs:.2e} {dt_s:.0f}s')

print('\n(B) Fine-tuned...')
t0=time.time(); Wf,hf=notears(brca_X,init_W=W_bb,max_outer=30)
nz_f=int((np.abs(Wf)>0.3).sum()); dt_f=time.time()-t0
print(f'  Edges={nz_f} h={hf:.2e} {dt_f:.0f}s')

print(f'\n=== RESULTS d={d} ===')
print(f'  Scratch:   {nz_s:4d} edges h={hs:.2e} {dt_s:.0f}s')
print(f'  Fine-tuned:{nz_f:4d} edges h={hf:.2e} {dt_f:.0f}s')
print(f'  Delta:     {nz_f-nz_s:+4d}')
e_s=set(); e_f=set()
for i in range(d):
    for j in range(d):
        if i!=j and abs(Ws[i,j])>0.3: e_s.add((i,j))
        if i!=j and abs(Wf[i,j])>0.3: e_f.add((i,j))
print(f'  Shared:{len(e_s&e_f)} S-only:{len(e_s-e_s)} F-only:{len(e_f-e_s)}')
if dt_s>0: print(f'  Speedup: {dt_s/dt_f:.1f}x')
