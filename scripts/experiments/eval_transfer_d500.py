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
CausalGPT: d=500 transfer experiment.
Source: TCGA(-BRCA) NOTEARS → warm-start BRCA NOTEARS.
"""
import numpy as np, time, os, pandas as pd, json
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

TCGA_DIR = DATA_DIR
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)

d = 500
print(f'Selecting top-{d} genes from non-BRCA...')
other = []
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    try:
        other.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
    except:
        continue
other = pd.concat(other,axis=0)
mads = np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads = np.nan_to_num(mads,nan=0)
top_genes = list(other.columns[np.argsort(mads)[-d:]])
tcga_X = other[top_genes].values.astype(np.float64)
tcga_X = np.nan_to_num(tcga_X,nan=0)
tcga_X = (tcga_X-tcga_X.mean(0))/(tcga_X.std(0)+1e-12)
print(f'  TCGA(-BRCA): {tcga_X.shape}')

brca = pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common = [g for g in top_genes if g in brca.columns]
d_actual = len(common)
brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X,nan=0)
brca_X = (brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)
tcga_ci = [list(top_genes).index(g) for g in common]
tcga_X = tcga_X[:,tcga_ci]
print(f'  BRCA: {brca_X.shape} ({d_actual} shared genes)')
d = d_actual

def notears(X, init_W=None, lam=0.01, max_outer=40):
    n,d_=X.shape; cov=X.T@X/n
    w_vec=init_W.flatten() if init_W is not None else np.random.randn(d_*d_)*0.1
    rho,alpha=0.1,0.0; best_w,best_h=w_vec.copy(),1e10; steps=[]
    for it in range(max_outer):
        def lag(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
            return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T
            return (cov@(W-np.eye(d_))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':150,'ftol':1e-10})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w=abs(hv),w_vec.copy()
        steps.append({'iter':it,'h':float(hv)})
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if it%10==0: print(f'    iter {it:2d}/{max_outer} h={hv:.2e}',flush=True)
    return best_w.reshape(d_,d_),best_h,steps

# A: Source domain
print('\n(A) TCGA(-BRCA) source NOTEARS...')
t0=time.time()
W_src,h_src,steps_src=notears(tcga_X,max_outer=30)
nz_src=int((np.abs(W_src)>0.3).sum())
print(f'  Edges={nz_src} h={h_src:.2e} {time.time()-t0:.0f}s')

# B: BRCA scratch
print('\n(B) BRCA scratch...')
t0=time.time()
W_sc,h_sc,steps_sc=notears(brca_X,max_outer=30)
nz_sc=int((np.abs(W_sc)>0.3).sum())
print(f'  Edges={nz_sc} h={h_sc:.2e} {time.time()-t0:.0f}s')

# C: BRCA transfer
print('\n(C) BRCA transfer (TCGA warm)...')
t0=time.time()
W_tr,h_tr,steps_tr=notears(brca_X,init_W=W_src,max_outer=30)
nz_tr=int((np.abs(W_tr)>0.3).sum())
print(f'  Edges={nz_tr} h={h_tr:.2e} {time.time()-t0:.0f}s')

print(f'\n{"="*60}')
print(f'RESULTS d={d}')
print(f'{"="*60}')
print(f'  TCGA source:  {nz_src:5d} edges  iters={len(steps_src)}')
print(f'  BRCA scratch: {nz_sc:5d} edges  iters={len(steps_sc)}')
print(f'  BRCA transfer:{nz_tr:5d} edges  iters={len(steps_tr)}')
print(f'  Delta:        {nz_tr-nz_sc:+5d}')

e_src=set(); e_sc=set(); e_tr=set()
for i in range(d):
    for j in range(d):
        if i!=j and abs(W_src[i,j])>0.3: e_src.add((i,j))
        if i!=j and abs(W_sc[i,j])>0.3: e_sc.add((i,j))
        if i!=j and abs(W_tr[i,j])>0.3: e_tr.add((i,j))
print(f'  Overlap: src∩scratch={len(e_src&e_sc)} src∩transfer={len(e_src&e_tr)} sc∩tr={len(e_sc&e_tr)}')

from scipy.stats import pearsonr
fv=W_src.flatten(); fs=W_sc.flatten(); ft=W_tr.flatten()
rs=pearsonr(fv,fs)[0]; rt=pearsonr(fv,ft)[0]
print(f'  Pearson r with source: scratch={rs:.4f} transfer={rt:.4f}')

json.dump({'d':d,'src_edges':nz_src,'scratch_edges':nz_sc,'transfer_edges':nz_tr,
           'delta':nz_tr-nz_sc,'r_scratch':rs,'r_transfer':rt,
           'src_scratch_overlap':len(e_src&e_sc),'src_transfer_overlap':len(e_src&e_tr),
           'scratch_transfer_overlap':len(e_sc&e_tr),
           'src_iters':len(steps_src),'scratch_iters':len(steps_sc),'transfer_iters':len(steps_tr)},
          open(os.path.join(OUT,'transfer_d500.json'),'w'),indent=2)
