# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""CausalGPT d=1000 scaling experiment: TCGA->BRCA transfer only."""
import numpy as np, time, os, pandas as pd, json
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation, pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)
TCGA_DIR = DATA_DIR
DATA_DIR = os.path.join(ROOT,'data')
d = 1000; MAX_OUTER = 25

print(f'd={d} scaling experiment...')
other_dfs=[]
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    try: other_dfs.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
    except: continue
other=pd.concat(other_dfs,axis=0)
mads=np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads=np.nan_to_num(mads,nan=0)
top_genes=list(other.columns[np.argsort(mads)[-d:]])
brca=pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
with open(os.path.join(DATA_DIR,'gene_list.txt')) as f: all_genes=f.read().strip().split('\n')
common_brca=[g for g in top_genes if g in brca.columns]
common_dm=[g for g in top_genes if g in all_genes]
common=[g for g in common_brca if g in common_dm]
d_actual=len(common)
tcga_X=other[common].values.astype(np.float64); tcga_X=np.nan_to_num(tcga_X,nan=0); tcga_X=(tcga_X-tcga_X.mean(0))/(tcga_X.std(0)+1e-12)
brca_X=brca[common].values.astype(np.float64); brca_X=np.nan_to_num(brca_X,nan=0); brca_X=(brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)
print(f'TCGA: {tcga_X.shape} BRCA: {brca_X.shape}')

def nt(X,init_W=None,lam=0.01,max_outer=MAX_OUTER,label=''):
    n,d_=X.shape; cov=X.T@X/n
    w_vec=init_W.flatten() if init_W is not None else np.random.randn(d_*d_)*0.1
    rho,alpha=0.1,0.0; bw,bh=w_vec.copy(),1e10
    for it in range(max_outer):
        def lag(w): W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w): W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T; return (cov@(W-np.eye(d_))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':100})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; alpha+=rho*hv
        if abs(hv)<bh: bh,bw=abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if it%5==0: print(f'  [{label}] iter{it} h={hv:.2e}',flush=True)
    return bw.reshape(d_,d_),bh

print('\n(A) TCGA source...'); t0=time.time(); W_src,hs=nt(tcga_X,label='src'); dt_src=time.time()-t0
print(f'  {int((np.abs(W_src)>0.3).sum())} edges {dt_src:.0f}s')

print('\n(B) BRCA scratch...'); t0=time.time(); W_sc,hs=nt(brca_X,label='sc'); dt_sc=time.time()-t0
print(f'  {int((np.abs(W_sc)>0.3).sum())} edges {dt_sc:.0f}s')

print('\n(C) BRCA warm-start...'); t0=time.time(); W_tr,ht=nt(brca_X,init_W=W_src,label='tr'); dt_tr=time.time()-t0
print(f'  {int((np.abs(W_tr)>0.3).sum())} edges {dt_tr:.0f}s')

nz_src=int((np.abs(W_src)>0.3).sum()); nz_sc=int((np.abs(W_sc)>0.3).sum()); nz_tr=int((np.abs(W_tr)>0.3).sum())
e_src=set(); e_sc=set(); e_tr=set()
for i in range(d_actual):
    for j in range(d_actual):
        if i!=j and abs(W_src[i,j])>0.3: e_src.add((i,j))
        if i!=j and abs(W_sc[i,j])>0.3: e_sc.add((i,j))
        if i!=j and abs(W_tr[i,j])>0.3: e_tr.add((i,j))
ov_sc=len(e_src&e_sc); ov_tr=len(e_src&e_tr)
r_sc=pearsonr(W_src.flatten(),W_sc.flatten())[0]; r_tr=pearsonr(W_src.flatten(),W_tr.flatten())[0]

print(f'\n=== d={d_actual} RESULTS ===')
print(f'  TCGA src: {nz_src} edges')
print(f'  Scratch:  {nz_sc} edges ov={ov_sc}/{nz_src} r={r_sc:.4f}')
print(f'  Warm-st:  {nz_tr} edges ov={ov_tr}/{nz_src} r={r_tr:.4f}')
print(f'  Delta: ov +{ov_tr-ov_sc} r +{r_tr-r_sc:.4f}')

json.dump({'d':d_actual,'src_edges':nz_src,'scratch_edges':nz_sc,'warm_edges':nz_tr,
           'ov_scratch':ov_sc,'ov_warm':ov_tr,'r_scratch':r_sc,'r_warm':r_tr,
           'time_src':dt_src,'time_sc':dt_sc,'time_tr':dt_tr},
          open(os.path.join(OUT,f'scaling_d{d_actual}.json'),'w'),indent=2)
np.save(os.path.join(OUT,f'W_src_d{d_actual}.npy'),W_src)
np.save(os.path.join(OUT,f'W_sc_d{d_actual}.npy'),W_sc)
np.save(os.path.join(OUT,f'W_tr_d{d_actual}.npy'),W_tr)
print(f'Done! Saved to {OUT}/')
