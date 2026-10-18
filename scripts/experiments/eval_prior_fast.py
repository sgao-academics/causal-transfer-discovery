# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""Prior regularization sweep: lambda_prior vs overlap/r/sparsity."""
import numpy as np, time, os, pandas as pd, json
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation, pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)
TCGA_DIR = DATA_DIR
d = 500

# Load data + W_src
W_src = np.load(os.path.join(OUT,'W_src_d500.npy'))
nz_src = int((np.abs(W_src)>0.3).sum())
print(f'W_src loaded: {nz_src} edges')

other=[]; 
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    try: other.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
    except: continue
other=pd.concat(other,axis=0)
mads=np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads=np.nan_to_num(mads,nan=0)
top_genes=list(other.columns[np.argsort(mads)[-d:]])
brca=pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common=[g for g in top_genes if g in brca.columns]; d=len(common)
brca_X=brca[common].values.astype(np.float64); brca_X=np.nan_to_num(brca_X,nan=0)
brca_X=(brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)
tcga_ci=[list(top_genes).index(g) for g in common]
W_src=W_src[np.ix_(tcga_ci,tcga_ci)]  # subset to common genes
nz_src=int((np.abs(W_src)>0.3).sum())
print(f'BRCA: {brca_X.shape} W_src(subset): {nz_src} edges')

# Sparse prior
W_prior=W_src.copy(); W_prior[np.abs(W_prior)<0.3]=0; np.fill_diagonal(W_prior,0)
print(f'W_prior nonzeros: {(W_prior!=0).sum()}')
print(f'Number of edges: {int((np.abs(W_prior)>0.3).sum())}')

def notears_prior(X, W_prior, lam_p=1.0, max_outer=15):
    n,d_=X.shape; cov=X.T@X/n
    w_vec=np.random.randn(d_*d_)*0.1; rho,alpha=0.1,0.0
    best_w,best_h=w_vec.copy(),1e10; steps=[]
    for it in range(max_outer):
        def lag(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
            pl=lam_p*np.sum((W-W_prior)**2)/(d_*d_)
            return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+0.01*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv+pl
        def lag_grad(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T
            pg=2*lam_p*(W-W_prior)/(d_*d_)
            return (cov@(W-np.eye(d_))+0.01*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()+pg.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':100})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w=abs(hv),w_vec.copy()
        steps.append({'iter':it,'h':float(hv)})
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
    return best_w.reshape(d_,d_),best_h,steps

# Baseline: scratch
print('\n(A) Scratch (no prior)...')
t0=time.time(); W_sc,h_sc,st_sc=notears_prior(brca_X,W_prior,lam_p=0.0,max_outer=15)
nz_sc=int((np.abs(W_sc)>0.3).sum()); dt_sc=time.time()-t0
e_sc=set(); e_src=set()
for i in range(d):
    for j in range(d):
        if i!=j and abs(W_sc[i,j])>0.3: e_sc.add((i,j))
        if i!=j and abs(W_src[i,j])>0.3: e_src.add((i,j))
ov_sc=len(e_sc&e_src); r_sc=pearsonr(W_sc.flatten(),W_src.flatten())[0]
print(f'  tau0.3={nz_sc} ov={ov_sc}/{nz_src} r={r_sc:.4f} {dt_sc:.0f}s')

# Sweep lambda
results={}
for lam_p in [0.05, 0.1, 0.2, 0.5, 1.0]:
    print(f'\nlambda={lam_p}...')
    t0=time.time(); W_p,h_p,st=notears_prior(brca_X,W_prior,lam_p=lam_p,max_outer=15)
    dt=time.time()-t0; nz03=int((np.abs(W_p)>0.3).sum())
    e_p=set()
    for i in range(d):
        for j in range(d):
            if i!=j and abs(W_p[i,j])>0.3: e_p.add((i,j))
    ov=len(e_p&e_src); r_val=pearsonr(W_p.flatten(),W_src.flatten())[0]
    results[lam_p]={'tau03':nz03,'overlap':ov,'r':r_val,'h':float(h_p),'time':dt,'iters':len(st)}
    print(f'  tau0.3={nz03} ov={ov}/{nz_src} r={r_val:.4f} {dt:.0f}s')

# Summary
print(f'\n{"="*65}')
print(f'PRIOR REGULARIZATION: d={d}, W_src={nz_src} edges')
print(f'{"="*65}')
print(f'  | {"Method":12s} {"lambda":>7s} {"tau0.3":>7s} {"overlap":>8s} {"%src":>5s} {"r":>7s} {"time":>6s} |')
print(f'  |{"-"*61}|')
print(f'  | {"Scratch":12s} {"0":>7s} {nz_sc:7d} {ov_sc:8d} {ov_sc/nz_src*100:4.0f}% {r_sc:7.4f} {dt_sc:6.0f}s |')
for lam_p in sorted(results.keys()):
    r=results[lam_p]
    print(f'  | {"Prior":12s} {lam_p:7.2f} {r["tau03"]:7d} {r["overlap"]:8d} {r["overlap"]/nz_src*100:4.0f}% {r["r"]:7.4f} {r["time"]:6.0f}s |')

best=max(results.keys(),key=lambda k:results[k]['overlap'])
print(f'\n  Best: lambda={best:.2f} with {results[best]["overlap"]}/{nz_src} overlap (+{results[best]["overlap"]-ov_sc:+d} vs scratch)')

json.dump({str(k):v for k,v in results.items()},open(os.path.join(OUT,'prior_sweep.json'),'w'),indent=2)
