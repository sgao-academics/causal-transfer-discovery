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
CausalGPT Full Experiment: d=500, 40 outer iters, dual-source transfer.
Sources: TCGA(-BRCA) + DepMap → Target: BRCA

Metrics: edge count, source overlap, weight correlation, STRING precision.
"""
import numpy as np, time, os, pandas as pd, json, gzip
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation, pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results'); os.makedirs(OUT,exist_ok=True)
TCGA_DIR = DATA_DIR
DATA_DIR = os.path.join(ROOT,'data')
STR_DIR = os.path.join(TCGA_DIR,'validation')
d = 500; MAX_OUTER = 35

# ============================================================
# 1. DATA
# ============================================================
print(f'=== Loading d={d} genes ===')
# Top-d genes by MAD in TCGA(-BRCA)
other_dfs = []
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    try: other_dfs.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
    except: continue
other = pd.concat(other_dfs,axis=0)
mads = np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads = np.nan_to_num(mads,nan=0)
top_genes = list(other.columns[np.argsort(mads)[-d:]])

# BRCA
brca = pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common_brca = [g for g in top_genes if g in brca.columns]

# DepMap
depmap = pd.read_csv(os.path.join(DATA_DIR,'X_depmap_final.npy'), allow_pickle=True) if False else None
# DepMap stored as .npy not .csv — need gene_list.txt to map columns
with open(os.path.join(DATA_DIR,'gene_list.txt')) as f:
    all_genes = f.read().strip().split('\n')
common_dm = [g for g in top_genes if g in all_genes]

# Intersection: genes present in ALL 3 datasets
common = [g for g in common_brca if g in common_dm]
d_actual = len(common)
print(f'  TCGA(-BRCA) genes: {len(top_genes)}, shared all 3: {d_actual}')

# Build data matrices
tcga_X = other[common].values.astype(np.float64)
tcga_X = np.nan_to_num(tcga_X,nan=0)
tcga_X = (tcga_X-tcga_X.mean(0))/(tcga_X.std(0)+1e-12)

brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X,nan=0)
brca_X = (brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)

# DepMap: load .npy and subset by gene index
dm_gene_idx = [all_genes.index(g) for g in common]
X_dm_full = np.load(os.path.join(DATA_DIR,'X_depmap_final.npy')).astype(np.float64)
dm_X = X_dm_full[:, dm_gene_idx]
dm_X = np.nan_to_num(dm_X,nan=0)
dm_X = (dm_X-dm_X.mean(0))/(dm_X.std(0)+1e-12)

print(f'  TCGA: {tcga_X.shape}  DepMap: {dm_X.shape}  BRCA: {brca_X.shape}')
d = d_actual

# ============================================================
# 2. NOTEARS
# ============================================================
def notears(X, init_W=None, lam=0.01, max_outer=MAX_OUTER, label=''):
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
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':120})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w=abs(hv),w_vec.copy()
        steps.append({'iter':it,'h':float(hv),'rho':rho})
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if it%10==0: print(f'  [{label}] iter {it:2d}/{max_outer} h={hv:.2e} rho={rho:.1e}',flush=True)
    W_final=best_w.reshape(d_,d_)
    nz=int((np.abs(W_final)>0.3).sum())
    print(f'  [{label}] DONE: {nz} edges, h={best_h:.2e}, {len(steps)} iters')
    return W_final, best_h, steps

# ============================================================
# 3. EXPERIMENTS
# ============================================================
results = {}

# (A) TCGA(-BRCA) source
print('\n(A) TCGA(-BRCA) source...')
t0=time.time()
W_tcga, h_tcga, st_tcga = notears(tcga_X, label='TCGA')
results['tcga']={'W':W_tcga,'h':float(h_tcga),'time':time.time()-t0,'iters':len(st_tcga)}

# (B) DepMap source
print('\n(B) DepMap source...')
t0=time.time()
W_dm, h_dm, st_dm = notears(dm_X, label='DepMap')
results['depmap']={'W':W_dm,'h':float(h_dm),'time':time.time()-t0,'iters':len(st_dm)}

# (C) BRCA scratch
print('\n(C) BRCA scratch...')
t0=time.time()
W_sc, h_sc, st_sc = notears(brca_X, label='Scratch')
results['scratch']={'W':W_sc,'h':float(h_sc),'time':time.time()-t0,'iters':len(st_sc)}

# (D) BRCA warm-started with TCGA
print('\n(D) BRCA warm-start (TCGA)...')
t0=time.time()
W_tw, h_tw, st_tw = notears(brca_X, init_W=W_tcga, label='TCGA-warm')
results['tcga_warm']={'W':W_tw,'h':float(h_tw),'time':time.time()-t0,'iters':len(st_tw)}

# (E) BRCA warm-started with DepMap
print('\n(E) BRCA warm-start (DepMap)...')
t0=time.time()
W_dw, h_dw, st_dw = notears(brca_X, init_W=W_dm, label='DM-warm')
results['dm_warm']={'W':W_dw,'h':float(h_dw),'time':time.time()-t0,'iters':len(st_dw)}

# ============================================================
# 4. COMPARISON
# ============================================================
def edge_stats(W):
    nz03=int((np.abs(W)>0.3).sum()); nz01=int((np.abs(W)>0.1).sum()); nz005=int((np.abs(W)>0.05).sum())
    return nz03,nz01,nz005

def overlap_edges(W1, W2, tau=0.3):
    d_=W1.shape[0]; e1=set(); e2=set()
    for i in range(d_):
        for j in range(d_):
            if i!=j and abs(W1[i,j])>tau: e1.add((i,j))
            if i!=j and abs(W2[i,j])>tau: e2.add((i,j))
    return len(e1&e2), len(e1), len(e2)

print(f'\n{"="*75}')
print(f'FULL RESULTS: d={d}, max_outer={MAX_OUTER}')
print(f'{"="*75}')

# Header
print(f'  {"Method":18s} {"tau=0.3":>7s} {"tau=0.1":>7s} {"tau=0.05":>7s} {"h(W)":>9s} {"time":>6s} {"iters":>5s}')
print(f'  {"-"*63}')

all_W = {'TCGA src':W_tcga,'DepMap src':W_dm,'Scratch':W_sc,'TCGA-warm':W_tw,'DM-warm':W_dw}
for name,W in all_W.items():
    nz03,nz01,nz005 = edge_stats(W)
    r = results[name.replace(' ','_').lower().replace('-','_')] if name!='TCGA src' and name!='DepMap src' else results['tcga' if 'TCGA' in name else 'depmap']
    if name=='TCGA src': r=results['tcga']
    elif name=='DepMap src': r=results['depmap']
    h_val = float(np.trace(expm(W*W))-d)
    print(f'  {name:18s} {nz03:7d} {nz01:7d} {nz005:7d} {h_val:9.2e} {r["time"]:6.0f}s {r["iters"]:5d}')

# Overlap analysis
print(f'\n  {"="*55}')
print(f'  OVERLAP WITH SOURCE (tau=0.3)')
print(f'  {"="*55}')
for src_name,src_W in [('TCGA',W_tcga),('DepMap',W_dm)]:
    src_total = int((np.abs(src_W)>0.3).sum())
    for tgt_name,tgt_W in [('Scratch',W_sc),('TCGA-warm',W_tw),('DM-warm',W_dw)]:
        ov,_,_ = overlap_edges(src_W, tgt_W, 0.3)
        print(f'  {src_name:>10s} ∩ {tgt_name:<12s} = {ov:4d}/{src_total:4d} ({ov/src_total*100:5.1f}%)')

# Weight correlation
print(f'\n  {"="*55}')
print(f'  WEIGHT CORRELATION (Pearson r)')
print(f'  {"="*55}')
for src_name,src_W in [('TCGA',W_tcga),('DepMap',W_dm)]:
    for tgt_name,tgt_W in [('Scratch',W_sc),('TCGA-warm',W_tw),('DM-warm',W_dw)]:
        r_val = pearsonr(src_W.flatten(), tgt_W.flatten())[0]
        print(f'  r({src_name:>10s}, {tgt_name:<12s}) = {r_val:.4f}')

# ============================================================
# 5. STRING VALIDATION
# ============================================================
print(f'\n  {"="*55}')
print(f'  STRING VALIDATION')
print(f'  {"="*55}')

# Build gene->pid mapping (cached if possible)
print('  Building STRING mapping...')
info={}
with gzip.open(os.path.join(STR_DIR,'string_info.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    next(f)
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)>=2: info[parts[0]]=parts[1]
gene2pid={}
with gzip.open(os.path.join(STR_DIR,'string_aliases.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    next(f)
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)>=2 and parts[0] in info: gene2pid[parts[1]]=parts[0]

ppi={}
with gzip.open(os.path.join(STR_DIR,'string_ppi_full.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    next(f)
    for line in f:
        parts=line.strip().split()
        if len(parts)>=10:
            try:
                p1,p2,cs=parts[0],parts[1],int(parts[9])
                ppi[(p1,p2)]=cs; ppi[(p2,p1)]=cs
            except: continue

def string_precision_top(W, gene2pid, ppi, common, thresh=700, top_n=200):
    d_=W.shape[0]; abs_w=np.abs(W).copy(); np.fill_diagonal(abs_w,0)
    idx=np.argsort(-abs_w.flatten())[:top_n]
    sup=0; mappable=0
    for k in idx:
        i=int(k//d_); j=int(k%d_)
        gi,gj=common[i],common[j]
        if gi in gene2pid and gj in gene2pid:
            mappable+=1
            if ppi.get((gene2pid[gi],gene2pid[gj]),0)>=thresh: sup+=1
    return sup,mappable,sup/mappable*100 if mappable>0 else 0

for thresh,name in [(400,'low'),(700,'med'),(900,'high')]:
    print(f'  STRING >= {thresh} ({name}):')
    for label,W in [('TCGA',W_tcga),('DM',W_dm),('Scratch',W_sc),('TCGA-w',W_tw),('DM-w',W_dw)]:
        s,m,p=string_precision_top(W,gene2pid,ppi,common,thresh)
        print(f'    {label:10s} {s:3d}/{m:3d} = {p:4.1f}%')

# Save everything
export = {}
for name in ['tcga','depmap','scratch','tcga_warm','dm_warm']:
    W = results[name]['W']
    nz03,nz01,nz005 = edge_stats(W)
    export[name] = {
        'edges_03':nz03,'edges_01':nz01,'edges_005':nz005,
        'h':results[name]['h'],'time':results[name]['time'],'iters':results[name]['iters']
    }
# Add cross-correlations
for src_k,src_n in [('tcga','TCGA'),('depmap','DepMap')]:
    for tgt_k,tgt_n in [('scratch','Scratch'),('tcga_warm','TCGA-warm'),('dm_warm','DM-warm')]:
        ov,_,_ = overlap_edges(results[src_k]['W'], results[tgt_k]['W'], 0.3)
        rv = pearsonr(results[src_k]['W'].flatten(), results[tgt_k]['W'].flatten())[0]
        export[f'{src_k}_to_{tgt_k}'] = {'overlap':ov,'r':rv}

with open(os.path.join(OUT,'full_experiment.json'),'w') as f:
    json.dump(export,f,indent=2)

# Save W matrices
for name,key in [('W_tcga','tcga'),('W_depmap','depmap'),('W_scratch','scratch'),
                  ('W_tcga_warm','tcga_warm'),('W_dm_warm','dm_warm')]:
    np.save(os.path.join(OUT,f'{name}_d{d}.npy'), results[key]['W'])

print(f'\nAll results saved to {OUT}/')
