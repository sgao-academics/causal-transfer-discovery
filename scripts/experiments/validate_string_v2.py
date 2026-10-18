# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""STRING validation v2: correct file format parsing."""
import numpy as np, os, pickle, pandas as pd, gzip, json

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results')
TCGA_DIR = DATA_DIR
STR_DIR = os.path.join(TCGA_DIR,'validation')
d = 500

# ====== 1. Load gene names ======
from scipy.stats import median_abs_deviation
other=[]
for f in sorted(os.listdir(TCGA_DIR)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    try: other.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
    except: continue
other=pd.concat(other,axis=0)
mads=np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64)) for j in range(other.shape[1])])
mads=np.nan_to_num(mads,nan=0)
top_genes=list(other.columns[np.argsort(mads)[-d:]])
brca=pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
common=[g for g in top_genes if g in brca.columns]
d_actual=len(common)
print(f'Common genes: {d_actual}')

# ====== 2. Build gene symbol -> STRING protein_id ======
print('Building gene->protein mapping...')
# info: protein_id -> preferred_name
info={}
with gzip.open(os.path.join(STR_DIR,'string_info.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    next(f)  # skip header
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)>=2: info[parts[0]]=parts[1]
print(f'  Info proteins: {len(info)}')

# aliases: alias -> protein_id (only keep if protein_id in info)
gene2pid={}
with gzip.open(os.path.join(STR_DIR,'string_aliases.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    next(f)
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)>=2 and parts[0] in info:
            gene2pid[parts[1]]=parts[0]  # alias -> protein_id
print(f'  Aliases: {len(gene2pid)} gene symbols mapped')

# How many of our genes mapped?
mapped=sum(1 for g in common if g in gene2pid)
print(f'  Our genes mapped: {mapped}/{d_actual}')

# ====== 3. STRING PPI ======
print('Loading PPI scores...')
ppi={}
with gzip.open(os.path.join(STR_DIR,'string_ppi_full.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    next(f)
    for line in f:
        parts=line.strip().split()
        if len(parts)>=10:
            try:
                p1,p2,combined=parts[0],parts[1],int(parts[9])
                ppi[(p1,p2)]=combined; ppi[(p2,p1)]=combined
            except: continue
n_pairs=len(set(k for k in ppi))//2
print(f'  PPI pairs: {n_pairs:,}')

# ====== 4. Load W matrices ======
W_src=np.load(os.path.join(OUT,'W_src_d500.npy'))
W_src=W_src[np.ix_(list(range(d_actual)),list(range(d_actual)))]

# Recompute scratch and warm-start
print('\nRecomputing W...')
from scipy.linalg import expm
from scipy.optimize import minimize

brca_X=brca[common].values.astype(np.float64)
brca_X=np.nan_to_num(brca_X,nan=0)
brca_X=(brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)

def qn(X, init_W=None, mx=15):
    n,d_=X.shape; cov=X.T@X/n
    w_vec=init_W.flatten() if init_W is not None else np.random.randn(d_*d_)*0.1
    rho,alpha=0.1,0.0; bw=w_vec.copy()
    for it in range(mx):
        def lag(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
            return 0.5*np.trace((np.eye(d_)-W).T@cov@(np.eye(d_)-W))+0.01*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W=w.reshape(d_,d_); hv=np.trace(expm(W*W))-d_; dh=2*W*expm(W*W).T
            return (cov@(W-np.eye(d_))+0.01*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res=minimize(lag,w_vec,method='L-BFGS-B',jac=lag_grad,options={'maxiter':100})
        w_vec=res.x; W=w_vec.reshape(d_,d_); hv=np.trace(expm(W*W))-d_
        alpha+=rho*hv
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        bw=w_vec.copy()
    return bw.reshape(d_,d_)

print('  Scratch...'); W_sc=qn(brca_X,mx=15)
print('  Warm-start...'); W_tr=qn(brca_X,init_W=W_src,mx=15)

# ====== 5. STRING validation ======
def string_precision(W, gene2pid, ppi, common, score_thresh=700):
    d_=W.shape[0]
    abs_w=np.abs(W).copy(); np.fill_diagonal(abs_w,0)
    flat=abs_w.flatten(); top_idx=np.argsort(-flat)[:200]
    supported=0; mappable=0
    for idx_ in top_idx:
        i=int(idx_//d_); j=int(idx_%d_)
        gi=common[i]; gj=common[j]
        if gi in gene2pid and gj in gene2pid:
            mappable+=1
            s=ppi.get((gene2pid[gi],gene2pid[gj]),0)
            if s>=score_thresh: supported+=1
    return supported,mappable,supported/mappable*100 if mappable>0 else 0

# Evaluate at multiple STRING thresholds
for thresh,name in [(400,'low'),(700,'medium'),(900,'high')]:
    s_src,m_src,p_src=string_precision(W_src,gene2pid,ppi,common,thresh)
    s_sc,m_sc,p_sc=string_precision(W_sc,gene2pid,ppi,common,thresh)
    s_tr,m_tr,p_tr=string_precision(W_tr,gene2pid,ppi,common,thresh)
    print(f'\nSTRING combined >= {thresh} ({name}):')
    print(f'  TCGA src:   {s_src}/{m_src} = {p_src:.1f}%')
    print(f'  Scratch:    {s_sc}/{m_sc} = {p_sc:.1f}%')
    print(f'  Warm-start: {s_tr}/{m_tr} = {p_tr:.1f}%')

# STRING-validated edge overlap
def string_edges(W, gene2pid, ppi, common, thresh=700):
    d_=W.shape[0]; se=set()
    for i in range(d_):
        for j in range(i+1,d_):
            if abs(W[i,j])>0.3:
                gi,gj=common[i],common[j]
                if gi in gene2pid and gj in gene2pid:
                    if ppi.get((gene2pid[gi],gene2pid[gj]),0)>=thresh:
                        se.add((min(gi,gj),max(gi,gj)))
    return se

s_src=string_edges(W_src,gene2pid,ppi,common,700)
s_sc=string_edges(W_sc,gene2pid,ppi,common,700)
s_tr=string_edges(W_tr,gene2pid,ppi,common,700)
nz_src=int((np.abs(W_src)>0.3).sum()); nz_sc=int((np.abs(W_sc)>0.3).sum()); nz_tr=int((np.abs(W_tr)>0.3).sum())

print(f'\n=== STRING-VALIDATED EDGE OVERLAP (tau=0.3, STRING>=700) ===')
print(f'  src={len(s_src)} sc={len(s_sc)} tr={len(s_tr)} (total: src={nz_src} sc={nz_sc} tr={nz_tr})')
print(f'  src∩sc={len(s_src&s_sc)} src∩tr={len(s_src&s_tr)} sc∩tr={len(s_sc&s_tr)}')
print(f'  3-way={len(s_src&s_sc&s_tr)}')

json.dump({
    'genes_mapped':mapped,'total_genes':d_actual,
    'src_edges_total':nz_src,'sc_edges_total':nz_sc,'tr_edges_total':nz_tr,
    'string_src':len(s_src),'string_sc':len(s_sc),'string_tr':len(s_tr),
    'src_tr_overlap':len(s_src&s_tr),'src_sc_overlap':len(s_src&s_sc),
},open(os.path.join(OUT,'string_validation.json'),'w'),indent=2)
print(f'\nSaved: {os.path.join(OUT,"string_validation.json")}')
