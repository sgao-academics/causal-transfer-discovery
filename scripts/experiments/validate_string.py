# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""STRING validation of transferred edges."""
import numpy as np, os, pickle, pandas as pd, gzip

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT,'results')
TCGA_DIR = DATA_DIR

# Load results from transfer experiment
import json
with open(os.path.join(OUT,'transfer_d500.json')) as f:
    res = json.load(f)
print(f'transfer_d500: src={res["src_edges"]} sc={res["scratch_edges"]} tr={res["transfer_edges"]}')

# Load gene names for d=500
d=500
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

# Load STRING data
print('Loading STRING...')
STR_DIR = os.path.join(TCGA_DIR,'validation')
# string_info: ENSEMBL protein ID -> gene symbol
info={}
with gzip.open(os.path.join(STR_DIR,'string_info.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)>=2:
            info[parts[0]]=parts[1]  # protein_id -> preferred_name
print(f'  Info: {len(info)} proteins')

# string_aliases: alias -> protein_id
aliases={}
with gzip.open(os.path.join(STR_DIR,'string_aliases.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        parts=line.strip().split('\t')
        if len(parts)>=2 and parts[1] in info:
            aliases[parts[0]]=parts[1]
print(f'  Aliases: {len(aliases)} entries')

# Build gene -> protein_id mapping
gene2pid={}
for gene in common:
    if gene in aliases:
        gene2pid[gene]=aliases[gene]

# Load STRING PPI as dict: (pid1,pid2) -> combined_score
print('Loading STRING PPI...')
ppi_scores={}
first=True
with gzip.open(os.path.join(STR_DIR,'string_ppi_full.txt.gz'),'rt',encoding='utf-8',errors='ignore') as f:
    for line in f:
        if first: first=False; continue  # skip header
        parts=line.strip().split()
        if len(parts)>=3:
            try:
                p1,p2,score=parts[0],parts[1],int(parts[2])
                ppi_scores[(p1,p2)]=score
                ppi_scores[(p2,p1)]=score
            except: continue
print(f'  PPI: {len(set(k for k in ppi_scores))//2} unique pairs')

# Load W matrices from transfer experiment
W_src=np.load(os.path.join(OUT,'W_src_d500.npy'))
W_src=W_src[np.ix_(list(range(d_actual)),list(range(d_actual)))]  # d_actual x d_actual

# We need W_sc and W_tr from the transfer experiment
# They're not saved as npy. Let's recompute with fewer iters
print('\nRecomputing W matrices (15 outer iters)...')
from scipy.linalg import expm
from scipy.optimize import minimize

tcga_X=other[common].values.astype(np.float64)
tcga_X=np.nan_to_num(tcga_X,nan=0)
tcga_X=(tcga_X-tcga_X.mean(0))/(tcga_X.std(0)+1e-12)
brca_X=brca[common].values.astype(np.float64)
brca_X=np.nan_to_num(brca_X,nan=0)
brca_X=(brca_X-brca_X.mean(0))/(brca_X.std(0)+1e-12)

def quick_notears(X, init_W=None, max_outer=15):
    n,d_=X.shape; cov=X.T@X/n
    w_vec=init_W.flatten() if init_W is not None else np.random.randn(d_*d_)*0.1
    rho,alpha=0.1,0.0; best_w=w_vec.copy()
    for it in range(max_outer):
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
        best_w=w_vec.copy()
    return best_w.reshape(d_,d_)

print('  Scratch...'); W_sc=quick_notears(brca_X,max_outer=15)
print('  Warm-start...'); W_tr=quick_notears(brca_X,init_W=W_src,max_outer=15)

# Extract top edges per method
def get_top_edges(W, n_top=200):
    d_=W.shape[0]; abs_w=np.abs(W).copy(); np.fill_diagonal(abs_w,0)
    flat=abs_w.flatten(); idx=np.argsort(-flat)[:n_top]
    return [(int(i//d_),int(i%d_),float(W[i//d_,i%d_])) for i in idx]

e_src=get_top_edges(W_src,200); e_sc=get_top_edges(W_sc,200); e_tr=get_top_edges(W_tr,200)

# STRING validation
def validate_edges(edges, gene2pid, ppi_scores, score_threshold=700):
    """Count edges with STRING support."""
    supported=0; total=0
    for i,j,w in edges:
        gi=common[i]; gj=common[j]
        if gi in gene2pid and gj in gene2pid:
            p1=gene2pid[gi]; p2=gene2pid[gj]
            total+=1
            score=ppi_scores.get((p1,p2),0)
            if score>=score_threshold: supported+=1
    mapped=total
    return supported,mapped

sup_src,map_src=validate_edges(e_src,gene2pid,ppi_scores)
sup_sc,map_sc=validate_edges(e_sc,gene2pid,ppi_scores)
sup_tr,map_tr=validate_edges(e_tr,gene2pid,ppi_scores)

n_src=int((np.abs(W_src)>0.3).sum()); n_sc=int((np.abs(W_sc)>0.3).sum()); n_tr=int((np.abs(W_tr)>0.3).sum())

print(f'\n{"="*60}')
print(f'STRING VALIDATION (top 200 edges, combined_score >= 700)')
print(f'{"="*60}')
print(f'  Method           tau0.3  STRING_hits  Mapped  Precision')
print(f'  {"-"*55}')
print(f'  TCGA src          {n_src:5d}   {sup_src:5d}       {map_src:5d}  {sup_src/map_src*100:.1f}%' if map_src>0 else '')
print(f'  Scratch           {n_sc:5d}   {sup_sc:5d}       {map_sc:5d}  {sup_sc/map_sc*100:.1f}%' if map_sc>0 else '')
print(f'  Warm-start        {n_tr:5d}   {sup_tr:5d}       {map_tr:5d}  {sup_tr/map_tr*100:.1f}%' if map_tr>0 else '')

# Edge overlap among STRING-validated edges
def string_edges(edges, gene2pid, ppi_scores, thresh=700):
    se=set()
    for i,j,w in edges:
        gi=common[i]; gj=common[j]
        if gi in gene2pid and gj in gene2pid:
            if ppi_scores.get((gene2pid[gi],gene2pid[gj]),0)>=thresh:
                se.add((gi,gj))
    return se

s_src=string_edges(e_src,gene2pid,ppi_scores)
s_sc=string_edges(e_sc,gene2pid,ppi_scores)
s_tr=string_edges(e_tr,gene2pid,ppi_scores)
print(f'\n  STRING-validated edge sets:')
print(f'    src={len(s_src)} sc={len(s_sc)} tr={len(s_tr)}')
print(f'    src∩sc={len(s_src&s_sc)} src∩tr={len(s_src&s_tr)} sc∩tr={len(s_sc&s_tr)}')
print(f'    3-way={len(s_src&s_sc&s_tr)}')
