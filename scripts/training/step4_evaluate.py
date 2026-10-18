# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""Step 4: Quick evaluation - NOTEARS from scratch vs fine-tuned on BRCA."""
import os, json, time, numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import median_abs_deviation

CHECKPOINT = os.path.join(PKG_ROOT, "checkpoints")
DATA = os.path.join(PKG_ROOT, "data")
tcga_dir = DATA_DIR
RESULTS = PKG_RESULTS
os.makedirs(RESULTS, exist_ok=True)
RANDOM_SEED = 42

def h_constraint(W):
    return np.trace(expm(W * W)) - W.shape[0]

def notears_lbfgs(X, init_W=None, lam=0.01, max_outer=50, seed=42):
    """NOTEARS with optional warm start."""
    n, d = X.shape
    cov = X.T @ X / n
    if init_W is None:
        np.random.seed(seed)
        w_vec = np.random.randn(d*d) * 0.01
    else:
        w_vec = init_W.flatten()
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    for it in range(max_outer):
        def lag(w):
            W = w.reshape(d,d); hv = np.trace(expm(W*W))-d
            return 0.5*np.trace((np.eye(d)-W).T@cov@(np.eye(d)-W))+lam*np.sum(np.abs(W))+0.5*rho*hv**2+alpha*hv
        def lag_grad(w):
            W = w.reshape(d,d); hv = np.trace(expm(W*W))-d
            dh = 2*W*expm(W*W).T
            return (cov@(W-np.eye(d))+lam*np.sign(W)).flatten()+rho*hv*dh.flatten()+alpha*dh.flatten()
        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
                      options={'maxiter':200,'ftol':1e-12,'gtol':1e-12})
        w_vec = res.x; W = w_vec.reshape(d,d); hv = np.trace(expm(W*W))-d
        alpha += rho*hv
        if abs(hv) < best_h: best_h, best_w = abs(hv), w_vec.copy()
        if abs(hv) < 1e-7: break
        if hv > 0 and abs(hv) > 1e-10: rho = min(rho*2, 1e10)
    return best_w.reshape(d,d), best_h

# Load BRCA data (d=500)
print('Loading BRCA...')
brca = pd.read_csv(os.path.join(tcga_dir, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0).T
other = []
for f in sorted(os.listdir(tcga_dir)):
    if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
    other.append(pd.read_csv(os.path.join(tcga_dir, f), sep='\t', index_col=0).T)
other = pd.concat(other, axis=0)
# Top-500 genes by MAD in non-BRCA
mads = np.array([median_abs_deviation(other.iloc[:,j].values.astype(np.float64))
                 for j in range(other.shape[1])])
mads = np.nan_to_num(mads, nan=0)
top_genes = list(other.columns[np.argsort(mads)[-500:]])
common = [g for g in top_genes if g in brca.columns]
brca_X = brca[common].values.astype(np.float64)
brca_X = np.nan_to_num(brca_X, nan=0)
brca_X = (brca_X - brca_X.mean(0)) / (brca_X.std(0) + 1e-12)
print(f'BRCA: {brca_X.shape} ({len(common)} genes)')

# Load backbone
U0 = np.load(os.path.join(CHECKPOINT, 'causal-gpt-v1', 'U0.npy'))
V0 = np.load(os.path.join(CHECKPOINT, 'causal-gpt-v1', 'V0.npy'))
gene_list = open(os.path.join(DATA, 'gene_list.txt')).read().strip().split('\n')
gene_to_idx = {g: i for i, g in enumerate(gene_list)}
bb_idx = [gene_to_idx[g] for g in common if g in gene_to_idx]
print(f'Backbone genes mapped: {len(bb_idx)}/{len(common)}')
U_sub = U0[bb_idx]; V_sub = V0[bb_idx]
W_backbone = U_sub @ V_sub.T

# ========== NOTEARS from scratch ==========
print('\nNOTEARS from scratch...')
t0 = time.time()
W_s, h_s = notears_lbfgs(brca_X, init_W=None, max_outer=50)
nz_s = int(np.sum(np.abs(W_s) > 0.3))
dt_s = time.time() - t0
print(f'  Edges: {nz_s}, h: {h_s:.2e}, time: {dt_s:.0f}s')

# ========== CausalGPT fine-tuned ==========
print('\nCausalGPT fine-tuned (100 steps)...')
t0 = time.time()
W_ft, h_ft = notears_lbfgs(brca_X, init_W=W_backbone, max_outer=50)
nz_ft = int(np.sum(np.abs(W_ft) > 0.3))
dt_ft = time.time() - t0
print(f'  Edges: {nz_ft}, h: {h_ft:.2e}, time: {dt_ft:.0f}s')

# ========== Results ==========
print(f'\n{"="*50}')
print(f'RESULTS (BRCA held-out, d=500, tau=0.3)')
print(f'{"="*50}')
print(f'  Scratch:   {nz_s:4d} edges | h={h_s:.2e} | {dt_s:.0f}s')
print(f'  Fine-tuned: {nz_ft:4d} edges | h={h_ft:.2e} | {dt_ft:.0f}s')
print(f'  Delta:     {nz_ft-nz_s:+4d} edges')
if dt_s > 0 and dt_ft > 0:
    print(f'  Speedup:   {dt_s/dt_ft:.1f}x')

# Edge overlap
edges_s = set()
edges_ft = set()
for i in range(500):
    for j in range(500):
        if i != j and abs(W_s[i,j]) > 0.3: edges_s.add((i,j))
        if i != j and abs(W_ft[i,j]) > 0.3: edges_ft.add((i,j))
shared = len(edges_s & edges_ft)
print(f'  Shared:    {shared} edges')
print(f'  Scratch-only: {len(edges_s - edges_ft)}')
print(f'  FT-only:   {len(edges_ft - edges_s)}')

json.dump({
    'scratch': {'edges': nz_s, 'h': h_s, 'time': dt_s},
    'fine_tuned': {'edges': nz_ft, 'h': h_ft, 'time': dt_ft},
    'shared_edges': shared,
    'delta_edges': nz_ft - nz_s,
}, open(os.path.join(RESULTS, 'eval.json'), 'w'), indent=2)
print(f'Saved: {os.path.join(RESULTS, "eval.json")}')
