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
DAGMA + GOLEM comparison benchmark.
Extends the GPU pipeline with two additional methods using same warm-start framework.
Checkpointing: each config done → immediate json.dump. Auto-skip on restart.
"""
import os, json, time, sys
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr, median_abs_deviation

ROOT = PKG_ROOT
CACHE_DIR = os.path.join(ROOT, 'results', 'cache')
CHECKPOINT_FILE = os.path.join(ROOT, 'results', 'benchmark_v6_checkpoint.json')
TCGA_DIR = DATA_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# NOTEARS (existing, copy from GPU pipeline for completeness)
# ============================================================
def notears(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    n, d = X.shape
    cov = X.T @ X / n
    w_vec = init_W.flatten().copy() if init_W is not None else np.random.randn(d*d)*0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    
    for it in range(max_outer):
        def lag(w):
            W = w.reshape(d, d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            hv = torch.trace(torch.linalg.matrix_exp(W_t * W_t)).item() - d
            base = 0.5*np.trace((np.eye(d)-W).T@cov@(np.eye(d)-W)) + lam*np.sum(np.abs(W))
            return base + 0.5*rho*hv**2 + alpha*hv
        def lag_grad(w):
            W = w.reshape(d, d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            W2 = W_t * W_t
            expW2 = torch.linalg.matrix_exp(W2)
            hv = torch.trace(expW2).item() - d
            dh = (2 * W_t * expW2.T).cpu().numpy()
            return (cov@(W-np.eye(d)) + lam*np.sign(W) + rho*hv*dh + alpha*dh).flatten()
        
        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
                       options={'maxiter':120,'ftol':1e-10,'gtol':1e-8})
        w_vec = res.x; W = w_vec.reshape(d,d)
        hv = torch.trace(torch.linalg.matrix_exp(
            torch.tensor(W,device=DEVICE,dtype=torch.float64)**2)).item() - d
        alpha += rho*hv
        if abs(hv)<best_h: best_h,best_w = abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho = min(rho*2,1e10)
        if verbose and it%5==0:
            print(f'  [{it:2d}] h={hv:.2e} nz={int((np.abs(W)>0.3).sum())}',flush=True)
    W = best_w.reshape(d,d)
    return W, best_h, time.time()-t0, it+1

# ============================================================
# DAGMA: log-det acyclicity constraint h(W) = -log(det(I - W*W))
# ============================================================
def dagma(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    n, d = X.shape
    cov = X.T @ X / n
    w_vec = init_W.flatten().copy() if init_W is not None else np.random.randn(d*d)*0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    
    for it in range(max_outer):
        s = max(1.01, 10.0 * (0.85 ** it))  # Annealing: start safe, converge to 1
        def lag(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            M = s * torch.eye(d, device=DEVICE, dtype=torch.float64) - W_t * W_t
            hv = -torch.log(torch.clamp(torch.det(M), min=1e-20)).item() + d * np.log(s)
            base = 0.5*np.trace((np.eye(d)-W).T@cov@(np.eye(d)-W)) + lam*np.sum(np.abs(W))
            return base + 0.5*rho*hv**2 + alpha*hv
        
        def lag_grad(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            M = s * torch.eye(d, device=DEVICE, dtype=torch.float64) - W_t * W_t
            hv = -torch.log(torch.clamp(torch.det(M), min=1e-20)).item() + d * np.log(s)
            try:
                Minv = torch.linalg.inv(M)
            except:
                try:
                    Minv = torch.linalg.pinv(M, rcond=1e-8)
                except:
                    Minv = torch.zeros_like(M)
            dh = (-2.0 * W_t @ Minv).cpu().numpy()
            return (cov@(W-np.eye(d)) + lam*np.sign(W) + rho*hv*dh + alpha*dh).flatten()
        
        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
                       options={'maxiter':120,'ftol':1e-10,'gtol':1e-8})
        w_vec = res.x; W = w_vec.reshape(d,d)
        
        W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
        M = s * torch.eye(d, device=DEVICE, dtype=torch.float64) - W_t * W_t
        hv = -torch.log(torch.clamp(torch.det(M), min=1e-20)).item() + d * np.log(s)
        
        alpha += rho*hv
        if abs(hv)<best_h: best_h,best_w = abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho = min(rho*2,1e10)
        if verbose and it%5==0:
            print(f'  [{it:2d}] h={hv:.2e} nz={int((np.abs(W)>0.3).sum())}',flush=True)
    W = best_w.reshape(d,d)
    return W, best_h, time.time()-t0, it+1

# ============================================================
# GOLEM-EV: likelihood loss + augmented Lagrangian DAG constraint
# ============================================================
def golem(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    n, d = X.shape
    w_vec = init_W.flatten().copy() if init_W is not None else np.random.randn(d*d)*0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    # Precompute RSS scaling: d/2 * log(RSS/(d*n))
    X_t = torch.tensor(X, device=DEVICE, dtype=torch.float64)
    
    for it in range(max_outer):
        def loss_golem(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            # GOLEM-EV loss: n/2 * log(||X - XW||^2/n) - log|det(I-W)| + lam*||W||₁
            diff = X_t - X_t @ W_t
            rss = max(torch.sum(diff*diff).item(), 1e-20)
            # log-likelihood term
            ll = 0.5 * d * np.log(rss / n)
            # log-det term
            I = torch.eye(d, device=DEVICE, dtype=torch.float64)
            M = I - W_t
            logdet = torch.log(torch.clamp(torch.abs(torch.det(M)), min=1e-20)).item()
            l1 = lam * np.sum(np.abs(W))
            
            # DAG constraint
            W2 = W_t * W_t
            hv = torch.trace(torch.linalg.matrix_exp(W2)).item() - d
            
            return ll - logdet + l1 + 0.5*rho*hv**2 + alpha*hv
        
        def loss_golem_grad(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            
            # Gradient of log-likelihood term
            diff = X_t - X_t @ W_t
            rss = max(torch.sum(diff*diff).item(), 1e-20)
            # d(ll)/dW = -d * X^T*(X-XW) / RSS
            grad_ll = -d * (X_t.T @ diff).cpu().numpy() / rss
            
            # Gradient of log-det term: -log|det(I-W)| has gradient (I-W)^{-T}
            I = torch.eye(d, device=DEVICE, dtype=torch.float64)
            M = I - W_t
            try:
                grad_logdet = torch.linalg.inv(M).T.cpu().numpy()
            except:
                try:
                    grad_logdet = torch.linalg.pinv(M, rcond=1e-8).T.cpu().numpy()
                except:
                    grad_logdet = np.zeros((d, d))
            
            # L1 gradient
            grad_l1 = lam * np.sign(W)
            
            # DAG constraint gradient
            W2 = W_t * W_t
            expW2 = torch.linalg.matrix_exp(W2)
            hv = torch.trace(expW2).item() - d
            dh = (2 * W_t * expW2.T).cpu().numpy()
            
            return (grad_ll + grad_logdet + grad_l1 + rho*hv*dh + alpha*dh).flatten()
        
        res = minimize(loss_golem, w_vec, method='L-BFGS-B', jac=loss_golem_grad,
                       options={'maxiter':120,'ftol':1e-10,'gtol':1e-8})
        w_vec = res.x; W = w_vec.reshape(d,d)
        
        W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
        hv = torch.trace(torch.linalg.matrix_exp(W_t*W_t)).item() - d
        
        alpha += rho*hv
        if abs(hv)<best_h: best_h,best_w = abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho = min(rho*2,1e10)
        if verbose and it%5==0:
            print(f'  [{it:2d}] h={hv:.2e} nz={int((np.abs(W)>0.3).sum())}',flush=True)
    W = best_w.reshape(d,d)
    return W, best_h, time.time()-t0, it+1

# ============================================================
# Data + Eval (same as GPU pipeline)
# ============================================================
def load_data(d):
    brca = pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
    others=[]
    for f in sorted(os.listdir(TCGA_DIR)):
        if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'): continue
        try: others.append(pd.read_csv(os.path.join(TCGA_DIR,f),sep='\t',index_col=0).T)
        except: continue
    tcga_all=pd.concat(others,axis=0)
    mads=np.array([median_abs_deviation(tcga_all.iloc[:,j].values.astype(np.float64)) for j in range(tcga_all.shape[1])])
    mads=np.nan_to_num(mads,nan=0)
    top=list(tcga_all.columns[np.argsort(mads)[-d:]])
    brca_g=[g for g in top if g in brca.columns]; d_actual=len(brca_g)
    ci=[list(tcga_all.columns).index(g) for g in brca_g]
    X_tcga=tcga_all.iloc[:,ci].values.astype(np.float64); X_brca=brca[brca_g].values.astype(np.float64)
    X_tcga=np.nan_to_num(X_tcga,nan=0); X_brca=np.nan_to_num(X_brca,nan=0)
    X_tcga=(X_tcga-X_tcga.mean(0))/(X_tcga.std(0)+1e-12)
    X_brca=(X_brca-X_brca.mean(0))/(X_brca.std(0)+1e-12)
    return X_tcga,X_brca,d_actual,brca_g

def evaluate(W_src,W_tgt,tau=0.3):
    d=W_src.shape[0]; se=set(); te=set()
    for i in range(d):
        for j in range(d):
            if i!=j and abs(W_src[i,j])>tau: se.add((i,j))
            if i!=j and abs(W_tgt[i,j])>tau: te.add((i,j))
    ov=len(se&te); ret=ov/len(se)*100 if se else 0
    m=np.ones((d,d),bool); np.fill_diagonal(m,False)
    r,_=pearsonr(W_src[m],W_tgt[m])
    return {'src_edges':len(se),'tgt_edges':len(te),'overlap':ov,'retention_pct':round(ret,1),'pearson_r':round(r,4)}

def make_key(d,seed,method,algo):
    return f"d{d}_s{seed}_{algo}_{method}"

# ============================================================
# Main
# ============================================================
def main():
    ckpt = json.load(open(CHECKPOINT_FILE)) if os.path.exists(CHECKPOINT_FILE) else {'completed':{}}
    
    # Run DAGMA + GOLEM on d=200, 454, 600, 750, 889 (same as NOTEARS)
    dims = [200, 454, 600, 750, 889]
    seeds = [42, 123, 456]
    algorithms = {
        'dagma': dagma,
        'golem': golem
    }
    methods = ['scratch', 'warm']
    
    total = len(dims) * len(seeds) * len(algorithms) * len(methods)
    print(f'DAGMA + GOLEM benchmark: {total} configs (d={dims}, 3 seeds)')
    print(f'Device: {DEVICE}')
    
    for d in dims:
        d_actual_key = f'd{d}_actual'
        if d_actual_key not in ckpt:
            print(f'\nLoading d={d}...')
            tcga_X, brca_X, d_actual, genes = load_data(d)
            ckpt[d_actual_key] = d_actual
            with open(CHECKPOINT_FILE,'w') as f: json.dump(ckpt,f,indent=2)
        else:
            d_actual = ckpt[d_actual_key]
            print(f'\nReusing d={d} (actual={d_actual})...')
            tcga_X, brca_X, d_actual, genes = load_data(d)
        
        # NOTEARS source W (reuse from GPU pipeline if exists)
        src_key = make_key(d, 42, 'source', 'notears')
        if src_key not in ckpt:
            print('[NOTEARS source] needed...')
            if d <= 200:
                from scipy.linalg import expm
            W_src, h_src, dt_src, iters = notears(tcga_X, max_outer=35, verbose=False)
            np.save(os.path.join(CACHE_DIR, f'src_d{d_actual}.npy'), W_src)
            ckpt[src_key] = {'h':float(h_src),'time':dt_src,'edges':int((np.abs(W_src)>0.3).sum())}
            with open(CHECKPOINT_FILE,'w') as f: json.dump(ckpt,f,indent=2)
            print(f'  source done: {int((np.abs(W_src)>0.3).sum())} edges, {dt_src:.0f}s')
        
        W_src = np.load(os.path.join(CACHE_DIR, f'src_d{d_actual}.npy'))
        W_src = W_src[:d_actual,:d_actual]
        
        for algo_name, algo_fn in algorithms.items():
            for seed in seeds:
                for method in methods:
                    key = make_key(d, seed, method, algo_name)
                    if key in ckpt:
                        continue
                    
                    print(f'\n[{algo_name.upper():6s}] d={d_actual} {method} seed={seed}')
                    if method == 'scratch':
                        W, h, dt, iters = algo_fn(brca_X, max_outer=35, verbose=True)
                        np.save(os.path.join(CACHE_DIR, f'{algo_name}_scratch_d{d_actual}_s{seed}.npy'), W)
                        ev = evaluate(W_src, W)
                    else:
                        W, h, dt, iters = algo_fn(brca_X, max_outer=35, init_W=W_src, verbose=True)
                        np.save(os.path.join(CACHE_DIR, f'{algo_name}_warm_d{d_actual}_s{seed}.npy'), W)
                        ev = evaluate(W_src, W)
                    
                    ckpt[key] = {'h':float(h),'time':dt,**ev}
                    with open(CHECKPOINT_FILE,'w') as f: json.dump(ckpt,f,indent=2)
                    print(f'  ret={ev["retention_pct"]}% ov={ev["overlap"]}/{ev["src_edges"]} r={ev["pearson_r"]:.4f} dt={dt:.0f}s')
    
    # Summary
    print('\n' + '='*70)
    print('DAGMA + GOLEM FULL SUMMARY')
    print('='*70)
    for d in dims:
        d_actual = ckpt.get(f'd{d}_actual', d)
        print(f'\n--- d={d_actual} ---')
        for algo_name in ['dagma','golem']:
            for method in ['scratch','warm']:
                vals = []
                for seed in seeds:
                    key = make_key(d, seed, method, algo_name)
                    if key in ckpt:
                        vals.append(ckpt[key])
                if vals:
                    rets = [v['retention_pct'] for v in vals]
                    rs = [v['pearson_r'] for v in vals]
                    print(f'  {algo_name:6s} {method:8s} ret={np.mean(rets):5.1f}+/-{np.std(rets):4.1f}%  '
                          f'r={np.mean(rs):.4f}+/-{np.std(rs):.4f}  n={len(vals)}')

if __name__ == '__main__':
    main()
