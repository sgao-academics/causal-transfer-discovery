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
DAGMA + GOLEM d=454/600/750/889 benchmark (d=200 already done in checkpoint).
Race-condition fix: reload checkpoint before each save.
"""
import os, json, time, sys, numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from scipy.stats import pearsonr, median_abs_deviation

ROOT = PKG_ROOT
CACHE_DIR = os.path.join(ROOT, 'results', 'cache')
CHECKPOINT_FILE = os.path.join(ROOT, 'results', 'benchmark_v6_checkpoint.json')
TCGA_DIR = DATA_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = 'cuda'

def make_key(d, seed, method, algo):
    return f"d{d}_s{seed}_{algo}_{method}"

def safe_save(key, value):
    """Reload checkpoint, update, save — avoids race conditions."""
    ckpt = json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
    ckpt['completed'][key] = value
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f, indent=2)

def evaluate(W_src, W_tgt, tau=0.3):
    d = W_src.shape[0]; se=set(); te=set()
    for i in range(d):
        for j in range(d):
            if i!=j and abs(W_src[i,j])>tau: se.add((i,j))
            if i!=j and abs(W_tgt[i,j])>tau: te.add((i,j))
    ov=len(se&te); ret=ov/len(se)*100 if se else 0
    m=np.ones((d,d),bool); np.fill_diagonal(m,False)
    r,_=pearsonr(W_src[m],W_tgt[m])
    return {'src_edges':len(se),'tgt_edges':len(te),'overlap':ov,'retention_pct':round(ret,1),'pearson_r':round(r,4)}

# ============================================================
# DAGMA (fixed: use NOTEARS trace-exp constraint for numerical stability)
# ============================================================
def dagma_v2(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    """DAGMA loss (log-det) with NOTEARS acyclicity constraint (trace-exp) for stability."""
    n, d = X.shape
    cov = X.T @ X / n
    w_vec = init_W.flatten().copy() if init_W is not None else np.random.randn(d*d)*0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    
    for it in range(max_outer):
        def lag(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            # DAGMA loss: -log|det(I-W)| + ℓ₁ + NOTEARS constraint
            I = torch.eye(d, device=DEVICE, dtype=torch.float64)
            M = I - W_t
            logdet = torch.log(torch.clamp(torch.abs(torch.det(M)), min=1e-20)).item()
            l1 = lam * np.sum(np.abs(W))
            W2 = W_t * W_t
            hv = torch.trace(torch.linalg.matrix_exp(W2)).item() - d
            return 0.5*np.trace((np.eye(d)-W).T@cov@(np.eye(d)-W)) - logdet + l1 + 0.5*rho*hv**2 + alpha*hv
        
        def lag_grad(w):
            W = w.reshape(d,d)
            W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
            # MSE + L1 gradient
            base_grad = cov @ (W - np.eye(d)) + lam * np.sign(W)
            # Log-det gradient: (I-W)^{-T}
            I = torch.eye(d, device=DEVICE, dtype=torch.float64)
            M = I - W_t
            try:
                grad_logdet = torch.linalg.inv(M).T.cpu().numpy()
            except:
                try:
                    grad_logdet = torch.linalg.pinv(M, rcond=1e-8).T.cpu().numpy()
                except:
                    grad_logdet = np.zeros((d,d))
            # DAG constraint gradient
            W2 = W_t * W_t
            expW2 = torch.linalg.matrix_exp(W2)
            hv = torch.trace(expW2).item() - d
            dh = (2 * W_t * expW2.T).cpu().numpy()
            return (base_grad + grad_logdet + rho*hv*dh + alpha*dh).flatten()
        
        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
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
    return best_w.reshape(d,d), best_h, time.time()-t0, it+1

# GOLEM (existing, already stable)
def golem(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    n, d = X.shape
    w_vec = init_W.flatten().copy() if init_W is not None else np.random.randn(d*d)*0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    X_t = torch.tensor(X, device=DEVICE, dtype=torch.float64)
    for it in range(max_outer):
        def loss_g(w):
            W=w.reshape(d,d); W_t=torch.tensor(W,device=DEVICE,dtype=torch.float64)
            diff=X_t-X_t@W_t; rss=max(torch.sum(diff*diff).item(),1e-20)
            ll=0.5*d*np.log(rss/n)
            I=torch.eye(d,device=DEVICE,dtype=torch.float64)
            M=I-W_t; logdet=torch.log(torch.clamp(torch.abs(torch.det(M)),min=1e-20)).item()
            l1=lam*np.sum(np.abs(W))
            W2=W_t*W_t; hv=torch.trace(torch.linalg.matrix_exp(W2)).item()-d
            return ll-logdet+l1+0.5*rho*hv**2+alpha*hv
        def loss_g_grad(w):
            W=w.reshape(d,d); W_t=torch.tensor(W,device=DEVICE,dtype=torch.float64)
            diff=X_t-X_t@W_t; rss=max(torch.sum(diff*diff).item(),1e-20)
            grad_ll=-d*(X_t.T@diff).cpu().numpy()/rss
            I=torch.eye(d,device=DEVICE,dtype=torch.float64)
            M=I-W_t
            try: grad_logdet=torch.linalg.inv(M).T.cpu().numpy()
            except:
                try: grad_logdet=torch.linalg.pinv(M,rcond=1e-8).T.cpu().numpy()
                except: grad_logdet=np.zeros((d,d))
            grad_l1=lam*np.sign(W)
            W2=W_t*W_t; expW2=torch.linalg.matrix_exp(W2)
            hv=torch.trace(expW2).item()-d; dh=(2*W_t*expW2.T).cpu().numpy()
            return (grad_ll+grad_logdet+grad_l1+rho*hv*dh+alpha*dh).flatten()
        res=minimize(loss_g,w_vec,method='L-BFGS-B',jac=loss_g_grad,options={'maxiter':120,'ftol':1e-10,'gtol':1e-8})
        w_vec=res.x; W=w_vec.reshape(d,d)
        W_t=torch.tensor(W,device=DEVICE,dtype=torch.float64)
        hv=torch.trace(torch.linalg.matrix_exp(W_t*W_t)).item()-d
        alpha+=rho*hv
        if abs(hv)<best_h: best_h,best_w=abs(hv),w_vec.copy()
        if abs(hv)<1e-7: break
        if hv>0: rho=min(rho*2,1e10)
        if verbose and it%5==0: print(f'  [{it:2d}] h={hv:.2e} nz={int((np.abs(W)>0.3).sum())}',flush=True)
    return best_w.reshape(d,d),best_h,time.time()-t0,it+1

def load_data(d):
    brca=pd.read_csv(os.path.join(TCGA_DIR,'TCGA_BRCA_HiSeqV2.tsv'),sep='\t',index_col=0).T
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
    X_tcga=(X_tcga-X_tcga.mean(0))/(X_tcga.std(0)+1e-12); X_brca=(X_brca-X_brca.mean(0))/(X_brca.std(0)+1e-12)
    return X_tcga,X_brca,d_actual,brca_g

def main():
    dims = [454, 600, 750, 889]
    seeds = [42, 123, 456]
    algorithms = {'dagma': dagma_v2, 'golem': golem}
    
    for d in dims:
        print(f'\nLoading d={d}...')
        tcga_X, brca_X, d_actual, genes = load_data(d)
        
        # Check if NOTEARS source exists for this dim
        ckpt = json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
        src_seed42_key = make_key(d, 42, 'source', 'notears')
        if src_seed42_key not in ckpt['completed']:
            # Compute source NOTEARS
            print('[NOTEARS SOURCE] computing...')
            from run_dagma_golem import notears
            W_src, h, dt, it = notears(tcga_X, max_outer=35, verbose=True)
            np.save(os.path.join(CACHE_DIR, f'src_d{d_actual}.npy'), W_src)
            safe_save(src_seed42_key, {'h':float(h),'time':dt,'edges':int((np.abs(W_src)>0.3).sum())})
        else:
            src_path = os.path.join(CACHE_DIR, f'src_d{d_actual}.npy')
            if not os.path.exists(src_path):
                # Try different source file patterns
                for f in os.listdir(CACHE_DIR):
                    if f.startswith('src_d') and str(d_actual) in f and f.endswith('.npy'):
                        W_src = np.load(os.path.join(CACHE_DIR, f))[:d_actual,:d_actual]
                        break
            else:
                W_src = np.load(src_path)[:d_actual,:d_actual]
        
        for algo_name, algo_fn in algorithms.items():
            for seed in seeds:
                for method in ['scratch', 'warm']:
                    key = make_key(d, seed, method, algo_name)
                    # Reload ckpt each time to check
                    ckpt = json.load(open(CHECKPOINT_FILE, encoding='utf-8'))
                    if key in ckpt['completed']:
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
                    
                    safe_save(key, {'h':float(h),'time':dt,**ev})
                    print(f'  ret={ev["retention_pct"]}% ov={ev["overlap"]}/{ev["src_edges"]} dt={dt:.0f}s')
    
    print('\nALL DONE!')

if __name__ == '__main__':
    main()
