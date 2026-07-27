"""
GPU-accelerated benchmark pipeline with checkpointing.
Uses torch.linalg.matrix_exp for d>200 (~2500x speedup on GPU).
"""
import os, json, time, sys
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr, median_abs_deviation

# Resolve repository root (2 levels up from scripts/experiments/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, 'results', 'cache')
CHECKPOINT_FILE = os.path.join(ROOT, 'results', 'benchmark_v6_checkpoint.json')
TCGA_DIR = os.environ.get('TCGA_DATA_DIR', os.path.join(ROOT, 'data'))
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ============================================================
# NOTEARS (CPU fallback for d<=200, GPU for d>200)
# ============================================================
def notears(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    n, d = X.shape
    cov = X.T @ X / n
    if init_W is not None:
        w_vec = init_W.flatten().copy()
    else:
        w_vec = np.random.randn(d * d) * 0.1
    
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    use_gpu = (d > 200 and DEVICE == 'cuda')
    
    for it in range(max_outer):
        def lag(w):
            W = w.reshape(d, d)
            if use_gpu:
                W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
                hv = torch.trace(torch.linalg.matrix_exp(W_t * W_t)).item() - d
            else:
                from scipy.linalg import expm
                hv = np.trace(expm(W * W)) - d
            base = 0.5 * np.trace((np.eye(d) - W).T @ cov @ (np.eye(d) - W)) + lam * np.sum(np.abs(W))
            return base + 0.5 * rho * hv**2 + alpha * hv
        
        def lag_grad(w):
            W = w.reshape(d, d)
            if use_gpu:
                W_t = torch.tensor(W, device=DEVICE, dtype=torch.float64)
                W2 = W_t * W_t
                expW2 = torch.linalg.matrix_exp(W2)
                hv = torch.trace(expW2).item() - d
                dh = (2 * W_t * expW2.T).cpu().numpy()
            else:
                from scipy.linalg import expm
                hv = np.trace(expm(W * W)) - d
                dh = 2 * W * expm(W * W).T
            base_grad = cov @ (W - np.eye(d)) + lam * np.sign(W)
            return (base_grad + rho * hv * dh + alpha * dh).flatten()
        
        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
                       options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-8})
        w_vec = res.x
        W = w_vec.reshape(d, d)
        
        if use_gpu:
            hv = torch.trace(torch.linalg.matrix_exp(
                torch.tensor(W, device=DEVICE, dtype=torch.float64) ** 2)).item() - d
        else:
            from scipy.linalg import expm
            hv = np.trace(expm(W * W)) - d
        
        alpha += rho * hv
        if abs(hv) < best_h:
            best_h, best_w = abs(hv), w_vec.copy()
        if abs(hv) < 1e-7:
            break
        if hv > 0:
            rho = min(rho * 2, 1e10)
        if verbose:
            print(f'  [{it:2d}] h={hv:.2e} rho={rho:.1e} nz={int((np.abs(W)>0.3).sum())}', flush=True)
    
    W = best_w.reshape(d, d)
    dt = time.time() - t0
    return W, best_h, dt, it + 1

# ============================================================
# Data loading
# ============================================================
def load_data(d):
    brca = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0).T
    others = []
    for f in sorted(os.listdir(TCGA_DIR)):
        if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'):
            continue
        try:
            others.append(pd.read_csv(os.path.join(TCGA_DIR, f), sep='\t', index_col=0).T)
        except:
            continue
    tcga_all = pd.concat(others, axis=0)
    mads = np.array([median_abs_deviation(tcga_all.iloc[:, j].values.astype(np.float64))
                     for j in range(tcga_all.shape[1])])
    mads = np.nan_to_num(mads, nan=0)
    top_genes_all = list(tcga_all.columns[np.argsort(mads)[-d:]])
    brca_genes = [g for g in top_genes_all if g in brca.columns]
    d_actual = len(brca_genes)
    tcga_ci = [list(tcga_all.columns).index(g) for g in brca_genes]
    tcga_X = tcga_all.iloc[:, tcga_ci].values.astype(np.float64)
    brca_X = brca[brca_genes].values.astype(np.float64)
    tcga_X = np.nan_to_num(tcga_X, nan=0)
    brca_X = np.nan_to_num(brca_X, nan=0)
    tcga_X = (tcga_X - tcga_X.mean(0)) / (tcga_X.std(0) + 1e-12)
    brca_X = (brca_X - brca_X.mean(0)) / (brca_X.std(0) + 1e-12)
    return tcga_X, brca_X, d_actual, brca_genes

def evaluate(W_source, W_target, tau=0.3):
    d = W_source.shape[0]
    src_edges = set()
    tgt_edges = set()
    for i in range(d):
        for j in range(d):
            if i != j and abs(W_source[i, j]) > tau:
                src_edges.add((i, j))
            if i != j and abs(W_target[i, j]) > tau:
                tgt_edges.add((i, j))
    overlap = len(src_edges & tgt_edges)
    total_src = len(src_edges)
    retention = overlap / total_src * 100 if total_src > 0 else 0
    mask = np.ones((d, d), bool)
    np.fill_diagonal(mask, False)
    r_val, _ = pearsonr(W_source[mask], W_target[mask])
    return {'src_edges': total_src, 'tgt_edges': len(tgt_edges), 'overlap': overlap,
            'retention_pct': round(retention, 1), 'pearson_r': round(r_val, 4)}

# ============================================================
# Checkpoint
# ============================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed': {}, 'results': {}}

def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)

def make_key(d, seed, method):
    return f"d{d}_s{seed}_{method}"

# ============================================================
# Main
# ============================================================
def main():
    ckpt = load_checkpoint()
    
    # Dimension + seeds + methods
    dims = [200, 454, 600, 750, 889]
    seeds = [42, 123, 456]
    methods = ['source', 'scratch', 'warm', 'shuffled']
    
    total_configs = len(dims) * len(seeds) * len(methods)
    done_before = len(ckpt['completed'])
    print(f'Total configs: {total_configs}, Already done: {done_before}')
    
    data_cache = {}
    config_count = 0
    
    for d in dims:
        for seed in seeds:
            # Load data once per (d, seed)
            cache_key = f'd{d}_s{seed}'
            if cache_key not in data_cache:
                print(f'\nLoading d={d} seed={seed}...')
                tcga_X, brca_X, d_actual, genes = load_data(d)
                data_cache[cache_key] = (tcga_X, brca_X, d_actual)
            tcga_X, brca_X, d_actual = data_cache[cache_key]
            
            # Must have source first
            src_key = make_key(d, seed, 'source')
            if src_key not in ckpt['completed']:
                print(f'\n[SOURCE] d={d_actual} s={seed}')
                W_src, h, dt, iters = notears(tcga_X, max_outer=35, verbose=True)
                np.save(os.path.join(CACHE_DIR, f'src_d{d_actual}_s{seed}.npy'), W_src)
                ckpt['completed'][src_key] = {'h': float(h), 'time': dt, 'd_actual': d_actual,
                                               'edges': int((np.abs(W_src) > 0.3).sum())}
                save_checkpoint(ckpt)
            
            W_src = np.load(os.path.join(CACHE_DIR, f'src_d{d_actual}_s{seed}.npy'))
            
            for method in ['scratch', 'warm', 'shuffled']:
                key = make_key(d, seed, method)
                if key in ckpt['completed']:
                    continue
                
                config_count += 1
                print(f'\n[{config_count}] {method.upper()} d={d_actual} s={seed}')
                
                if method == 'scratch':
                    W, h, dt, iters = notears(brca_X, max_outer=35, verbose=True)
                    np.save(os.path.join(CACHE_DIR, f'scratch_d{d_actual}_s{seed}.npy'), W)
                    ev = evaluate(W_src[:d_actual,:d_actual], W, tau=0.3)
                elif method == 'warm':
                    W, h, dt, iters = notears(brca_X, max_outer=35, init_W=W_src[:d_actual,:d_actual], verbose=True)
                    np.save(os.path.join(CACHE_DIR, f'warm_d{d_actual}_s{seed}.npy'), W)
                    ev = evaluate(W_src[:d_actual,:d_actual], W, tau=0.3)
                elif method == 'shuffled':
                    idx = np.random.RandomState(seed).permutation(d_actual)
                    W_shuf = W_src[:d_actual,:d_actual][idx, :][:, idx]
                    W, h, dt, iters = notears(brca_X, max_outer=35, init_W=W_shuf, verbose=True)
                    np.save(os.path.join(CACHE_DIR, f'shuf_d{d_actual}_s{seed}.npy'), W)
                    ev = evaluate(W_src[:d_actual,:d_actual], W, tau=0.3)
                
                ckpt['completed'][key] = {'h': float(h), 'time': dt, **ev}
                save_checkpoint(ckpt)
                print(f'  ret={ev["retention_pct"]}% ov={ev["overlap"]}/{ev["src_edges"]} '
                      f'r={ev["pearson_r"]:.4f} dt={dt:.0f}s')
    
    # ---- Summary ----
    print('\n' + '='*60)
    print('SUMMARY (mean +/- std)')
    print('='*60)
    summary = {}
    for d in dims:
        summary[f'd{d}'] = {}
        for method in ['scratch', 'warm', 'shuffled']:
            values = []
            for seed in seeds:
                key = make_key(d, seed, method)
                if key in ckpt['completed']:
                    values.append(ckpt['completed'][key])
            if values:
                rets = [v['retention_pct'] for v in values]
                rs = [v['pearson_r'] for v in values]
                summary[f'd{d}'][method] = {
                    'mean_retention': round(np.mean(rets), 1),
                    'std_retention': round(np.std(rets), 1),
                    'mean_r': round(np.mean(rs), 4),
                    'std_r': round(np.std(rs), 4),
                    'n_seeds': len(values)
                }
                print(f'd={d:3d} {method:10s} ret={np.mean(rets):5.1f}+/-{np.std(rets):4.1f}%  '
                      f'r={np.mean(rs):.4f}+/-{np.std(rs):.4f}  n={len(values)}')
    
    with open(os.path.join(ROOT, 'results', 'benchmark_v6_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved. Total configs: {len(ckpt["completed"])}')

if __name__ == '__main__':
    main()
