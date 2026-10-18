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
Full benchmark pipeline with checkpointing.
V6 audit fixes: d=600/750, multi-seed, shuffled control, synthetic DAG.
Each config completed → immediate json.dump to checkpoint.
Crash-safe: auto-skip completed configs on restart.
"""
import os, json, time, sys, hashlib
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import pearsonr, median_abs_deviation

ROOT = PKG_ROOT
CACHE_DIR = os.path.join(ROOT, 'results', 'cache')
CHECKPOINT_FILE = os.path.join(ROOT, 'results', 'benchmark_v6_checkpoint.json')
TCGA_DIR = DATA_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# NOTEARS Implementation (matches reference implementation)
# ============================================================
def notears(X, lam=0.01, max_outer=35, init_W=None, verbose=False):
    n, d = X.shape
    cov = X.T @ X / n
    if init_W is not None:
        w_vec = init_W.flatten().copy()
    else:
        np.random.seed(int(time.time() * 1e6) % (2**31))
        w_vec = np.random.randn(d * d) * 0.1
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    t0 = time.time()
    for it in range(max_outer):
        def lag(w):
            W = w.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            return 0.5 * np.trace((np.eye(d) - W).T @ cov @ (np.eye(d) - W)) + lam * np.sum(np.abs(W)) + 0.5 * rho * hv**2 + alpha * hv
        def lag_grad(w):
            W = w.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            dh = 2 * W * expm(W * W).T
            return (cov @ (W - np.eye(d)) + lam * np.sign(W)).flatten() + rho * hv * dh.flatten() + alpha * dh.flatten()
        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad, 
                       options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-8})
        w_vec = res.x
        W = w_vec.reshape(d, d)
        hv = np.trace(expm(W * W)) - d
        alpha += rho * hv
        if abs(hv) < best_h:
            best_h, best_w = abs(hv), w_vec.copy()
        if abs(hv) < 1e-7:
            break
        if hv > 0:
            rho = min(rho * 2, 1e10)
        if verbose and it % 5 == 0:
            nz = int((np.abs(W) > 0.3).sum())
            print(f'  [{it:2d}] h={hv:.2e} rho={rho:.1e} edges={nz}', flush=True)
    W = best_w.reshape(d, d)
    dt = time.time() - t0
    return W, best_h, dt, it + 1

# ============================================================
# Data loading
# ============================================================
def load_data(d):
    """Load TCGA(-BRCA), DepMap, BRCA data at dimension d."""
    # Load BRCA
    brca = pd.read_csv(os.path.join(TCGA_DIR, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0).T
    
    # Load other cancer types
    others = []
    for f in sorted(os.listdir(TCGA_DIR)):
        if 'BRCA' in f or not f.startswith('TCGA_') or not f.endswith('.tsv'):
            continue
        try:
            others.append(pd.read_csv(os.path.join(TCGA_DIR, f), sep='\t', index_col=0).T)
        except:
            continue
    tcga_all = pd.concat(others, axis=0)
    
    # MAD ranking on TCGA
    mads = np.array([median_abs_deviation(tcga_all.iloc[:, j].values.astype(np.float64))
                     for j in range(tcga_all.shape[1])])
    mads = np.nan_to_num(mads, nan=0)
    
    # Select common genes
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

def generate_synthetic_dag(d, seed=42):
    """Generate random DAG with Erdos-Renyi structure."""
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.rand() < 2.0 / (d - 1):  # avg 2d edges
                W_true[i, j] = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    return W_true

def generate_synthetic_data(W_true, n=500, seed=42):
    """Generate data from linear SEM."""
    rng = np.random.RandomState(seed)
    d = W_true.shape[0]
    X = rng.randn(n, d)
    I = np.eye(d)
    inv = np.linalg.inv(I - W_true.T)
    eps = np.sqrt(0.5) * rng.randn(n, d)
    X = eps @ inv
    return X.astype(np.float64)

# ============================================================
# Evaluation metrics
# ============================================================
def evaluate(W_source, W_target, tau=0.3):
    """Compute overlap and correlation metrics."""
    d = W_source.shape[0]
    mask = np.ones((d, d), dtype=bool)
    np.fill_diagonal(mask, False)
    
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
    total_tgt = len(tgt_edges)
    retention = overlap / total_src * 100 if total_src > 0 else 0
    
    r_val, _ = pearsonr(W_source[mask], W_target[mask])
    
    return {
        'src_edges': total_src,
        'tgt_edges': total_tgt,
        'overlap': overlap,
        'retention_pct': round(retention, 1),
        'pearson_r': round(r_val, 4)
    }

# ============================================================
# Checkpoint management
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
# Single config runner
# ============================================================
def run_config(d, seed, method, X_source, X_target, W_source=None):
    """Run one configuration. Returns dict with results."""
    np.random.seed(seed)
    
    if method == 'scratch':
        W, h, dt, iters = notears(X_target, max_outer=35, verbose=True)
    elif method == 'warm':
        W, h, dt, iters = notears(X_target, max_outer=35, init_W=W_source, verbose=True)
    elif method == 'shuffled':
        # Shuffle rows of W_source (randomize structure, preserve sparsity)
        d = W_source.shape[0]
        idx = np.random.permutation(d)
        W_shuffled = W_source[idx, :]
        W_shuffled = W_shuffled[:, idx]
        W, h, dt, iters = notears(X_target, max_outer=35, init_W=W_shuffled, verbose=True)
    elif method == 'source':
        # Just compute source graph on X_source
        W, h, dt, iters = notears(X_source, max_outer=35, verbose=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {'W': W, 'h': float(h), 'time': dt, 'iters': iters}

# ============================================================
# Pipeline
# ============================================================
def main():
    ckpt = load_checkpoint()
    
    # ----- Experiment definitions -----
    # Synthetic DAG experiments
    synth_configs = []
    for d in [200, 500]:
        for seed in [42, 123, 456]:
            synth_configs.append({'d': d, 'seed': seed})
    
    # Real data experiments
    real_configs = []
    for d in [200, 454, 600, 750, 889]:
        for seed in [42, 123, 456]:
            for method in ['source', 'scratch', 'warm', 'shuffled']:
                real_configs.append({'d': d, 'seed': seed, 'method': method})
    
    # ---- Process synthetic DAGs ----
    print("=" * 60)
    print("SYNTHETIC DAG EXPERIMENTS")
    print("=" * 60)
    synth_results = {}
    for cfg in synth_configs:
        d, seed = cfg['d'], cfg['seed']
        key_scratch = make_key(d, seed, 'synth_scratch')
        key_warm = make_key(d, seed, 'synth_warm')
        
        if key_scratch in ckpt['completed'] and key_warm in ckpt['completed']:
            print(f'SKIP synthetic d={d} s={seed} (both done)')
            continue
        
        # Generate ground truth
        W_true = generate_synthetic_dag(d, seed=42)  # same DAG for all seeds
        X_train = generate_synthetic_data(W_true, n=500, seed=seed)
        X_test = generate_synthetic_data(W_true, n=200, seed=seed + 1000)
        
        # Scratch on test set
        if key_scratch not in ckpt['completed']:
            print(f'\n[SYNTH] d={d} s={seed} scratch')
            res = run_config(d, seed, 'scratch', None, X_test)
            W_scr = res['W']
            np.save(os.path.join(CACHE_DIR, f'synth_d{d}_s{seed}_scratch.npy'), W_scr)
            ev = evaluate(W_true, W_scr, tau=0.3)
            ckpt['completed'][key_scratch] = {'h': res['h'], 'time': res['time'], **ev}
            save_checkpoint(ckpt)
        
        # Warm-start on test set
        if key_warm not in ckpt['completed']:
            print(f'\n[SYNTH] d={d} s={seed} warm')
            res = run_config(d, seed, 'warm', None, X_test, W_source=W_true)
            W_warm = res['W']
            np.save(os.path.join(CACHE_DIR, f'synth_d{d}_s{seed}_warm.npy'), W_warm)
            ev = evaluate(W_true, W_warm, tau=0.3)
            ckpt['completed'][key_warm] = {'h': res['h'], 'time': res['time'], **ev}
            save_checkpoint(ckpt)
        
        # Record scratch reference for comparison
        if key_scratch in ckpt['completed'] and key_warm in ckpt['completed']:
            s_data = ckpt['completed'][key_scratch]
            w_data = ckpt['completed'][key_warm]
            synth_results[f'd{d}_s{seed}'] = {
                'scratch': s_data, 'warm': w_data,
                'delta_overlap': w_data['overlap'] - s_data['overlap'],
                'delta_retention': round(w_data['retention_pct'] - s_data['retention_pct'], 1)
            }
            print(f'  SYNTH d={d} s={seed}: scratch={s_data["overlap"]}/{s_data["src_edges"]} '
                  f'warm={w_data["overlap"]}/{w_data["src_edges"]} '
                  f'(delta={synth_results[f"d{d}_s{seed}"]["delta_overlap"]:+d})')
    
    # ---- Process real data ----
    print("\n" + "=" * 60)
    print("REAL DATA EXPERIMENTS")
    print("=" * 60)
    
    # Group by (d, seed) for efficient data loading
    data_cache = {}
    
    for cfg in real_configs:
        d, seed, method = cfg['d'], cfg['seed'], cfg['method']
        key = make_key(d, seed, method)
        
        if key in ckpt['completed']:
            continue
        
        # Load data (cache per d)
        if (d, seed) not in data_cache:
            print(f'\n  Loading d={d} seed={seed}...')
            tcga_X, brca_X, d_actual, genes = load_data(d)
            data_cache[(d, seed)] = (tcga_X, brca_X, d_actual)
        
        tcga_X, brca_X, d_actual = data_cache[(d, seed)]
        
        # Load or compute source W
        source_key = make_key(d, seed, 'source')
        if source_key in ckpt['completed']:
            W_source = np.load(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'))
        elif method == 'source':
            pass  # will be computed below
        else:
            # Need source first
            if source_key not in ckpt['completed']:
                print(f'\n  Computing source d={d} s={seed} (prerequisite)...')
                res = run_config(d_actual, seed, 'source', tcga_X, None)
                W_source = res['W']
                np.save(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'), W_source)
                ckpt['completed'][source_key] = {'h': res['h'], 'time': res['time'], 
                                                  'edges': int((np.abs(W_source) > 0.3).sum())}
                save_checkpoint(ckpt)
            else:
                W_source = np.load(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'))
        
        # Run method
        print(f'\n[{method.upper()}] d={d} s={seed}')
        if method == 'source':
            if source_key in ckpt['completed']:
                continue
            res = run_config(d_actual, seed, method, tcga_X, None)
            W = res['W']
            np.save(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'), W)
            ckpt['completed'][key] = {'h': res['h'], 'time': res['time'],
                                       'edges': int((np.abs(W) > 0.3).sum())}
        elif method == 'scratch':
            res = run_config(d_actual, seed, method, None, brca_X)
            W = res['W']
            np.save(os.path.join(CACHE_DIR, f'scratch_d{d}_s{seed}.npy'), W)
            W_source = np.load(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'))
            ev = evaluate(W_source, W, tau=0.3)
            ckpt['completed'][key] = {'h': res['h'], 'time': res['time'], **ev}
        elif method == 'warm':
            W_source = np.load(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'))
            res = run_config(d_actual, seed, method, None, brca_X, W_source=W_source)
            W = res['W']
            np.save(os.path.join(CACHE_DIR, f'warm_d{d}_s{seed}.npy'), W)
            ev = evaluate(W_source, W, tau=0.3)
            ckpt['completed'][key] = {'h': res['h'], 'time': res['time'], **ev}
        elif method == 'shuffled':
            W_source = np.load(os.path.join(CACHE_DIR, f'src_d{d}_s{seed}.npy'))
            res = run_config(d_actual, seed, method, None, brca_X, W_source=W_source)
            W = res['W']
            np.save(os.path.join(CACHE_DIR, f'shuf_d{d}_s{seed}.npy'), W)
            ev = evaluate(W_source, W, tau=0.3)
            ckpt['completed'][key] = {'h': res['h'], 'time': res['time'], **ev}
        
        save_checkpoint(ckpt)
        print(f'  edges={ckpt["completed"][key].get("tgt_edges","?")} '
              f'ret={ckpt["completed"][key].get("retention_pct","?")}% '
              f'r={ckpt["completed"][key].get("pearson_r","?")} '
              f'time={ckpt["completed"][key].get("time",0):.0f}s')
    
    # ---- Save summary ----
    # Collect all real results by (d, seed)
    summary = {}
    for d in [200, 454, 600, 750, 889]:
        summary[f'd{d}'] = {}
        for seed in [42, 123, 456]:
            summary[f'd{d}'][f's{seed}'] = {}
            for method in ['scratch', 'warm', 'shuffled']:
                key = make_key(d, seed, method)
                if key in ckpt['completed']:
                    summary[f'd{d}'][f's{seed}'][method] = ckpt['completed'][key]
    
    # Compute means
    summary['means'] = {}
    for d_str in summary:
        if d_str == 'means' or d_str.startswith('d') is False:
            continue
        d = int(d_str[1:])
        summary['means'][f'd{d}'] = {}
        for method in ['scratch', 'warm', 'shuffled']:
            values = []
            for seed_str in ['s42', 's123', 's456']:
                if seed_str in summary[d_str] and method in summary[d_str][seed_str]:
                    values.append(summary[d_str][seed_str][method])
            if values:
                summary['means'][f'd{d}'][method] = {
                    'mean_retention': round(np.mean([v['retention_pct'] for v in values]), 1),
                    'std_retention': round(np.std([v['retention_pct'] for v in values]), 1),
                    'mean_r': round(np.mean([v['pearson_r'] for v in values]), 4),
                    'std_r': round(np.std([v['pearson_r'] for v in values]), 4),
                    'mean_edges': round(np.mean([v['tgt_edges'] for v in values]), 1),
                    'n_seeds': len(values)
                }
    
    summary['synthetic'] = synth_results
    
    with open(os.path.join(ROOT, 'results', 'benchmark_v6_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for d_str in sorted(summary.keys()):
        if d_str in ['means', 'synthetic']:
            continue
        d = int(d_str[1:])
        if d_str in summary.get('means', {}):
            m = summary['means'][d_str]
            print(f'\nd={d}:')
            for method in ['scratch', 'warm', 'shuffled']:
                if method in m:
                    mm = m[method]
                    print(f'  {method:10s} ret={mm["mean_retention"]:5.1f}±{mm["std_retention"]:4.1f}%  '
                          f'r={mm["mean_r"]:.4f}±{mm["std_r"]:.4f}  '
                          f'edges={mm["mean_edges"]:.0f}  n={mm["n_seeds"]}')
    
    if 'synthetic' in summary:
        print(f'\nSYNTHETIC: {len(summary["synthetic"])} configurations')

if __name__ == '__main__':
    main()
