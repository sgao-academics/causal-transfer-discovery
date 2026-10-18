#!/usr/bin/env python3
# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(1):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""
Fixed warm-start: scaled source graph to avoid float32 overflow in matrix_exp.
Source: correlation proxy, threshold 0.7, scaled to weight 0.3.
Re-runs warm-start only (scratch already succeeded: 715 edges).
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

BASE = PKG_ROOT
DATA_DIR = DATA_DIR
CKPT_DIR = os.path.join(BASE, "results", "d5000_checkpoints")
CACHE_DIR = os.path.join(BASE, "results", "d5000_cache")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

d_val = 5000
OUTER = 20
INNER = 100
LR = 0.002
LAMBDA1 = 0.01
W_THRESHOLD = 0.3
SEED = 42
SRC_CORR_THRESHOLD = 0.7   # higher threshold = sparser, higher-confidence source
SRC_WEIGHT = 0.3            # scale factor to avoid matrix_exp overflow

torch.manual_seed(SEED); np.random.seed(SEED)

# ============ DATA CACHING ============
print("Loading data (with cache)...", flush=True)
t0 = time.time()

cache_brca = os.path.join(CACHE_DIR, "X_brca_d5000.npy")
cache_src = os.path.join(CACHE_DIR, "X_source_d5000.npy")
cache_genes = os.path.join(CACHE_DIR, "selected_genes_d5000.txt")
cache_wsrc = os.path.join(CACHE_DIR, "W_src_d5000.npy")

tcga_files = [f for f in os.listdir(DATA_DIR) if f.startswith('TCGA_') and '_HiSeqV2.tsv' in f]
brca_file = [f for f in tcga_files if 'BRCA' in f][0]
source_files = [f for f in tcga_files if 'BRCA' not in f]

if all(os.path.exists(p) for p in [cache_brca, cache_src, cache_wsrc]):
    X_brca = np.load(cache_brca)
    X_source = np.load(cache_src)
    W_src = np.load(cache_wsrc)
    print(f"  Loaded from cache: BRCA {X_brca.shape}, Source {X_source.shape}", flush=True)
else:
    df_brca = pd.read_csv(os.path.join(DATA_DIR, brca_file), sep='\t', index_col=0)
    X_brca_arr = df_brca.values.T.astype(np.float32)
    var = np.var(X_brca_arr, axis=0)
    top_idx = np.argsort(var)[-d_val:]
    selected_genes = df_brca.index[top_idx].tolist()
    X_brca = X_brca_arr[:, top_idx]
    X_brca = (X_brca - X_brca.mean(0)) / (X_brca.std(0) + 1e-8)
    
    source_chunks = []
    for fname in source_files:
        fp = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fp, sep='\t', index_col=0)
            available = [g for g in selected_genes if g in df.index]
            if len(available) < d_val * 0.5: continue
            sub = df.loc[available].values.T.astype(np.float32)
            if sub.shape[1] < d_val:
                full = np.zeros((sub.shape[0], d_val), dtype=np.float32)
                match_idx = [selected_genes.index(g) for g in available]
                full[:, match_idx] = sub
                sub = full
            sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)
            source_chunks.append(sub)
        except: continue
    X_source = np.vstack(source_chunks)
    
    corr = np.corrcoef(X_source.T)
    np.fill_diagonal(corr, 0)
    W_src = (np.abs(corr) > SRC_CORR_THRESHOLD).astype(np.float32) * SRC_WEIGHT
    
    # Save cache
    np.save(cache_brca, X_brca)
    np.save(cache_src, X_source)
    np.save(cache_wsrc, W_src)
    with open(cache_genes, 'w') as f:
        f.write('\n'.join(selected_genes))
    print(f"  Built + cached: BRCA {X_brca.shape}, Source {X_source.shape}", flush=True)

n_src = int(np.sum(W_src > 0))
W_sq = W_src * W_src
max_row = W_sq.sum(axis=1).max()
print(f"  Source edges (thr={SRC_CORR_THRESHOLD}): {n_src}", flush=True)
print(f"  Max row sum of W*W: {max_row:.3f} (exp({max_row:.1f})={np.exp(max_row):.0f}, safe if < 80)", flush=True)
print(f"  Data ready: {time.time()-t0:.0f}s", flush=True)

# ============ WARM-START (checkpointed) ============
def run_warm(X_np, W_init_np, label="warm", outer=OUTER, inner=INNER, lr=LR, lambda1=LAMBDA1):
    n, d = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{label}_d{d}.json")
    W_path = os.path.join(CKPT_DIR, f"W_{label}_d{d}.npy")
    
    start_outer = 0
    rho, alpha = 1.0, 0.0
    h_old = float('inf')
    total_time = 0.0
    
    if os.path.exists(ckpt_path) and os.path.exists(W_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        start_outer = ckpt.get("last_outer", 0)
        rho = ckpt.get("rho", 1.0)
        alpha = ckpt.get("alpha", 0.0)
        h_old = ckpt.get("h_old", float('inf'))
        total_time = ckpt.get("total_time", 0.0)
        W = torch.tensor(np.load(W_path), dtype=torch.float32, device=device, requires_grad=True)
        print(f"  Resumed from outer {start_outer}", flush=True)
    else:
        W = torch.tensor(W_init_np, dtype=torch.float32, device=device, requires_grad=True)
        print(f"  Starting fresh with {int(np.sum(np.abs(W_init_np)>0))} init edges", flush=True)
    
    for o in range(start_outer, outer):
        t_loop = time.time()
        optimizer = torch.optim.Adam([W], lr=lr)
        
        for i in range(inner):
            optimizer.zero_grad()
            W_sq = W * W
            M_exp = torch.linalg.matrix_exp(W_sq)
            h = torch.trace(M_exp) - d
            XW = X_t @ W
            mse = 0.5 / n * torch.sum((X_t - XW) ** 2)
            l1 = lambda1 * torch.sum(torch.abs(W))
            loss = mse + l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 10.0)
            optimizer.step()
        
        with torch.no_grad():
            W_sq = W * W
            h_val = torch.trace(torch.linalg.matrix_exp(W_sq)).item() - d
        
        if h_val > 0.25 * h_old:
            rho = min(rho * 10, 1e12)
            alpha = alpha + rho * h_val
        
        n_edges = int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
        dt = time.time() - t_loop
        total_time += dt
        
        print(f"  [{label}] outer {o+1}/{outer}: h={h_val:.2e}, rho={rho:.1e}, edges={n_edges}, {dt:.0f}s", flush=True)
        
        W_np_current = W.detach().cpu().numpy()
        np.save(W_path, W_np_current)
        with open(ckpt_path, 'w') as f:
            json.dump({"last_outer": o+1, "rho": rho, "alpha": alpha, "h_old": h_val,
                       "total_time": total_time, "last_h": h_val, "last_edges": n_edges}, f)
        
        h_old = h_val
        if abs(h_val) < 1e-6:
            print(f"  Converged! h={h_val:.2e}", flush=True)
            break
    
    W_np = W.detach().cpu().numpy()
    n_edges = int(np.sum(np.abs(W_np) > W_THRESHOLD))
    return W_np, n_edges

print(f"\n=== Warm-start (scaled source, OUTER={OUTER}, INNER={INNER}) ===", flush=True)
torch.cuda.empty_cache()
W_warm, n_warm = run_warm(X_brca, W_src, label="warm")

# ============ RESULTS ============
print(f"\n=== RESULTS ===", flush=True)
W_scratch = np.load(os.path.join(CKPT_DIR, "W_scratch_d5000.npy"))
n_scratch = int(np.sum(np.abs(W_scratch) > W_THRESHOLD))

src_set = set((i,j) for i in range(d_val) for j in range(d_val) if i!=j and W_src[i,j]>0)
scratch_set = set((i,j) for i in range(d_val) for j in range(d_val) if i!=j and abs(W_scratch[i,j])>W_THRESHOLD)
warm_set = set((i,j) for i in range(d_val) for j in range(d_val) if i!=j and abs(W_warm[i,j])>W_THRESHOLD)

ret_s = len(src_set & scratch_set)
ret_w = len(src_set & warm_set)
ret_s_pct = 100*ret_s/max(1,len(src_set))
ret_w_pct = 100*ret_w/max(1,len(src_set))

print(f"  Source edges: {n_src}", flush=True)
print(f"  Scratch edges: {n_scratch}", flush=True)
print(f"  Warm edges: {n_warm}", flush=True)
print(f"  Retention scratch: {ret_s}/{len(src_set)} ({ret_s_pct:.1f}%)", flush=True)
print(f"  Retention warm: {ret_w}/{len(src_set)} ({ret_w_pct:.1f}%)", flush=True)
print(f"  Transfer gain: {ret_w_pct-ret_s_pct:+.1f}pp", flush=True)

result = {
    "d": d_val,
    "n_source": X_source.shape[0], "n_target": X_brca.shape[0],
    "src_corr_threshold": SRC_CORR_THRESHOLD, "src_weight": SRC_WEIGHT,
    "source_edges": n_src,
    "scratch_edges": n_scratch, "warm_edges": n_warm,
    "retention_scratch": ret_s, "retention_warm": ret_w,
    "retention_scratch_pct": round(ret_s_pct, 2),
    "retention_warm_pct": round(ret_w_pct, 2),
    "transfer_gain_pp": round(ret_w_pct-ret_s_pct, 2),
    "warm_scratch_ratio": round(n_warm/max(1,n_scratch), 3),
    "outer": OUTER, "inner": INNER, "lr": LR, "device": str(device),
}

with open(os.path.join(BASE, "results", "d5000_results.json"), 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: results/d5000_results.json", flush=True)
