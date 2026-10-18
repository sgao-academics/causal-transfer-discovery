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
Run NOTEARS on the SOURCE data (non-BRCA, 9241 samples) to obtain a TRUE
causal source graph, then warm-start BRCA from it.

Key optimization: MSE loss computed via the X^T X closed form, so cost is
O(d^3) per step (independent of sample count n). This makes source-graph
NOTEARS as fast as the target (1218-sample) run despite 7.6x more samples.

Pipeline:
  1. Load cached X_source (9241 x 5000) and X_brca (1218 x 5000).
  2. Source NOTEARS (X^T X form) -> W_src_causal  [label="source"]
  3. Warm-start BRCA from W_src_causal (X@W form) [label="warm_causal"]
  4. Compare scratch vs warm_causal retention.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3} GB)", flush=True)

BASE = PKG_ROOT
CKPT_DIR = os.path.join(BASE, "results", "d5000_checkpoints")
CACHE_DIR = os.path.join(BASE, "results", "d5000_cache")
os.makedirs(CKPT_DIR, exist_ok=True)

D = 5000
OUTER = 20
INNER = 100
LR = 0.002
LAMBDA1 = 0.01
W_THRESHOLD = 0.3
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)

# ============ LOAD CACHED DATA ============
print("Loading cached data...", flush=True)
X_brca = np.load(os.path.join(CACHE_DIR, "X_brca_d5000.npy"))
X_source = np.load(os.path.join(CACHE_DIR, "X_source_d5000.npy"))
print(f"  BRCA: {X_brca.shape}, Source: {X_source.shape}", flush=True)

# ============ SOURCE NOTEARS (X^T X closed form) ============
def run_source_notears_xtx(X_np, label="source"):
    """NOTEARS with MSE via X^T X (cost independent of n)."""
    n, d = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    # Precompute S = X^T X / n  (d x d)
    S = (X_t.T @ X_t) / n
    tr_S = torch.trace(S).item()
    print(f"  Precomputed S = X^T X / n ({d}x{d}), trace(S)={tr_S:.1f}", flush=True)

    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{label}_d{d}.json")
    W_path = os.path.join(CKPT_DIR, f"W_{label}_d{d}.npy")

    start_outer, rho, alpha, h_old = 0, 1.0, 0.0, float('inf')
    if os.path.exists(ckpt_path) and os.path.exists(W_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        start_outer = ck.get("last_outer", 0)
        rho, alpha = ck.get("rho", 1.0), ck.get("alpha", 0.0)
        h_old = ck.get("h_old", float('inf'))
        W = torch.tensor(np.load(W_path), dtype=torch.float32, device=device, requires_grad=True)
        print(f"  [{label}] resumed from outer {start_outer}", flush=True)
    else:
        W = torch.zeros(d, d, dtype=torch.float32, device=device, requires_grad=True)
        print(f"  [{label}] starting fresh", flush=True)

    for o in range(start_outer, OUTER):
        t_loop = time.time()
        opt = torch.optim.Adam([W], lr=LR)
        for _ in range(INNER):
            opt.zero_grad()
            W_sq = W * W
            h = torch.trace(torch.linalg.matrix_exp(W_sq)) - d
            # MSE via X^T X: ||X - XW||^2 / (2n)
            SW = S @ W
            mse = 0.5 * (tr_S - 2.0 * torch.trace(SW) + torch.trace(W.T @ SW))
            l1 = LAMBDA1 * torch.sum(torch.abs(W))
            loss = mse + l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 10.0)
            opt.step()

        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)).item() - d

        if h_val > 0.25 * h_old:
            rho = min(rho * 10, 1e12)
            alpha = alpha + rho * h_val

        n_edges = int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
        print(f"  [{label}] outer {o+1}/{OUTER}: h={h_val:.2e}, rho={rho:.1e}, "
              f"edges={n_edges}, {time.time()-t_loop:.0f}s", flush=True)

        np.save(W_path, W.detach().cpu().numpy())
        with open(ckpt_path, 'w') as f:
            json.dump({"last_outer": o+1, "rho": rho, "alpha": alpha,
                       "h_old": h_val, "last_h": h_val, "last_edges": n_edges}, f)

        h_old = h_val
        if abs(h_val) < 1e-6:
            print(f"  [{label}] converged (h={h_val:.2e})", flush=True)
            break

    W_np = W.detach().cpu().numpy()
    return W_np, int(np.sum(np.abs(W_np) > W_THRESHOLD))

# ============ WARM-START (X@W form, target samples) ============
def run_warm_xtarget(X_np, W_init_np, label="warm_causal"):
    n, d = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{label}_d{d}.json")
    W_path = os.path.join(CKPT_DIR, f"W_{label}_d{d}.npy")

    start_outer, rho, alpha, h_old = 0, 1.0, 0.0, float('inf')
    if os.path.exists(ckpt_path) and os.path.exists(W_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        start_outer = ck.get("last_outer", 0)
        rho, alpha = ck.get("rho", 1.0), ck.get("alpha", 0.0)
        h_old = ck.get("h_old", float('inf'))
        W = torch.tensor(np.load(W_path), dtype=torch.float32, device=device, requires_grad=True)
        print(f"  [{label}] resumed from outer {start_outer}", flush=True)
    else:
        W = torch.tensor(W_init_np, dtype=torch.float32, device=device, requires_grad=True)
        print(f"  [{label}] starting fresh with {int(np.sum(np.abs(W_init_np)>W_THRESHOLD))} init edges", flush=True)

    for o in range(start_outer, OUTER):
        t_loop = time.time()
        opt = torch.optim.Adam([W], lr=LR)
        for _ in range(INNER):
            opt.zero_grad()
            W_sq = W * W
            h = torch.trace(torch.linalg.matrix_exp(W_sq)) - d
            XW = X_t @ W
            mse = 0.5 / n * torch.sum((X_t - XW) ** 2)
            l1 = LAMBDA1 * torch.sum(torch.abs(W))
            loss = mse + l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 10.0)
            opt.step()

        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)).item() - d

        if h_val > 0.25 * h_old:
            rho = min(rho * 10, 1e12)
            alpha = alpha + rho * h_val

        n_edges = int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
        print(f"  [{label}] outer {o+1}/{OUTER}: h={h_val:.2e}, rho={rho:.1e}, "
              f"edges={n_edges}, {time.time()-t_loop:.0f}s", flush=True)

        np.save(W_path, W.detach().cpu().numpy())
        with open(ckpt_path, 'w') as f:
            json.dump({"last_outer": o+1, "rho": rho, "alpha": alpha,
                       "h_old": h_val, "last_h": h_val, "last_edges": n_edges}, f)

        h_old = h_val
        if abs(h_val) < 1e-6:
            print(f"  [{label}] converged (h={h_val:.2e})", flush=True)
            break

    W_np = W.detach().cpu().numpy()
    return W_np, int(np.sum(np.abs(W_np) > W_THRESHOLD))

# ============ RUN ============
print(f"\n[1/2] Source NOTEARS (non-BRCA, {X_source.shape[0]} samples, X^T X form)...", flush=True)
torch.cuda.empty_cache()
W_src_causal, n_src = run_source_notears_xtx(X_source, label="source")

# Sanity check on source graph
W_sq = W_src_causal * W_src_causal
max_row = W_sq.sum(axis=1).max()
print(f"  Source causal graph: {n_src} edges, max row sum W*W = {max_row:.3f}", flush=True)
if max_row > 80:
    print(f"  WARNING: row sum too high, may overflow. Scaling...", flush=True)
    W_src_causal = W_src_causal * (np.sqrt(30/max_row))
    n_src = int(np.sum(np.abs(W_src_causal) > W_THRESHOLD))
    print(f"  Rescaled: {n_src} edges", flush=True)

print(f"\n[2/2] Warm-start BRCA from causal source...", flush=True)
torch.cuda.empty_cache()
W_warm_causal, n_warm = run_warm_xtarget(X_brca, W_src_causal, label="warm_causal")

# ============ RESULTS ============
print(f"\n=== RESULTS ===", flush=True)
W_scratch = np.load(os.path.join(CKPT_DIR, "W_scratch_d5000.npy"))
n_scratch = int(np.sum(np.abs(W_scratch) > W_THRESHOLD))

src_set = {(i,j) for i in range(D) for j in range(D) if i!=j and abs(W_src_causal[i,j])>W_THRESHOLD}
scratch_set = {(i,j) for i in range(D) for j in range(D) if i!=j and abs(W_scratch[i,j])>W_THRESHOLD}
warm_set = {(i,j) for i in range(D) for j in range(D) if i!=j and abs(W_warm_causal[i,j])>W_THRESHOLD}

ret_s = len(src_set & scratch_set)
ret_w = len(src_set & warm_set)
ret_s_pct = 100*ret_s/max(1,len(src_set))
ret_w_pct = 100*ret_w/max(1,len(src_set))

print(f"  Source causal edges: {n_src}", flush=True)
print(f"  Scratch edges: {n_scratch}", flush=True)
print(f"  Warm-causal edges: {n_warm}", flush=True)
print(f"  Retention scratch: {ret_s}/{len(src_set)} ({ret_s_pct:.1f}%)", flush=True)
print(f"  Retention warm-causal: {ret_w}/{len(src_set)} ({ret_w_pct:.1f}%)", flush=True)
print(f"  Transfer gain: {ret_w_pct-ret_s_pct:+.1f}pp", flush=True)

result = {
    "d": D, "n_source": int(X_source.shape[0]), "n_target": int(X_brca.shape[0]),
    "source_type": "NOTEARS-causal (non-BRCA, X^T X form)",
    "source_edges": n_src,
    "scratch_edges": n_scratch, "warm_edges": n_warm,
    "retention_scratch": ret_s, "retention_warm": ret_w,
    "retention_scratch_pct": round(ret_s_pct, 2),
    "retention_warm_pct": round(ret_w_pct, 2),
    "transfer_gain_pp": round(ret_w_pct-ret_s_pct, 2),
    "warm_scratch_ratio": round(n_warm/max(1,n_scratch), 3),
    "seed": SEED, "outer": OUTER, "inner": INNER, "lr": LR,
    "lambda1": LAMBDA1, "w_threshold": W_THRESHOLD, "device": str(device),
}

with open(os.path.join(BASE, "results", "d5000_causal_results.json"), 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: results/d5000_causal_results.json", flush=True)
