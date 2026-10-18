#!/usr/bin/env python3
# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(0):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""
run_all.py — One-command reproducible pipeline for the d=5000 NOTEARS
warm-start experiment (JBCB-1496 revision, reviewer point #6).

Reproduces the full high-dimensional feasibility + transfer study:

  Mode 1 (scratch):       NOTEARS from zero init on TCGA BRCA (d=5000).
  Mode 2 (warm-correl):   warm-start from a correlation-threshold source.
  Mode 3 (warm-causal):   warm-start from a NOTEARS-learned CAUSAL source.

Usage:
  python run_all.py                    # all three modes (resumes checkpoints)
  python run_all.py --scratch          # scratch only
  python run_all.py --warm-correl      # correlation warm-start only
  python run_all.py --warm-causal      # causal-source warm-start only

Reproducibility guarantees:
  - Fixed seeds (torch + numpy, seed=42).
  - Deterministic gene selection (top-variance), cached to results/d5000_cache/.
  - Per-outer-loop checkpoints to results/d5000_checkpoints/ (resumable).
  - Two MSE forms, both verified numerically equivalent:
      * X@W form   (used for target data, 1218 samples)
      * X^T X form (used for source data, 9241 samples; cost independent of n)
  - All numbers written to results/d5000_*.json.

Requirements: torch (CUDA), numpy, pandas.
Hardware: NVIDIA RTX 5060 Laptop GPU (8 GB VRAM).
"""

import os, sys, json, time, warnings, argparse
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = DATA_DIR
CKPT_DIR = os.path.join(BASE, "results", "d5000_checkpoints")
CACHE_DIR = os.path.join(BASE, "results", "d5000_cache")

D = 5000
OUTER = 20
INNER = 100
LR = 0.002
LAMBDA1 = 0.01
W_THRESHOLD = 0.3
SEED = 42
SRC_CORR_THRESHOLD = 0.7   # correlation source
SRC_WEIGHT = 0.3           # scale factor (float32 stability)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DATA (CACHED, DETERMINISTIC) =====================
def load_data():
    cache_brca = os.path.join(CACHE_DIR, "X_brca_d5000.npy")
    cache_src = os.path.join(CACHE_DIR, "X_source_d5000.npy")
    cache_wsrc = os.path.join(CACHE_DIR, "W_src_d5000.npy")
    cache_genes = os.path.join(CACHE_DIR, "selected_genes_d5000.txt")

    if all(os.path.exists(p) for p in [cache_brca, cache_src, cache_wsrc]):
        X_brca = np.load(cache_brca)
        X_source = np.load(cache_src)
        W_src_correl = np.load(cache_wsrc)
        print(f"  [cache] BRCA {X_brca.shape}, Source {X_source.shape}")
        return X_brca, X_source, W_src_correl

    tcga_files = [f for f in os.listdir(DATA_DIR)
                  if f.startswith("TCGA_") and "_HiSeqV2.tsv" in f]
    brca_file = [f for f in tcga_files if "BRCA" in f][0]
    source_files = [f for f in tcga_files if "BRCA" not in f]

    df_brca = pd.read_csv(os.path.join(DATA_DIR, brca_file), sep="\t", index_col=0)
    X_arr = df_brca.values.T.astype(np.float32)
    var = np.var(X_arr, axis=0)
    top_idx = np.argsort(var)[-D:]
    selected_genes = df_brca.index[top_idx].tolist()

    X_brca = X_arr[:, top_idx]
    X_brca = (X_brca - X_brca.mean(0)) / (X_brca.std(0) + 1e-8)

    chunks = []
    for fname in source_files:
        fp = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fp, sep="\t", index_col=0)
            avail = [g for g in selected_genes if g in df.index]
            if len(avail) < D * 0.5:
                continue
            sub = df.loc[avail].values.T.astype(np.float32)
            if sub.shape[1] < D:
                full = np.zeros((sub.shape[0], D), dtype=np.float32)
                match = [selected_genes.index(g) for g in avail]
                full[:, match] = sub
                sub = full
            sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)
            chunks.append(sub)
        except Exception:
            continue
    X_source = np.vstack(chunks)

    corr = np.corrcoef(X_source.T)
    np.fill_diagonal(corr, 0)
    W_src_correl = (np.abs(corr) > SRC_CORR_THRESHOLD).astype(np.float32) * SRC_WEIGHT

    np.save(cache_brca, X_brca)
    np.save(cache_src, X_source)
    np.save(cache_wsrc, W_src_correl)
    with open(cache_genes, "w") as f:
        f.write("\n".join(selected_genes))
    print(f"  [built] BRCA {X_brca.shape}, Source {X_source.shape}")
    return X_brca, X_source, W_src_correl

# ===================== NOTEARS CORE =====================
def _h(W, d):
    return torch.trace(torch.linalg.matrix_exp(W * W)) - d

def _load_or_init(label, d, W_init_np, device):
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{label}_d{d}.json")
    W_path = os.path.join(CKPT_DIR, f"W_{label}_d{d}.npy")
    if os.path.exists(ckpt_path) and os.path.exists(W_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        W = torch.tensor(np.load(W_path), dtype=torch.float32,
                         device=device, requires_grad=True)
        return ck.get("last_outer", 0), ck.get("rho", 1.0), ck.get("alpha", 0.0), \
               ck.get("h_old", float("inf")), W
    W = (torch.tensor(W_init_np, dtype=torch.float32, device=device, requires_grad=True)
         if W_init_np is not None
         else torch.zeros(d, d, dtype=torch.float32, device=device, requires_grad=True))
    return 0, 1.0, 0.0, float("inf"), W

def _save_ckpt(label, d, o, rho, alpha, h_val, n_edges, W):
    np.save(os.path.join(CKPT_DIR, f"W_{label}_d{d}.npy"), W.detach().cpu().numpy())
    with open(os.path.join(CKPT_DIR, f"ckpt_{label}_d{d}.json"), "w") as f:
        json.dump({"last_outer": o + 1, "rho": rho, "alpha": alpha,
                   "h_old": h_val, "last_h": h_val, "last_edges": n_edges}, f)

def run_notears(X_np, W_init_np=None, label="scratch", use_xtx=False):
    """NOTEARS with per-outer-loop checkpointing.
    use_xtx=True uses the X^T X closed form (source data, many samples).
    """
    n, d = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

    tr_S = None
    if use_xtx:
        S = (X_t.T @ X_t) / n
        tr_S = torch.trace(S).item()

    start_outer, rho, alpha, h_old, W = _load_or_init(label, d, W_init_np, device)
    if start_outer >= OUTER:
        print(f"  [{label}] already complete", flush=True)
        return W.detach().cpu().numpy(), int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
    print(f"  [{label}] starting from outer {start_outer}", flush=True)

    for o in range(start_outer, OUTER):
        t_loop = time.time()
        opt = torch.optim.Adam([W], lr=LR)
        for _ in range(INNER):
            opt.zero_grad()
            h = _h(W, d)
            if use_xtx:
                SW = S @ W
                mse = 0.5 * (tr_S - 2.0 * torch.trace(SW) + torch.trace(W.T @ SW))
            else:
                XW = X_t @ W
                mse = 0.5 / n * torch.sum((X_t - XW) ** 2)
            l1 = LAMBDA1 * torch.sum(torch.abs(W))
            loss = mse + l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 10.0)
            opt.step()

        with torch.no_grad():
            h_val = _h(W, d).item()

        if h_val > 0.25 * h_old:
            rho = min(rho * 10, 1e12)
            alpha = alpha + rho * h_val

        n_edges = int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
        print(f"  [{label}] outer {o+1}/{OUTER}: h={h_val:.2e}, rho={rho:.1e}, "
              f"edges={n_edges}, {time.time()-t_loop:.0f}s", flush=True)

        _save_ckpt(label, d, o, rho, alpha, h_val, n_edges, W)
        h_old = h_val
        if abs(h_val) < 1e-6:
            print(f"  [{label}] converged (h={h_val:.2e})", flush=True)
            break

    W_np = W.detach().cpu().numpy()
    return W_np, int(np.sum(np.abs(W_np) > W_THRESHOLD))

# ===================== METRICS =====================
def retention(W_src, W_other, threshold=W_THRESHOLD):
    src = {(i, j) for i in range(D) for j in range(D)
           if i != j and abs(W_src[i, j]) > threshold}
    oth = {(i, j) for i in range(D) for j in range(D)
           if i != j and abs(W_other[i, j]) > threshold}
    return len(src & oth), len(src)

def edge_count(W, threshold=W_THRESHOLD):
    return int(np.sum(np.abs(W) > threshold))

# ===================== MAIN =====================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--warm-correl", action="store_true")
    p.add_argument("--warm-causal", action="store_true")
    args = p.parse_args()
    run_all = not (args.scratch or args.warm_correl or args.warm_causal)

    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory//1024**3} GB)", flush=True)

    X_brca, X_source, W_src_correl = load_data()

    results = {"d": D, "n_source": int(X_source.shape[0]),
               "n_target": int(X_brca.shape[0]), "seed": SEED,
               "outer": OUTER, "inner": INNER, "lr": LR, "lambda1": LAMBDA1,
               "w_threshold": W_THRESHOLD, "device": str(device)}

    # --- Scratch ---
    if run_all or args.scratch:
        print("\n[scratch] NOTEARS from zero init...", flush=True)
        torch.cuda.empty_cache()
        W_scratch, _ = run_notears(X_brca, label="scratch")
        results["scratch_edges"] = edge_count(W_scratch)
    else:
        W_scratch = np.load(os.path.join(CKPT_DIR, "W_scratch_d5000.npy"))
        results["scratch_edges"] = edge_count(W_scratch)

    # --- Warm-start from CORRELATION source ---
    if run_all or args.warm_correl:
        print("\n[warm-correl] warm-start from correlation source...", flush=True)
        torch.cuda.empty_cache()
        W_wc, _ = run_notears(X_brca, W_init_np=W_src_correl, label="warm")
        results["warm_correl_edges"] = edge_count(W_wc)
        r, total = retention(W_src_correl, W_wc)
        rs, _ = retention(W_src_correl, W_scratch)
        results["correl_retention_scratch"] = rs
        results["correl_retention_warm"] = r
        results["correl_retention_total"] = total
        results["correl_source_edges"] = edge_count(W_src_correl, threshold=0.0)

    # --- Warm-start from CAUSAL source ---
    if run_all or args.warm_causal:
        print("\n[source] NOTEARS causal source graph (non-BRCA)...", flush=True)
        torch.cuda.empty_cache()
        W_src_causal, n_src = run_notears(X_source, label="source", use_xtx=True)

        # float32 stability check
        max_row = (W_src_causal * W_src_causal).sum(axis=1).max()
        if max_row > 80:
            W_src_causal = W_src_causal * np.sqrt(30 / max_row)
            n_src = edge_count(W_src_causal)
            print(f"  [source] rescaled for stability: {n_src} edges", flush=True)

        print("\n[warm-causal] warm-start from causal source...", flush=True)
        torch.cuda.empty_cache()
        W_wc, _ = run_notears(X_brca, W_init_np=W_src_causal, label="warm_causal")

        results["causal_source_edges"] = n_src
        results["warm_causal_edges"] = edge_count(W_wc)
        r, total = retention(W_src_causal, W_wc)
        rs, _ = retention(W_src_causal, W_scratch)
        results["causal_retention_scratch"] = rs
        results["causal_retention_warm"] = r
        results["causal_retention_total"] = total

    out = os.path.join(BASE, "results", "d5000_final_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nSaved: {out}", flush=True)

if __name__ == "__main__":
    main()
