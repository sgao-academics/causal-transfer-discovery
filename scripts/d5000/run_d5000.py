#!/usr/bin/env python3
"""
run_d5000.py — High-dimensional (d=5000) NOTEARS warm-start extension.

This script reproduces the "High-Dimensional Extension" subsection of the
revised manuscript (reviewer point #6). It runs three modes:

  Mode 1 (scratch):       NOTEARS from zero init on TCGA BRCA (d=5000).
  Mode 2 (warm-correl):   warm-start from a correlation-threshold source.
  Mode 3 (warm-causal):   warm-start from a NOTEARS-learned CAUSAL source.

Usage:
    python run_d5000.py --scratch          # scratch only
    python run_d5000.py --warm-correl      # correlation warm-start only
    python run_d5000.py --warm-causal      # causal-source warm-start only
    python run_d5000.py                    # all three (resumes checkpoints)

Requirements:
    - TCGA data (BRCA + 32 non-breast cancer types), set via TCGA_DATA_DIR
      or place under <repo>/data/
    - torch (CUDA), numpy, pandas
    - NVIDIA GPU with >= 8 GB VRAM (used: RTX 5060 Laptop, 8 GB)

Reproducibility:
    - Fixed seed (42).
    - Deterministic gene selection (top-variance), cached to results/d5000_cache/.
    - Per-outer-loop checkpoints to results/d5000_checkpoints/ (resumable).
    - Two MSE forms, numerically equivalent:
        X@W form   (target data, 1218 samples)
        X^T X form (source data, 9241 samples; O(d^3), independent of n)
"""

import os, sys, json, time, warnings, argparse
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ===================== PATHS =====================
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("TCGA_DATA_DIR", os.path.join(REPO, "data"))
CKPT_DIR = os.path.join(REPO, "results", "d5000_checkpoints")
CACHE_DIR = os.path.join(REPO, "results", "d5000_cache")

# ===================== CONFIG =====================
D = 5000
OUTER = 20
INNER = 100
LR = 0.002
LAMBDA1 = 0.01
W_THRESHOLD = 0.3
SEED = 42
SCRATCH_SEEDS = [42, 123, 456]
SCRATCH_SIGMA = 0.05  # reduced from 0.1: at d=5000 sigma=0.1 gives W^2 row-sum ~50 > float32 matrix_exp bound (~30)
SRC_CORR_THRESHOLD = 0.7
SRC_WEIGHT = 0.3

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load (or build + cache) the deterministic d=5000 gene selection and data."""
    cache_brca = os.path.join(CACHE_DIR, "X_brca_d5000.npy")
    cache_src = os.path.join(CACHE_DIR, "X_source_d5000.npy")
    cache_wsrc = os.path.join(CACHE_DIR, "W_src_d5000.npy")
    cache_genes = os.path.join(CACHE_DIR, "selected_genes_d5000.txt")

    if all(os.path.exists(p) for p in [cache_brca, cache_src, cache_wsrc]):
        return (np.load(cache_brca), np.load(cache_src), np.load(cache_wsrc))

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
    print(f"  [built cache] BRCA {X_brca.shape}, Source {X_source.shape}")
    return X_brca, X_source, W_src_correl


def _h(W, d):
    return torch.trace(torch.linalg.matrix_exp(W * W)) - d


def _load_or_init(label, d, W_init_np):
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
    """NOTEARS with per-outer-loop checkpointing."""
    n, d = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

    tr_S = None
    if use_xtx:
        S = (X_t.T @ X_t) / n
        tr_S = torch.trace(S).item()

    start_outer, rho, alpha, h_old, W = _load_or_init(label, d, W_init_np)
    if start_outer >= OUTER:
        print(f"  [{label}] already complete")
        return W.detach().cpu().numpy(), int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
    print(f"  [{label}] starting from outer {start_outer}")

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
              f"edges={n_edges}, {time.time()-t_loop:.0f}s")

        _save_ckpt(label, d, o, rho, alpha, h_val, n_edges, W)
        h_old = h_val
        if abs(h_val) < 1e-6:
            print(f"  [{label}] converged (h={h_val:.2e})")
            break

    W_np = W.detach().cpu().numpy()
    return W_np, int(np.sum(np.abs(W_np) > W_THRESHOLD))


def retention(W_src, W_other, threshold=W_THRESHOLD):
    src = {(i, j) for i in range(D) for j in range(D)
           if i != j and abs(W_src[i, j]) > threshold}
    oth = {(i, j) for i in range(D) for j in range(D)
           if i != j and abs(W_other[i, j]) > threshold}
    return len(src & oth), len(src)


def edge_count(W, threshold=W_THRESHOLD):
    return int(np.sum(np.abs(W) > threshold))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--warm-correl", action="store_true")
    p.add_argument("--warm-causal", action="store_true")
    args = p.parse_args()
    run_all = not (args.scratch or args.warm_correl or args.warm_causal)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory//1024**3} GB)")

    X_brca, X_source, W_src_correl = load_data()

    results = {"d": D, "n_source": int(X_source.shape[0]),
               "n_target": int(X_brca.shape[0]), "seed": SEED,
               "outer": OUTER, "inner": INNER, "lr": LR, "lambda1": LAMBDA1,
               "w_threshold": W_THRESHOLD, "device": str(device)}

    if run_all or args.scratch:
        print("\n[scratch] NOTEARS from random init (3 seeds)...")
        torch.cuda.empty_cache()
        scratch_edges = []
        scratch_Ws = []
        for seed in SCRATCH_SEEDS:
            np.random.seed(seed)
            W_init = (np.random.randn(D * D).astype(np.float32) * SCRATCH_SIGMA).reshape(D, D)
            W_s, _ = run_notears(X_brca, W_init_np=W_init, label=f"scratch_s{seed}")
            scratch_edges.append(edge_count(W_s))
            scratch_Ws.append(W_s)
        results["scratch_edges_mean"] = round(float(np.mean(scratch_edges)), 1)
        results["scratch_edges_std"] = round(float(np.std(scratch_edges)), 1)
        W_scratch = scratch_Ws[0]
    else:
        scratch_edges = []
        scratch_Ws = []
        for seed in SCRATCH_SEEDS:
            W_s = np.load(os.path.join(CKPT_DIR, f"W_scratch_s{seed}_d{D}.npy"))
            scratch_edges.append(edge_count(W_s))
            scratch_Ws.append(W_s)
        results["scratch_edges_mean"] = round(float(np.mean(scratch_edges)), 1)
        results["scratch_edges_std"] = round(float(np.std(scratch_edges)), 1)
        W_scratch = scratch_Ws[0]

    if run_all or args.warm_correl:
        print("\n[warm-correl] warm-start from correlation source...")
        torch.cuda.empty_cache()
        W_wc, _ = run_notears(X_brca, W_init_np=W_src_correl, label="warm")
        results["warm_correl_edges"] = edge_count(W_wc)
        r, total = retention(W_src_correl, W_wc)
        rs, _ = retention(W_src_correl, W_scratch)
        results["correl_retention_scratch"] = rs
        results["correl_retention_warm"] = r
        results["correl_retention_total"] = total
        results["correl_source_edges"] = edge_count(W_src_correl, threshold=0.0)

    if run_all or args.warm_causal:
        print("\n[source] NOTEARS causal source graph (non-BRCA)...")
        torch.cuda.empty_cache()
        W_src_causal, n_src = run_notears(X_source, label="source", use_xtx=True)

        max_row = (W_src_causal * W_src_causal).sum(axis=1).max()
        if max_row > 80:
            W_src_causal = W_src_causal * np.sqrt(30 / max_row)
            n_src = edge_count(W_src_causal)
            print(f"  [source] rescaled for stability: {n_src} edges")

        print("\n[warm-causal] warm-start from causal source...")
        torch.cuda.empty_cache()
        W_wc, _ = run_notears(X_brca, W_init_np=W_src_causal, label="warm_causal")

        results["causal_source_edges"] = n_src
        results["warm_causal_edges"] = edge_count(W_wc)
        r, total = retention(W_src_causal, W_wc)
        rs_list = [retention(W_src_causal, W_s)[0] for W_s in scratch_Ws]
        results["causal_retention_scratch_mean"] = round(float(np.mean(rs_list)), 1)
        results["causal_retention_scratch_pct_mean"] = round(100 * float(np.mean(rs_list)) / total, 1)
        results["causal_retention_warm"] = r
        results["causal_retention_total"] = total

    out = os.path.join(REPO, "results", "d5000_final_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
