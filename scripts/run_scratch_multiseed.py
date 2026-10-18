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
d=5000 scratch NOTEARS, RANDOM init (N(0, 0.1^2)), 3 seeds (42/123/456).
Aligns the scratch baseline with the main experiment (run_benchmark_gpu.py
uses np.random.randn(d*d)*0.1), replacing the old deterministic zero-init.

The causal source graph (W_source_d5000.npy, 634 edges) and warm-causal
result remain fixed and deterministic (zero-init source + warm from source),
so they are NOT re-run here.

Checkpoints: results/d5000_checkpoints/ckpt_scratch_s{seed}_d5000.json + W_*.npy
Per-outer-loop resume; crash-safe.
"""
import os, sys, json, time, warnings, argparse
import numpy as np
import torch
warnings.filterwarnings('ignore')

BASE = PKG_ROOT
CACHE_DIR = os.path.join(BASE, "results", "d5000_cache")
CKPT_DIR = os.path.join(BASE, "results", "d5000_checkpoints")

D = 5000
OUTER = 20
INNER = 100
LR = 0.002
LAMBDA1 = 0.01
W_THRESHOLD = 0.3
SEEDS = [42, 123, 456]


def _h(W):
    return torch.trace(torch.linalg.matrix_exp(W * W)) - D


def run_scratch(X_brca, seed, device, dry_run=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    label = f"scratch_s{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{label}_d{D}.json")
    W_path = os.path.join(CKPT_DIR, f"W_{label}_d{D}.npy")
    n, d = X_brca.shape
    X_t = torch.tensor(X_brca, dtype=torch.float32, device=device)

    start_outer = 0
    rho, alpha = 1.0, 0.0
    h_old = float('inf')

    if os.path.exists(ckpt_path) and os.path.exists(W_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        start_outer = ck.get("last_outer", 0)
        rho = ck.get("rho", 1.0)
        alpha = ck.get("alpha", 0.0)
        h_old = ck.get("h_old", float('inf'))
        W = torch.tensor(np.load(W_path), dtype=torch.float32,
                         device=device, requires_grad=True)
        print(f"[seed {seed}] resumed from outer {start_outer}/{OUTER}", flush=True)
    else:
        # random init. Main experiment uses sigma=0.1 (d<=889: row-sum of W^2 ~ 8.9,
        # safe). At d=5000 sigma=0.1 gives row-sum ~50 > float32 matrix_exp safe limit
        # (~30), so we scale sigma to 0.05 -> row-sum ~12.5, safe.
        np.random.seed(seed)
        sigma = 0.05
        W_np = (np.random.randn(d * d).astype(np.float32) * sigma).reshape(d, d)
        rs = float((W_np * W_np).sum(axis=1).max())
        print(f"[seed {seed}] random init N(0, {sigma}^2), row-sum(W^2)={rs:.1f}", flush=True)
        W = torch.tensor(W_np, dtype=torch.float32, device=device, requires_grad=True)

    if start_outer >= OUTER:
        return W.detach().cpu().numpy()

    max_outer = 1 if dry_run else OUTER
    for o in range(start_outer, max_outer):
        t_loop = time.time()
        opt = torch.optim.Adam([W], lr=LR)
        for _ in range(INNER):
            opt.zero_grad()
            h = _h(W)
            XW = X_t @ W
            mse = 0.5 / n * torch.sum((X_t - XW) ** 2)
            l1 = LAMBDA1 * torch.sum(torch.abs(W))
            loss = mse + l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 10.0)
            opt.step()

        with torch.no_grad():
            h_val = _h(W).item()

        if h_val > 0.25 * h_old:
            rho = min(rho * 10, 1e12)
            alpha = alpha + rho * h_val

        n_edges = int(torch.sum(torch.abs(W) > W_THRESHOLD).item())
        dt = time.time() - t_loop

        if not dry_run:
            np.save(W_path, W.detach().cpu().numpy())
            with open(ckpt_path, 'w') as f:
                json.dump({"last_outer": o + 1, "rho": rho, "alpha": alpha,
                           "h_old": h_val, "last_h": h_val, "last_edges": n_edges}, f)

        h_old = h_val
        print(f"[seed {seed}] outer {o+1}/{max_outer}: h={h_val:.2e}, "
              f"edges={n_edges}, {dt:.0f}s", flush=True)

        if abs(h_val) < 1e-6:
            print(f"[seed {seed}] converged (h={h_val:.2e})", flush=True)
            break

    return W.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=str, default="42,123,456")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory//1024**3} GB)",
              flush=True)

    X_brca = np.load(os.path.join(CACHE_DIR, "X_brca_d5000.npy"))
    print(f"Target: {X_brca.shape}", flush=True)

    W_src = np.load(os.path.join(CKPT_DIR, "W_source_d5000.npy"))
    src_set = {(i, j) for i in range(D) for j in range(D)
               if i != j and abs(W_src[i, j]) > W_THRESHOLD}
    print(f"Source edges (fixed causal source): {len(src_set)}", flush=True)

    results = {"d": D, "seeds": seeds,
               "init": "N(0, 0.05^2) random (sigma scaled for float32 stability at d=5000)",
               "source_edges": len(src_set)}

    for seed in seeds:
        print(f"\n=== scratch seed {seed} ===", flush=True)
        torch.cuda.empty_cache()
        try:
            W = run_scratch(X_brca, seed, device, dry_run=args.dry_run)
            if args.dry_run:
                print(f"[seed {seed}] dry-run OK", flush=True)
                continue
            n_edges = int(np.sum(np.abs(W) > W_THRESHOLD))
            scr_set = {(i, j) for i in range(D) for j in range(D)
                       if i != j and abs(W[i, j]) > W_THRESHOLD}
            ret = len(src_set & scr_set)
            ret_pct = 100 * ret / len(src_set)
            results[f"seed{seed}_edges"] = n_edges
            results[f"seed{seed}_retention"] = ret
            results[f"seed{seed}_retention_pct"] = round(ret_pct, 1)
            print(f"[seed {seed}] edges={n_edges}, "
                  f"retention={ret}/{len(src_set)} ({ret_pct:.1f}%)", flush=True)
        except Exception as e:
            print(f"[seed {seed}] ERROR: {e}", flush=True)
            torch.cuda.empty_cache()

    if not args.dry_run:
        edges = [results[f"seed{s}_edges"] for s in seeds if f"seed{s}_edges" in results]
        rets = [results[f"seed{s}_retention_pct"] for s in seeds
                if f"seed{s}_retention_pct" in results]
        if edges:
            results["scratch_edges_mean"] = round(float(np.mean(edges)), 1)
            results["scratch_edges_std"] = round(float(np.std(edges)), 1)
            results["scratch_retention_mean"] = round(float(np.mean(rets)), 1)
            results["scratch_retention_std"] = round(float(np.std(rets)), 1)
        out = os.path.join(BASE, "results", "d5000_scratch_multiseed.json")
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out}", flush=True)
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
