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
run_d5000_multiseed.py -- d=5000 NOTEARS warm-start transfer, multi-seed.

Purpose (JBCB-1496 R1, reviewer point 2):
  Report mean +/- std across seeds for the genome-scale (d=5000) results,
  matching the format of the main tables.

Design note (IMPORTANT):
  In the d<=889 benchmark (scripts/experiments/run_benchmark_gpu.py), the
  nonzero warm-start variance comes from the SOURCE graph differing between
  independent runs: the source NOTEARS starts from a random draw
  (np.random.randn * 0.1), so each run yields a different source graph W_S,
  and warm-start operates from that seed-specific graph.

  The previous d=5000 script (run_all.py) initialised the source from
  torch.zeros, i.e. deterministic, which is why only ONE warm-start value
  exists. To reproduce the d<=889 protocol at d=5000, each seed here gets:
      (1) its own randomly-initialised source graph  W_S^(s)
      (2) a warm-start run initialised at W_S^(s)
  Scratch results are reused from run_scratch_multiseed.py (same protocol).

Usage:
  python run_d5000_multiseed.py                 # all seeds in SEEDS
  python run_d5000_multiseed.py --seeds 42      # one seed
  python run_d5000_multiseed.py --smoke         # tiny smoke test (d=500)

Outputs:
  results/d5000_multiseed_transfer.json
  checkpoints in results/d5000_checkpoints/ (label prefix: ms_)
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
BASE = PKG_ROOT
CACHE_DIR = os.path.join(BASE, "results", "d5000_cache")
CKPT_DIR = os.path.join(BASE, "results", "d5000_checkpoints")
OUT_JSON = os.path.join(BASE, "results", "d5000_multiseed_transfer.json")

D = 5000
SEEDS = [42, 123, 456]
OUTER = 20
INNER = 100
LR = 0.002
LAMBDA1 = 0.01
W_THRESHOLD = 0.3
INIT_SIGMA = 0.05      # matches the d=5000 scratch init (float32 stability)

os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(msg, flush=True)


# ===================== NOTEARS CORE =====================
def _h(W, d):
    return torch.trace(torch.linalg.matrix_exp(W * W)) - d


def run_notears(X_np, W_init_np=None, label="run", d_override=None,
                outer=OUTER, inner=INNER, lr=LR, use_xtx=False):
    """NOTEARS with augmented Lagrangian + per-outer checkpointing.

    use_xtx=True uses the X^T X closed form (source data, many samples), so the
    cost is independent of n.
    """
    n, d = X_np.shape
    if d_override:
        d = d_override
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

    tr_S = None
    if use_xtx:
        S = (X_t.T @ X_t) / n
        tr_S = torch.trace(S).item()

    ckpt_path = os.path.join(CKPT_DIR, "ckpt_%s_d%d.json" % (label, d))
    W_path = os.path.join(CKPT_DIR, "W_%s_d%d.npy" % (label, d))

    if os.path.exists(ckpt_path) and os.path.exists(W_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        start_outer = ck.get("last_outer", 0)
        rho = ck.get("rho", 1.0)
        alpha = ck.get("alpha", 0.0)
        h_old = ck.get("h_old", float("inf"))
        W = torch.tensor(np.load(W_path), dtype=torch.float32,
                         device=device, requires_grad=True)
        log("  [%s] resume outer %d/%d" % (label, start_outer, outer))
    else:
        start_outer, rho, alpha, h_old = 0, 1.0, 0.0, float("inf")
        if W_init_np is not None:
            W = torch.tensor(np.asarray(W_init_np, dtype=np.float32),
                             dtype=torch.float32, device=device, requires_grad=True)
        else:
            W = torch.zeros(d, d, dtype=torch.float32, device=device, requires_grad=True)

    if start_outer >= outer:
        log("  [%s] already complete" % label)
        return W.detach().cpu().numpy()

    for o in range(start_outer, outer):
        t_loop = time.time()
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(inner):
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
        log("  [%s] outer %d/%d h=%.2e rho=%.1e edges=%d %.0fs"
            % (label, o + 1, outer, h_val, rho, n_edges, time.time() - t_loop))

        np.save(W_path, W.detach().cpu().numpy())
        with open(ckpt_path, "w") as f:
            json.dump({"last_outer": o + 1, "rho": rho, "alpha": alpha,
                       "h_old": h_val, "last_h": h_val, "last_edges": n_edges}, f)
        h_old = h_val
        if abs(h_val) < 1e-6:
            log("  [%s] converged" % label)
            break

    return W.detach().cpu().numpy()


def edge_set(W, d, tau=W_THRESHOLD):
    a = np.abs(W)
    mask = a > tau
    np.fill_diagonal(mask, False)
    idx = np.argwhere(mask)
    return set((int(i), int(j)) for i, j in idx)


# ===================== MAIN =====================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    d_val = D
    outer = OUTER

    log("Device: %s" % device)
    if torch.cuda.is_available():
        log("GPU: %s (%d GB)" % (torch.cuda.get_device_name(0),
                                 torch.cuda.get_device_properties(0).total_memory // 1024 ** 3))

    # ---- data (cached) ----
    X_brca = np.load(os.path.join(CACHE_DIR, "X_brca_d5000.npy")).astype(np.float32)
    X_source = np.load(os.path.join(CACHE_DIR, "X_source_d5000.npy")).astype(np.float32)
    log("BRCA %s  Source %s" % (X_brca.shape, X_source.shape))

    if args.smoke:
        # tiny smoke test: truncate dimension AND shorten the schedule
        d_val = 500
        outer = 3
        X_brca = X_brca[:, :d_val]
        X_source = X_source[:, :d_val]
        log("SMOKE MODE: d=%d outer=%d" % (d_val, outer))

    out = {"d": d_val, "seeds": seeds, "outer": outer, "inner": INNER, "lr": LR,
           "lambda1": LAMBDA1, "w_threshold": W_THRESHOLD,
           "init_sigma": INIT_SIGMA, "device": str(device), "per_seed": {}}

    warm_rets = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        log("\n===== seed %d =====" % seed)

        # (1) seed-specific source graph, random init (mirrors d<=889 protocol)
        W0 = (np.random.randn(d_val, d_val) * INIT_SIGMA).astype(np.float32)
        log("[source] NOTEARS on non-BRCA, random init N(0, %.2f^2)" % INIT_SIGMA)
        W_src = run_notears(X_source, W_init_np=W0, label="ms_source_s%d" % seed,
                            d_override=d_val, outer=outer, use_xtx=True)

        # float32 stability guard for the warm start
        max_row = float((W_src * W_src).sum(axis=1).max())
        if max_row > 30:
            W_src = (W_src * np.sqrt(30.0 / max_row)).astype(np.float32)
            log("  [source] rescaled (row-sum %.1f -> 30)" % max_row)

        src_edges = edge_set(W_src, d_val)
        log("  [source] %d edges @tau=%.1f" % (len(src_edges), W_THRESHOLD))

        # (2) warm-start from that source graph
        log("[warm] NOTEARS on BRCA from source graph")
        W_warm = run_notears(X_brca, W_init_np=W_src, label="ms_warm_s%d" % seed,
                             d_override=d_val, outer=outer)

        warm_edges = edge_set(W_warm, d_val)
        kept = len(src_edges & warm_edges)
        ret = 100.0 * kept / max(1, len(src_edges))
        warm_rets.append(ret)
        log("  [warm] %d edges, retained %d/%d = %.2f%%"
            % (len(warm_edges), kept, len(src_edges), ret))

        out["per_seed"][str(seed)] = {
            "source_edges": len(src_edges),
            "warm_edges": len(warm_edges),
            "retained": kept,
            "warm_retention_pct": round(ret, 2),
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)

    out["warm_retention_mean"] = round(float(np.mean(warm_rets)), 2)
    out["warm_retention_std"] = round(float(np.std(warm_rets)), 2)
    out["n_seeds"] = len(warm_rets)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    log("\n===== SUMMARY =====")
    for s, v in out["per_seed"].items():
        log("  seed %s: %.2f%% (%d/%d)" % (s, v["warm_retention_pct"],
                                           v["retained"], v["source_edges"]))
    log("  warm retention: %.2f +/- %.2f%% (n=%d)"
        % (out["warm_retention_mean"], out["warm_retention_std"], out["n_seeds"]))
    log("Saved: %s" % OUT_JSON)


if __name__ == "__main__":
    main()
