#!/usr/bin/env python3
# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""Real-data baselines that a referee will demand for a transfer-learning paper.

Three alternative explanations for "warm-start beats scratch" are tested on the
same TCGA -> BRCA task, seeds and solver as Table 1:

  pool      concatenate source (9,241 non-breast) and target (1,218 BRCA)
            samples, fit ONE graph, score it against the same source graph.
  tgtcorr   |corr| skeleton of the BRCA matrix itself (top-m pairs, sign
            oriented, cycles broken) -- needs no second dataset.
  tgtreg    ridge coefficients of each BRCA column on the others, top-m
            entries -- the strongest target-side structure prior available
            without an external cohort.
  degmatch  random DAG with the source graph's edge count and out-degree
            profile, weights resampled -- controls for "any sparse hint".

WHY RETENTION ALONE IS NOT THE VERDICT
--------------------------------------
Pooling is scored on the same retention scale, but that number is high by
construction: pooling has seen 88% of the samples used to learn the source
graph, so it is expected to reproduce it.  The question a referee actually
asks is "does pooling produce a graph that describes the TARGET better".

Because TCGA has no ground truth, two target-side criteria are reported:

  bic_full     Gaussian BIC of the thresholded graph on the BRCA matrix
               (fit + complexity penalty, so a dense graph cannot win for free)
  nrmse_hold   normalised residual on a held-out 20% of BRCA, after
               thresholding -- out-of-sample fit

A method can therefore lose on retention and still be the honest winner; the
record keeps all three numbers so the table can be read either way.

Usage
-----
  python run_real_baselines.py --dims 454,889 --seeds 42,123,456
  python run_real_baselines.py --arms pool,tgtcorr
"""
import os
import sys
import json
import time
import argparse

import numpy as np

ROOT = PKG_ROOT
EXP_DIR = os.path.join(ROOT, "scripts", "experiments")
sys.path.insert(0, EXP_DIR)

from run_benchmark_gpu import notears, evaluate  # noqa: E402
from load_tcga_safe import load_data  # noqa: E402
from run_synthetic_baselines import (_break_cycles, target_corr_init,  # noqa: E402
                                     target_ridge_init, degree_matched_init)

CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "real_baselines.json")
TAU = 0.3


def threshold(W, tau=TAU):
    M = (np.abs(W) > tau).astype(np.float64)
    np.fill_diagonal(M, 0.0)
    return W * M


def bic(X, W, tau=TAU):
    """Gaussian BIC of the thresholded graph: n*d*log(RSS/(n*d)) + k*log(n*d)."""
    Wt = threshold(W, tau)
    R = X - X @ Wt
    rss = float((R ** 2).sum())
    n, d = X.shape
    k = int((Wt != 0).sum())
    if rss <= 0:
        return float("inf")
    return n * d * np.log(rss / (n * d)) + k * np.log(n * d)


def nrmse(X, W, tau=TAU):
    Wt = threshold(W, tau)
    R = X - X @ Wt
    return float((R ** 2).mean() / (X ** 2).mean())


def score(W_src, W, X_tgt, X_hold):
    ev = evaluate(W_src, W)
    ev["bic_full"] = round(bic(X_tgt, W), 1)
    ev["nrmse_hold"] = round(nrmse(X_hold, W), 5)
    ev["nrmse_full"] = round(nrmse(X_tgt, W), 5)
    ev["edges_thresh"] = int((threshold(W) != 0).sum())
    return ev


def one_cell(d_req, seed, arms, st):
    X_tcga, X_brca, d_actual, genes = load_data(d_req)
    rng = np.random.default_rng(seed)
    # held-out 20% of the target, fixed per seed so all arms see the same rows
    n = X_brca.shape[0]
    idx = rng.permutation(n)
    n_hold = max(50, int(0.2 * n))
    hold = idx[:n_hold]
    X_hold = X_brca[hold]

    sp = os.path.join(CACHE, "src_d%d_s%d.npy" % (d_actual, seed))
    W_src = np.load(sp)[:d_actual, :d_actual]
    m_prior = int((np.abs(W_src) > TAU).sum())

    tag = "d%d_s%d" % (d_actual, seed)
    print("[%s] d=%d n_src=%d n_tgt=%d src_edges=%d n_hold=%d"
          % (tag, d_actual, X_tcga.shape[0], n, m_prior, n_hold), flush=True)

    # ---- put the archived arms on the same three scales --------------------
    # Without this the new baselines could not be compared with warm/scratch:
    # the checkpoint stores retention but not BIC or held-out residual.
    for arm, fname in (("scratch", "scratch_d%d_s%d.npy"),
                       ("warm", "warm_d%d_s%d.npy"),
                       ("shuffled", "shuf_d%d_s%d.npy")):
        key = "%s_%s" % (tag, arm)
        if key in st:
            continue
        fp = os.path.join(CACHE, fname % (d_actual, seed))
        if not os.path.exists(fp):
            print("   %-9s archive missing (%s)" % (arm, os.path.basename(fp)), flush=True)
            continue
        W = np.load(fp)[:d_actual, :d_actual]
        ev = score(W_src, W, X_brca, X_hold)
        ev["from_archive"] = os.path.basename(fp)
        st[key] = ev
        _save(st)
        print("   %-9s ret=%5.1f%% (%d/%d) r=%.3f BIC=%.0f nrmse_hold=%.4f edges=%d"
              % (arm, ev["retention_pct"], ev["overlap"], ev["src_edges"],
                 ev["pearson_r"], ev["bic_full"], ev["nrmse_hold"],
                 ev["edges_thresh"]), flush=True)

    for arm in arms:
        key = "%s_%s" % (tag, arm)
        if key in st:
            print("   %-9s done (ret=%.1f%%)" % (arm, st[key]["retention_pct"]), flush=True)
            continue
        t0 = time.time()
        if arm == "pool":
            X_pool = np.vstack([X_tcga, X_brca])
            X_pool = (X_pool - X_pool.mean(0)) / (X_pool.std(0) + 1e-12)
            W, h, dt, it = notears(X_pool, lam=0.01, max_outer=35, verbose=False)
            ev = score(W_src, W, X_brca, X_hold)
            ev["retention_is_tautological"] = True
            ev["n_pool"] = int(X_pool.shape[0])
            np.save(os.path.join(CACHE, "pool_d%d_s%d.npy" % (d_actual, seed)), W)
        else:
            ev = {}
            np.random.seed(seed)
            if arm == "tgtcorr":
                W0 = target_corr_init(X_brca, m_prior, rng)
            elif arm == "tgtreg":
                W0 = target_ridge_init(X_brca, m_prior, rng)
            elif arm == "degmatch":
                W0 = degree_matched_init(W_src, rng)
            else:
                raise ValueError(arm)
            ev["init_edges"] = int((np.abs(W0) > 0).sum())
            ev["init_is_acyclic"] = bool(_is_dag(W0))
            W, h, dt, it = notears(X_brca, lam=0.01, max_outer=35, init_W=W0)
            ev.update(score(W_src, W, X_brca, X_hold))
            np.save(os.path.join(CACHE, "%s_d%d_s%d.npy" % (arm, d_actual, seed)), W)
        ev.update({"h": float(h), "time": float(dt), "outer": int(it)})
        st[key] = ev
        _save(st)
        print("   %-9s ret=%5.1f%% (%d/%d) r=%.3f BIC=%.0f nrmse_hold=%.4f "
              "edges=%d %.0fs"
              % (arm, ev["retention_pct"], ev["overlap"], ev["src_edges"],
                 ev["pearson_r"], ev["bic_full"], ev["nrmse_hold"],
                 ev["edges_thresh"], ev["time"]), flush=True)


def _is_dag(W):
    d = W.shape[0]
    M = W != 0
    # Kahn's algorithm
    indeg = M.sum(0).astype(int)
    stack = [i for i in range(d) if indeg[i] == 0]
    seen = 0
    while stack:
        a = stack.pop()
        seen += 1
        for b in np.nonzero(M[a])[0]:
            indeg[b] -= 1
            if indeg[b] == 0:
                stack.append(b)
    return seen == d


def _save(st):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, OUT)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="454,889")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--arms", default="pool,tgtcorr,tgtreg,degmatch")
    # separate output files let several workers run on disjoint dimension
    # blocks without ever writing the same JSON concurrently
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.out:
        OUT = os.path.join(ROOT, "results", args.out)

    dims = [int(x) for x in args.dims.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    arms = args.arms.split(",")

    st = {}
    if os.path.exists(OUT):
        try:
            st = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            st = {}

    for d_req in dims:
        for seed in seeds:
            one_cell(d_req, seed, arms, st)
    print("DONE cells=%d" % len(st))


if __name__ == "__main__":
    main()
