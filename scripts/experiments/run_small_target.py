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
"""Does the transfer benefit grow as the target cohort shrinks?

The manuscript motivates transfer with "small cohorts, d >> n", but every
reported experiment uses the largest available target (BRCA, n = 1,218).  A
referee can therefore say the motivation and the evidence do not meet.

This script subsamples BRCA to n in {150, 300, 600} (plus the full 1,218 for
reference), refits the warm and scratch arms on each subsample, and reports
retention against the SAME per-seed source graph used by Table 1.  Subsampling
is seeded, so the same rows are used for both arms -- the comparison stays
paired.

Two readings are recorded on purpose:

  retention_pct   fidelity to the source graph (the paper's metric)
  nrmse_target    normalised SEM residual on the subsample -- if warm-start
                  merely freezes the prior, the residual will be WORSE than
                  scratch; if it transfers structure that the small sample
                  cannot recover on its own, it will be better

Reporting both is what keeps the claim honest: retention can rise for a trivial
reason (weak data constraint leaves the initialisation in place), and the
residual says whether that is a help or a harm.

Usage
-----
  python run_small_target.py --dims 454,889 --ns 150,300,600
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
from run_real_baselines import threshold, bic, nrmse  # noqa: E402

CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "small_target.json")
TAU = 0.3


def _save(st):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="454,889")
    ap.add_argument("--ns", default="150,300,600,1218")
    ap.add_argument("--seeds", default="42,123,456")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    st = {}
    if os.path.exists(OUT):
        try:
            st = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            st = {}

    for d_req in dims:
        X_tcga, X_brca, d_actual, genes = load_data(d_req)
        n_full = X_brca.shape[0]
        print("d=%d loaded: d_actual=%d n_tgt=%d" % (d_req, d_actual, n_full), flush=True)

        for seed in seeds:
            sp = os.path.join(CACHE, "src_d%d_s%d.npy" % (d_actual, seed))
            if not os.path.exists(sp):
                print("  missing %s -- skip" % sp, flush=True)
                continue
            W_src = np.load(sp)[:d_actual, :d_actual]

            for n in ns:
                if n > n_full:
                    continue
                tag = "d%d_s%d_n%d" % (d_actual, seed, n)
                if all("%s_%s" % (tag, a) in st for a in ("warm", "scratch")):
                    print("  %s done" % tag, flush=True)
                    continue
                # seeded subsample, identical rows for both arms
                if n == n_full:
                    X_sub = X_brca
                else:
                    rng = np.random.default_rng(seed * 1000 + n)
                    rows = rng.choice(n_full, size=n, replace=False)
                    X_sub = X_brca[np.sort(rows)]

                for arm in ("warm", "scratch"):
                    key = "%s_%s" % (tag, arm)
                    if key in st:
                        continue
                    t0 = time.time()
                    np.random.seed(seed)
                    init = W_src if arm == "warm" else None
                    W, h, dt, it = notears(X_sub, lam=0.01, max_outer=35,
                                           init_W=init, verbose=False)
                    ev = evaluate(W_src, W)
                    rec = {"n_target": n, "d_over_n": round(d_actual / n, 3),
                           "retention_pct": ev["retention_pct"],
                           "overlap": ev["overlap"], "src_edges": ev["src_edges"],
                           "tgt_edges": ev["tgt_edges"],
                           "pearson_r": ev["pearson_r"],
                           "nrmse_target": round(nrmse(X_sub, W), 5),
                           "bic_target": round(bic(X_sub, W), 1),
                           "h": float(h), "time": float(dt), "outer": int(it),
                           "d_actual": d_actual}
                    st[key] = rec
                    _save(st)
                    print("  %-22s %-7s ret=%5.1f%% (%d/%d) nrmse=%.4f edges=%d %.0fs"
                          % (tag, arm, rec["retention_pct"], rec["overlap"],
                             rec["src_edges"], rec["nrmse_target"],
                             rec["tgt_edges"], rec["time"]), flush=True)

    print("DONE cells=%d" % len(st))


if __name__ == "__main__":
    main()
