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
"""Ten-seed robustness sweep for the NOTEARS benchmark.

WHY A SEPARATE FILE
-------------------
The published Table 1 rests on three seeds.  A referee may ask whether the
warm/scratch gap survives a wider draw.  Re-running all ten seeds in place
would change every number in the manuscript, which is not acceptable in a minor
revision, so this script appends seven NEW seeds to a separate checkpoint and
leaves results/benchmark_v6_checkpoint.json untouched.  The ten-seed means can
then be reported in the SI as a robustness check while the main table keeps the
three-seed protocol that the rest of the paper is built on.

PROTOCOL NOTES (must be stated in the SI, not hidden)
-----------------------------------------------------
* Seeds 42/123/456 are the ones behind Table 1; their source graphs were
  produced by an unseeded random start, i.e. the seed number is a run label.
* The seven new seeds set RNG state explicitly before the source graph is
  learned, so the new half of the sweep is bit-reproducible from the seed
  alone.
* Both halves share the identical estimator, penalty, tau and outer-loop
  budget; only the random start differs.  Pairing is therefore preserved:
  warm and scratch for a given (d, seed) start from the same data and are
  scored against that seed's own source graph.

Usage
-----
  python run_extraseeds.py --dims 454,889
  python run_extraseeds.py --dims 200,454,600,750,889
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

CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "bench_v6_extraseeds.json")
TAU = 0.3

NEW_SEEDS = [789, 1011, 2024, 3141, 4242, 5555, 6789]
OLD_SEEDS = [42, 123, 456]
DIMS = [200, 454, 600, 750, 889]


def _save(st):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, OUT)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="454,889")
    ap.add_argument("--seeds", default="")
    # disjoint seed blocks may run concurrently as long as each worker owns
    # its own output file; the merge step concatenates them
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.out:
        OUT = os.path.join(ROOT, "results", args.out)

    dims = [int(x) for x in args.dims.split(",")]
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else NEW_SEEDS)

    st = {}
    if os.path.exists(OUT):
        try:
            st = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            st = {}
    st.setdefault("config", {"small_runner": NEW_SEEDS, "table1_seeds": OLD_SEEDS,
                             "tau": TAU, "note": "source graphs for the new seeds "
                             "use an explicit RNG seed; Table 1 seeds do not"})

    for d_req in dims:
        X_tcga, X_brca, d_actual, genes = load_data(d_req)
        print("d=%d loaded: d_actual=%d n_src=%d n_tgt=%d"
              % (d_req, d_actual, X_tcga.shape[0], X_brca.shape[0]), flush=True)

        for seed in seeds:
            tag = "d%d_s%d" % (d_actual, seed)
            ks, kw, kc = ("%s_%s" % (tag, a) for a in ("source", "warm", "scratch"))
            kf = "%s_shuffled" % tag

            # ---- source graph ------------------------------------------
            sp = os.path.join(CACHE, "src_d%d_s%d.npy" % (d_actual, seed))
            if ks not in st:
                np.random.seed(seed)
                t0 = time.time()
                W_src, h, dt, it = notears(X_tcga, lam=0.01, max_outer=35)
                np.save(sp, W_src)
                st[ks] = {"h": float(h), "time": float(dt), "outer": int(it),
                          "edges": int((np.abs(W_src) > TAU).sum()),
                          "explicit_seed": True}
                _save(st)
                print("  [src] %s edges=%d %.0fs" % (tag, st[ks]["edges"], dt), flush=True)
            W_src = np.load(sp)[:d_actual, :d_actual]

            # ---- scratch ----------------------------------------------
            if kc not in st:
                np.random.seed(seed)
                t0 = time.time()
                W, h, dt, it = notears(X_brca, lam=0.01, max_outer=35)
                ev = evaluate(W_src, W)
                np.save(os.path.join(CACHE, "scratch_d%d_s%d.npy" % (d_actual, seed)), W)
                st[kc] = {"h": float(h), "time": float(dt), "outer": int(it), **ev}
                _save(st)
                print("  [scr] %s ret=%5.1f%% r=%.3f %.0fs"
                      % (tag, ev["retention_pct"], ev["pearson_r"], dt), flush=True)

            # ---- warm -------------------------------------------------
            if kw not in st:
                np.random.seed(seed)
                t0 = time.time()
                W, h, dt, it = notears(X_brca, lam=0.01, max_outer=35, init_W=W_src)
                ev = evaluate(W_src, W)
                np.save(os.path.join(CACHE, "warm_d%d_s%d.npy" % (d_actual, seed)), W)
                st[kw] = {"h": float(h), "time": float(dt), "outer": int(it), **ev}
                _save(st)
                print("  [wrm] %s ret=%5.1f%% r=%.3f %.0fs"
                      % (tag, ev["retention_pct"], ev["pearson_r"], dt), flush=True)

            # ---- shuffled (negative control) ---------------------------
            if kf not in st:
                np.random.seed(seed)
                idx = np.random.RandomState(seed).permutation(d_actual)
                W0 = W_src[idx, :][:, idx]
                t0 = time.time()
                W, h, dt, it = notears(X_brca, lam=0.01, max_outer=35, init_W=W0)
                ev = evaluate(W_src, W)
                np.save(os.path.join(CACHE, "shuf_d%d_s%d.npy" % (d_actual, seed)), W)
                st[kf] = {"h": float(h), "time": float(dt), "outer": int(it), **ev}
                _save(st)
                print("  [shf] %s ret=%5.1f%% r=%.3f %.0fs"
                      % (tag, ev["retention_pct"], ev["pearson_r"], dt), flush=True)

    print("DONE cells=%d" % (len(st) - 1))


if __name__ == "__main__":
    main()
