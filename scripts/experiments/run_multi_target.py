#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the transfer benefit depend on WHICH tissue is the target?

WHY
---
Every reported transfer experiment uses BRCA (n = 1,218) as the target, for the
simple reason that it is the largest cohort and the one the main benchmark was
built around.  That leaves one question a referee can ask and we cannot answer
from the manuscript: is the advantage a property of the method, or a property of
breast cancer?

This script answers it by repeating the d = 454 core experiment with each of
several TCGA cohorts in the target role.

DESIGN -- leave-one-organ-out
-----------------------------
For target organ O, the source is *every other* cohort in the panel (BRCA
included).  The source graph is therefore always learned without any sample from
O, and the "source and target share zero samples" claim survives for every
target, not just for BRCA.

The temptation is to reuse the single archived source graph (the non-BRCA one)
for every target and save 2.5 h per organ.  That would be wrong: the source graph
would then contain the target's own samples, which is exactly the leak the
paper's protocol exists to rule out.  Each target gets its own source graph.

Gene panel
----------
One panel for all targets: the same d genes used by the main benchmark, derived
from the MAD ranking of the source matrix.  Holding the coordinate system fixed
is what makes retention numbers comparable across organs -- a per-organ panel
would make every column index mean something different.

Normalisation follows load_data() exactly (cohort matrices stacked, then z-scored
as one matrix; target z-scored on its own), so the BRCA row of the output has to
reproduce the archived d = 454 numbers.  That is the built-in check that this
script has not silently changed the protocol.

Usage
-----
  python run_multi_target.py --organs KIRC,LUAD,THCA,LUSC,LIHC --seeds 42
  python run_multi_target.py --organs KIRC,LUAD --seeds 42,123,456

Output
------
results/multi_target_d454.json, rewritten after every cell (resumable).
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from run_benchmark_gpu import notears, evaluate  # noqa: E402

# Override with the TCGA_DIR environment variable when the cohort files live
# elsewhere: this is the only machine-specific path in the script.
TCGA_DIR = os.environ.get("TCGA_DIR", r"D:\NO.1\cancer_application\data")
CACHE = os.path.join(ROOT, "results", "cache")
OUT = os.path.join(ROOT, "results", "multi_target_d454.json")
TAU = 0.3
D = 454
LAM = 0.01
MAX_OUTER = 35

os.makedirs(CACHE, exist_ok=True)


def cohorts():
    """Whitelist of expression files (never *_clinical.tsv)."""
    out = []
    for f in sorted(os.listdir(TCGA_DIR)):
        if f.startswith("TCGA_") and f.endswith("_HiSeqV2.tsv"):
            out.append(f[len("TCGA_"):-len("_HiSeqV2.tsv")])
    return out


def read_organ(cohort, genes):
    """Samples x genes, raw values (no normalisation yet)."""
    fp = os.path.join(TCGA_DIR, "TCGA_%s_HiSeqV2.tsv" % cohort)
    df = pd.read_csv(fp, sep="\t", index_col=0).T
    X = df[genes].values.astype(np.float64)
    return np.nan_to_num(X, nan=0.0)


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def build_panel(d=D):
    """Same gene set and same ordering as the main benchmark.

    Deliberately NOT re-derived here: the archived source graphs and every
    retention number in the paper are expressed in this exact column order, so
    the panel is taken from the main loader (MAD ranking of the source matrix,
    restricted to genes present in BRCA) and cached instead of recomputed.
    """
    gp = os.path.join(CACHE, "panel_d%d.txt" % d)
    if os.path.exists(gp):
        return [x.strip() for x in open(gp, encoding="utf-8") if x.strip()]

    from load_tcga_safe import load_data as _load      # whitelist loader
    _, _, d_actual, genes = _load(d)
    with open(gp, "w", encoding="utf-8") as f:
        f.write("\n".join(genes))
    return genes


def load_source(exclude, genes):
    """Stack every cohort except `exclude`, then z-score the stacked matrix.

    One z-score AFTER stacking, not one per cohort: the archived benchmark was
    built that way, and z-scoring each cohort first erases the between-cohort
    mean/variance differences the source graph was learned on.  Doing it per
    cohort silently produced a sparser source graph (59 edges instead of 74 at
    d = 454, seed 42) and would have made every retention number incomparable.
    """
    parts = []
    used = []
    for c in cohorts():
        if c == exclude:
            continue
        parts.append(read_organ(c, genes))
        used.append(c)
    return zscore(np.vstack(parts)), used


def _save(st):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organs", default="BRCA,KIRC,LUAD,THCA,LUSC,LIHC")
    ap.add_argument("--seeds", default="42")
    args = ap.parse_args()

    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    genes = build_panel()
    d_actual = len(genes)
    print("panel %d genes (requested d=%d), organs=%s, seeds=%s"
          % (d_actual, D, organs, seeds), flush=True)

    st = {}
    if os.path.exists(OUT):
        try:
            st = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            st = {}
    st.setdefault("meta", {"d_actual": d_actual, "tau": TAU, "lam": LAM,
                           "max_outer": MAX_OUTER, "protocol": "leave-one-organ-out"})
    st["meta"]["d_actual"] = d_actual

    for organ in organs:
        for seed in seeds:
            key = "%s_s%d" % (organ, seed)
            if key in st and "warm" in st[key]:
                print("  %-22s done" % key, flush=True)
                continue

            t0 = time.time()
            src_path = os.path.join(CACHE, "srclo_%s_d%d_s%d.npy" % (organ, d_actual, seed))
            if os.path.exists(src_path):
                W_src = np.load(src_path)
                print("  %-22s source cached (%d edges)"
                      % (key, int((np.abs(W_src) > TAU).sum())), flush=True)
            else:
                X_src, used = load_source(organ, genes)
                print("  %-22s source n=%d from %d cohorts, learning graph..."
                      % (key, X_src.shape[0], len(used)), flush=True)
                np.random.seed(seed)
                W_src, h_src, dt_src, it_src = notears(X_src, lam=LAM,
                                                       max_outer=MAX_OUTER, verbose=False)
                np.save(src_path, W_src)
                print("  %-22s source graph: h=%.2e edges=%d %.0fs"
                      % (key, h_src, int((np.abs(W_src) > TAU).sum()), dt_src), flush=True)
                del X_src

            X_tgt = zscore(read_organ(organ, genes))
            rec = {"n_target": int(X_tgt.shape[0]), "d_actual": d_actual,
                   "src_edges": int((np.abs(W_src) > TAU).sum())}

            for arm in ("scratch", "warm"):
                np.random.seed(seed)
                init = W_src if arm == "warm" else None
                W, h, dt, it = notears(X_tgt, lam=LAM, max_outer=MAX_OUTER,
                                       init_W=init, verbose=False)
                ev = evaluate(W_src, W, tau=TAU)
                rec[arm] = {"edges": ev["tgt_edges"], "overlap": ev["overlap"],
                            "retention_pct": ev["retention_pct"],
                            "pearson_r": ev["pearson_r"], "h": float(h),
                            "time": float(dt), "outer": int(it)}
                print("  %-22s %-7s ret=%5.1f%% (%d/%d) edges=%d r=%.3f %.0fs"
                      % (key, arm, ev["retention_pct"], ev["overlap"], ev["src_edges"],
                         ev["tgt_edges"], ev["pearson_r"], dt), flush=True)

            rec["gain_pp"] = round(rec["warm"]["retention_pct"] - rec["scratch"]["retention_pct"], 1)
            rec["seed"] = seed
            rec["wall_s"] = round(time.time() - t0, 1)
            st[key] = rec
            _save(st)
            print("  %-22s GAIN %+.1f pp  (wall %.0fs)"
                  % (key, rec["gain_pp"], rec["wall_s"]), flush=True)

    print("\nDONE cells=%d -> %s" % (len([k for k in st if k != "meta"]), OUT), flush=True)


if __name__ == "__main__":          # multiprocessing-safe guard (V24)
    main()
