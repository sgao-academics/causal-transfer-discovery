#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d = 5,000 transfer with a chosen target cohort (leave-one-organ-out).

WHY
---
Section 5.4 reports the genome-adjacent experiment with BRCA as the only target.
A reader cannot tell from that whether the +36 pp gain is a property of the
method or of breast cancer.  This script runs the identical protocol with any
TCGA cohort in the target role.

PROTOCOL -- copied from the archived seed-matched runs, not from run_all.py
---------------------------------------------------------------------------
The published d=5000 numbers (81.8 +/- 0.4 % warm vs 45.4 +/- 2.9 % scratch,
624 +/- 4 source edges) come from the seed-matched scripts, and the solver is
imported from `run_d5000_multiseed.py` rather than reimplemented.  Details that
must match or the numbers stop being comparable:

  * Random initialisation N(0, 0.05^2) for BOTH the source graph and the
    scratch arm, seeded per seed.  sigma is 0.05 and not the d<=889 value of 0.1
    because at d=5000 sigma=0.1 gives a row-sum of W^2 ~ 50, above the float32
    matrix_exp limit (~30) -> NaN.  The same W0 is handed to both arms.
  * The learned source graph is rescaled by sqrt(30/row_sum) when row_sum > 30,
    before it is used as the warm-start.
  * OUTER = 20 as a CAP, not a target: every archived d=5000 run converged to
    h = 0.00e+00 and hit the h < 1e-6 break at loop 10-12 (all ckpt_*_d5000.json
    record last_h = 0).  Forcing a fixed 11 would truncate a run that needs 12
    (ms_warm_s42 landed on 12), so the loop is left to converge on its own.
  * Gene panel: the frozen 5,000 BRCA top-variance genes; every target uses the
    same columns, so retention is measured in one coordinate system.
  * Source: every cohort except the target, each z-scored separately and then
    stacked -- the d=5000 convention (the d<=889 loader z-scores after stacking;
    the two scales were built separately and each keeps its own, or the archived
    numbers stop reproducing).
  * Acyclicity / sparsity / threshold: INNER=100, lr=0.002, lambda1=0.01, tau=0.3.

COST (measured on the archived runs, RTX 5060 8 GB)
---------------------------------------------------
  source ~3.4 h + scratch ~3.4 h + warm ~0.8 h ~ 7.6 h per organ.
  The source graph cannot be shared between targets: it would then contain the
  target's own samples, which is exactly the leak the protocol rules out.

  Seeds: default is 42 only.  Three seeds x five organs is ~5 days of GPU; these
  runs are therefore single-seed exploratory and must be labelled as such in the
  SI, with the three-seed BRCA numbers kept as the confirmatory result.

Usage
-----
  python run_d5000_multitarget.py --organ KIRC
  python run_d5000_multitarget.py --organ KIRC --smoke     # d=300 plumbing test
  python run_d5000_multitarget.py --organ LUAD --seeds 42,123

Output: results/d5000_multi/{ORGAN}[_{seed}]/result.json  (resumable per outer loop)
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_d5000_multiseed as MS   # noqa: E402  -- solver, imported not copied

# Override with the TCGA_DIR environment variable when the cohort files live
# elsewhere: this is the only machine-specific path in the script.
TCGA_DIR = os.environ.get("TCGA_DIR", r"D:\NO.1\cancer_application\data")
PANEL = os.path.join(MS.CACHE_DIR, "selected_genes_d5000.txt")
OUTER_DEFAULT = 20          # the cap; every archived run converged and broke at 10-12
INIT_SIGMA = 0.05
ROWSUM_MAX = 30.0


def expression_cohorts():
    out = []
    for f in sorted(os.listdir(TCGA_DIR)):
        if f.startswith("TCGA_") and f.endswith("_HiSeqV2.tsv"):
            out.append(f[len("TCGA_"):-len("_HiSeqV2.tsv")])
    return out


def panel(d_use):
    genes = [x.strip() for x in open(PANEL, encoding="utf-8") if x.strip()]
    return genes[:d_use] if d_use < len(genes) else genes


def read_z(cohort, genes):
    """Samples x genes, z-scored within the cohort (d=5000 convention)."""
    fp = os.path.join(TCGA_DIR, "TCGA_%s_HiSeqV2.tsv" % cohort)
    df = pd.read_csv(fp, sep="\t", index_col=0).T
    X = df[genes].values.astype(np.float32)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def edge_set(W, tau):
    m = np.abs(W) > tau
    np.fill_diagonal(m, False)
    return set(map(tuple, np.argwhere(m)))


def run_one(organ, seed, d_use, outer, inner, smoke):
    # keep plumbing runs out of the directory the real run will use
    suffix = "_smoke" if smoke else (("_mini%d" % d_use) if d_use != MS.D else "")
    tag = organ + suffix + (("_s%d" % seed) if not suffix else "")
    out_dir = os.path.join(MS.BASE, "results", "d5000_multi", tag)
    ck_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ck_dir, exist_ok=True)

    MS.CKPT_DIR = ck_dir            # solver reads this module global
    MS.D = d_use
    MS.INNER = inner

    torch.manual_seed(seed)
    np.random.seed(seed)

    genes = panel(d_use)

    # ---- target ----
    ct = os.path.join(MS.CACHE_DIR, "tgt_%s_d%d%s.npy"
                      % (organ, d_use, "_smoke" if smoke else ""))
    if os.path.exists(ct):
        X_tgt = np.load(ct)
    else:
        X_tgt = read_z(organ, genes)
        np.save(ct, X_tgt)

    # ---- source, leave-one-organ-out ----
    cs = os.path.join(MS.CACHE_DIR, "src_ex_%s_d%d%s.npy"
                      % (organ, d_use, "_smoke" if smoke else ""))
    used = [c for c in expression_cohorts() if c != organ]
    if os.path.exists(cs):
        X_src = np.load(cs)
    else:
        t0 = time.time()
        X_src = np.vstack([read_z(c, genes) for c in used])
        np.save(cs, X_src)
        print("  source built: %d x %d from %d cohorts (%.0fs)"
              % (X_src.shape[0], X_src.shape[1], len(used), time.time() - t0), flush=True)

    print("target=%s d=%d seed=%d outer=%d inner=%d | target %s | source %s"
          % (organ, d_use, seed, outer, inner, X_tgt.shape, X_src.shape), flush=True)

    # float32 warmup
    Tw = torch.randn(d_use, d_use, dtype=torch.float32, device=MS.device)
    _ = torch.linalg.matrix_exp(Tw * Tw)
    torch.cuda.synchronize()
    del Tw
    torch.cuda.empty_cache()

    res = {"organ": organ, "d": d_use, "seed": seed, "outer": outer,
           "inner": inner, "lr": MS.LR, "lambda1": MS.LAMBDA1,
           "w_threshold": MS.W_THRESHOLD, "init_sigma": INIT_SIGMA,
           "n_source": int(X_src.shape[0]), "n_target": int(X_tgt.shape[0]),
           "n_source_cohorts": len(used), "smoke": bool(smoke),
           "protocol": "leave-one-organ-out, seed-matched, d=5000 convention"}

    W0 = (np.random.randn(d_use, d_use) * INIT_SIGMA).astype(np.float32)

    print("[source] NOTEARS on the %d-cohort source (X^T X)" % len(used), flush=True)
    torch.cuda.empty_cache()
    W_src = MS.run_notears(X_src, W_init_np=W0, label="src",
                           d_override=d_use, outer=outer, use_xtx=True)

    max_row = float((W_src * W_src).sum(axis=1).max())
    if max_row > ROWSUM_MAX:
        W_src = (W_src * np.sqrt(ROWSUM_MAX / max_row)).astype(np.float32)
        print("  rescaled for float32 stability (row-sum %.1f -> %.1f)" % (max_row, ROWSUM_MAX),
              flush=True)
    src_set = edge_set(W_src, MS.W_THRESHOLD)
    print("  source edges: %d" % len(src_set), flush=True)
    res["source_edges"] = len(src_set)
    res["src_rowsum_max"] = round(max_row, 2)

    del X_src

    print("[scratch] NOTEARS on %s from the same W0" % organ, flush=True)
    torch.cuda.empty_cache()
    W_scr = MS.run_notears(X_tgt, W_init_np=W0, label="scr",
                           d_override=d_use, outer=outer, use_xtx=False)

    print("[warm] NOTEARS on %s from the source graph" % organ, flush=True)
    torch.cuda.empty_cache()
    W_warm = MS.run_notears(X_tgt, W_init_np=W_src, label="warm",
                            d_override=d_use, outer=outer, use_xtx=False)

    scr_set = edge_set(W_scr, MS.W_THRESHOLD)
    warm_set = edge_set(W_warm, MS.W_THRESHOLD)
    kept_s, kept_w = len(src_set & scr_set), len(src_set & warm_set)
    res.update({"scratch_edges": len(scr_set), "warm_edges": len(warm_set),
                "scratch_kept": kept_s, "warm_kept": kept_w,
                "scratch_retention_pct": round(100.0 * kept_s / max(1, len(src_set)), 2),
                "warm_retention_pct": round(100.0 * kept_w / max(1, len(src_set)), 2),
                "gain_pp": round(100.0 * (kept_w - kept_s) / max(1, len(src_set)), 2)})

    out = os.path.join(out_dir, "result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("\n=== %s seed %d ===" % (organ, seed), flush=True)
    for k, v in res.items():
        print("  %-22s %s" % (k, v), flush=True)
    print("saved: %s" % out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--outer", type=int, default=OUTER_DEFAULT)
    ap.add_argument("--dim", type=int, default=0, help="truncate d (plumbing tests only)")
    ap.add_argument("--inner", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    d_use = args.dim if args.dim else MS.D
    outer = 1 if args.smoke else args.outer
    inner = args.inner if args.inner else (5 if args.smoke else MS.INNER)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("organ=%s d=%d seeds=%s outer=%d inner=%d smoke=%s device=%s"
          % (args.organ.upper(), d_use, seeds, outer, inner, args.smoke, MS.device), flush=True)

    for s in seeds:
        run_one(args.organ.upper(), s, d_use, outer, inner, args.smoke)


if __name__ == "__main__":      # guard (V24): the imported solver touches CUDA
    main()
