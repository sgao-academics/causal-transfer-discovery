# -*- coding: utf-8 -*-
# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""
Per-seed DAGMA / GOLEM transfer arms.

WHY THIS EXISTS
---------------
Table 5 of the manuscript reported a single warm-start value per (algorithm, d):
the warm arm was always initialised from the SAME fixed source graph
(cache/src_d{d}.npy), so the three "seeds" produced bit-identical solutions and
30 of the 45 cells behind the "45/45 wins" claim carried n = 1.  The d <= 889
NOTEARS benchmark, by contrast, uses a seed-matched protocol: each seed learns
its own source graph and every retention ratio is measured against that seed's
own graph.

This script replays the NOTEARS protocol for DAGMA and GOLEM:
  * each seed takes its own NOTEARS source graph, cache/src_d{d}_s{seed}.npy
    (these are exactly the graphs that produced Table 1 -- verified: the
    NOTEARS warm retentions recomputed from them match the checkpoint to the
    first decimal at all 15 cells);
  * warm start initialises from that graph, scratch does not;
  * both are scored against that same graph, i.e. seed-matched denominators.

Nothing about the algorithms is re-implemented: dagma_v2, golem, load_data and
evaluate are imported verbatim from run_dagma_golem_v2.py.  Original artefacts
(cache/src_d{d}.npy, cache/dagma_warm_d{d}_s{seed}.npy, ...) are never
overwritten; the new matrices go to *_seedmatched_* filenames.

USAGE
  python run_dagma_golem_seedmatched.py                # all d, all seeds
  python run_dagma_golem_seedmatched.py --dims 200     # just one dimension
  python run_dagma_golem_seedmatched.py --algos dagma

Resumable: every finished cell is appended to results/dagma_golem_seedmatched.json
and skipped on the next launch.
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

# --- reuse the manuscript's own implementations; do not rewrite them --------
from run_dagma_golem_v2 import dagma_v2, golem, evaluate  # noqa: E402
# load_data in run_dagma_golem_v2 admits any TCGA_*.tsv that is not BRCA, which
# now picks up TCGA_HNSC_clinical.tsv (added to the data directory after the
# benchmark was run) and crashes.  load_tcga_safe uses an explicit
# TCGA_<cohort>_HiSeqV2.tsv whitelist and was verified to reproduce the same
# d_actual and the same coordinate system as the archived graphs (residual of
# the known source graph is 1/4 of its column-permuted null).
from load_tcga_safe import load_data  # noqa: E402

CACHE = os.path.join(ROOT, "results", "cache")
OUT_JSON = os.path.join(ROOT, "results", "dagma_golem_seedmatched.json")
LOG = os.path.join(ROOT, "_dagma_golem_seedmatched_log.txt")
# set by --state / --log so that several workers can run concurrently
STATE_PATH = OUT_JSON
LOG_PATH = LOG

ALGOS = {"dagma": dagma_v2, "golem": golem}
ALL_DIMS = [200, 454, 600, 750, 889]
ALL_SEEDS = [42, 123, 456]
TAU = 0.3


# --------------------------------------------------------------------------
# logging / checkpointing
# --------------------------------------------------------------------------
def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(st):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_PATH)


def key(d, seed, algo, arm):
    return "d%d_s%d_%s_%s" % (d, seed, algo, arm)


# --------------------------------------------------------------------------
def src_path(d, seed):
    return os.path.join(CACHE, "src_d%d_s%d.npy" % (d, seed))


def run(args):
    dims = [int(x) for x in args.dims.split(",")] if args.dims else ALL_DIMS
    algos = args.algos.split(",") if args.algos else list(ALGOS)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else ALL_SEEDS

    st = load_state()
    log("=" * 76)
    log("START  dims=%s algos=%s seeds=%s  already_done=%d"
        % (dims, algos, seeds, len(st)))

    for d_req in dims:
        # loading the TCGA matrices is the slow part -- do it once per dimension
        X_tcga, X_brca, d_actual, genes = load_data(d_req)
        log("d=%d loaded: d_actual=%d  n_src=%d  n_tgt=%d"
            % (d_req, d_actual, X_tcga.shape[0], X_brca.shape[0]))

        for seed in seeds:
            sp = src_path(d_actual, seed)
            if not os.path.exists(sp):
                log("  !! missing per-seed source graph %s -- SKIP" % sp)
                continue
            W_src = np.load(sp)[:d_actual, :d_actual]

            for algo in algos:
                if algo not in ALGOS:
                    log("  !! unknown algorithm %s" % algo)
                    continue
                fn = ALGOS[algo]

                # ---------------- warm arm (the point of this script) -----
                k_warm = key(d_actual, seed, algo, "warm")
                if k_warm not in st:
                    log("[%s] d=%d seed=%d WARM from own source graph" % (algo, d_actual, seed))
                    np.random.seed(seed)          # determinism for any residual randomness
                    t0 = time.time()
                    W, h, dt, iters = fn(X_brca, max_outer=35, init_W=W_src, verbose=True)
                    ev = evaluate(W_src, W)
                    out_npy = os.path.join(
                        CACHE, "%s_warm_seedmatched_d%d_s%d.npy" % (algo, d_actual, seed))
                    np.save(out_npy, W)
                    st[k_warm] = {"h": float(h), "time": float(dt), "outer": int(iters),
                                  "denominator": "seed_matched",
                                  "source_edges": int(ev["src_edges"]), **ev}
                    save_state(st)
                    log("  -> ret=%.1f%% (%d/%d)  tgt_edges=%d  r=%.4f  h=%.2e  %.0fs"
                        % (ev["retention_pct"], ev["overlap"], ev["src_edges"],
                           ev["tgt_edges"], ev["pearson_r"], h, time.time() - t0))
                else:
                    log("[%s] d=%d seed=%d WARM already done (ret=%.1f%%)"
                        % (algo, d_actual, seed, st[k_warm]["retention_pct"]))

                # ---------------- scratch arm -------------------------------
                # Two cases:
                #  * the archived scratch solution was produced by the same
                #    implementation described in Methods -> reuse it, only the
                #    denominator changes (seed-matched);
                #  * otherwise (d=200 DAGMA was produced by run_dagma_golem.py,
                #    which enforces acyclicity via log-det instead of the
                #    trace-exponential form stated in Methods) -> re-run it here
                #    with --force-scratch.
                k_scr = key(d_actual, seed, algo, "scratch")
                if k_scr in st:
                    log("  scratch already done (ret=%.1f%%)" % st[k_scr]["retention_pct"])
                else:
                    scr_npy = os.path.join(
                        CACHE, "%s_scratch_d%d_s%d.npy" % (algo, d_actual, seed))
                    force = args.force_scratch or (d_actual in args.force_scratch_dims)
                    if os.path.exists(scr_npy) and not force:
                        W_scr = np.load(scr_npy)[:d_actual, :d_actual]
                        ev = evaluate(W_src, W_scr)
                        st[k_scr] = {"denominator": "seed_matched",
                                     "reused_solution": os.path.basename(scr_npy),
                                     "source_edges": int(ev["src_edges"]), **ev}
                        save_state(st)
                        log("  scratch re-scored: ret=%.1f%% (%d/%d)"
                            % (ev["retention_pct"], ev["overlap"], ev["src_edges"]))
                    else:
                        log("  scratch RE-RUN with the Methods implementation")
                        np.random.seed(seed * 7 + 1)
                        W_scr, h_s, dt_s, it_s = fn(X_brca, max_outer=35, verbose=False)
                        ev = evaluate(W_src, W_scr)
                        np.save(os.path.join(
                            CACHE, "%s_scratch_seedmatched_d%d_s%d.npy"
                            % (algo, d_actual, seed)), W_scr)
                        st[k_scr] = {"h": float(h_s), "time": float(dt_s), "outer": int(it_s),
                                     "denominator": "seed_matched",
                                     "rerun_with_methods_implementation": True,
                                     "source_edges": int(ev["src_edges"]), **ev}
                        save_state(st)
                        log("  -> scratch ret=%.1f%% (%d/%d)  tgt_edges=%d  %.0fs"
                            % (ev["retention_pct"], ev["overlap"], ev["src_edges"],
                               ev["tgt_edges"], dt_s))

    log("DONE  cells=%d" % len(st))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="", help="comma list, e.g. 200,454")
    p.add_argument("--algos", default="", help="comma list: dagma,golem")
    p.add_argument("--seeds", default="", help="comma list: 42,123,456")
    p.add_argument("--force-scratch", action="store_true",
                   help="re-run the scratch arm instead of reusing the archived "
                        "solution (needed where the archive used a different "
                        "acyclicity constraint)")
    p.add_argument("--force-scratch-dims", default="", type=str,
                   help="comma list of dimensions whose scratch arm must be re-run, "
                        "e.g. 200 (only d=200 DAGMA used the log-det constraint)")
    p.add_argument("--state", default="", help="alternate checkpoint JSON (for parallel workers)")
    p.add_argument("--log", default="", help="alternate log file (for parallel workers)")
    args = p.parse_args()
    args.force_scratch_dims = ([int(x) for x in args.force_scratch_dims.split(",")]
                               if args.force_scratch_dims else [])
    if args.state:
        STATE_PATH = args.state
    if args.log:
        LOG_PATH = args.log
    run(args)
