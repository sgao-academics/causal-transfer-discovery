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
"""Synthetic ground-truth test of causal transfer (addresses the
"retention is fidelity, not accuracy" objection).

Design
------
A known DAG is generated for the SOURCE and for the TARGET. The two graphs
share a controlled fraction rho of the target's true edges; the remainder is
idiosyncratic to each dataset. Data are sampled from linear SEMs, so the truth
is exact.

Four initialisations are compared on the same target data:
  scratch   random N(0, 0.1^2)
  warm      the learned source graph W_S
  shuffled  row/column permutation of W_S (same weights, no structure)
  oracle    the TRUE target graph (upper bound, not a competitor)

Reported per configuration: F1 / SHD / precision / recall against the target
truth, plus source-edge retention and, crucially, HARM = retained source edges
that are false in the target. The last quantity is what a fidelity metric hides.

Protocol and solver are identical to scripts/experiments/run_benchmark_gpu.py
(L-BFGS-B inner, augmented Lagrangian, lambda=0.01, rho0=0.1, tau=0.3).

Usage
-----
  python run_synthetic_truth.py --pilot      # 1 seed, 2 overlap levels (timing)
  python run_synthetic_truth.py              # full 3 seeds x 5 levels
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark_gpu import notears  # noqa: E402  (paper's own solver)

OUT = os.path.join(PKG_RESULTS, "synthetic_truth.json")
# Defaults match the paper's operating regime: edge density 0.16 per node and
# d/n_tgt ~= 0.7.  Both matter.  With d/n = 0.1 (e.g. d=50, n=500) the target
# problem is over-identified and the solver overrides ANY initialisation, so
# the experiment would silently stop testing transfer at all.
D = 100
M_EDGES = 16          # 0.16 edges per node, as in 73/454 and 107/889
N_SRC = 2000
N_TGT = 140           # d/n = 0.71, close to 889/1218 = 0.73
TAU = 0.3
RHO_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 456]


def random_dag(d, m, rng, forbid=None):
    """Return a weighted adjacency matrix with exactly m edges, acyclic."""
    order = rng.permutation(d)
    cand = []
    for i in range(d):
        for j in range(i + 1, d):
            a, b = order[i], order[j]      # a -> b respects the order
            cand.append((a, b))
    rng.shuffle(cand)
    edges = []
    for a, b in cand:
        if forbid is not None and (a, b) in forbid:
            continue
        edges.append((a, b))
        if len(edges) == m:
            break
    W = np.zeros((d, d))
    for a, b in edges:
        W[a, b] = rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0])
    return W, set(edges)


def make_pair(d, m, rho, rng):
    """Target truth and a source truth sharing `rho` of the target's edges."""
    W_t, e_t = random_dag(d, m, rng)
    n_shared = int(round(rho * m))
    pool = sorted(e_t)
    idx = rng.choice(len(pool), size=n_shared, replace=False)
    shared = set(pool[i] for i in idx) if n_shared > 0 else set()
    W_s = np.zeros((d, d))
    for (a, b) in shared:
        W_s[a, b] = W_t[a, b] * rng.uniform(0.8, 1.2)
    # top up the source with its own idiosyncratic edges
    extra, _ = random_dag(d, m, rng)
    need = m - len(shared)
    added = 0
    for a, b in zip(*np.nonzero(extra)):
        if (a, b) in shared or W_s[a, b] != 0:
            continue
        W_s[a, b] = extra[a, b]
        added += 1
        if added >= need:
            break
    return W_t, e_t, W_s, shared


def sample_sem(W, n, rng):
    """X = E (I - W)^{-1}; unit-variance noise, then z-score columns."""
    d = W.shape[0]
    E = rng.standard_normal((n, d))
    X = E @ np.linalg.inv(np.eye(d) - W)
    X = X / (X.std(0) + 1e-12)
    return X


def edges_of(W, tau=TAU):
    m = np.abs(W) > tau
    np.fill_diagonal(m, False)
    return set(map(tuple, np.argwhere(m).tolist()))


def prf(pred, true):
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    shd = len(pred - true) + len(true - pred)
    return p, r, f1, shd, tp


def one_config(seed, rho):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    W_t, e_t, W_s_true, shared = make_pair(D, M_EDGES, rho, rng)
    X_src = sample_sem(W_s_true, N_SRC, rng)
    X_tgt = sample_sem(W_t, N_TGT, rng)

    # (1) learn the source graph from source data (same solver as the paper)
    W_S, h_s, t_s, it_s = notears(X_src, lam=0.01, max_outer=35)

    res = {"seed": seed, "rho": rho, "shared_true": len(shared),
           "src_learn_time": round(t_s, 1), "src_outer": it_s}

    # quality of the prior itself, against the TARGET truth
    p, r, f1, shd, tp = prf(edges_of(W_S), e_t)
    res["prior"] = {"precision": p, "recall": r, "f1": f1, "shd": shd}

    perm = np.random.permutation(D)
    inits = {"scratch": None, "warm": W_S, "shuffled": W_S[np.ix_(perm, perm)],
             "oracle": W_t}
    fitted = {}
    for name, W0 in inits.items():
        W_T, h, dt, it = notears(X_tgt, lam=0.01, max_outer=35, init_W=W0)
        fitted[name] = W_T
        pred = edges_of(W_T)
        p, r, f1, shd, tp = prf(pred, e_t)
        src_edges = edges_of(W_S)
        retained = len(src_edges & pred)
        retention = 100.0 * retained / max(1, len(src_edges))
        harm = len((src_edges & pred) - e_t)   # retained but false in target
        res[name] = {"precision": p, "recall": r, "f1": f1, "shd": shd,
                     "tp": tp, "n_edges": len(pred), "retention_pct": retention,
                     "retained": retained, "harm": harm,
                     "harm_frac_of_retained": harm / max(1, retained),
                     "time": round(dt, 1), "outer": it}

    # ---- diagnostics: did the initialisation actually survive? ----
    src_edges = edges_of(W_S)
    res["diag"] = {
        # how far the warm solution moved away from its own starting point
        "move_from_init": float(np.abs(fitted["warm"] - W_S).mean()),
        # is warm numerically identical to scratch?  (would mean no transfer)
        "warm_vs_scratch_maxdiff": float(np.abs(fitted["warm"] - fitted["scratch"]).max()),
        "oracle_vs_scratch_maxdiff": float(np.abs(fitted["oracle"] - fitted["scratch"]).max()),
        # does scratch recover source edges on its own?  if so, retention is not evidence
        "scratch_srcret_pct": 100.0 * len(src_edges & edges_of(fitted["scratch"]))
                               / max(1, len(src_edges)),
        "n_src_learned": len(src_edges),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    seeds = [42] if args.pilot else SEEDS
    rhos = [0.0, 1.0] if args.pilot else RHO_LEVELS

    out = {"config": {"d": D, "m_edges": M_EDGES, "n_src": N_SRC,
                      "n_tgt": N_TGT, "tau": TAU, "rhos": rhos, "seeds": seeds},
           "runs": []}
    if os.path.exists(OUT) and not args.pilot:
        try:
            out = json.load(open(OUT, encoding="utf-8"))
            print("resuming: %d runs already done" % len(out["runs"]))
        except Exception:
            pass
    done = {(r["seed"], r["rho"]) for r in out["runs"]}

    t0 = time.time()
    for rho in rhos:
        for seed in seeds:
            if (seed, rho) in done:
                print("skip seed=%d rho=%.2f" % (seed, rho), flush=True)
                continue
            t = time.time()
            r = one_config(seed, rho)
            out["runs"].append(r)
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print("seed %3d rho %.2f | priorF1 %.3f | F1 scr %.3f warm %.3f shuf %.3f "
                  "oracle %.3f | ret %5.1f%% harm %d/%d | %.0fs"
                  % (seed, rho, r["prior"]["f1"], r["scratch"]["f1"],
                     r["warm"]["f1"], r["shuffled"]["f1"], r["oracle"]["f1"],
                     r["warm"]["retention_pct"], r["warm"]["harm"],
                     r["warm"]["retained"], time.time() - t), flush=True)
    print("total %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
