#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate every number printed in the manuscript against the archived results.

No GPU, no external data, no re-fitting: this reads
results/benchmark_v6_checkpoint.json (the single source of truth for Tables 1
and 4.4) and results/full_experiment.json, recomputes each reported mean and
sample standard deviation, and compares against the values typeset in the
manuscript.  Anything that disagrees by more than TOL is reported as an error.

Usage:
    python validate_paper.py
Exit code 0 iff every check passes.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, "results", "benchmark_v6_checkpoint.json")
FULL = os.path.join(BASE, "results", "full_experiment.json")
SEEDSMATCH = os.path.join(BASE, "results", "dagma_golem_seedmatched.json")

DIMS = [200, 454, 600, 750, 889]
SEEDS = [42, 123, 456]
TOL = 0.15          # printed values are rounded to one decimal

# ---- values as typeset in the manuscript (Table 1) -------------------------
T1 = {  # d: (scratch mean, scratch sd, warm mean, warm sd)
    200: (19.9, 5.0, 31.9, 2.0),
    454: (33.4, 1.9, 55.1, 1.7),
    600: (33.3, 2.0, 66.6, 4.7),
    750: (36.5, 2.8, 66.8, 2.2),
    889: (34.0, 3.3, 68.1, 1.3),
}

# ---- Table 4.4 -------------------------------------------------------------
T44 = {  # d: (D_scr, D_scr_sd, D_warm, D_warm_sd, G_scr, G_scr_sd, G_warm, G_warm_sd)
    200: (14.7, 1.0, 78.5, 2.4, 16.8, 6.1, 41.9, 4.3),
    454: (25.4, 1.8, 88.8, 2.2, 30.1, 5.0, 72.0, 2.2),
    600: (27.6, 4.0, 88.1, 0.9, 35.3, 1.1, 74.7, 2.6),
    750: (25.0, 3.8, 90.9, 2.5, 37.5, 5.1, 76.1, 3.8),
    889: (29.2, 3.3, 89.9, 3.1, 35.8, 3.8, 75.1, 2.2),
}

errors = []
checks = 0


def cmp(tag, got, want):
    global checks
    checks += 1
    ok = abs(got - want) <= TOL
    print("  %-42s %6.1f (paper %5.1f) %s" % (tag, got, want, "[OK]" if ok else "[FAIL]"))
    if not ok:
        errors.append("%s: computed %.2f, paper %.2f" % (tag, got, want))


def vals(comp, key):
    return [comp["d%d_s%d" % (d, s)][key]["retention_pct"] for s in SEEDS]


def main():
    comp = json.load(open(CKPT, encoding="utf-8"))["completed"]

    print("=" * 78)
    print("Stage 1: NOTEARS main results (Table 1, 3 seeds, seed-matched)")
    print("=" * 78)
    for d in DIMS:
        s = np.array([comp["d%d_s%d_scratch" % (d, x)]["retention_pct"] for x in SEEDS])
        w = np.array([comp["d%d_s%d_warm" % (d, x)]["retention_pct"] for x in SEEDS])
        cmp("d=%d scratch mean" % d, s.mean(), T1[d][0])
        cmp("d=%d scratch SD" % d, s.std(ddof=1), T1[d][1])
        cmp("d=%d warm mean" % d, w.mean(), T1[d][2])
        cmp("d=%d warm SD" % d, w.std(ddof=1), T1[d][3])

    print()
    print("=" * 78)
    print("Stage 2: DAGMA/GOLEM results (Table 4.4, 3 seeds, seed-matched)")
    print("=" * 78)
    for d in DIMS:
        for i, algo in enumerate(("dagma", "golem")):
            s = np.array([comp["d%d_s%d_%s_scratch" % (d, x, algo)]["retention_pct"]
                          for x in SEEDS])
            w = np.array([comp["d%d_s%d_%s_warm" % (d, x, algo)]["retention_pct"]
                          for x in SEEDS])
            base = 4 * i
            cmp("d=%d %s scratch mean" % (d, algo.upper()), s.mean(), T44[d][base])
            cmp("d=%d %s scratch SD" % (d, algo.upper()), s.std(ddof=1), T44[d][base + 1])
            cmp("d=%d %s warm mean" % (d, algo.upper()), w.mean(), T44[d][base + 2])
            cmp("d=%d %s warm SD" % (d, algo.upper()), w.std(ddof=1), T44[d][base + 3])

    print()
    print("=" * 78)
    print("Stage 3: 45/45 claim (each cell an independent run)")
    print("=" * 78)
    wins = total = 0
    for d in DIMS:
        for s in SEEDS:
            total += 1
            if comp["d%d_s%d_warm" % (d, s)]["retention_pct"] > \
               comp["d%d_s%d_scratch" % (d, s)]["retention_pct"]:
                wins += 1
        for algo in ("dagma", "golem"):
            for s in SEEDS:
                total += 1
                if comp["d%d_s%d_%s_warm" % (d, s, algo)]["retention_pct"] > \
                   comp["d%d_s%d_%s_scratch" % (d, s, algo)]["retention_pct"]:
                    wins += 1
    global checks
    checks += 1
    print("  %-42s %d/%d" % ("warm > scratch", wins, total))
    if wins != total or total != 45:
        errors.append("45/45 claim: got %d/%d" % (wins, total))

    # no cell may be a duplicate of another (the n=1 problem)
    dup = 0
    for d in DIMS:
        for algo in ("dagma", "golem", "notears"):
            if algo == "notears":
                ks = ["d%d_s%d_warm" % (d, s) for s in SEEDS]
            else:
                ks = ["d%d_s%d_%s_warm" % (d, s, algo) for s in SEEDS]
            v = [json.dumps(comp[k], sort_keys=True) for k in ks]
            dup += len(v) - len(set(v))
    checks += 1
    print("  %-42s %d" % ("duplicate warm entries across seeds", dup))
    if dup:
        errors.append("%d warm entries are identical across seeds (n=1 cells)" % dup)

    print()
    print("=" * 78)
    print("Stage 4: cross-domain DepMap transfer")
    print("=" * 78)
    full = json.load(open(FULL, encoding="utf-8"))
    for tag, got, want in [
        ("DepMap source edges", full["depmap"]["edges_03"], 20),
        ("scratch overlap", full["depmap_to_scratch"]["overlap"], 0),
        ("TCGA-warm overlap", full["depmap_to_tcga_warm"]["overlap"], 2),
        ("DepMap-warm overlap", full["depmap_to_dm_warm"]["overlap"], 7),
    ]:
        checks += 1
        ok = got == want
        print("  %-42s %3d (paper %3d) %s" % (tag, got, want, "[OK]" if ok else "[FAIL]"))
        if not ok:
            errors.append("%s: got %d, paper %d" % (tag, got, want))

    print()
    print("=" * 78)
    print("Stage 5: pooling, target-sample size, ten seeds (S9-S11)")
    print("=" * 78)
    mk = json.load(open(os.path.join(BASE, "results", "matched_k_eval.json"), encoding="utf-8"))
    by_cell = {}
    for r in mk:
        by_cell.setdefault((r["d"], r["seed"]), {})[r["arm"]] = r["nrmse_k70"]
    penalties = []
    for key, arms in by_cell.items():
        if "pool" in arms and "warm" in arms:
            penalties.append(arms["pool"] - arms["warm"])
    checks += 1
    ok = len(penalties) == 11 and all(p > 0 for p in penalties)
    print("  %-42s %d cells, %d positive %s"
          % ("pool worse than warm at k=70", len(penalties),
             sum(1 for x in penalties if x > 0), "[OK]" if ok else "[FAIL]"))
    if not ok:
        errors.append("pooling penalty not positive in all cells")
    checks += 1
    lo, hi = min(penalties), max(penalties)
    ok = abs(hi - 0.031) < 5e-4
    print("  %-42s %.3f..%.3f %s" % ("penalty range (paper 0.003..0.031)",
                                     lo, hi, "[OK]" if ok else "[FAIL]"))
    if not ok:
        errors.append("pooling penalty range differs from the SI table")

    st = json.load(open(os.path.join(BASE, "results", "small_target.json"), encoding="utf-8"))
    agg = {}
    for k, v in st.items():
        parts = k.split("_")
        if parts[0] != "d454":
            continue
        agg.setdefault(int(parts[2][1:]), {}).setdefault(parts[3], []).append(
            v["retention_pct"])
    for n, want in ((150, 37.2), (1218, 21.2)):
        sc = sum(agg[n]["scratch"]) / len(agg[n]["scratch"])
        wm = sum(agg[n]["warm"]) / len(agg[n]["warm"])
        checks += 1
        ok = abs((wm - sc) - want) < 0.15
        print("  %-42s %+5.1f (paper %+5.1f) %s"
              % ("gain at n=%d" % n, wm - sc, want, "[OK]" if ok else "[FAIL]"))
        if not ok:
            errors.append("downsample gain at n=%d differs" % n)

    ex = json.load(open(os.path.join(BASE, "results", "bench_v6_extraseeds.json"), encoding="utf-8"))
    ck = json.load(open(os.path.join(BASE, "results", "benchmark_v6_checkpoint.json"),
                        encoding="utf-8"))["completed"]
    wins = tot = 0
    for seed in (42, 123, 456, 789, 1011, 2024, 3141, 4242, 5555, 6789):
        for d in (454, 889):
            k1 = "d%d_s%d_scratch" % (d, seed)
            k2 = "d%d_s%d_warm" % (d, seed)
            src = ck if seed in (42, 123, 456) else ex
            a, b = src.get(k1), src.get(k2)
            if isinstance(a, dict) and isinstance(b, dict):
                tot += 1
                wins += int(b["retention_pct"] > a["retention_pct"])
    checks += 1
    ok = wins == tot == 20
    print("  %-42s %d/%d %s" % ("ten seeds, both dimensions", wins, tot,
                                "[OK]" if ok else "[FAIL]"))
    if not ok:
        errors.append("ten-seed sweep: %d/%d" % (wins, tot))

    print()
    print("=" * 78)
    if errors:
        print("FAILED: %d problem(s) out of %d checks" % (len(errors), checks))
        for e in errors:
            print("   -", e)
        return 1
    print("ALL %d CHECKS PASSED" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
