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
Expression-only TCGA loader (drop-in replacement for load_data in
run_dagma_golem_v2.py).

WHY
---
load_data() in run_dagma_golem_v2.py selects "everything starting with TCGA_
that ends in .tsv and does not mention BRCA".  When TCGA_HNSC_clinical.tsv was
later copied into D:\\NO.1\\cancer_application\\data, that rule silently began
ingesting a clinical table whose columns are sample barcodes
('TCGA-4P-AA8J-01'), and the loader crashed on the first float conversion.

This module uses a positive whitelist instead -- `TCGA_<COHORT>_HiSeqV2.tsv` --
which is exactly the set the original rule admitted when the benchmark was run
(every other .tsv in that directory is either a clinical table or a cohort that
does not match the pattern).

GENE-ORDER GUARD
----------------
Re-deriving the top-d gene set must reproduce the gene ordering used by the
archived source graphs, otherwise every retention number would be measured on
a permuted coordinate system.  verify_gene_order() checks this empirically:
the known source graph must explain substantially more variance on the freshly
loaded matrix than on a column-permuted copy of it.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

TCGA_DIR = DATA_DIR


def _expression_files():
    out = []
    for f in sorted(os.listdir(TCGA_DIR)):
        if not f.startswith("TCGA_"):
            continue
        if not f.endswith("_HiSeqV2.tsv"):      # drops .gz and *_clinical.tsv
            continue
        if "BRCA" in f:
            continue                             # BRCA is the target cohort
        out.append(f)
    return out


def load_data(d, verbose=False):
    """Same contract as run_dagma_golem_v2.load_data: X_tcga, X_brca, d_actual, genes."""
    files = _expression_files()
    if verbose:
        print("  source cohorts (%d): %s" % (len(files), files))

    brca = pd.read_csv(os.path.join(TCGA_DIR, "TCGA_BRCA_HiSeqV2.tsv"),
                       sep="\t", index_col=0).T
    others = []
    for f in files:
        try:
            others.append(pd.read_csv(os.path.join(TCGA_DIR, f),
                                      sep="\t", index_col=0).T)
        except Exception as e:                    # pragma: no cover
            if verbose:
                print("  skip %s (%s)" % (f, e))
            continue
    if not others:
        raise RuntimeError("no source cohorts found in %s" % TCGA_DIR)
    tcga_all = pd.concat(others, axis=0)

    mads = np.array([median_abs_deviation(tcga_all.iloc[:, j].values.astype(np.float64))
                     for j in range(tcga_all.shape[1])])
    mads = np.nan_to_num(mads, nan=0)
    top = list(tcga_all.columns[np.argsort(mads)[-d:]])

    brca_g = [g for g in top if g in brca.columns]
    d_actual = len(brca_g)
    ci = [list(tcga_all.columns).index(g) for g in brca_g]

    X_tcga = tcga_all.iloc[:, ci].values.astype(np.float64)
    X_brca = brca[brca_g].values.astype(np.float64)
    X_tcga = np.nan_to_num(X_tcga, nan=0)
    X_brca = np.nan_to_num(X_brca, nan=0)
    X_tcga = (X_tcga - X_tcga.mean(0)) / (X_tcga.std(0) + 1e-12)
    X_brca = (X_brca - X_brca.mean(0)) / (X_brca.std(0) + 1e-12)
    return X_tcga, X_brca, d_actual, brca_g


def _nres(X, W):
    """Normalised residual of the linear SEM X ~ X W."""
    R = X - X @ W
    return float((R ** 2).mean() / (X ** 2).mean())


def verify_gene_order(X, W, rng=None):
    """
    Is W expressed in the same column order as X?
    Returns (nres_true, nres_permuted).  Correct ordering => nres_true clearly
    smaller.  A permuted copy is the null.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    d = X.shape[1]
    perm = rng.permutation(d)
    return _nres(X, W), _nres(X[:, perm], W)
