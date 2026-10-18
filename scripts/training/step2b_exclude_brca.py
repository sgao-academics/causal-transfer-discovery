# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(2):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""Rebuild TCGA matrix excluding BRCA for held-out evaluation."""
import os, json, numpy as np, pandas as pd
from scipy.stats import median_abs_deviation

OUT = os.path.join(PKG_ROOT, "data")
tcga_dir = DATA_DIR

# Load all non-BRCA TCGA
tcga_files = sorted([f for f in os.listdir(tcga_dir)
                     if f.startswith('TCGA_') and f.endswith('_HiSeqV2.tsv')
                     and 'BRCA' not in f])
parts = []
for f in tcga_files:
    df = pd.read_csv(os.path.join(tcga_dir, f), sep='\t', index_col=0).T
    parts.append(df)
tcga_nobrca = pd.concat(parts, axis=0)

# Load BRCA separately
brca = pd.read_csv(os.path.join(tcga_dir, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0).T

# Find common genes with DepMap
common = sorted(set(open(os.path.join(OUT, 'common_genes.txt')).read().strip().split('\n')))
tcga_common = [g for g in common if g in tcga_nobrca.columns]
tcga_nobrca = tcga_nobrca[tcga_common].values.astype(np.float64)
tcga_nobrca = np.nan_to_num(tcga_nobrca, nan=0)
tcga_nobrca = (tcga_nobrca - tcga_nobrca.mean(0)) / (tcga_nobrca.std(0) + 1e-12)
np.save(os.path.join(OUT, 'X_tcga_nobrca.npy'), tcga_nobrca)
print(f'TCGA (no BRCA): {tcga_nobrca.shape}')

# Save BRCA separately
brca_common = [g for g in common if g in brca.columns]
brca = brca[brca_common].values.astype(np.float64)
brca = np.nan_to_num(brca, nan=0)
brca = (brca - brca.mean(0)) / (brca.std(0) + 1e-12)
np.save(os.path.join(OUT, 'X_brca_heldout.npy'), brca)
np.save(os.path.join(OUT, 'brca_genes.npy'), np.array(brca_common, dtype=object))
print(f'BRCA held-out: {brca.shape} ({len(brca_common)} genes)')
print('DONE')
