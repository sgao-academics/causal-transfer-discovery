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
Step 2: Build unified gene expression/dependency matrices for pre-training.
Output: data/X_depmap.npy, data/X_tcga.npy, data/gene_list.npy
"""
import os, json, re, numpy as np, pandas as pd
from scipy.stats import median_abs_deviation

OUT = os.path.join(PKG_ROOT, "data")
os.makedirs(OUT, exist_ok=True)

# Load common genes
common_genes = sorted(open(os.path.join(OUT, 'common_genes.txt')).read().strip().split('\n'))
print(f'Common genes: {len(common_genes)}')
print(f'Sample: {common_genes[:5]}')

# ---------------------------------------------------------------------------
# 1. DepMap matrix
# ---------------------------------------------------------------------------
print("\n1. Building DepMap matrix...")
depmap = pd.read_csv(os.path.join(DATA_DIR, "depmap", "CRISPRGeneEffect.csv"), index_col=0)
dm_cols = [re.sub(r'\s*\(\d+\)\s*$', '', str(c)) for c in depmap.columns]
depmap.columns = dm_cols
depmap = depmap.loc[:, ~depmap.columns.duplicated(keep='first')]

# Subset to common genes (some DepMap genes won't be in TCGA)
dm_common = [g for g in common_genes if g in depmap.columns]
print(f'   {len(dm_common)}/{len(common_genes)} genes in DepMap')
X_depmap = depmap[dm_common].values.astype(np.float64)
X_depmap = np.nan_to_num(X_depmap, nan=0.0)
# Standardize per gene (column)
X_depmap = (X_depmap - X_depmap.mean(axis=0)) / (X_depmap.std(axis=0) + 1e-12)
np.save(os.path.join(OUT, 'X_depmap.npy'), X_depmap)
print(f'   Saved: {X_depmap.shape} ({X_depmap.nbytes//1024//1024}MB)')

# ---------------------------------------------------------------------------
# 2. TCGA matrix
# ---------------------------------------------------------------------------
print("\n2. Building TCGA matrix...")
tcga_dir = DATA_DIR
tcga_files = sorted([f for f in os.listdir(tcga_dir)
                     if f.startswith('TCGA_') and f.endswith('_HiSeqV2.tsv')])
tcga_parts = []
for f in tcga_files:
    df = pd.read_csv(os.path.join(tcga_dir, f), sep='\t', index_col=0)
    tcga_parts.append(df.T)
tcga_full = pd.concat(tcga_parts, axis=0)
tcga_full = tcga_full.loc[:, ~tcga_full.columns.duplicated(keep='first')]

tcga_common = [g for g in common_genes if g in tcga_full.columns]
print(f'   {len(tcga_common)}/{len(common_genes)} genes in TCGA')
X_tcga = tcga_full[tcga_common].values.astype(np.float64)
X_tcga = np.nan_to_num(X_tcga, nan=0.0)
X_tcga = (X_tcga - X_tcga.mean(axis=0)) / (X_tcga.std(axis=0) + 1e-12)
np.save(os.path.join(OUT, 'X_tcga.npy'), X_tcga)
print(f'   Saved: {X_tcga.shape} ({X_tcga.nbytes//1024//1024}MB)')

# ---------------------------------------------------------------------------
# 3. Final intersection (genes present in BOTH matrices)
# ---------------------------------------------------------------------------
final_genes = sorted(set(dm_common) & set(tcga_common))
print(f'\n3. Final gene intersection: {len(final_genes)} genes')

# Re-save with final gene set
dm_idx = [dm_common.index(g) for g in final_genes]
tcga_idx = [tcga_common.index(g) for g in final_genes]
X_dm = X_depmap[:, dm_idx]
X_tc = X_tcga[:, tcga_idx]

np.save(os.path.join(OUT, 'X_depmap_final.npy'), X_dm)
np.save(os.path.join(OUT, 'X_tcga_final.npy'), X_tc)

# Save gene list as text and npy
with open(os.path.join(OUT, 'gene_list.txt'), 'w') as f:
    f.write('\n'.join(final_genes))
np.save(os.path.join(OUT, 'gene_list.npy'), np.array(final_genes, dtype=object))

# Summary
summary = {
    'depmap_samples': X_dm.shape[0],
    'tcga_samples': X_tc.shape[0],
    'n_genes': len(final_genes),
    'depmap_shape': list(X_dm.shape),
    'tcga_shape': list(X_tc.shape),
}
with open(os.path.join(OUT, 'matrix_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'   DepMap: {X_dm.shape}')
print(f'   TCGA:   {X_tc.shape}')
print(f'   Total samples: {X_dm.shape[0] + X_tc.shape[0]}')
print(f'   Gene list saved: {len(final_genes)} genes')
print('DONE — Step 2')
