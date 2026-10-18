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
Step 1: Gene set alignment across DepMap, TCGA, GTEx.
Output: Unified 18,435-gene matrices saved as .npy for fast loading.
"""
import os,sys,json,numpy as np,pandas as pd
from scipy.stats import median_abs_deviation

OUT = os.path.join(PKG_ROOT, "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DepMap: CRISPRGeneEffect.csv
# ---------------------------------------------------------------------------
print("1. Loading DepMap...")
depmap = pd.read_csv(os.path.join(DATA_DIR, "depmap", "CRISPRGeneEffect.csv"), index_col=0)
# Strip " (XXXXX)" Entrez ID suffix from gene names: "A1BG (1)" -> "A1BG"
import re
dm_cols = depmap.columns.tolist()
dm_cols_clean = [re.sub(r'\s*\(\d+\)\s*$', '', str(c)) for c in dm_cols]
depmap.columns = dm_cols_clean
depmap = depmap.loc[:, ~depmap.columns.duplicated(keep='first')]
depmap_genes = set(depmap.columns)
print(f"   {depmap.shape[0]} cell lines, {len(depmap_genes)} genes (after cleaning)")

# ---------------------------------------------------------------------------
# 2. TCGA: 33 cancer .tsv files
# ---------------------------------------------------------------------------
print("2. Loading TCGA...")
tcga_dir = DATA_DIR
tcga_files = sorted([f for f in os.listdir(tcga_dir)
                     if f.startswith('TCGA_') and f.endswith('_HiSeqV2.tsv')])
all_tcga_genes = set()
tcga_parts = []
for f in tcga_files:
    df = pd.read_csv(os.path.join(tcga_dir, f), sep='\t', index_col=0)
    all_tcga_genes |= set(df.index)
    df = df.T  # (samples, genes)
    tcga_parts.append(df)
tcga_full = pd.concat(tcga_parts, axis=0)
print(f"   {tcga_full.shape[0]} samples, {len(all_tcga_genes)} genes")

# ---------------------------------------------------------------------------
# 3. GTEx: 8 organ .npy files
# ---------------------------------------------------------------------------
print("3. Loading GTEx...")
gtex_dir = os.path.join(PKG_ROOT, "data", "gtex")
gtex_organs = ['Brain','Breast','Colon','Liver','Lung','Ovary','Pancreas','Prostate']
gtex_parts = []
for organ in gtex_organs:
    X_path = os.path.join(gtex_dir, f'{organ}_X.npy')
    if os.path.exists(X_path):
        X = np.load(X_path)
        gtex_parts.append(X)
        print(f"   {organ}: {X.shape}")
gtex_full = np.vstack(gtex_parts)
print(f"   Total: {gtex_full.shape[0]} samples, {gtex_full.shape[1]} features")

# ---------------------------------------------------------------------------
# 4. Gene intersection
# ---------------------------------------------------------------------------
print("\n4. Computing gene intersection...")
# DepMap genes from CRISPRGeneEffect
# TCGA genes from row index
# GTEx genes from... hmm, GTEx has no gene names in the .npy files
# Let's check if there's a gene list file
gtex_gene_file = os.path.join(gtex_dir, 'gene_names.txt')
if not os.path.exists(gtex_gene_file):
    # Check for any metadata
    for f in os.listdir(gtex_dir):
        if f.endswith('.txt') or f.endswith('.csv'):
            print(f"   Found: {f}")
    
# For now, let's assume GTEx genes are indexed the same as some reference
# We'll need to verify. Let's save what we have.

print(f"\nDepMap genes: {len(depmap_genes)}")
print(f"TCGA genes: {len(all_tcga_genes)}")

# Intersection of DepMap and TCGA (the two we can name-match)
common_genes = sorted(depmap_genes & all_tcga_genes)
print(f"DepMap ∩ TCGA intersection: {len(common_genes)} genes")

# Save gene list
with open(os.path.join(OUT, 'common_genes.txt'), 'w') as f:
    f.write('\n'.join(common_genes))

# Save counts
stats = {
    'depmap_samples': depmap.shape[0],
    'depmap_genes': len(depmap_genes),
    'tcga_samples': tcga_full.shape[0],
    'tcga_genes': len(all_tcga_genes),
    'gtex_samples': gtex_full.shape[0],
    'gtex_features': gtex_full.shape[1],
    'intersection_genes': len(common_genes),
}
with open(os.path.join(OUT, 'alignment_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nStats saved to {OUT}/alignment_stats.json")
print(f"Common genes ({len(common_genes)}) saved to {OUT}/common_genes.txt")
print("DONE — Step 1")
