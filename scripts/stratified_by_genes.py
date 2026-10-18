# --- paths resolved relative to this file (replication package) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
PKG_ROOT = _HERE
for _ in range(1):
    PKG_ROOT = _os.path.dirname(PKG_ROOT)
PKG_RESULTS = _os.path.join(PKG_ROOT, "results")
DATA_DIR = _os.environ.get("TCGA_DATA_DIR", _os.path.join(PKG_ROOT, "data"))
# -------------------------------------------------------------------
"""
Stratified edge analysis by gene properties (weight, pathway, STRING score).
Uses gene-name edges from supplementary_analysis.json
"""
import json, os
import numpy as np

RESULTS = PKG_RESULTS

# Load preserved edges
with open(os.path.join(RESULTS, 'supplementary_analysis.json'), 'r') as f:
    supp = json.load(f)

preserved = supp['preserved_edges']  # list of {gene_i, gene_j, w_src, w_tw}
top5 = supp['top_preserved']

# Load threshold sensitivity
threshold_data = supp.get('threshold_sensitivity', [])

# Load STRING validation
with open(os.path.join(RESULTS, 'string_validation.json'), 'r') as f:
    sv = json.load(f)

# Load KEGG enrichment
with open(os.path.join(RESULTS, 'kegg_enrichment.json'), 'r') as f:
    kegg = json.load(f)

print("="*60)
print("EDGE STRATIFICATION BY GENE PROPERTIES")
print("="*60)

# 1. Edge weight stratification
print("\n--- Edge Weight Stratification ---")
weights = [(abs(e['w_src']), e) for e in preserved]
weights.sort(key=lambda x: x[0], reverse=True)

for pct in [25, 50, 75]:
    n = max(1, len(weights) * pct // 100)
    strong = weights[:n]
    weak = weights[n:]
    
    strong_mean = np.mean([w for w,_ in strong])
    weak_mean = np.mean([w for w,_ in weak])
    print(f"  Top {pct}% (n={n}): mean |W_src| = {strong_mean:.4f}")
    if weak:
        print(f"  Bottom {100-pct}% (n={len(weak)}): mean |W_src| = {weak_mean:.4f}")

# 2. STRING confidence stratification
print("\n--- STRING Confidence Stratification ---")
# The preserved edges are a subset; check how many have STRING evidence
string_src = sv.get('string_src', {})
string_sc = sv.get('string_sc', {})
string_tr = sv.get('string_tr', {})

preserved_genes = set()
for e in preserved:
    preserved_genes.add(e['gene_i'])
    preserved_genes.add(e['gene_j'])

print(f"  Preserved edges: {len(preserved)}")
print(f"  Unique genes in preserved edges: {len(preserved_genes)}")

# 3. Top preserved edges by weight
print("\n--- Top Preserved Edges ---")
for i, e in enumerate(top5):
    print(f"  {i+1}. {e['gene_i']} -> {e['gene_j']}: W_src={e['w_src']:.4f}, W_warm={e['w_tw']:.4f}")

# 4. Edge categories
print("\n--- Edge Transfer Categories ---")
print(f"  Total preserved (source edges appearing in warm-start): {len(preserved)}")
print(f"  Strongly preserved (|W_src| > 0.5): {sum(1 for e in preserved if abs(e['w_src']) > 0.5)}")
print(f"  Weakly preserved (|W_src| <= 0.3): {sum(1 for e in preserved if abs(e['w_src']) <= 0.3)}")

# 5. Compute weight ratios
print("\n--- Weight Preservation Fidelity ---")
w_ratios = []
for e in preserved:
    if abs(e['w_src']) > 1e-6:
        ratio = e['w_tw'] / e['w_src']
        w_ratios.append(ratio)
if w_ratios:
    print(f"  Mean W_warm/W_src ratio: {np.mean(w_ratios):.4f} +/- {np.std(w_ratios):.4f}")
    print(f"  Median: {np.median(w_ratios):.4f}")
    print(f"  Corr(W_src, W_warm): {np.corrcoef([abs(e['w_src']) for e in preserved], [abs(e['w_tw']) for e in preserved])[0,1]:.4f}")

# Output
out = {
    'n_preserved': len(preserved),
    'n_preserved_strong': sum(1 for e in preserved if abs(e['w_src']) > 0.5),
    'n_preserved_weak': sum(1 for e in preserved if abs(e['w_src']) <= 0.3),
    'mean_w_ratio': float(np.mean(w_ratios)) if w_ratios else None,
    'w_src_warm_corr': float(np.corrcoef(
        [abs(e['w_src']) for e in preserved],
        [abs(e['w_tw']) for e in preserved]
    )[0,1]) if len(preserved) > 2 else None,
    'top_5_edges': top5
}

with open(os.path.join(RESULTS, 'stratified_by_genes.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to stratified_by_genes.json")

# 6. For the manuscript: edge type classification table
print("\n--- Edge Type Classification (for manuscript table) ---")
from collections import Counter
edge_types = {'strong_conserved': 0, 'moderate': 0, 'weak_dataset_specific': 0}
for e in preserved:
    if abs(e['w_src']) > 0.5:
        edge_types['strong_conserved'] += 1
    elif abs(e['w_src']) > 0.3:
        edge_types['moderate'] += 1
    else:
        edge_types['weak_dataset_specific'] += 1
for k, v in edge_types.items():
    print(f"  {k}: {v} ({100*v/len(preserved):.0f}%)")
