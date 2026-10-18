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
Stratified edge analysis for JBCB revision (Reviewer #1, Item 3 + 5)
Analyzes which edge properties predict successful transfer.
"""
import json, os, sys
import numpy as np

RESULTS = PKG_RESULTS

def load_w(name):
    """Load a weight matrix from results dir"""
    pats = [f'W_{name}_d', f'W_{name}']
    for f in sorted(os.listdir(RESULTS)):
        for p in pats:
            if f.startswith(p) and f.endswith('.npy'):
                return np.load(os.path.join(RESULTS, f))
    raise FileNotFoundError(f"No .npy found for {name}")

def edges_from_w(W, threshold=0.3):
    """Extract edges (i,j,weight) with |weight| > threshold"""
    edges = []
    d = W.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i,j]) > threshold:
                edges.append((i, j, W[i,j]))
    return edges

def main():
    print("="*60)
    print("STRATIFIED EDGE ANALYSIS")
    print("="*60)
    
    # Load matrices at d=454 (main analysis dimension)
    try:
        W_src = load_w('src_d454')  # source (TCGA)
        W_scr = load_w('scratch_d454')  # scratch (BRCA)
        W_warm = load_w('warm_d454')  # warm-start (TCGA->BRCA)
    except:
        # Fallback to d=500
        W_src = load_w('src_d500')
        W_scr = load_w('sc_d889')[:500, :500]  # subset
        W_warm = load_w('dm_warm_d454')
        print("Using fallback d=454 matrices")
    
    d = W_src.shape[0]
    tau = 0.3
    
    # Extract all edges
    src_edges = edges_from_w(W_src, tau)
    scr_edges = edges_from_w(W_scr, tau)
    warm_edges = edges_from_w(W_warm, tau)
    
    # Edge sets
    src_set = {(i,j) for i,j,_ in src_edges}
    scr_set = {(i,j) for i,j,_ in scr_edges}
    warm_set = {(i,j) for i,j,_ in warm_edges}
    
    # Three categories
    universal = src_set & warm_set  # edges in both source and warm-start
    lost = src_set - warm_set       # in source but NOT transferred
    new = warm_set - src_set        # discovered only by warm-start
    
    print(f"\nSource edges: {len(src_set)}")
    print(f"Scratch edges: {len(scr_set)} (overlap w/ source: {len(src_set & scr_set)})")
    print(f"Warm-start edges: {len(warm_set)} (overlap w/ source: {len(universal)})")
    print(f"Lost edges (source, not transferred): {len(lost)}")
    print(f"New edges (warm-start only): {len(new)}")
    
    # === STRATIFICATION 1: By edge weight ===
    print("\n" + "-"*40)
    print("STRATIFICATION BY SOURCE EDGE WEIGHT")
    print("-"*40)
    
    weights = [abs(w) for _,_,w in src_edges]
    percentiles = [25, 50, 75, 90]
    
    for p in percentiles:
        thresh = np.percentile(weights, p)
        strong_edges = {(i,j) for i,j,w in src_edges if abs(w) >= thresh}
        weak_edges = {(i,j) for i,j,w in src_edges if abs(w) < thresh}
        
        strong_transferred = len(strong_edges & warm_set)
        weak_transferred = len(weak_edges & warm_set)
        
        strong_rate = 100 * strong_transferred / len(strong_edges) if strong_edges else 0
        weak_rate = 100 * weak_transferred / len(weak_edges) if weak_edges else 0
        
        print(f"  Percentile {p} ({thresh:.3f}):")
        print(f"    Strong (|W|>={thresh:.3f}): {strong_transferred}/{len(strong_edges)} ({strong_rate:.1f}%) transferred")
        print(f"    Weak   (|W|<{thresh:.3f}): {weak_transferred}/{len(weak_edges)} ({weak_rate:.1f}%) transferred")
        print(f"    Delta: {strong_rate - weak_rate:+.1f} pp")
    
    # === STRATIFICATION 2: Edge property summary ===
    print("\n" + "-"*40)
    print("EDGE PROPERTY SUMMARY")
    print("-"*40)
    
    # Compare weight distributions
    universal_weights = [abs(W_src[i,j]) for i,j in universal]
    lost_weights = [abs(W_src[i,j]) for i,j in lost if (i,j) in {(a,b) for a,b,_ in src_edges}]
    
    if universal_weights:
        print(f"  Universal edges: mean |W| = {np.mean(universal_weights):.4f} +/- {np.std(universal_weights):.4f}")
    if lost_weights:
        print(f"  Lost edges:      mean |W| = {np.mean(lost_weights):.4f} +/- {np.std(lost_weights):.4f}")
    
    # Correlation between source and warm-start weights for universal edges
    src_wts = [W_src[i,j] for i,j in universal]
    warm_wts = [W_warm[i,j] for i,j in universal]
    if len(src_wts) > 1:
        corr = np.corrcoef(src_wts, warm_wts)[0,1]
        print(f"  Source vs Warm-start weight correlation (universal edges): r = {corr:.4f}")
    
    # === STRATIFICATION 3: Scratch vs Warm-start overlap ===
    print("\n" + "-"*40)
    print("OVERLAP ANALYSIS")
    print("-"*40)
    
    both = scr_set & warm_set
    scr_only = scr_set - warm_set
    warm_only = warm_set - scr_set
    
    print(f"  Both methods: {len(both)}")
    print(f"  Scratch only: {len(scr_only)}")
    print(f"  Warm-start only: {len(warm_only)}")
    
    # How many scratch edges are "wrong" (= not in source)?
    scr_not_in_src = scr_set - src_set
    warm_not_in_src = warm_set - src_set
    print(f"  Scratch edges NOT in source: {len(scr_not_in_src)}")
    print(f"  Warm-start edges NOT in source: {len(warm_not_in_src)}")
    
    # === OUTPUT ===
    out_path = os.path.join(RESULTS, 'stratified_edge_analysis.json')
    output = {
        'd': d, 'tau': tau,
        'counts': {
            'source_edges': len(src_set),
            'scratch_edges': len(scr_set),
            'warm_edges': len(warm_set),
            'universal': len(universal),
            'lost': len(lost),
            'new': len(new),
            'both_methods': len(both),
            'scratch_only': len(scr_only),
            'warm_only': len(warm_only)
        },
        'weight_stratification': {
            'universal_mean_weight': float(np.mean(universal_weights)) if universal_weights else None,
            'lost_mean_weight': float(np.mean(lost_weights)) if lost_weights else None,
            'src_warm_correlation': float(corr) if len(src_wts) > 1 else None
        },
        'top_preserved': sorted([
            {'edge': f'{i}->{j}', 'w_src': float(W_src[i,j]), 'w_warm': float(W_warm[i,j])}
            for i,j in universal
        ], key=lambda x: abs(x['w_src']), reverse=True)[:10]
    }
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput saved to: {out_path}")

if __name__ == '__main__':
    main()
