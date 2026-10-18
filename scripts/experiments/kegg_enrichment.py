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
KEGG pathway enrichment analysis for transferred causal edges.
Uses STRING API for enrichment (no local KEGG download needed).
Analyzes: (1) Top-200 edges per method, (2) Transferred-but-not-scratch edges.
"""
import numpy as np, json, os, requests, time, sys

ROOT = PKG_ROOT
CACHE_DIR = os.path.join(ROOT, 'results', 'cache')
OUT_PATH = os.path.join(ROOT, 'results', 'kegg_enrichment.json')

def load_existing_w(d=454):
    """Load existing W matrices from full_experiment.json era."""
    res_dir = os.path.join(ROOT, 'results')
    W_src = np.load(os.path.join(res_dir, 'W_src_d500.npy'))
    W_scr = np.load(os.path.join(res_dir, 'W_scratch_d454.npy'))
    W_warm = np.load(os.path.join(res_dir, 'W_tcga_warm_d454.npy'))
    W_dm = np.load(os.path.join(res_dir, 'W_dm_warm_d454.npy'))
    return W_src, W_scr, W_warm, W_dm

def get_top_edges(W, tau=0.3, top_k=200):
    """Get top edges by absolute weight. Returns list of (i,j,weight)."""
    d = W.shape[0]
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i,j]) > tau:
                edges.append((i, j, W[i,j]))
    edges.sort(key=lambda x: -abs(x[2]))
    return edges[:top_k] if len(edges) >= top_k else edges

def get_transferred_only(W_src, W_scratch, W_warm, tau=0.3):
    """Get edges preserved by warm-start but missed by scratch."""
    d = W_src.shape[0]
    src_edges = set()
    scr_edges = set()
    for i in range(d):
        for j in range(d):
            if i != j and abs(W_src[i,j]) > tau: src_edges.add((i,j))
            if i != j and abs(W_scratch[i,j]) > tau: scr_edges.add((i,j))
    transferred = set()
    for i,j in src_edges:
        if (i,j) not in scr_edges and abs(W_warm[i,j]) > tau:
            transferred.add((i,j, W_warm[i,j]))
    return sorted(transferred, key=lambda x: -abs(x[2]))

# Load gene list
with open(os.path.join(ROOT, 'data', 'gene_list.txt')) as f:
    all_genes = f.read().strip().split('\n')

# Load gene_idx from checkpoints
ck_dir = os.path.join(ROOT, 'checkpoints', 'causal-gpt-v1-d500')
gene_idx = np.load(os.path.join(ck_dir, 'gene_idx.npy'))
available_genes = [all_genes[i] for i in gene_idx]

print(f'Genes available: {len(available_genes)}')
print(f'First 5: {available_genes[:5]}')

# Load W matrices
W_src, W_scr, W_warm, W_dm = load_existing_w()
d_actual = min(W_src.shape[0], W_scr.shape[0], W_warm.shape[0], W_dm.shape[0])
# Truncate all to same dimension
W_src = W_src[:d_actual, :d_actual]
W_scr = W_scr[:d_actual, :d_actual]
W_warm = W_warm[:d_actual, :d_actual]
W_dm = W_dm[:d_actual, :d_actual]
available_genes = available_genes[:d_actual]
print(f'Common d: {d_actual}, W_src edges: {int((np.abs(W_src)>0.3).sum())}')

# Extract edge sets
src_edges = get_top_edges(W_src, top_k=200)
scr_edges = get_top_edges(W_scr, top_k=200)
warm_edges = get_top_edges(W_warm, top_k=200)
dm_edges = get_top_edges(W_dm, top_k=200)
transferred = get_transferred_only(W_src, W_scr, W_warm)

print(f'Top edges: src={len(src_edges)}, scr={len(scr_edges)}, warm={len(warm_edges)}, dm={len(dm_edges)}')
print(f'Transferred-only edges: {len(transferred)}')

# Map edge indices to gene symbols
def edges_to_gene_list(edges):
    genes = set()
    for i,j,_ in edges:
        if i < d_actual and j < d_actual:
            genes.add(available_genes[i])
            genes.add(available_genes[j])
    return list(genes)

# STRING enrichment API
def enrich_string(genes, method_name):
    """Use STRING functional enrichment API."""
    url = "https://string-db.org/api/json/enrichment"
    params = {
        "identifiers": "%0d".join(genes[:500]),  # max 500
        "species": 9606,
        "caller_identity": "causal_gpt_benchmark"
    }
    try:
        r = requests.post(url, data={"identifiers": "%0d".join(genes[:500]), "species": "9606"})
        if r.status_code == 200:
            results = r.json()
            # Filter KEGG pathways
            kegg = [x for x in results if x.get('category') == 'KEGG']
            return kegg[:20]  # top 20
        else:
            print(f'  STRING API error for {method_name}: {r.status_code}')
            return []
    except Exception as e:
        print(f'  STRING API error for {method_name}: {e}')
        return []

# Run enrichment for each edge set
results = {}

for name, edges in [('source', src_edges), ('scratch', scr_edges), ('warm', warm_edges), 
                     ('dm_warm', dm_edges), ('transferred_only', transferred)]:
    print(f'\nEnriching: {name} ({len(edges)} edges)...')
    genes = edges_to_gene_list(edges)
    print(f'  Genes: {len(genes)}')
    enriched = enrich_string(genes, name)
    results[name] = {
        'n_edges': len(edges),
        'n_genes': len(genes),
        'top_pathways': [{'term': e['term'], 'description': e['description'], 
                          'fdr': e.get('fdr', 1.0), 'n_genes': e.get('number_of_genes', 0),
                          'genes': e.get('inputGenes', '')[:100]} 
                         for e in enriched[:10]]
    }
    for e in enriched[:5]:
        print(f'    {e.get("description","?")} FDR={e.get("fdr",1.0):.1e} n={e.get("number_of_genes",0)}')
    time.sleep(1)  # rate limit

# Save
with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved to {OUT_PATH}')

# Print summary table
print('\n' + '='*60)
print('KEGG ENRICHMENT SUMMARY')
print('='*60)
for name, data in results.items():
    print(f'\n{name} ({data["n_edges"]} edges, {data["n_genes"]} genes):')
    for p in data['top_pathways'][:5]:
        print(f'  {p["term"]:12s} {p["description"][:60]:60s} FDR={p["fdr"]:.1e}')
