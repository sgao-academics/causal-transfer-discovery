# -*- coding: utf-8 -*-
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
stratification_tau03.py -- Unified edge-stratification table at tau = 0.3.

Addresses JBCB-1496 reviewer point 1: combine the threshold table (main-text
Table 3) and the pathway-membership statements into a single table reporting,
for each source-edge strength band at the operating threshold tau = 0.3:
  - number of source edges,
  - scratch retention (%),
  - warm-start retention (%),
  - KEGG pathway membership of the genes touched by those edges.

Inputs (d = 454, TCGA -> BRCA):
  results/W_src_d454.npy        source graph W_S
  results/W_scratch_d454.npy    scratch graph
  results/W_tcga_warm_d454.npy  warm-start graph
  data/genecol_brca454.npy      gene symbols in column order  (454)

Outputs:
  results/edge_stratification_tau03.json
  results/kegg_gene2path.json   (cached gene -> pathway annotation)
"""
import io
import json
import os
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np

BASE = PKG_ROOT
RES = os.path.join(BASE, "results")
TAU = 0.3
BANDS = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]


def get(url, timeout=30, tries=3):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            return urllib.request.urlopen(req, timeout=timeout).read().decode('utf-8', 'ignore')
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise last


def main():
    genes = [str(g) for g in np.load(os.path.join(BASE, 'data', 'genecol_brca454.npy'),
                                     allow_pickle=True)]
    d = len(genes)
    # The TCGA->BRCA source graph actually used by the main experiment
    # (full_experiment.json: tcga edges_03 = 73) is W_tcga_d454.npy.
    src_candidates = [os.path.join(RES, 'W_tcga_d454.npy'),
                      os.path.join(RES, 'cache', 'src_d454_s42.npy')]
    src_path = next(p for p in src_candidates if os.path.exists(p))
    print('using source graph: %s' % src_path)
    W_src = np.load(src_path)
    W_scr = np.load(os.path.join(RES, 'W_scratch_d454.npy'))
    W_warm = np.load(os.path.join(RES, 'W_tcga_warm_d454.npy'))
    assert W_src.shape[0] == d, (W_src.shape, d)

    # ---- source edges at tau ----
    src = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W_src[i, j]) > TAU:
                src.append((i, j, float(W_src[i, j])))
    out = {'d': d, 'tau': TAU, 'n_source_edges': len(src)}

    # retention per edge
    def kept(W, i, j):
        return abs(W[i, j]) > TAU

    scr_kept = sum(1 for i, j, _ in src if kept(W_scr, i, j))
    warm_kept = sum(1 for i, j, _ in src if kept(W_warm, i, j))
    out['scratch_overlap'] = scr_kept
    out['scratch_retention_pct'] = round(100.0 * scr_kept / len(src), 1)
    out['warm_overlap'] = warm_kept
    out['warm_retention_pct'] = round(100.0 * warm_kept / len(src), 1)
    print('source=%d scratch=%d(%.1f%%) warm=%d(%.1f%%)'
          % (len(src), scr_kept, out['scratch_retention_pct'],
             warm_kept, out['warm_retention_pct']))

    # ---- strength bands ----
    bands = []
    for lo, hi in BANDS:
        sel = [(i, j, w) for i, j, w in src if lo <= abs(w) < hi]
        if not sel:
            continue
        sk = sum(1 for i, j, _ in sel if kept(W_scr, i, j))
        wk = sum(1 for i, j, _ in sel if kept(W_warm, i, j))
        bands.append({
            'band': '[%.1f, %.1f)' % (lo, hi),
            'n': len(sel),
            'scratch_kept': sk,
            'scratch_pct': round(100.0 * sk / len(sel), 1),
            'warm_kept': wk,
            'warm_pct': round(100.0 * wk / len(sel), 1),
            'genes': sorted({genes[i] for i, j, _ in sel} | {genes[j] for i, j, _ in sel}),
        })
        print('  band %s n=%d scratch=%.1f%% warm=%.1f%%'
              % (bands[-1]['band'], len(sel), bands[-1]['scratch_pct'], bands[-1]['warm_pct']))
    out['bands'] = bands

    # ---- KEGG annotation of all genes touched by source edges ----
    all_genes = sorted({genes[i] for i, j, _ in src} | {genes[j] for i, j, _ in src})
    cache_path = os.path.join(RES, 'kegg_gene2path.json')
    if os.path.exists(cache_path):
        g2p = json.load(open(cache_path))
        print('loaded KEGG cache (%d genes)' % len(g2p))
    else:
        g2p = {}
        print('querying KEGG for %d genes...' % len(all_genes))
        for k, g in enumerate(all_genes):
            try:
                # /find/hsa/<symbol> returns 'hsa:<entrez>\tSYMBOL; description'
                t = get('https://rest.kegg.jp/find/hsa/%s' % g)
                if not t.strip():
                    g2p[g] = {'entrez': None, 'pathways': []}
                    continue
                entrez = t.split('\t')[0].strip().split(':')[1].split(';')[0]
                t2 = get('https://rest.kegg.jp/link/pathway/hsa:%s' % entrez)
                pws = sorted({ln.split('\t')[1].split(':')[1].strip()
                              for ln in t2.strip().split('\n') if '\t' in ln})
                g2p[g] = {'entrez': entrez, 'pathways': pws}
            except Exception as e:
                g2p[g] = {'entrez': None, 'pathways': [], 'error': repr(e)[:80]}
            if k % 20 == 0:
                print('  %d/%d' % (k, len(all_genes)), flush=True)
            time.sleep(0.05)
        json.dump(g2p, open(cache_path, 'w'), indent=1)
    out['n_genes_annotated'] = sum(1 for g, v in g2p.items() if v.get('pathways'))

    # pathway names
    names = {}
    try:
        t = get('https://rest.kegg.jp/list/pathway/hsa')
        for ln in t.strip().split('\n'):
            if '\t' in ln:
                pid, nm = ln.split('\t', 1)
                pid = pid.split(':')[-1]           # 'path:hsa05200' or 'hsa05200'
                names[pid] = nm.split(' - Homo sapiens')[0]
        json.dump(names, open(os.path.join(RES, 'kegg_pathway_names.json'), 'w'), indent=1)
    except Exception as e:
        print('pathway names failed: %r' % (e,))

    # per-band pathway membership
    for b in bands:
        cnt = Counter()
        for g in b['genes']:
            for p in g2p.get(g, {}).get('pathways', []):
                cnt[p] += 1
        b['pathways'] = [{'id': p, 'name': names.get(p, ''), 'n_genes': n}
                         for p, n in cnt.most_common(8)]
        b['n_genes_in_core_cancer'] = sum(
            1 for g in b['genes'] if 'hsa05200' in g2p.get(g, {}).get('pathways', []))

    # core-cancer-pathway split (hsa05200 "Pathways in cancer")
    core = [(i, j, w) for i, j, w in src
            if 'hsa05200' in g2p.get(genes[i], {}).get('pathways', [])
            or 'hsa05200' in g2p.get(genes[j], {}).get('pathways', [])]
    noncore = [(i, j, w) for i, j, w in src if (i, j, w) not in core]
    for label, sel in [('core_cancer', core), ('other', noncore)]:
        if sel:
            wk = sum(1 for i, j, _ in sel if kept(W_warm, i, j))
            sk = sum(1 for i, j, _ in sel if kept(W_scr, i, j))
            out[label] = {'n': len(sel),
                          'scratch_pct': round(100.0 * sk / len(sel), 1),
                          'warm_pct': round(100.0 * wk / len(sel), 1)}
            print('  %s: n=%d scratch=%.1f%% warm=%.1f%%'
                  % (label, len(sel), out[label]['scratch_pct'], out[label]['warm_pct']))

    json.dump(out, open(os.path.join(RES, 'edge_stratification_tau03.json'), 'w'), indent=1)
    print('saved results/edge_stratification_tau03.json')


if __name__ == '__main__':
    main()
