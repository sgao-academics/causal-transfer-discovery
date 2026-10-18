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
Extended external validation for JBCB revision (Reviewer #1, Item 1)
Robust retry with progress bars. Validates causal edges against BioGRID + PathwayCommons.
"""
import json, os, sys, urllib.request, zipfile, io, time
from collections import defaultdict

RESULTS = PKG_RESULTS

def download_with_progress(url, desc, timeout=120, retries=3):
    """Download with retry and progress reporting"""
    for attempt in range(retries):
        try:
            print(f"  [{desc}] Attempt {attempt+1}/{retries}...")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; CausalGPT/1.0)',
                'Accept': '*/*'
            })
            resp = urllib.request.urlopen(req, timeout=timeout)
            total = int(resp.headers.get('Content-Length', 0))
            chunks = []
            downloaded = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (1024*1024) < 8192:
                    pct = 100 * downloaded / total
                    print(f"    {downloaded//1024}KB / {total//1024}KB ({pct:.0f}%)", end='\r')
            data = b''.join(chunks)
            print(f"    Done: {len(data)//1024}KB")
            return data
        except Exception as e:
            print(f"    Failed: {e}")
            if attempt < retries - 1:
                wait = (attempt + 1) * 5
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
    return None

def download_biogrid():
    """Download BioGRID human interactions (latest release, tab3 format)"""
    # BioGRID hosts tab3 files by organism. Try known URLs.
    urls = [
        # Direct organism-specific file (human = 9606)
        "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-Latest-Release.tab3.zip",
        # Alternative: mitab format for all organisms
        "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-Latest-Release.mitab.zip",
    ]
    
    for url in urls:
        data = download_with_progress(url, "BioGRID", timeout=120, retries=2)
        if not data:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                print(f"    ZIP contents: {z.namelist()[:5]}...")
                for name in z.namelist():
                    if name.endswith('.tab3.txt') or name.endswith('.mitab.txt'):
                        content = z.read(name).decode('utf-8', errors='ignore')
                        return parse_biogrid_tab3(content)
            # If no .txt in zip, try raw content
        except Exception as e:
            print(f"    ZIP parse failed: {e}")
            continue
    
    print("  [BioGRID] All URLs failed. Using empty result.")
    return {}

def parse_biogrid_tab3(content):
    """Parse BioGRID TAB3 format into human gene-pair evidence"""
    pairs = defaultdict(int)
    human_count = 0
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 10:
            continue
        
        # TAB3 format: cols 15,16 = Organism ID (NCBI tax ID)
        try:
            tax_a = parts[15].strip()
            tax_b = parts[16].strip()
        except IndexError:
            continue
        
        # Human only
        if tax_a != '9606' or tax_b != '9606':
            continue
        
        human_count += 1
        # Cols 7,8 = Official Symbol for Interactor A, B
        try:
            gene_a = parts[7].strip().upper()
            gene_b = parts[8].strip().upper()
        except IndexError:
            continue
        
        if gene_a and gene_b:
            pair = tuple(sorted([gene_a, gene_b]))
            pairs[pair] += 1
    
    print(f"    Human interactions: {human_count}, unique pairs: {len(pairs)}")
    return dict(pairs)

def download_pathwaycommons(gene_pairs):
    """Query PathwayCommons via REST API for specific gene pairs"""
    results = {}
    batch_size = 10
    total = len(gene_pairs)
    
    for i in range(0, total, batch_size):
        batch = gene_pairs[i:i+batch_size]
        genes_a = ",".join([p[0] for p in batch])
        genes_b = ",".join([p[1] for p in batch])
        
        url = (f"https://www.pathwaycommons.org/pc2/graph?"
               f"source={urllib.request.quote(genes_a)}"
               f"&target={urllib.request.quote(genes_b)}"
               f"&kind=neighborhood&format=JSON")
        
        pct = 100 * min(i + batch_size, total) / total
        print(f"  [PathwayCommons] Batch {i//batch_size+1}/{(total+batch_size-1)//batch_size} ({pct:.0f}%)", end='\r')
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; CausalGPT/1.0)'
            })
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            
            for edge in data.get('edges', []):
                subj = edge.get('subject', {}).get('name', '')
                obj = edge.get('object', {}).get('name', '')
                if subj and obj:
                    pair = tuple(sorted([subj.upper(), obj.upper()]))
                    results[pair] = results.get(pair, 0) + 1
        except Exception as e:
            print(f"\n    PC batch {i//batch_size+1} error: {e}")
            time.sleep(2)  # rate limit
    
    print(f"\n    Done: {len(results)} interactions found")
    return results

def main():
    print("="*60)
    print("EXTENDED EXTERNAL VALIDATION (with download progress)")
    print("="*60)
    
    # Load preserved edges
    with open(os.path.join(RESULTS, 'supplementary_analysis.json'), 'r') as f:
        supp = json.load(f)
    
    top5 = supp.get('top_preserved', [])
    gene_pairs = [(e['gene_i'].upper(), e['gene_j'].upper()) for e in top5]
    all_genes = set()
    for g1, g2 in gene_pairs:
        all_genes.add(g1)
        all_genes.add(g2)
    
    print(f"\nAnalyzing {len(gene_pairs)} top preserved edges ({len(all_genes)} unique genes)")
    print(f"Edges: {', '.join([f'{a}-{b}' for a,b in gene_pairs])}")
    
    # 1. BioGRID
    print("\n" + "-"*40)
    print("1. BioGRID Download")
    print("-"*40)
    biogrid = download_biogrid()
    
    # 2. PathwayCommons
    print("\n" + "-"*40)
    print("2. PathwayCommons Query")
    print("-"*40)
    pc = download_pathwaycommons(gene_pairs)
    
    # 3. Results table
    print("\n" + "-"*40)
    print("3. Validation Results")
    print("-"*40)
    print(f"  {'Edge':<30s} {'BioGRID':>8s} {'PC':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    
    validated_edges = []
    for gi, gj in gene_pairs:
        pair = tuple(sorted([gi, gj]))
        bg = "YES" if biogrid.get(pair, 0) > 0 else "no"
        pc_result = "YES" if pc.get(pair, 0) > 0 else "no"
        print(f"  {gi}-{gj:<{30-len(gi)-len(gj)-1}} {bg:>8s} {pc_result:>8s}")
        if biogrid.get(pair, 0) > 0 or pc.get(pair, 0) > 0:
            validated_edges.append(f"{gi}-{gj}")
    
    print(f"\n  STRING v12 precision: 3-5% (from manuscript)")
    print(f"  TRRUST precision: 0% (expected)")
    print(f"  BioGRID validated: {sum(1 for p in gene_pairs if biogrid.get(tuple(sorted([p[0],p[1]])),0)>0)}/{len(gene_pairs)}")
    print(f"  PathwayCommons validated: {sum(1 for p in gene_pairs if pc.get(tuple(sorted([p[0],p[1]])),0)>0)}/{len(gene_pairs)}")
    
    # Save
    output = {
        'biogrid_validated': [f"{gi}-{gj}" for gi,gj in gene_pairs 
                             if biogrid.get(tuple(sorted([gi,gj])),0) > 0],
        'pathwaycommons_validated': [f"{gi}-{gj}" for gi,gj in gene_pairs 
                                      if pc.get(tuple(sorted([gi,gj])),0) > 0],
        'edge_count': len(gene_pairs),
        'note': 'Top-5 preserved edges validated against additional databases'
    }
    
    out_path = os.path.join(RESULTS, 'extended_validation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == '__main__':
    main()
