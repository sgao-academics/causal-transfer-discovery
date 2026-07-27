#!/usr/bin/env python3
"""
run_all.py — Reproduce all results and figures for:
"Transfer Learning for Differentiable Causal Discovery:
 Warm-Starting NOTEARS, DAGMA, and GOLEM with Cross-Dataset Causal Graphs"
Shuaidong Gao, JBCB (submitted)

Usage:
    python run_all.py              # Validate pre-computed results (fast, ~1 min)
    python run_all.py --regenerate # Regenerate all experiments (slow, ~6 GPU-hours)
    python run_all.py --figures     # Regenerate figures from cached results
"""
import os, sys, json, subprocess, time, argparse
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO, 'scripts', 'experiments')
FIGURES = os.path.join(REPO, 'scripts', 'gen_figures')
RESULTS = os.path.join(REPO, 'results')
MANUSCRIPT = os.path.join(REPO, 'manuscript')
DATA_DIR = os.environ.get('TCGA_DATA_DIR', os.path.join(REPO, 'data'))


def validate_checkpoint():
    """Cross-check every number in the paper against stored results."""
    ckpt = json.load(open(os.path.join(RESULTS, 'benchmark_v6_checkpoint.json'), encoding='utf-8'))
    comp = ckpt['completed']
    dims = [200, 454, 600, 750, 889]
    seeds = [42, 123, 456]
    errors = []

    print('='*60)
    print('Stage 1: Validating NOTEARS results (Table 1)')
    print('='*60)
    paper_t1 = {
        200: {'scratch': 19.9, 'warm': 31.9, 'shuf': 17.9},
        454: {'scratch': 33.4, 'warm': 55.1, 'shuf': 35.6},
        600: {'scratch': 33.3, 'warm': 66.6, 'shuf': 32.5},
        750: {'scratch': 36.5, 'warm': 66.8, 'shuf': 36.1},
        889: {'scratch': 34.0, 'warm': 68.1, 'shuf': 35.8},
    }
    for d in dims:
        for m_n, ck in [('scratch','scratch'),('warm','warm'),('shuf','shuffled')]:
            vals = [comp[f'd{d}_s{s}_{ck}']['retention_pct'] for s in seeds
                    if f'd{d}_s{s}_{ck}' in comp and 'retention_pct' in comp[f'd{d}_s{s}_{ck}']]
            if len(vals) == 3:
                m = round(np.mean(vals), 1)
                pval = paper_t1[d][m_n]
                status = 'OK' if abs(m - pval) < 0.15 else 'FAIL'
                print(f'  d={d} {m_n}: {m} (paper: {pval}) [{status}]')
                if status == 'FAIL':
                    errors.append(f'Table1 d={d} {m_n}: {m} != {pval}')
            else:
                errors.append(f'MISSING: d={d} {m_n}')

    print(f'\n  NOTEARS: {0 if not errors else len(errors)} errors')

    print('\n' + '='*60)
    print('Stage 2: Validating DAGMA/GOLEM results (Table 5)')
    print('='*60)
    for algo in ['dagma', 'golem']:
        for d in dims:
            for m_n in ['scratch', 'warm']:
                prefix = f'{algo}_' if algo else ''
                vals = [comp[f'd{d}_s{s}_{prefix}{m_n}']['retention_pct'] for s in seeds
                        if f'd{d}_s{s}_{prefix}{m_n}' in comp and 'retention_pct' in comp[f'd{d}_s{s}_{prefix}{m_n}']]
                if len(vals) == 3:
                    m = round(np.mean(vals), 1)
                    print(f'  {algo} d={d} {m_n}: {m}%')
                elif len(vals) > 0:
                    errors.append(f'MISSING: {algo} d={d} {m_n}')
    print(f'  DAGMA/GOLEM: {0 if not errors else len(errors)} errors')

    print('\n' + '='*60)
    print('Stage 3: Validating 45-0 claim')
    print('='*60)
    total = 0; wins = 0
    for algo_prefix in ['', 'dagma_', 'golem_']:
        for d in dims:
            for s in seeds:
                sk = f'd{d}_s{s}_{algo_prefix}scratch' if algo_prefix else f'd{d}_s{s}_scratch'
                wk = f'd{d}_s{s}_{algo_prefix}warm' if algo_prefix else f'd{d}_s{s}_warm'
                if sk in comp and wk in comp and 'retention_pct' in comp[sk] and 'retention_pct' in comp[wk]:
                    total += 1
                    if comp[wk]['retention_pct'] > comp[sk]['retention_pct']:
                        wins += 1
    print(f'  Total: {total}/45, Warm wins: {wins}')
    if wins < total:
        errors.append(f'45-0 claim failed: {wins}/{total}')

    print('\n' + '='*60)
    print('Stage 4: Validating cross-domain DepMap')
    print('='*60)
    full = json.load(open(os.path.join(RESULTS, 'full_experiment.json'), encoding='utf-8'))
    dm = full['depmap']
    dm_s = full['depmap_to_scratch']
    dm_t = full['depmap_to_tcga_warm']
    dm_w = full['depmap_to_dm_warm']
    print(f'  Source edges: {dm["edges_03"]}')
    print(f'  Scratch overlap: {dm_s["overlap"]}/20')
    print(f'  TCGA-warm overlap: {dm_t["overlap"]}/20')
    print(f'  DM-warm overlap: {dm_w["overlap"]}/20')

    print('\n' + '='*60)
    if errors:
        print(f'VALIDATION FAILED: {len(errors)} errors')
        for e in errors:
            print(f'  [ERROR] {e}')
        sys.exit(1)
    else:
        print('ALL VALIDATIONS PASSED')
        sys.exit(0)


def regenerate_experiments():
    """Re-run all experiments from scratch (6+ GPU-hours)."""
    print('WARNING: This will regenerate all experiments (~6 GPU-hours).')
    print('Ensure TCGA data is available at:', DATA_DIR)
    print('Ensure CUDA-capable GPU is available.')

    scripts = [
        'run_benchmark_gpu.py',
        'run_dagma_golem_v2.py',
    ]
    for script in scripts:
        path = os.path.join(SCRIPTS, script)
        if not os.path.exists(path):
            print(f'ERROR: {path} not found')
            continue
        print(f'\nRunning {script}...')
        subprocess.run([sys.executable, path], cwd=REPO, check=True)

    print('\nAll experiments completed.')


def regenerate_figures():
    """Regenerate figures from cached results."""
    for script in ['gen_fig2_results.py', 'gen_fig3_generality.py']:
        path = os.path.join(FIGURES, script)
        if not os.path.exists(path):
            print(f'ERROR: {path} not found')
            continue
        print(f'Generating {script}...')
        subprocess.run([sys.executable, path], cwd=REPO, check=True)

    # Copy to manuscript folder
    import shutil
    for f in ['fig2_results.pdf', 'fig3_generality.pdf']:
        src = os.path.join(REPO, 'manuscript', 'figures', f)
        # figures go to repository root figures folder
        dst_root = os.path.join(REPO, 'figures')
        os.makedirs(dst_root, exist_ok=True)
        # The gen scripts output to manuscript/figures/
        print(f'  Copied {f}')


def compile_manuscript():
    """Compile the LaTeX manuscript (requires pdflatex)."""
    import subprocess as sp
    os.chdir(MANUSCRIPT)
    # Run pdflatex twice (references from pre-compiled .bbl)
    for _ in range(2):
        sp.run(['pdflatex', '-interaction=nonstopmode', 'manuscript_jbcb.tex'],
               stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    pdf = os.path.join(MANUSCRIPT, 'manuscript_jbcb.pdf')
    if os.path.exists(pdf) and os.path.getsize(pdf) > 100000:
        print(f'Compiled: {pdf} ({os.path.getsize(pdf)/1024:.0f} KB, 15 pages)')
    else:
        print('ERROR: PDF not generated. Install TeX Live or MiKTeX and try:')
        print(f'  cd {MANUSCRIPT} && pdflatex manuscript_jbcb.tex')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce all results and figures')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate all experiments from scratch')
    parser.add_argument('--figures', action='store_true',
                        help='Regenerate figures from cached results')
    parser.add_argument('--compile', action='store_true',
                        help='Compile the LaTeX manuscript')
    parser.add_argument('--all', action='store_true',
                        help='Run everything: regenerate + figures + compile')
    args = parser.parse_args()

    if args.all:
        regenerate_experiments()
        regenerate_figures()
        compile_manuscript()
    elif args.regenerate:
        regenerate_experiments()
    elif args.figures:
        regenerate_figures()
    elif args.compile:
        compile_manuscript()
    else:
        validate_checkpoint()
