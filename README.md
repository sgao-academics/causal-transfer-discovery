# Replication Package

**Paper:** Causal Transfer Learning: Warm-Starting NOTEARS, DAGMA, and GOLEM with
Cross-Dataset Causal Graphs
**Author:** Shuaidong Gao
**Journal:** Journal of Bioinformatics and Computational Biology (JBCB), JBCB-1496
**Revision:** second revision (October 2026)
**Manuscript:** 15 typeset pages; Supplementary Material: Sections S1–S11

---

## Quick Start

```bash
# 1. Validate every number printed in the manuscript (~5 s; no GPU, no data files)
python validate_paper.py            # 71 checks; exit code 0 on success

# 2. Regenerate Figures 1-3 from the archived checkpoint (~30 s)
python scripts/gen_figures/gen_fig1_overview.py
python scripts/gen_figures/gen_fig2_results.py
python scripts/gen_figures/gen_fig3_generality.py

#    grayscale twins for print (the journal charges for colour illustrations)
FIG_MONO=1 python scripts/gen_figures/gen_fig1_overview.py
FIG_MONO=1 python scripts/gen_figures/gen_fig2_results.py
FIG_MONO=1 python scripts/gen_figures/gen_fig3_generality.py

# 3. Compile the manuscript
pdflatex manuscript_jbcb && bibtex manuscript_jbcb && pdflatex manuscript_jbcb
```

`validate_paper.py` is the validation entry point. It re-derives every mean and
sample SD in Table 2 (NOTEARS) and Table 5 (DAGMA and GOLEM) from
`results/benchmark_v6_checkpoint.json`, checks that the 45 configurations are 45
*distinct* runs (no seed produces a duplicate solution), and verifies the DepMap
counts.

`run_all.py` is **not** a validation script. It re-runs the genome-scale
(d = 5000) experiment from its per-outer checkpoints and needs a GPU plus the
TCGA expression matrices (~3.5 h per seed). Use it only for full regeneration:

```bash
python run_all.py --help
```

## Requirements

- Python 3.8+ with `torch`, `numpy`, `scipy`, `matplotlib`, `pandas`
- For compilation: `pdflatex` + `bibtex` (TeX Live or MiKTeX)
- For full regeneration: NVIDIA GPU with CUDA (8 GB VRAM recommended)

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── validate_paper.py                   # Re-derives every printed number (71 checks)
├── run_all.py                          # Genome-scale (d=5000) regeneration only
├── README.md, requirements.txt, SHA256SUMS
├── manuscript_jbcb.tex / .pdf / .bbl   # Main text (15 typeset pages)
├── supplementary.tex / .pdf            # Supplementary Material, S1–S11
├── refs.bib, ws-jbcb.cls, ws-jbcb.bst  # Bibliography and JBCB style files
├── author_photo.pdf                    # Author biography portrait
├── figures/                            # All three figures, colour and grayscale
│   ├── fig1_overview.{pdf,png}         #   Fig. 1 pipeline schematic (vector)
│   ├── fig2_results.{pdf,png}          #   Fig. 2 NOTEARS transfer (5 panels)
│   ├── fig3_generality.{pdf,png}       #   Fig. 3 generality + validation (6 panels)
│   └── *_bw.{pdf,png}                  #   grayscale twins used for print
├── scripts/
│   ├── gen_figures/
│   │   ├── gen_fig1_overview.py        # Figure 1 (5 panels, drawn to print width)
│   │   ├── gen_fig2_results.py         # Figure 2 (5 panels)
│   │   ├── gen_fig3_generality.py      # Figure 3 (6 panels)
│   │   └── figstyle.py, palette.py, housefig.py   # shared house style
│   ├── experiments/
│   │   ├── run_benchmark_gpu.py        # NOTEARS benchmark (d=200..889, 3 seeds)
│   │   ├── run_dagma_golem_v2.py       # DAGMA & GOLEM benchmark (Methods impl.)
│   │   ├── run_dagma_golem_seedmatched.py  # per-seed source graphs for Table 5
│   │   ├── run_real_baselines.py       # pooling + target-only priors (S9)
│   │   ├── run_small_target.py         # BRCA downsample n=150..1218 (S10)
│   │   ├── run_extraseeds.py           # seven fresh seeds (S11)
│   │   ├── run_synthetic_baselines.py  # synthetic overlap sweep (not in the paper)
│   │   ├── load_tcga_safe.py           # expression-only TCGA loader (whitelist)
│   │   ├── kegg_enrichment.py          # KEGG enrichment (Section S4)
│   │   ├── validate_string.py          # STRING/BioGRID validation (Section S4)
│   │   ├── run_multi_target.py         # Six-organ target sweep, d=454 (Section S12)
│   │   └── ...
│   ├── run_d5000_multiseed.py          # Genome-scale (d=5000) transfer, 3 seeds
│   ├── run_scratch_multiseed.py        # Genome-scale scratch baseline, 3 seeds
│   ├── run_d5000_multitarget.py        # Five-organ target sweep, d=5000 (Section S12)
│   ├── stratification_tau03.py         # Edge-strength stratification (Table 3 / S3)
│   └── fix_depmap_edges.py             # Regenerates the DepMap retained-edge list
├── results/
│   ├── benchmark_v6_checkpoint.json    # All main-benchmark results (d<=889)
│   ├── benchmark_v6_summary.json
│   ├── dagma_golem_seedmatched.json    # Per-seed DAGMA/GOLEM runs (Table 5)
│   ├── table5_seedmatched.json         # Table 5 summary: mean, SD, delta, wins
│   ├── real_baselines.json             # Pooling / target-only arms (Tables S9, S10)
│   ├── matched_k_eval.json             # Held-out residual at matched edge count
│   ├── small_target.json               # BRCA downsample (Table S11)
│   ├── bench_v6_extraseeds.json        # Ten-seed sweep (Table S12)
│   ├── synthetic_baselines.json        # Synthetic overlap sweep (not in the paper)
│   ├── d5000_seedmatched.json          # Genome-scale, seed-matched (Table S5)
│   ├── d5000_multiseed_transfer.json   # Genome-scale warm-start, 3 seeds
│   ├── d5000_scratch_multiseed.json    # Genome-scale scratch, 3 seeds
│   ├── edge_stratification_tau03.json  # Per-band retention + pathway annotation
│   ├── depmap_edges_d454.json          # 7 transferred DepMap edges (regenerated)
│   ├── full_experiment.json            # DepMap cross-domain data
│   ├── external_validation.json        # STRING multi-threshold precision
│   ├── d5000_checkpoints/              # Per-outer checkpoints (JSON only)
│   ├── multi_target_d454.json          # Six-organ sweep, d=454, 3 seeds (Table S13)
│   ├── d5000_multi/                    # Five-organ sweep, d=5000, seed 42 (Table S14)
│   └── cache/                          # d=454 weight matrices used for figures
└── data/                               # Place TCGA data here (see below)
```

Table numbering follows the second revision: **Table 1** datasets, **Table 2**
NOTEARS transfer, **Table 3** edge-strength stratification, **Table 4** cross-domain
transfer, **Table 5** DAGMA and GOLEM. In the first submission the text cited the
last two by section number (Tables 4.4 and 4.1); the tables themselves are unchanged.

Large `d=5000` weight matrices (100 MB each) are **not** included; they are regenerated
by `scripts/run_d5000_multiseed.py` and `scripts/run_scratch_multiseed.py`, which resume
from the per-outer checkpoints in `results/d5000_checkpoints/`.

## Data Availability

Pre-computed results are in `results/`. Every table and figure in the paper can be
validated without external data.

To re-run experiments, download:

1. **TCGA pan-cancer expression data** from [UCSC Xena](https://xenabrowser.net/)
   - HiSeqV2 pan-cancer normalized expression: `TCGA_XXX_HiSeqV2.tsv` for 33 cancer types
   - Place all `.tsv` files in one directory and set `TCGA_DATA_DIR`
     (defaults to `./data` if unset)
2. **DepMap CRISPR dependency data** from [DepMap Portal](https://depmap.org/)
   - `CRISPR_gene_effect.csv` (23Q2 release), in `$TCGA_DATA_DIR/depmap/`
3. **STRING v12** (`https://string-db.org/`), **TRRUST v2**
   (`https://www.grnpedia.org/trrust/`), **BioGRID** — only for the validation stage;
   the cached validation output is included.

All experiment scripts resolve the package root from their own location and read
expression data from `$TCGA_DATA_DIR`, so the package can be placed anywhere.

## Validation Output

Running `python validate_paper.py` produces (abridged):

```
Stage 1: NOTEARS main results (Table 2, 3 seeds, seed-matched)
  d=889 scratch mean   34.0 (paper  34.0) [OK]
  d=889 scratch SD      3.3 (paper   3.3) [OK]
  d=889 warm mean      68.1 (paper  68.1) [OK]
  d=889 warm SD         1.3 (paper   1.3) [OK]
  ... (60 more mean/SD checks, all [OK])

Stage 2: DAGMA/GOLEM results (Table 5, 3 seeds, seed-matched)
  d=889 DAGMA warm mean  89.9 (paper 89.9) [OK]
  d=889 GOLEM warm mean  75.1 (paper 75.1) [OK]
  ... (38 more checks, all [OK])

Stage 3: 45/45 claim (each cell an independent run)
  warm > scratch                             45/45
  duplicate warm entries across seeds        0

Stage 4: cross-domain DepMap transfer
  DepMap source edges    20 (paper  20) [OK]
  scratch overlap         0 (paper   0) [OK]
  TCGA-warm overlap       2 (paper   2) [OK]
  DepMap-warm overlap     7 (paper   7) [OK]

Stage 5: pooling, target-sample size, ten seeds (S9-S11)
  pool worse than warm at k=70               11 cells, 11 positive [OK]
  penalty range (paper 0.003..0.031)         0.003..0.031 [OK]
  gain at n=150                              +37.2 (paper +37.2) [OK]
  gain at n=1218                             +21.2 (paper +21.2) [OK]
  ten seeds, both dimensions                 20/20 [OK]

Stage 6: multi-organ extension (S12)
  S13 <organ> mean gain                      23.2 / 17.2 / 21.0 / 25.3 / 27.8 / 20.3 [OK]
  cells with positive gain                   18/18 [OK]
  S14 <organ> source/scratch/warm/gain       5 organs x 4 values [OK]

ALL 98 CHECKS PASSED
```

The `duplicate warm entries across seeds` counter is the direct test that no
algorithm–dimension cell rests on a single effective run: it compares the stored
result record of the three seeds for each algorithm and dimension and reports how
many are byte-identical. It must be zero.

## Key Results

| Metric | Value |
|:--|:--|
| NOTEARS warm vs scratch (d=889) | 68.1 ± 1.3 % vs 34.0 ± 3.3 % |
| Warm > Scratch across all configs | 45/45 (100 %), 45 independent runs |
| DAGMA edge retention (d ≥ 454) | 88.1–90.9 % (89.9 ± 3.1 % at d=889) |
| GOLEM edge retention (d ≥ 454) | 72.0–76.1 % (75.1 ± 2.2 % at d=889) |
| DAGMA / GOLEM gain (Δ) at d=889 | +60.7 / +39.3 pp |
| Power-law scaling R² | 0.91 (warm ∝ d^0.47, scratch ∝ d^0.33; exponent CIs overlap) |
| Shuffled control (negative) | indistinguishable from scratch (paired Wilcoxon p = 0.86); warm exceeds shuffled at all 15 pairs (p = 6.1 × 10⁻⁵) |
| Cross-domain DepMap retention | 35 % (7/20; Fisher p = 0.008) |
| Pooling vs warm, matched edge count | pool residual higher in 11/11 cells (+0.003 to +0.031) |
| Warm-start gain vs target size | +37.2 pp at n=150, +21.2 pp at n=1,218 (d=454) |
| Ten-seed sweep | 10/10 seeds at d=454 and at d=889 |
| **Genome-scale (d=5000), seed-matched** | **81.8 ± 0.4 % vs 45.4 ± 2.9 % (+36.4 pp, 3 seeds)** |
| Genome-scale, fixed 634-edge reference | 65.3 ± 0.7 % vs 38.1 ± 1.4 % (+27.2 pp) |
| Six target organs (d=454, 3 seeds) | 18/18 cells positive; Δ = +16.7 to +30.2 pp (mean +22.5 pp; sign test p = 3.8 × 10⁻⁶) |
| Five further targets (d=5000, seed 42) | 5/5 positive; Δ = +32.5 to +48.4 pp |
| Zero sample overlap (TCGA-BRCA) | verified |

### Genome-scale retention has two denominators — do not mix them

* **Seed-matched (used in the main text, Table S5).** Each seed learns its own source
  graph (619–627 edges) and both warm-start and scratch are scored against *that* graph.
  This is the protocol used at every dimension `d ≤ 889`: 81.8 ± 0.4 % vs 45.4 ± 2.9 %.
* **Fixed reference (for continuity with the first submission).** All runs scored against
  the single deterministic 634-edge source graph: 65.3 ± 0.7 % vs 38.1 ± 1.4 %. The
  lower warm value is purely geometric — each seed's own source graph overlaps that
  reference by only 77–79 %, and 0.78 × 0.82 ≈ 0.65.

Both denominators are reported in Supplementary Section S5.

### Multi-organ extension (Section S12)

Every main-text experiment uses BRCA as the target. Section S12 removes that
restriction: for each target organ the source graph is **re-learned from the remaining
cohorts**, because reusing the archived non-BRCA graph would put the target's own
samples into its prior.

```bash
python scripts/experiments/run_multi_target.py --organs KIRC,LUAD,THCA,LUSC,LIHC --seeds 42,123,456
python scripts/run_d5000_multitarget.py --organ KIRC          # ~5 h per organ
```

Six organs × three seeds at `d = 454` give 18/18 positive cells (Δ = +16.7 to
+30.2 pp). At `d = 5000`, five further organs are positive too (+32.5 to +48.4 pp),
but at one seed each, so those rows are labelled exploratory.

**Two things to know before quoting these numbers.**

1. *The BRCA row of Table S13 does not coincide with Table 2* (37.2 / 60.4 versus
   33.4 / 55.1). Re-running the arms on the archived source graphs reproduces the warm
   column exactly — 56.8, 53.4, 55.1 % for seeds 42/123/456 — so the difference is the
   source-graph draw, not the implementation. The scratch arm lands one to three edges
   away because it starts from a random initialisation taken at a different point in
   the RNG stream.
2. *At `d = 454` the retention level carries a source-graph-draw component of about
   ±5 pp*, while Δ is stable to about 3 pp. Compare Δ across organs; do not rank them.
   At `d = 5000` the component is negligible (81.8 ± 0.4 % over three seeds).

## Figures

All three figures are generated from archived results, by:

```bash
python scripts/gen_figures/gen_fig1_overview.py
python scripts/gen_figures/gen_fig2_results.py
python scripts/gen_figures/gen_fig3_generality.py
```

Each script writes both a PDF and a PNG. Set `FIG_MONO=1` to produce the grayscale
twin (`*_bw.pdf`, `*_bw.png`) used for print. Figure 1 is a vector schematic
(`figures/fig1_overview.pdf`) rather than TikZ, so the LaTeX source includes it with
`\includegraphics`.

All three are drawn on a canvas equal to their printed width (114.2 mm), so a label
set at 7–9.5 pt prints at 7–9.5 pt.

## Hardware & Runtime

| Mode | Hardware | Time |
|:--|:--|:--|
| Validation only | Any CPU | ~1 min |
| Figure regeneration | Any CPU | ~30 sec |
| Main benchmark regeneration (d ≤ 889) | GPU (8 GB VRAM) | ~6 hours |
| Genome-scale d=5000, per seed (source + warm) | GPU (8 GB VRAM, 7.7 GB used) | ~3.5 hours |
| Genome-scale d=5000, all three seeds | GPU (8 GB VRAM) | ~10.5 hours |

## Command-Line Options

```bash
python run_all.py                  # Validate pre-computed results
python run_all.py --figures        # Regenerate figures
python run_all.py --compile        # Compile manuscript PDF
python run_all.py --regenerate     # Re-run all experiments (slow)
python run_all.py --all            # Everything: regenerate + figures + compile
```

## Troubleshooting

**"CUDA not available"**: the scripts fall back to CPU automatically. Set
`CUDA_VISIBLE_DEVICES=""` to force CPU mode.

**"TCGA data not found"**: set `TCGA_DATA_DIR` (defaults to `./data`). Validation does
not require TCGA data.

**"File not found: benchmark_v6_checkpoint.json"**: run from the package root and keep
the `results/` folder intact.

**"pdflatex not found"**: install TeX Live or MiKTeX.

**Figure generation fails**: `results/cache/` must contain `src_d454.npy`,
`warm_d454_s42.npy` and `scratch_d454_s42.npy`; all three ship with the package.

**Genome-scale run restarts from scratch**: delete the stale files in
`results/d5000_checkpoints/` whose `last_outer` already equals 20; otherwise the script
resumes and reports `already complete` (this is intended behaviour).

## Integrity

`SHA256SUMS` lists the SHA-256 digest of every file in the package; verify with
`sha256sum -c SHA256SUMS` (Linux/macOS) or `Get-FileHash` (PowerShell).

## License

Provided for academic review and replication purposes.
