# Data directory

Validation (`validate_paper.py`) and figure regeneration do not need any data:
they read the archived results in `results/`.

To re-run the experiments, place the public inputs here and point
`TCGA_DATA_DIR` at this directory (it defaults to `./data`):

```
data/
  TCGA_XXX_HiSeqV2.tsv      # UCSC Xena, one file per cancer type (33 files)
  depmap/CRISPRGeneEffect.csv
```

STRING, TRRUST and BioGRID files are only needed for the validation stage in
`scripts/experiments/`; the cached output ships in `results/`.
