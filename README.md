# Statistical analysis and figure reproduction

Repository: <https://github.com/SecretFzzf/shure>

This repository contains the numerical reproduction materials accompanying the
Big Data Research manuscript: 8,640 recorded experimental result rows, statistical
analysis scripts, summary tables, and figure-generation scripts.

## Quick start

```bash
git clone https://github.com/SecretFzzf/shure.git
cd shure/analysis_support_upload
python -m pip install -r requirements.txt
python 02_transmission_analysis/recompute_three_seeds.py
python 02_transmission_analysis/recompute_model_seed_consistency.py
python 03_selection_analysis/recompute_selection.py
python 04_figure_generation/plot_three_seeds.py
```

See [the analysis package documentation](analysis_support_upload/README.md)
for file descriptions, table/figure mappings, and statistical scope.

## Scope and data availability

The scripts recompute analyses from archived metrics; they do not train models.
Upstream preprocessing and model-training code, original time-series datasets,
intermediate imputation arrays, and trained weights are not included. Neither raw
nor processed SWaT time-series data are redistributed. The directory
`01_raw_data` contains recorded metrics and configuration records.

Original experimental source and metadata manifests are retained as provenance
records, rather than as a claim that every referenced upstream file is included.
