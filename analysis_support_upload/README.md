# Reproduction Guide

## Better Imputation Does Not Guarantee Better Forecasting

**Transmission Efficiency in Two-Step Time Series Pipelines**

This guide describes the numerical reproduction materials for the study of
reconstruction-forecasting mismatch in two-step time series pipelines. The
package provides archived run-level metrics and scripts for the manuscript's
transmission, seed-stability, and imputer-selection analyses and Figures 1-4.

For the research overview, experimental design, and main findings, see the
[project README](../README.md). Repository: <https://github.com/SecretFzzf/shure>.

The `01_raw_data` directory contains per-run metrics, not original time-series
observations. This release covers numerical analysis and plotting; upstream
preprocessing and training code, raw time series, intermediate imputation arrays,
and model weights are outside its scope. Neither raw nor processed SWaT time
series are redistributed. Historical source and metadata manifests document the
original experiments and may refer to upstream files outside this package.

---

## 1. Directory Structure

```text
analysis_support_upload/
├── README.md
├── requirements.txt
│
├── 01_raw_data/                          # Primary recorded experimental metrics (8,640 runs)
│   ├── online_results_8640.csv           # Consolidated results (4 datasets × 3 models × 3 modes × 4 rates × 4 horizons × 5 imputers × 3 seeds)
│   └── input_audit.json                  # Completeness and configuration audit of the 8,640 runs
│
├── 02_transmission_analysis/             # Error transmission & inversion analysis (Tables 2–9 & Table E.16)
│   ├── recompute_three_seeds.py          # Main statistical computation script
│   ├── summary.json                      # Headline summary statistics and SHA-256 verification
│   ├── configuration_means_2880.csv      # Three-seed mean IE/FE across 2,880 configurations
│   ├── configuration_mean_sd_2880.csv    # Three-seed mean and sample SD (ddof=1) across 2,880 configurations
│   ├── main_contrasts_2304.csv           # All 2,304 non-Mean contrasts against the Mean anchor
│   ├── headline_and_dataset.csv          # Overall and per-dataset transmission/inversion metrics (Tables 3 & 5)
│   ├── method_performance.csv            # Imputation and forecasting performance by dataset and imputer (Table 2)
│   ├── threshold_sensitivity.csv         # Sensitivity across stability thresholds epsilon and MSE metric (Table 4)
│   ├── anchor_sensitivity.csv            # Sensitivity to baseline anchor choice: Mean vs. Spline (Table 6)
│   ├── strata.csv                        # Stratified pure inversion rates and cluster bootstrap CIs (Table 7)
│   ├── paired_comparisons.csv            # Paired sign-flip randomization tests and Holm-adjusted p-values (Table 8)
│   ├── recompute_model_seed_consistency.py # Model-stratified seed-summary script
│   ├── recompute_dataset_influence.py    # Leave-one-dataset-out composition check
│   ├── leave_one_dataset_out.csv         # Full grid and four dataset-omission summaries
│   ├── model_seed_consistency.csv         # Model-specific sign agreement and inversion stability
│   ├── seed_consistency.csv              # Cross-seed sign agreement and inversion stability (Table 9)
│   ├── seed_model_statistics.csv         # Per-seed, per-dataset, per-model breakdown (Appendix Table E.16)
│   ├── seed_specific_statistics.csv      # Per-seed overall and dataset-level statistics
│   ├── seed_units_2304.csv               # Unit-level seed-by-seed contrast signs across 2,304 contrasts
│   ├── heatmap.csv                       # Model × Imputer pure inversion matrix (for Figure 4)
│   ├── figure1_coordinates.csv           # Median forecasting MAE by model, imputer, and missing rate (for Figure 1)
│   ├── forecast_tolerance.csv            # Inversion counts under relative FE deterioration tolerances
│   ├── global_ie_medians.csv             # Global median IE by imputation method
│   ├── method_ranking.csv                # Unit-level best-IE vs. best-FE imputer comparison across 576 units
│   └── ranking_summary.csv               # Dataset-level summary of IE vs. FE optimal-imputer mismatches
│
├── 03_selection_analysis/                # Imputer selection & matched-missingness analysis (Sections 4.4 & 4.6)
│   ├── recompute_selection.py            # Selection-policy and matched-missingness computation script
│   ├── selection_input_audit.json        # Verification summary for Section 4.4 and Section 4.6
│   ├── policy_summary.csv                # Overall performance of 8 selection policies across 576 units (Table 10)
│   ├── paired_tests_with_holm.csv        # 7 paired horizon-group sign-flip tests with Holm correction (Section 4.6)
│   ├── policy_strata.csv                 # Selection-policy regret and worse-than-Mean rates by dataset and model
│   ├── policy_units_4608.csv             # Unit-level trace of all 8 policies × 576 selection units
│   ├── oracle_shares.csv                 # Dataset × model oracle imputer win shares (including exact ties)
│   └── matched_missingness.csv           # Matched intersection (615 configurations per mechanism) across MCAR, Block-MCAR, MNAR (Section 4.4)
│
├── 04_figure_generation/                 # Manuscript figure plotting script and generated PNGs (Figures 1–4)
│   ├── plot_three_seeds.py               # Plotting script reading from 02_transmission_analysis/
│   ├── fig1.png                          # Figure 1: Median forecasting MAE across missing rates
│   ├── fig2.png                          # Figure 2: Joint ΔIE vs. ΔFE scatter across architectures
│   ├── fig3.png                          # Figure 3: Inversion rate sensitivity to threshold ε
│   └── fig4.png                          # Figure 4: Model × Imputer pure inversion rate heatmap
│
└── 05_validation_manifests/              # Reproducibility manifests and SHA-256 checksums
    ├── source_code_manifest.json         # SHA-256 checksums of the experimental source scripts
    ├── source_metadata_validation.json   # SHA-256 checksums of the 36 per-seed/per-model raw result CSVs
    ├── revision_validation.json          # Original numerical verification summary
    ├── editorial_revision_20261018.json  # Manuscript editorial verification record
    ├── github_publication_validation.json # Reproduction checks and package checksums
    └── concise_revision_validation.json  # Concise manuscript and dataset-influence verification
```

---

## 2. Quick Reproduction Instructions

Install dependencies (`numpy`, `pandas`, `matplotlib`):

```bash
python -m pip install -r requirements.txt
```

Run the five scripts in order from the root of `analysis_support_upload/` (each script defaults to reading from `01_raw_data/` or `02_transmission_analysis/` and writing outputs into its own folder):

```bash
# 1. Recompute main transmission & inversion tables (Tables 2–9 & Table E.16)
python 02_transmission_analysis/recompute_three_seeds.py

# 2. Recompute model-stratified seed consistency (Section 4.5)
python 02_transmission_analysis/recompute_model_seed_consistency.py

# 3. Recompute the dataset influence check (Appendix Table F.17)
python 02_transmission_analysis/recompute_dataset_influence.py

# 4. Recompute imputer-selection policies & matched-missingness check (Section 4.4 & Table 10)
python 03_selection_analysis/recompute_selection.py

# 5. Regenerate manuscript Figures 1–4
python 04_figure_generation/plot_three_seeds.py
```

---

## 3. Manuscript Table & Figure Mapping

| Manuscript Item | Source File |
| :--- | :--- |
| **Table 1** (Experimental Grid) | Verified against `01_raw_data/online_results_8640.csv` & `01_raw_data/input_audit.json` |
| **Table 2** (Method Performance) | `02_transmission_analysis/method_performance.csv` |
| **Table 3** (Dataset Transmission Summary) | `02_transmission_analysis/headline_and_dataset.csv` |
| **Table 4** (Threshold & Metric Sensitivity) | `02_transmission_analysis/threshold_sensitivity.csv` |
| **Table 5** (Overall Inversion Summary) | `02_transmission_analysis/headline_and_dataset.csv` (`Overall` row) |
| **Table 6** (Anchor Sensitivity: Mean vs. Spline) | `02_transmission_analysis/anchor_sensitivity.csv` |
| **Table 7** (Stratified Inversion Rates) | `02_transmission_analysis/strata.csv` |
| **Table 8** (Paired Architecture/Imputer Tests) | `02_transmission_analysis/paired_comparisons.csv` |
| **Table 9** (Three-Seed Consistency) | `02_transmission_analysis/seed_consistency.csv` |
| **Section 4.5** (Model-stratified seed consistency) | `02_transmission_analysis/model_seed_consistency.csv` |
| **Section 4.4** (Matched-Mechanism Check, $n=615$) | `03_selection_analysis/matched_missingness.csv` |
| **Table 10 & Section 4.6** (Selection Policies) | `03_selection_analysis/policy_summary.csv` & `03_selection_analysis/paired_tests_with_holm.csv` |
| **Appendix Table F.17** (Dataset influence check) | `02_transmission_analysis/leave_one_dataset_out.csv` |
| **Appendix Table E.16** (Seed × Model Breakdown) | `02_transmission_analysis/seed_model_statistics.csv` |
| **Figures 1–4** | `04_figure_generation/fig1.png` – `fig4.png` |

## 4. Model-stratified forecasting-seed analysis

Section 4.5 additionally reports forecasting-error-change sign agreement and
all-three-seed pure inversion separately for DLinear, LSTM and PatchTST.
Reproduce these statistics after the main transmission analysis with:

```bash
python 02_transmission_analysis/recompute_model_seed_consistency.py
```

The script reads `02_transmission_analysis/seed_units_2304.csv` and writes
`02_transmission_analysis/model_seed_consistency.csv`. Sign-agreement proportions
use all 768 non-Mean configurations per model; pure-inversion proportions use
709 valid contrasts per model. These are descriptive statistics of the three
recorded forecasting seeds conditional on the fixed upstream inputs.

These summaries are computed from archived metrics and require no model training.

## 5. Dataset influence check

`recompute_dataset_influence.py` independently averages the archived 8,640
run-level metrics by configuration, forms the Mean contrasts at epsilon 0.01,
and reports the full grid and each dataset omission. Its output contains valid
counts, inversion counts and rates, median transmission ratios, relative error
changes, and model-specific inversion rates. It uses no additional model fitting
and makes no held-out-dataset prediction claim.
