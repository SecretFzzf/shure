# Better Imputation Does Not Guarantee Better Forecasting

**Transmission Efficiency in Two-Step Time Series Pipelines**

Yu Chang, Qi Li, Sheng Feng, Huiqiao Hou, and Yujie Guo

This project investigates whether improving missing-value reconstruction also
improves downstream time series forecasting. We study the common **impute first,
then forecast** pipeline and measure how changes in imputation error relate to
changes in forecasting error across datasets, imputers, and forecasting models.

## Research question

An imputer can reconstruct missing observations accurately yet produce inputs
that are less useful to a forecasting model. Selecting an imputer by
reconstruction error alone may therefore lead to a different choice from
selecting it by downstream forecasting performance.

Our study examines this mismatch through matched comparisons with Mean
imputation. It quantifies both the transmission of error changes and **pure
inversion**: cases in which reconstruction improves while forecasting worsens.

## Method

For each experimental configuration, we compare imputation error (IE) and
forecasting error (FE) with the corresponding Mean-imputation baseline:

$$
\Delta IE = IE_{\mathrm{method}} - IE_{\mathrm{Mean}}, \qquad
\Delta FE = FE_{\mathrm{method}} - FE_{\mathrm{Mean}}.
$$

The configuration-level transmission ratio is

$$
\tau = \frac{\Delta FE}{\Delta IE}.
$$

Pure inversion means $\Delta IE < 0$ and $\Delta FE > 0$. The primary
MAE analysis excludes contrasts with $|\Delta IE| \leq 0.01$ to limit
small-denominator instability. Forecasting errors are averaged across three seeds
within each configuration before these comparisons. Relative error changes,
threshold sensitivity, seed consistency, dataset influence checks, and imputer-selection policies provide
complementary views of downstream performance.

The experimental protocol uses historical-window inputs: no observation after
the forecast origin is used as an input. Scaling and learned imputation
parameters are fitted on training data and then fixed for validation and testing.

## Experimental design

| Component | Settings |
| :--- | :--- |
| Datasets | ETTh1, KDD-Beijing, SWaT, Weather |
| Imputers | Mean, Spline, KNN, BRITS, SAITS |
| Forecasters | LSTM, DLinear, PatchTST |
| Missingness mechanisms | MCAR, Block-MCAR, MNAR |
| Missing rates | 10%, 30%, 50%, 70% |
| Forecast horizons | 24, 48, 96, 192 steps |
| Forecasting seeds | 42, 123, 2026 |
| Experiment size | 2,880 configurations; 8,640 recorded runs |

The archived label `MAR_Block` corresponds to the Block-MCAR mechanism described
in the manuscript.

## Main findings

| Result | Value |
| :--- | ---: |
| Median relative change in reconstruction MAE | -41.22% |
| Median relative change in forecasting MAE | -2.90% |
| Median transmission ratio | 0.0449 |
| Pure inversion among valid contrasts | 415 / 2,127 (19.51%) |
| Pure inversion after omitting each dataset in turn | 17.98%-22.53% |
| Forecasting-error-change sign agreement across all three seeds | 1,720 / 2,304 (74.65%) |

The primary error, ratio, and inversion summaries use 2,127 valid non-Mean
comparisons after averaging forecasting errors across seeds. The seed-agreement result uses all 2,304
non-Mean configurations, with upstream inputs held fixed across forecasting
seeds. Pure inversion also varies by forecaster: 34.41% for LSTM, 6.91% for
DLinear, and 17.21% for PatchTST, each among its valid comparisons.

Omitting each dataset in turn preserves the ordering of model-specific inversion
rates (LSTM > PatchTST > DLinear). This checks sensitivity to the composition of
the existing dataset grid; it is not a held-out-dataset forecasting experiment.

These results show that reconstruction accuracy alone is insufficient to assess
downstream forecasting performance under the evaluated settings. The
transmission ratio is a descriptive diagnostic; it is not a causal effect or a
percentage of reconstruction improvement transferred to forecasting.

![Forecasting MAE across missing rates for LSTM, DLinear, and PatchTST](analysis_support_upload/04_figure_generation/fig1.png)

*Figure 1. Median forecasting MAE across missing rates, grouped by forecasting
model and imputation method.*

## Reproduce the analyses

Clone the repository and install the analysis dependencies:

```bash
git clone https://github.com/SecretFzzf/shure.git
cd shure/analysis_support_upload
python -m pip install -r requirements.txt
```

Run the analyses in this order:

```bash
# Transmission, inversion, threshold sensitivity, and seed-level summaries
python 02_transmission_analysis/recompute_three_seeds.py

# Forecasting-seed consistency by model
python 02_transmission_analysis/recompute_model_seed_consistency.py

# Influence of each dataset on pooled summaries
python 02_transmission_analysis/recompute_dataset_influence.py

# Imputer-selection policies and matched-missingness comparisons
python 03_selection_analysis/recompute_selection.py

# Figures 1-4
python 04_figure_generation/plot_three_seeds.py
```

The scripts resolve paths relative to their own locations and write regenerated
outputs beside the scripts. Dependencies are NumPy, pandas, and Matplotlib.

## Repository contents

```text
analysis_support_upload/
├── 01_raw_data/                 # Recorded metrics from 8,640 experimental runs
├── 02_transmission_analysis/    # Transmission, inversion, and seed analyses
├── 03_selection_analysis/       # Selection policies and matched missingness
├── 04_figure_generation/        # Plotting code and Figures 1-4
├── 05_validation_manifests/     # Checksums, provenance, and reproduction checks
├── README.md                   # Detailed reproduction guide and output mapping
└── requirements.txt
```

See the [detailed reproduction guide](analysis_support_upload/README.md) for the
mapping between scripts, output files, and manuscript tables and figures.

## Data and code availability

This release provides archived experimental metrics and the code used to
recompute the numerical analyses and figures. The `01_raw_data` directory holds
run-level result records rather than original time-series observations.

Original datasets should be obtained from the sources described in the
manuscript; access to SWaT is subject to its provider's request and approval
process. Neither raw nor processed SWaT time series are redistributed. Upstream
preprocessing and model-training code, intermediate imputation arrays, and trained
weights are outside this release. Historical source manifests document the
original experiments; some referenced upstream files are not included here.

## Manuscript

**Better Imputation Does Not Guarantee Better Forecasting: Transmission
Efficiency in Two-Step Time Series Pipelines**

Yu Chang, Qi Li, Sheng Feng, Huiqiao Hou, and Yujie Guo.

Repository: <https://github.com/SecretFzzf/shure>
