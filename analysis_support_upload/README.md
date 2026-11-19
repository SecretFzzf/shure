# Online Resource 1

This archive contains the 8,640 original run records, the analysis scripts,
and the figure assets used in the revised manuscript. These records support
numerical reproduction. The original series, historical preprocessing and
model-training implementations, imputed input windows and checkpoints are
outside this release. SWaT must be obtained from its provider under the
provider's data access conditions.

| Directory | Contents |
| --- | --- |
| 01_raw_data | Archived run metrics and configuration audit |
| 02_transmission_analysis | Paired contrasts, transmission, seed checks and dataset influence |
| 03_selection_analysis | Selection, regret and matched-mechanism comparisons |
| 04_figure_generation | Final figures and the original plotting code |
| 06_signal_diagnostics | Archived signal measures, held-out predictions and diagnostic code |
| 07_revision_analysis | Revision statistics, adjusted logistic regression, ETTh1 mask check and updated Figure 2 recipe |
| 08_validation_temporal | Validation selector inputs and summaries, reconstructed contrast labels and temporal intervals |
| 09_revision_checks | Coarser paired tests, signal sensitivities and updated single-column figure scripts |

## Recompute

From this directory, install the dependencies and run the archived analysis
in order. Scripts in directories 02 and 03 write their derived CSVs beside
the inputs by default. Use an extracted working copy.

```sh
python -m pip install -r requirements.txt
python 02_transmission_analysis/recompute_three_seeds.py
python 02_transmission_analysis/recompute_model_seed_consistency.py
python 02_transmission_analysis/recompute_dataset_influence.py
python 03_selection_analysis/recompute_selection.py
python 09_revision_checks/recompute_paired_and_signal.py
```

The final PNGs in 04 match the clean manuscript. Recreate Figure 1 using
```sh
python 04_figure_generation/plot_three_seeds.py --only-figure-one --out reproduced_figures
```
Figure 2 is recreated by the script in [07_revision_analysis](07_revision_analysis/README.md).
Figures 3 and 4 can be regenerated with
```sh
python 09_revision_checks/plot_single_column_figures.py --out reproduced_figures
```

[Validation and temporal follow-up](08_validation_temporal/README.md) contains
the commands and scope for the follow-up sensitivity checks.
[Coarser paired comparisons](09_revision_checks/README.md) describes the
latest script and derived evidence.

## Mapping to the revised manuscript

| Manuscript item | Evidence | Notes |
| --- | --- | --- |
| Table 1 · Dataset characteristics | [input_audit.json](01_raw_data/input_audit.json) | Metadata also in Section 3.1 |
| Table 2 · Experimental grid | [online_results_8640.csv](01_raw_data/online_results_8640.csv) | Original run-level input |
| Table 3 · Forecast errors by dataset and imputer | [method_performance.csv](02_transmission_analysis/method_performance.csv) |  |
| Table 4 · Transmission and inversion summaries | [headline_and_dataset.csv](02_transmission_analysis/headline_and_dataset.csv) |  |
| Table 5 · Threshold and metric sensitivity | [threshold_sensitivity.csv](02_transmission_analysis/threshold_sensitivity.csv) |  |
| Table 6 · Robust ratio summaries | [summary.json](02_transmission_analysis/summary.json) | Related statistics also in the run-level analysis outputs |
| Table 7 · Anchor sensitivity | [anchor_sensitivity.csv](02_transmission_analysis/anchor_sensitivity.csv) |  |
| Table 8 · Stratified inversion | [strata.csv](02_transmission_analysis/strata.csv) |  |
| Table 9 · Horizon-specific results | [strata.csv](07_revision_analysis/strata.csv) | Rows for prediction horizon |
| Table 10 · Original paired comparison tests | [paired_comparisons.csv](02_transmission_analysis/paired_comparisons.csv) |  |
| Table 11 · Forecasting-seed sign agreement | [seed_consistency.csv](02_transmission_analysis/seed_consistency.csv) |  |
| Table 12 · Imputer selection | [policy_summary.csv](03_selection_analysis/policy_summary.csv) | Paired tests in the same directory |
| Table 13 · Deep-imputer fitting budgets | [online_results_8640.csv](01_raw_data/online_results_8640.csv) | Fields: imputer_config |
| Table 14 · Shared SAITS settings | [online_results_8640.csv](01_raw_data/online_results_8640.csv) | Fields: imputer_config |
| Table 15 · Shared BRITS settings | [online_results_8640.csv](01_raw_data/online_results_8640.csv) | Fields: imputer_config |
| Table 16 · Forecasting architectures | [online_results_8640.csv](01_raw_data/online_results_8640.csv) | Fields: architecture_config |
| Table 17 · Forecasting training settings | [online_results_8640.csv](01_raw_data/online_results_8640.csv) | Fields: training_config |
| Table 18 · Seed and model breakdown | [seed_model_statistics.csv](02_transmission_analysis/seed_model_statistics.csv) |  |
| Table 19 · Dataset influence | [leave_one_dataset_out.csv](02_transmission_analysis/leave_one_dataset_out.csv) |  |
| Table 20 · Pooled signal diagnostics | [signal_associations.csv](06_signal_diagnostics/signal_associations.csv) | Out-of-fold results in signal_incremental_fit.csv |
| Table 21 · Signal diagnostics by architecture | [signal_by_model.csv](07_revision_analysis/signal_by_model.csv) |  |
| Table 22 · Full adjusted logistic regression | [adjusted_logistic.csv](07_revision_analysis/adjusted_logistic.csv) |  |
| Table 23 · Paired tests with 48 blocks | [paired_inversion_48blocks.csv](09_revision_checks/paired_inversion_48blocks.csv) | Selection comparisons in paired_selection_48blocks.csv |
| Figure 1 | [fig1.png](04_figure_generation/fig1.png) | Final manuscript asset |
| Figure 2 | [fig2.png](04_figure_generation/fig2.png) | Final manuscript asset |
| Figure 3 | [fig3.png](04_figure_generation/fig3.png) | Final manuscript asset |
| Figure 4 | [fig4.png](04_figure_generation/fig4.png) | Final manuscript asset |
