# Exploratory signal diagnostics

These materials support the manuscript's exploratory signal-diagnostic appendix
and its brief cross-reference in Results. No additional imputer or forecasting model is trained. The measures
are descriptive; no causal mechanism or new-dataset generality is claimed.

The 960 signal rows map to 2,304 non-Mean contrasts, of which 2,127 satisfy the
manuscript validity rule. Repeated horizons and forecasters are not independent
upstream cache realisations. Read the appendix for support rules and limitations.

## Numerical reproduction from included summaries

Install this directory's requirements in an isolated environment. From the root
of analysis_support_upload, copy this folder to a new output directory and run:

```powershell
Copy-Item -Recurse 06_signal_diagnostics new_signal_outputs
python 06_signal_diagnostics/run_signal_diagnostics.py --root . --results 01_raw_data/online_results_8640.csv --out new_signal_outputs --resume-metrics --bootstrap 2000
```

With --resume-metrics, --root is a required but unused argument: summaries are
recomputed from the complete included signal_metrics_960.csv. The command writes
new contrasts, associations, regression results and the scatter figure. The
included summary-bar image is an auxiliary rendering of the numerical tables.

## Repeating extraction

Without --resume-metrics, --root must point to the original dataset collection
containing ETT_online_42, KDD_online, SWaT_online and Weather_online, including
the matching raw CSVs and frozen NPZ caches. Those inputs are not redistributed.
The script reads arrays and masks without changing them. Spline uses V3 caches.

The original extraction audit records 32 unreadable KDD main-directory files
and the readable same-name cy/ alternatives used. A subsequent check of a fresh
KDD copy verified that all 60 test-array SHA-256 hashes match the arrays actually
used for these diagnostics, including those 32 alternatives.

signal_diagnostic_audit.json and independent_metric_check.json describe the
original calculation stage. Their historical manuscript_modified=false and
github_modified=false flags refer to that stage, before manuscript integration.

## Outputs

- signal_associations.csv: pooled/model-stratified correlations and sensitivities.
- signal_incremental_fit.csv and signal_oof_predictions.csv: grouped OOF results.
- signal_contrasts_2304.csv: configuration-level signals and inversion flags.
- signal_metrics_960.csv: extracted signal summaries and support coverage.
- signal_ie_reproduction.csv: agreement with archived reconstruction errors.
- signal_stratified_correlations.csv and signal_dataset_influence.csv: heterogeneity.
- signal_diagnostic_audit.json: cache selection, protocol and extraction audit.

All CSVs contain metrics or configuration records, not raw or reconstructed
time-series arrays. Neither raw nor processed SWaT time series are redistributed.

`signal_improvement_reference_groups.csv` reports descriptive improvement
proportions in pure inversions, all other valid contrasts, and the subset with
improved reconstruction but no forecast deterioration. These denominators give
context to the 97.11% autocorrelation-improvement proportion.
Recompute it from signal_contrasts_2304.csv with:

```python
import pandas as pd
c = pd.read_csv("signal_contrasts_2304.csv")
v = c[c.valid]
for name, g in [("Pure inversion", v[v.pure]),
                ("Other valid contrasts", v[~v.pure]),
                ("IE improved, FE not increased", v[(v.delta_ie < 0) & (v.delta_fe <= 0)])]:
    print(name, len(g), (g.delta_acf_distortion < 0).sum(),
          (g.delta_roughness_distortion < 0).sum())
```
