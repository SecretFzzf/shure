# Validation metric and temporal dependence analyses

This directory supports Sections 3.7, 4.4 and 4.6 and Appendix H. The
8,640 original run metrics remain in the same archive under
`analysis_support_upload/01_raw_data/`.

From the directory containing the extracted archive and its ZIP, run:

```sh
python analysis_support_upload/08_validation_temporal/recompute_server_followup.py --esm ESM_1.zip --inputs analysis_support_upload/08_validation_temporal --out reproduced_followup
```

This requires NumPy 2.3.5 and pandas 3.0.1. It matches the selected fields to
the original run archive, reconstructs the three-seed selections, reports
task-relative regrets, and computes the within-dataset paired block
intervals. For each task and metric the percentage regret is
100 times the excess test error divided by the lowest test error among the
same candidates. Error averages use the restored checkpoints. Tests enter
evaluation only. The temporal check matches the full-origin contrast labels
against the original archive.

`r13_mbb_summary_L4_8_16.csv` reports the temporal interval outputs. The
following command reruns the interval procedure if the original per-origin
error arrays are available:

```sh
python analysis_support_upload/08_validation_temporal/recompute_temporal_intervals.py --origin-stats origin_stats --out reproduced_temporal
```

Temporal reconstruction uses the original Mean, Spline and KNN imputations
and frozen checkpoints. It requires separate per-origin, per-feature absolute
error sums and eligible cell counts for both historical IE and target FE.
The NPZ inputs follow the names `{ETT,KDD,SWaT,Weather}_h{24,48,96,192}_{fe,ie}_per_origin.npz`.
The per-origin arrays, raw series, imputation windows and checkpoints are not
included here. `pilot_verification_36.csv` records the pre-recovery checks.
