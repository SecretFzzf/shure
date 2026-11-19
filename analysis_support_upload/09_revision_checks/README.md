# Paired sensitivity and signal diagnostics

From the parent analysis directory:

```sh
python 09_revision_checks/recompute_paired_and_signal.py
python 09_revision_checks/plot_single_column_figures.py --out reproduced_figures
```

- paired_inversion_48blocks.csv contains the original six paired
  pure-inversion comparisons on common valid configurations.
- paired_selection_48blocks.csv contains the original seven selection
  comparisons. Regret is averaged as a percentage per task.
- Both tables resample whole dataset × mechanism × rate blocks within dataset
  for percentile intervals and apply a two-sided sign-flip test to blockwise
  sums. Each uses 20,000 Monte Carlo draws (seed 42). Holm adjustment is
  computed separately for each original test family. Reported differences
  retain the original pairing and denominators.
- signal_oof_sensitivity.csv uses the archived five held-out folds. Clipping
  bounds and standardisation are estimated on training rows in each fold.
- acf_covariate_overlap.csv reports the in-sample R-squared from regressing
  the ACF diagnostic on the baseline covariates. This describes overlap.
- matched_continuous_acf.csv compares task-level group means in the same
  180 matched tasks. Within each task, imputers are averaged within each
  inversion group before tasks receive equal weight.
