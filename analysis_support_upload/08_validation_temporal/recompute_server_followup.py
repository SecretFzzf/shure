import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

UNIT = ["dataset", "model", "missing_mode", "missing_rate", "pred_len"]
KEY = UNIT + ["impute_method", "pred_seed"]
ARCHIVE = "analysis_support_upload/01_raw_data/online_results_8640.csv"
FEATURES = ["val_macro_mae", "val_pooled_mse", "test_fe_mae", "test_fe_mse"]


def match_original(rows, original):
    joined = rows.merge(
        original[KEY + ["best_epoch", "best_val_loss", "fe_mae", "fe_mse"]],
        on=KEY, how="left", suffixes=("", "_original"), validate="one_to_one",
    )
    pooled = "orig_best_val_loss" if "orig_best_val_loss" in rows else "val_pooled_mse"
    for source, archived in [
        ("best_epoch", "best_epoch_original"), (pooled, "best_val_loss"),
        ("test_fe_mae", "fe_mae"), ("test_fe_mse", "fe_mse"),
    ]:
        np.testing.assert_allclose(joined[source], joined[archived], rtol=0, atol=1e-14)


def outcomes(rows, rules, candidate_count):
    counts = rows.groupby(UNIT + ["impute_method"]).pred_seed.agg(set)
    assert counts.map(lambda seeds: seeds == {42, 123, 2026}).all()
    averaged = rows.groupby(UNIT + ["impute_method"])[
        rules + ["test_fe_mae", "test_fe_mse"]
    ].mean().unstack()
    assert (averaged.notna().sum(axis=1) == (len(rules) + 2) * candidate_count).all()
    choices = {rule: averaged[rule].idxmin(axis=1) for rule in rules}
    for candidate in ["spline", "mean"]:
        choices[f"fixed_{candidate}"] = pd.Series(candidate, index=averaged.index)
    frames = {}
    for rule, choice in choices.items():
        frame = pd.DataFrame({"selected": choice})
        for metric in ["mae", "mse"]:
            errors = averaged[f"test_fe_{metric}"]
            frame[metric] = errors.to_numpy()[
                np.arange(len(errors)), errors.columns.get_indexer(choice)
            ]
            oracle = errors.min(axis=1)
            frame[f"relative_{metric}_regret_pct"] = 100 * (frame[metric] - oracle) / oracle
        frame["better_mae_than_mean"] = frame.mae < averaged.test_fe_mae["mean"]
        frame["worse_mae_than_mean"] = frame.mae > averaged.test_fe_mae["mean"]
        frame["changed_from_archived_mse"] = choice != choices.get(
            "orig_best_val_loss", choices["val_pooled_mse"]
        )
        frames[rule] = frame
    return pd.concat(frames, names=["rule"])


def paired_intervals(tasks):
    columns = ["mae", "mse", "relative_mae_regret_pct", "relative_mse_regret_pct"]
    rng = np.random.default_rng(42)
    output = []
    for reference in ["val_pooled_mse", "fixed_spline"]:
        differences = tasks.loc["val_macro_mae", columns] - tasks.loc[reference, columns]
        blocks = differences.groupby(level=["dataset", "missing_mode", "missing_rate"]).mean()
        datasets = blocks.index.get_level_values("dataset").unique()
        draws = np.zeros((20_000, len(columns)))
        for dataset in datasets:
            values = blocks.loc[dataset].to_numpy()
            indices = rng.integers(0, len(values), (len(draws), len(values)))
            draws += values[indices].mean(axis=1) / len(datasets)
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        output.extend(zip([reference] * len(columns), columns, differences.mean(), low, high))
    return pd.DataFrame(output, columns=["reference", "metric", "difference", "lower", "upper"])


def verify_contrasts(recovered, original):
    classical = original[original.impute_method.isin(["mean", "spline", "knn"])]
    aggregated = classical.groupby(UNIT + ["impute_method"])[["ie_mae", "fe_mae"]].mean()
    baseline = aggregated.xs("mean", level="impute_method")
    contrast = aggregated.drop("mean", level="impute_method").sub(baseline)
    check = recovered.set_index(UNIT + ["impute_method"]).join(contrast, validate="one_to_one")
    assert len(check) == 1152 and check.ie_mae.notna().all()
    valid = check.ie_mae.abs() > 0.01
    invert = valid & (check.ie_mae < 0) & (check.fe_mae > 0)
    assert (valid == check.is_valid).all() and (invert == check.is_pure_inversion).all()
    for recovered_name, original_name in [("delta_ie", "ie_mae"), ("delta_fe", "fe_mae")]:
        assert (np.sign(check[recovered_name]) == np.sign(check[original_name])).all()
    print(f"Matched temporal subset: {len(check)} contrasts, {valid.sum()} valid, {invert.sum()} inversions.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.esm) as archive, archive.open(ARCHIVE) as file:
        original = pd.read_csv(file)
    args.out.mkdir(parents=True, exist_ok=True)
    five = pd.read_csv(args.inputs / "validation_macro_mae_original.csv")
    assert len(five) == 6480 and (five.val_loss_curve == five.val_pooled_mse).all()
    match_original(five, original)
    tasks = outcomes(five, ["val_macro_mae", "val_pooled_mse"], 5)
    assert len(tasks.loc["val_macro_mae"]) == 432
    tasks.to_csv(args.out / "selection_432_tasks.csv")
    tasks.drop(columns="selected").groupby(level="rule").mean().to_csv(
        args.out / "selection_432_summary.csv"
    )
    paired_intervals(tasks).to_csv(args.out / "selection_432_block_intervals.csv", index=False)
    for dataset in ["KDD", "ETT"]:
        three = pd.read_csv(args.inputs / f"{dataset}_val_summary_1296.csv")
        assert len(three) == 1296
        match_original(three, original)
        rules = ["val_macro_mae", "val_pooled_mse", "val_macro_mse", "orig_best_val_loss"]
        small = outcomes(three, rules, 3)
        assert len(small.loc["val_macro_mse"]) == 144
        assert not small.loc["val_pooled_mse", "changed_from_archived_mse"].any()
        small.drop(columns="selected").groupby(level="rule").mean().to_csv(
            args.out / f"selection_{dataset}_144_summary.csv"
        )
    verify_contrasts(pd.read_csv(args.inputs / "r13_full_origins_reconstructed_1152.csv"), original)


if __name__ == "__main__":
    main()
