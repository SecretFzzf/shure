"""Recompute grouped paired tests and signal checks from Online Resource 1."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DRAWS = 20_000
BLOCK = ["dataset", "missing_mode", "missing_rate"]
UNIT = ["dataset", "model", "missing_mode", "missing_rate", "pred_len"]
KEY = UNIT + ["impute_method"]
INVERSION_PAIRS = [
    ("model", "LSTM", "DLinear"),
    ("model", "LSTM", "PatchTST"),
    ("model", "PatchTST", "DLinear"),
    ("impute_method", "spline", "saits"),
    ("impute_method", "spline", "knn"),
    ("impute_method", "brits", "saits"),
]
SELECTION_PAIRS = [
    ("regret_mse", "Test IE-MAE"),
    ("regret_mse", "Fixed spline"),
    ("regret_mse", "Test IE-MSE"),
    ("regret_mae", "Test IE-MAE"),
    ("regret_mae", "Fixed spline"),
    ("worse_mae", "Test IE-MAE"),
    ("worse_mae", "Fixed spline"),
]


def clustered_summary(pairs):
    blocks = pairs.groupby(BLOCK).delta.agg(["sum", "size"]).reset_index()
    totals = blocks["sum"].to_numpy()
    sizes = blocks["size"].to_numpy()
    rng = np.random.RandomState(SEED)
    indices = []
    for _, group in blocks.groupby("dataset"):
        rows = group.index.to_numpy()
        indices.append(rows[rng.randint(len(rows), size=(DRAWS, len(rows)))])
    draws = np.column_stack(indices)
    bounds = np.quantile(totals[draws].sum(1) / sizes[draws].sum(1), [.025, .975])
    flips = np.random.RandomState(SEED).choice([-1, 1], size=(DRAWS, len(totals)))
    null = flips @ totals
    observed = totals.sum()
    extreme = np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)
    return dict(n=len(pairs), blocks=len(blocks), difference=observed / sizes.sum(),
                lower=bounds[0], upper=bounds[1], p=(1 + extreme) / (DRAWS + 1))


def holm(rows):
    frame = pd.DataFrame(rows)
    order = np.argsort(frame.p)
    adjusted = np.minimum(1, np.maximum.accumulate(
        frame.p.to_numpy()[order] * np.arange(len(order), 0, -1)
    ))
    frame.loc[order, "holm_p"] = adjusted
    return frame


def paired_inversion(contrasts):
    valid = contrasts[contrasts.valid]
    rows = []
    for factor, left, right in INVERSION_PAIRS:
        keys = [key for key in KEY if key != factor]
        paired = valid.loc[valid[factor].eq(left), keys + ["pure"]].merge(
            valid.loc[valid[factor].eq(right), keys + ["pure"]],
            on=keys, suffixes=("_a", "_b"), validate="one_to_one",
        )
        paired["delta"] = 100 * (paired.pure_a.astype(int) - paired.pure_b.astype(int))
        rows.append(dict(a=left, b=right, **clustered_summary(paired)))
    return holm(rows)


def paired_selection(tasks):
    rows = []
    for metric, right in SELECTION_PAIRS:
        paired = tasks.loc[tasks.rule.eq("Val MSE"), UNIT + [metric]].merge(
            tasks.loc[tasks.rule.eq(right), UNIT + [metric]],
            on=UNIT, suffixes=("_a", "_b"), validate="one_to_one",
        )
        paired["delta"] = paired[metric + "_a"].astype(float) - paired[metric + "_b"].astype(float)
        if metric == "worse_mae":
            paired["delta"] *= 100
        rows.append(dict(metric=metric, a="Val MSE", b=right, **clustered_summary(paired)))
    return holm(rows)


def signal_sensitivity(signal, fold_predictions):
    folds = fold_predictions.loc[
        fold_predictions.scope.eq("Overall")
        & fold_predictions.metric.eq("delta_roughness_distortion"),
        KEY + ["fold", "baseline_prediction", "augmented_prediction"],
    ]
    joined = signal.merge(folds, on=KEY, validate="one_to_one")
    assert len(joined) == 2127 and joined.valid.all()
    encoded = pd.get_dummies(joined[KEY].astype(str), drop_first=True, dtype=float).to_numpy()
    base = np.column_stack([np.ones(len(joined)), encoded, joined.delta_ie])
    target = joined.delta_fe.to_numpy()
    total = np.square(target - target.mean()).sum()

    def predict(feature=None, limits=None):
        prediction = np.empty(len(target))
        for fold in range(5):
            test = joined.fold.to_numpy() == fold
            train = ~test
            design = base
            if feature is not None:
                used = feature
                if limits is not None:
                    used = np.clip(feature, *np.quantile(feature[train], limits))
                used = (used - used[train].mean()) / used[train].std()
                design = np.column_stack([base, used])
            weights = np.linalg.lstsq(design[train], target[train], rcond=None)[0]
            prediction[test] = design[test] @ weights
        return 1 - np.square(target - prediction).sum() / total

    baseline = predict()
    np.testing.assert_allclose(baseline, 1 - np.square(target - joined.baseline_prediction).sum() / total,
                               rtol=0, atol=1e-10)
    specs = [
        ("original amplitude", "delta_roughness_distortion", None),
        ("amplitude clipped to training percentiles 1–99", "delta_roughness_distortion", (.01, .99)),
        ("amplitude clipped to training percentiles 5–95", "delta_roughness_distortion", (.05, .95)),
        ("amplitude excludes near-flat windows", "delta_roughness_nonflat", None),
        ("original autocorrelation", "delta_acf_distortion", None),
    ]
    rows = []
    for definition, metric, limits in specs:
        r2 = predict(joined[metric].to_numpy(), limits)
        rows.append(dict(definition=definition, baseline_r2=baseline, r2=r2, incremental_r2=r2-baseline))
    acf = joined.delta_acf_distortion.to_numpy()
    fitted = base @ np.linalg.lstsq(base, acf, rcond=None)[0]
    acf_r2 = 1 - np.square(acf - fitted).sum() / np.square(acf - acf.mean()).sum()
    return pd.DataFrame(rows), acf_r2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=here.parent)
    parser.add_argument("--out", type=Path, default=here)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load = lambda name: pd.read_csv(args.root / name)
    contrasts = load("02_transmission_analysis/main_contrasts_2304.csv")
    tasks = load("03_selection_analysis/policy_units_4608.csv")
    signal = load("06_signal_diagnostics/signal_contrasts_2304.csv")
    folds = load("06_signal_diagnostics/signal_oof_predictions.csv")
    paired_inversion(contrasts).to_csv(args.out / "paired_inversion_48blocks.csv", index=False)
    paired_selection(tasks).to_csv(args.out / "paired_selection_48blocks.csv", index=False)
    sensitivity, acf_r2 = signal_sensitivity(signal, folds)
    sensitivity.to_csv(args.out / "signal_oof_sensitivity.csv", index=False)
    pd.DataFrame([dict(n=2127, acf_baseline_overlap_r2=acf_r2)]).to_csv(
        args.out / "acf_covariate_overlap.csv", index=False
    )
    match = signal.loc[signal.delta_ie.lt(-.01)].groupby(UNIT + ["pure"]).acf_distortion.mean()
    matched = match.unstack().dropna()
    pd.DataFrame([dict(n=len(matched), inversion=matched[True].mean(),
                       non_inversion=matched[False].mean(),
                       inversion_greater_count=int((matched[True] > matched[False]).sum()))]).to_csv(
        args.out / "matched_continuous_acf.csv", index=False
    )


if __name__ == "__main__":
    main()
