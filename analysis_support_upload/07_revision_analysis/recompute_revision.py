"""Reproduce the JDSA revision checks from the original ESM and ETTh1 runs."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr, t

KEY = ["dataset", "model", "missing_mode", "missing_rate", "pred_len", "impute_method"]
BLOCK = ["dataset", "missing_mode", "missing_rate"]
FINE = [column for column in KEY if column != "pred_len"]
MODELS = ["LSTM", "DLinear", "PatchTST"]
EPS = 0.01
SEED_COLUMNS = ["missing_seed", "imputation_training_seed", "pred_seed"]
ERRORS = ["ie_mae", "fe_mae", "ie_mse", "fe_mse"]


def esm_csv(archive, suffix):
    path, = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    with archive.open(path) as stream:
        return pd.read_csv(stream)


def contrasts(runs):
    errors = runs.groupby(KEY, as_index=False)[ERRORS].mean()
    reference = errors[errors.impute_method == "mean"].drop(columns="impute_method")
    compared = errors[errors.impute_method != "mean"].merge(
        reference, on=KEY[:-1], suffixes=("", "_mean"), validate="many_to_one"
    )
    compared["delta_ie"] = compared.ie_mae - compared.ie_mae_mean
    compared["delta_fe"] = compared.fe_mae - compared.fe_mae_mean
    compared["valid"] = compared.delta_ie.abs() > EPS
    compared["directional_pure"] = (compared.delta_ie < 0) & (compared.delta_fe > 0)
    compared["directional_global"] = compared.delta_ie * compared.delta_fe < 0
    compared["pure"] = compared.valid & compared.directional_pure
    compared["tau"] = compared.delta_fe / compared.delta_ie.where(compared.valid)
    return compared


def threshold_stats(data):
    rows = []
    for threshold in [0, 0.001, 0.005, 0.01, 0.02, 0.05]:
        valid = data[data.delta_ie.abs() > threshold]
        rows.append(dict(epsilon=threshold, n=len(valid), excluded=len(data) - len(valid),
                         pure=int(valid.directional_pure.sum()),
                         pure_pct=100 * valid.directional_pure.mean(),
                         global_pct=100 * valid.directional_global.mean(),
                         tau_median=(valid.delta_fe / valid.delta_ie).median()))
    rows.append(dict(epsilon="no_filter_signs_only", n=len(data), excluded=0,
                     pure=int(data.directional_pure.sum()),
                     pure_pct=100 * data.directional_pure.mean(),
                     global_pct=100 * data.directional_global.mean(),
                     tau_median=np.nan))
    return pd.DataFrame(rows)


def grouped_interval(frame, grouping, numerator, denominator, stratify=True):
    grouped = frame.groupby(grouping)[[numerator, denominator]].sum().reset_index()
    rng = np.random.RandomState(42)
    draws = np.zeros((20_000, 2))
    strata = [part for _, part in grouped.groupby("dataset")] if stratify else [grouped]
    for part in strata:
        totals = part[[numerator, denominator]].to_numpy(dtype=float)
        sampled = rng.randint(len(totals), size=(len(draws), len(totals)))
        draws += totals[sampled].sum(axis=1)
    assert (draws[:, 1] > 0).all()
    return np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5])


def bootstrap_rates(data):
    subsets = [("Overall", data)] + list(data.groupby("model"))
    rows = []
    for label, subset in subsets:
        for grouping, name, stratify in [(FINE, "horizon_group", False),
                                         (BLOCK, "coarse_dataset_stratified", True)]:
            subset = subset[subset.valid]
            low, high = grouped_interval(subset, grouping, "pure", "valid", stratify)
            rows.append(dict(scope=label, grouping=name, pct=100 * subset.pure.mean(),
                             low=100 * low, high=100 * high,
                             groups=subset[grouping].drop_duplicates().shape[0]))
    return pd.DataFrame(rows)


def logistic(data):
    data = data[data.valid].copy()
    reference = {"dataset": "ETTh1", "model": "DLinear", "missing_mode": "MCAR",
                 "missing_rate": 0.1, "pred_len": 24, "impute_method": "spline"}
    predictors = {"intercept": np.ones(len(data))}
    for factor, baseline in reference.items():
        for level in sorted(set(data[factor]) - {baseline}):
            predictors[f"{factor}: {level} vs {baseline}"] = (data[factor] == level).to_numpy()
    x = np.stack(list(predictors.values()), axis=1).astype(float)
    y = data.pure.to_numpy(dtype=float)

    def objective(beta):
        z = x @ beta
        return (np.logaddexp(0, z) - y * z).sum(), x.T @ (expit(z) - y)

    def hessian(beta):
        p = expit(x @ beta)
        return x.T @ (x * (p * (1 - p))[:, None])

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, hess=hessian,
                   method="trust-exact", options={"gtol": 1e-7})
    assert np.linalg.norm(fit.jac, ord=np.inf) < 1e-5, fit.message
    bread = np.linalg.inv(hessian(fit.x))
    score = pd.DataFrame(x * (y - expit(x @ fit.x))[:, None])
    group = pd.MultiIndex.from_frame(data[BLOCK])
    sums = score.groupby(group).sum().to_numpy()
    g, n, p = len(sums), *x.shape
    covariance = (g / (g - 1)) * ((n - 1) / (n - p)) * bread @ sums.T @ sums @ bread
    se = np.sqrt(np.diag(covariance))
    critical = t.ppf(0.975, g - 1)
    return pd.DataFrame(dict(term=list(predictors), log_odds=fit.x, cluster_se=se,
                             odds_ratio=np.exp(fit.x),
                             low=np.exp(fit.x - critical * se),
                             high=np.exp(fit.x + critical * se), n=n, clusters=g))


def mask_check(main, alternate):
    fields = KEY + SEED_COLUMNS + ERRORS
    standard = main[(main.dataset == "ETTh1") & (main.pred_seed == 42)].copy()
    assert len(standard) == len(alternate) == 720
    assert standard[SEED_COLUMNS].drop_duplicates().values.tolist() == [[42, 42, 42]]
    assert alternate[SEED_COLUMNS].drop_duplicates().values.tolist() == [[62, 42, 42]]
    assert standard.status.eq("success").all() and alternate.status.eq("success").all()
    fixed = ["protocol", "seq_len", "imputation_variant", "forecast_model_variant",
             "architecture_config", "training_config", "train_windows", "val_windows",
             "test_windows", "ie_definition", "imputer_config"]
    alignment = standard[KEY + fixed].merge(alternate[KEY + fixed], on=KEY,
                                            validate="one_to_one", suffixes=("_42", "_62"))
    assert len(alignment) == 720
    for field in fixed:
        left, right = alignment[field + "_42"], alignment[field + "_62"]
        if field == "imputer_config":
            for method in ["brits", "saits"]:
                rows = alignment.impute_method.eq(method)
                assert {json.loads(value)["epochs"] for value in left[rows]} == (
                    {20} if method == "brits" else {24}
                )
                assert {json.loads(value)["epochs"] for value in right[rows]} == {100}
            keep = alignment.impute_method.isin(["mean", "knn", "spline"])
            left, right = left[keep], right[keep]
        pd.testing.assert_series_equal(left, right,
                                       check_names=False, check_dtype=False)
    methods = ["mean", "knn", "spline"]
    standard = standard[standard.impute_method.isin(methods)]
    comparable = alternate[alternate.impute_method.isin(methods)]
    assert len(standard) == len(comparable) == 432
    base, alt = contrasts(standard), contrasts(comparable)
    matched = base.merge(alt, on=KEY, validate="one_to_one", suffixes=("_42", "_62"))
    valid = matched[matched.valid_42 & matched.valid_62]
    assert len(matched) == 288
    summary = {"shared_valid": len(valid),
               "changed_pure_in_shared": int((valid.pure_42 != valid.pure_62).sum())}
    for seed, values in [(42, base), (62, alt)]:
        summary[str(seed)] = {
            "n": int(values.valid.sum()), "count": int(values.pure.sum()),
            "pct": 100 * values.loc[values.valid, "pure"].mean(),
            "tau": values.tau.median(),
            "shared_count": int(valid[f"pure_{seed}"].sum()),
            "shared_tau": valid[f"tau_{seed}"].median(),
        }
    audit_input = alternate[fields + fixed + ["status"]].copy()
    audit_input["imputer_config"] = audit_input.imputer_config.map(
        lambda entry: json.dumps({key: value for key, value in json.loads(entry).items()
                                  if key != "training_details"}, sort_keys=True)
    )
    return summary, audit_input


def signals(archive, data):
    metrics = ["delta_roughness_distortion", "delta_acf_distortion"]
    joined = data.merge(esm_csv(archive, "signal_contrasts_2304.csv")[KEY + metrics],
                        on=KEY, validate="one_to_one")
    correlations = pd.DataFrame([
        dict(scope=model, metric=metric, rho=spearmanr(part[metric], part.delta_fe).statistic)
        for model, part in joined[joined.valid].groupby("model") for metric in metrics
    ])
    fit = esm_csv(archive, "signal_incremental_fit.csv")
    per_model = correlations.merge(fit, on=["scope", "metric"], validate="one_to_one")
    assert len(per_model) == 6
    joined = joined[joined.delta_ie < -EPS].copy()
    assert len(joined) == 1998 and joined.pure.sum() == 415
    paired_keys = KEY[:-1]
    rows = []
    for metric in metrics:
        joined["improved"] = joined[metric] < 0
        matched = joined.groupby(paired_keys + ["pure"]).improved.mean().unstack().dropna()
        matched["difference"] = matched[True] - matched[False]
        matched["tasks"] = 1
        matched = matched.reset_index()
        low, high = grouped_interval(matched, BLOCK, "difference", "tasks")
        rows.append(dict(metric=metric, matched_tasks=len(matched),
                         blocks=matched[BLOCK].drop_duplicates().shape[0],
                         inversion_pct=100 * matched[True].mean(),
                         improvement_pct=100 * matched[False].mean(),
                         diff_pp=100 * matched.difference.mean(), low=100 * low, high=100 * high))
    return per_model, pd.DataFrame(rows)


def plot_figure(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "savefig.dpi": 350})
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0), sharex="row", sharey="row")
    categories = [("Ratio excluded", "#888888", 0.22, 7),
                  ("Other valid", "#0072B2", 0.29, 7),
                  ("Pure inversion", "#D55E00", 0.68, 10)]
    for i, model in enumerate(MODELS):
        part = data[data.model.eq(model)]
        for row, ax in enumerate(axes[:, i]):
            ax.axvspan(-EPS, EPS, color="#E8E8E8", zorder=0)
            ax.axvline(0, color="#525252", lw=0.65, zorder=1)
            ax.axhline(0, color="#525252", lw=0.65, zorder=1)
            for select, (_, color, alpha, size) in zip(
                [~part.valid, part.valid & ~part.pure, part.pure], categories
            ):
                ax.scatter(part.loc[select, "delta_ie"], part.loc[select, "delta_fe"],
                           s=size, color=color, alpha=alpha, linewidths=0)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(model)
                ax.tick_params(labelbottom=True)
            else:
                ax.set(xlim=(-0.16, 0.10), ylim=(-0.065, 0.065),
                       xticks=[-0.15, -0.05, 0.05], yticks=[-0.06, 0, 0.06])
    for row, label in enumerate(["Full range", "Near origin"]):
        axes[row, 0].set_ylabel(label + "\n" + r"$\Delta\overline{\mathrm{FE}}$")
    fig.supxlabel(r"$\Delta\mathrm{IE}$", y=.025)
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=color, ms=5, label=label)
                        for label, color, _, _ in categories],
               loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.13, right=.985, bottom=.13, top=.86, hspace=.35, wspace=.19)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--mask62-dir", type=Path)
    inputs.add_argument("--mask62-input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.esm) as archive:
        raw = esm_csv(archive, "online_results_8640.csv")
        assert len(raw) == 8640 and raw.status.eq("success").all()
        assert not raw.duplicated(KEY + ["pred_seed"]).any()
        assert raw.groupby(KEY).pred_seed.agg(lambda values: set(values) == {42, 123, 2026}).all()
        assert raw.groupby(KEY).ie_mae.nunique().max() == 1
        data = contrasts(raw)
        assert (len(data), int(data.valid.sum()), int(data.pure.sum())) == (2304, 2127, 415)
        tables = {"thresholds": threshold_stats(data), "group_intervals": bootstrap_rates(data),
                  "adjusted_logistic": logistic(data)}
        tables["strata"] = pd.concat([
            data.groupby(factor).agg(total=("valid", "size"), valid=("valid", "sum"),
                                     pure=("pure", "sum"), tau=("tau", "median"))
            .reset_index(names="level").assign(factor=factor)
            for factor in ["dataset", "model", "impute_method", "missing_mode", "pred_len", "missing_rate"]
        ], ignore_index=True).assign(excluded=lambda table: table.total - table.valid,
                                    pure_pct=lambda table: 100 * table.pure / table.valid)
        tables["signal_by_model"], tables["matched_signal"] = signals(archive, data)
    if args.mask62_dir:
        alternate = pd.concat(
            [pd.read_csv(path).assign(dataset="ETTh1")
             for path in sorted(args.mask62_dir.glob("results_*_full.csv"))],
            ignore_index=True
        )
    else:
        alternate = pd.read_csv(args.mask62_input)
    mask, mask_input = mask_check(raw, alternate)
    if args.mask62_dir:
        mask_input.to_csv(args.out / "mask62_input.csv", index=False)
    for name, table in tables.items():
        table.to_csv(args.out / (name + ".csv"), index=False, float_format="%.12g")
    (args.out / "mask62.json").write_text(json.dumps(mask, indent=2), encoding="utf-8")
    plot_figure(data, args.out / "Fig2.png")


if __name__ == "__main__":
    main()
