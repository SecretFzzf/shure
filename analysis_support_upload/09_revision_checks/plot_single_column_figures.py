"""Render the single-column threshold plot and inversion-rate heatmap."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def threshold(data, output):
    with (data / "threshold_sensitivity.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["metric"] == "MAE"]
    epsilon = [float(row["epsilon"]) for row in rows]
    pure = [float(row["pure_pct"]) for row in rows]
    overall = [float(row["global_pct"]) for row in rows]
    fig, ax = plt.subplots(figsize=(3.34, 2.38), layout="constrained")
    ax.plot(epsilon, pure, "o-", label="Pure inversion", color="#D55E00")
    ax.plot(epsilon, overall, "s-", label="Global inversion", color="#0072B2")
    ax.set(xlabel=r"Ratio threshold $\epsilon$", ylabel="Inversion rate (%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.23)
    fig.savefig(output)
    plt.close(fig)


def heatmap(data, output):
    with (data / "heatmap.csv").open(encoding="utf-8-sig", newline="") as file:
        rate = {(row["model"], row["impute_method"]): float(row["pure_pct"])
                for row in csv.DictReader(file)}
    imputers = ["spline", "knn", "brits", "saits"]
    models = ["LSTM", "DLinear", "PatchTST"]
    values = np.array([[rate[model, imputer] for imputer in imputers] for model in models])
    assert np.isfinite(values).all()
    fig, ax = plt.subplots(figsize=(3.34, 2.31), layout="constrained")
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto",
                      vmin=0, vmax=max(40, values.max()))
    ax.set_xticks(range(4), ["Spline", "KNN", "BRITS", "SAITS"])
    ax.set_yticks(range(3), models)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1)
    ax.tick_params(axis="both", which="both", length=0)
    for (row, column), value in np.ndenumerate(values):
        rgb = np.asarray(image.cmap(image.norm(value))[:3])
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        ink = "white" if linear @ np.array([.2126, .7152, .0722]) < .179 else "#161616"
        ax.text(column, row, f"{value:.1f}%", ha="center", va="center",
                color=ink, fontsize=9, fontweight="semibold")
    colorbar = fig.colorbar(image, ax=ax, fraction=.064, pad=.045)
    colorbar.set_label("Pure inversion (%)", labelpad=5)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(output)
    plt.close(fig)


def main():
    home = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=home.parent / "02_transmission_analysis")
    parser.add_argument("--out", type=Path, default=home)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 400,
    })
    threshold(args.data, args.out / "fig3.png")
    heatmap(args.data, args.out / "fig4.png")


if __name__ == "__main__":
    main()
