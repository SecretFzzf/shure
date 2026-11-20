"""Recompute the ETTh1 paired upstream-seed table from run-level metrics."""

import argparse
import csv
from decimal import Decimal
from itertools import product
from pathlib import Path
from statistics import median
import sys


CASES = {(42, 123): "A1", (42, 2026): "A2", (62, 42): "B"}
KEY = ("model", "missing_mode", "missing_rate", "pred_len", "impute_method")
THRESHOLD = Decimal("0.01")
EXPECTED = set(product(
    ("DLinear", "LSTM", "PatchTST"),
    ("MCAR", "MAR_Block", "MNAR"),
    ("0.1", "0.3", "0.5", "0.7"),
    ("24", "48", "96", "192"),
    ("brits", "saits"),
))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def index(rows, columns):
    values = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in values:
            raise ValueError(f"Duplicate configuration {key}")
        if row["status"] != "success":
            raise ValueError(f"Unsuccessful configuration {key}")
        values[key] = row
    return values


def contrast(run, mean):
    ie = Decimal(run["ie_mae"]) - Decimal(mean["ie_mae"])
    fe = Decimal(run["fe_mae"]) - Decimal(mean["fe_mae"])
    valid = abs(ie) > THRESHOLD
    return valid, ie < 0 and fe > 0, fe / ie if valid else None


def summarise(directory):
    original = [
        row for row in read_rows(directory.parent / "01_raw_data" / "online_results_8640.csv")
        if row["dataset"] == "ETTh1" and row["pred_seed"] == "42" and row["missing_seed"] == "42"
    ]
    mask62 = (
        row for row in read_rows(directory / "mask62_input.csv")
        if row["dataset"] == "ETTh1" and row["pred_seed"] == "42" and row["missing_seed"] == "62"
    )
    means = index(
        (row for row in [*original, *mask62] if row["impute_method"] == "mean"),
        ("missing_seed", *KEY[:-1]),
    )
    baseline = index(
        (row for row in original if row["impute_method"] in {"brits", "saits"}),
        KEY,
    )
    new = index(
        read_rows(directory / "r26_seed_sensitivity_run_metrics.csv"),
        ("missing_seed", "imputation_training_seed", *KEY),
    )
    if len(means) != 288 or set(baseline) != EXPECTED or len(new) != 864:
        raise ValueError("Unexpected ETTh1 grid or Mean reference coverage")

    for (mask, init), group in CASES.items():
        for method in ("all", "brits", "saits"):
            keys = sorted(key for key in EXPECTED if method == "all" or key[-1] == method)
            shared = []
            for key in keys:
                old = baseline[key]
                fresh = new[(str(mask), str(init), *key)]
                if old["imputation_training_seed"] != "42" or fresh["pred_seed"] != "42":
                    raise ValueError(f"Unexpected seed for {key}")
                a = contrast(old, means[("42", *key[:-1])])
                b = contrast(fresh, means[(str(mask), *key[:-1])])
                if a[0] and b[0]:
                    shared.append((a, b))
            n = len(shared)
            base_count = sum(a[1] for a, b in shared)
            new_count = sum(b[1] for a, b in shared)
            gains = sum(not a[1] and b[1] for a, b in shared)
            losses = sum(a[1] and not b[1] for a, b in shared)
            yield {
                "group": group, "imputer": method, "missing_seed": mask,
                "imputation_training_seed": init, "pred_seed": 42,
                "total": len(keys), "joint_valid": n,
                "baseline_pure": base_count, "new_pure": new_count,
                "baseline_pct": f"{100 * base_count / n:.4f}",
                "new_pct": f"{100 * new_count / n:.4f}",
                "difference_pp": f"{100 * (new_count - base_count) / n:.4f}",
                "changed": gains + losses, "changed_pct": f"{100 * (gains + losses) / n:.4f}",
                "gains": gains, "losses": losses,
                "baseline_tau_median": f"{median(a[2] for a, b in shared):.6f}",
                "new_tau_median": f"{median(b[2] for a, b in shared):.6f}",
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args()
    rows = list(summarise(Path(__file__).resolve().parent))
    writer = csv.DictWriter(args.out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
