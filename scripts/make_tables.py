"""Rebuild the paper's summary tables from per-image results.

Reads the tidy CSV written by ``evaluate.py`` and produces the localisation
summary (Tables 6 and 8), the paired significance tables (Table 7 and
Table A.14) and the advantage by target size (Table 10).

Usage
-----
    python -m scripts.make_tables --results results/raw_brisc.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

METHOD_ORDER = [
    "SVD-Fusion CAM (proposed)",
    "SVD-Fusion CAM (single layer)",
    "HiResCAM",
    "Grad-CAM++",
    "LayerCAM",
    "Grad-CAM",
    "SVD-Grad-CAM",
    "XGrad-CAM",
    "Eigen-CAM",
]


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Median EBPG and lift with their quartiles, mean PG and IoU per method."""
    table = results.groupby("method").agg(
        EBPG_median=("ebpg", lambda s: 100 * s.median()),
        EBPG_q1=("ebpg", lambda s: 100 * s.quantile(0.25)),
        EBPG_q3=("ebpg", lambda s: 100 * s.quantile(0.75)),
        lift_median=("lift", "median"),
        lift_q1=("lift", lambda s: s.quantile(0.25)),
        lift_q3=("lift", lambda s: s.quantile(0.75)),
        PG=("pg", lambda s: 100 * s.mean()),
        IoU=("iou", lambda s: 100 * s.mean()),
        n=("ebpg", "count"),
    )
    order = [m for m in METHOD_ORDER if m in table.index]
    return table.reindex(order)


def paired_table(results: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Wilcoxon signed-rank comparison of the reference against every method.

    Reports win rate, matched-pairs rank-biserial effect size, and p-value for
    lift, pointing game, and IoU.
    """
    rows = []
    for metric in ("lift", "pg", "iou"):
        pivot = results.pivot_table(index="image", columns="method", values=metric)
        for method in pivot.columns:
            if method == reference:
                continue
            pair = pivot[[reference, method]].dropna()
            if len(pair) < 30:
                continue
            diff = (pair[reference] - pair[method]).values
            nonzero = diff[diff != 0]
            try:
                _, p = wilcoxon(pair[reference], pair[method])
            except ValueError:
                p = np.nan
            if len(nonzero):
                ranks = rankdata(np.abs(nonzero))
                effect = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
            else:
                effect = 0.0
            rows.append({
                "metric": metric,
                "vs": method,
                "win_rate_%": 100 * (diff > 0).mean(),
                "effect_r": effect,
                "p": p,
            })
    return pd.DataFrame(rows)


SIZE_BINS = [(0.0, 0.05, "0-5%"), (0.05, 0.10, "5-10%"),
             (0.10, 0.20, "10-20%"), (0.20, 0.35, "20-35%"),
             (0.35, 1.01, "35-100%")]


def size_law_table(results: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Median lift per disjoint size bin, with gain over the best baseline.

    Gain is the relative improvement in median lift of the reference over the
    strongest baseline in that bin: (lift_ref - lift_best) / lift_best.
    """
    variants = [m for m in results["method"].unique() if "SVD-Fusion" in m]
    baselines = [m for m in results["method"].unique() if m not in variants]
    rows = []
    for lo, hi, name in SIZE_BINS:
        sub = results[(results["area"] >= lo) & (results["area"] < hi)]
        if sub.empty:
            continue
        ref_lift = sub[sub["method"] == reference]["lift"].median()
        bl_lifts = {m: sub[sub["method"] == m]["lift"].median() for m in baselines}
        best_bl = max(bl_lifts, key=bl_lifts.get)
        best_lift = bl_lifts[best_bl]
        rows.append({
            "size_bin": name,
            "mean_area_%": 100 * sub[sub["method"] == reference]["area"].mean(),
            "n": sub[sub["method"] == reference].shape[0],
            "lift_ours": ref_lift,
            "best_baseline": best_bl,
            "best_lift": best_lift,
            "gain_%": 100 * (ref_lift - best_lift) / best_lift,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="per-image CSV")
    parser.add_argument("--reference", default="SVD-Fusion CAM (proposed)")
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    if "lift" not in results.columns:
        results["lift"] = results["ebpg"] / results["area"].clip(lower=1e-8)

    print("=" * 70)
    print("Localisation summary")
    print("=" * 70)
    print(summary_table(results).round(2).to_string())

    print("\n" + "=" * 70)
    print(f"Paired comparison (reference: {args.reference})")
    print("=" * 70)
    paired = paired_table(results, args.reference)
    for metric in ("lift", "pg", "iou"):
        sub = paired[paired.metric == metric]
        print(f"\n-- {metric} --")
        print(sub[["vs", "win_rate_%", "effect_r", "p"]].to_string(
            index=False,
            formatters={"win_rate_%": "{:.1f}".format,
                        "effect_r": "{:+.3f}".format,
                        "p": "{:.1e}".format}))

    print("\n" + "=" * 70)
    print("Advantage by target-size bin")
    print("=" * 70)
    print(size_law_table(results, args.reference).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
