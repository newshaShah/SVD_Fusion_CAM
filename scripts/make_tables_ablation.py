"""Print the ablation tables and appendix tables from ``scripts/evaluate_ablation.py``.

Usage::

    python -m scripts.make_tables_ablation --results-dir results

For each dataset found in ``results/ablation_<dataset>.csv`` this prints:

* Table 3  - design axes at the deepest layer (Grad-CAM, SVD1 raw, SVD1
             sign-corrected, proposed single-layer variant);
* Table 4  - fusion applied to the scalar (SVD-Grad-CAM) and spatial weighting;
* App. C   - mean IoU at thresholds 0.3 / 0.5 / 0.7, full split and small targets;
* App. D   - retained gradient energy eta_k (median [IQR]) and the proposed
             method with k = 5;
* paired Wilcoxon tests of every method against the multi-scale variant, with
  Holm-corrected p-values.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

REFERENCE = "SVD-Fusion CAM (multi-scale)"
ORDER = [
    "SVD-Fusion CAM (multi-scale)", "SVD-Fusion CAM (single layer)",
    "SVD-Fusion CAM (multi-scale, k=5)",
    "HiResCAM", "Grad-CAM++", "LayerCAM", "Grad-CAM", "PCG-CAM", "SVD-Grad-CAM",
    "SVD-Grad-CAM (fused)", "XGrad-CAM", "Eigen-CAM",
    "SVD1 (raw sign)", "SVD1 (sign-corrected)",
]


def summary(df: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    table = df.groupby("method").agg(
        lift_median=("lift", "median"),
        PG=("pg", lambda s: 100 * s.mean()),
        IoU30=("iou30", lambda s: 100 * s.mean()),
        IoU50=("iou50", lambda s: 100 * s.mean()),
        IoU70=("iou70", lambda s: 100 * s.mean()),
        n=("ebpg", "count"),
    )
    return table.reindex([m for m in (methods or ORDER) if m in table.index])


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_tests(df: pd.DataFrame, metric_names=("lift", "pg", "iou50")) -> pd.DataFrame:
    rows = []
    for metric in metric_names:
        pivot = df.pivot_table(index="image", columns="method", values=metric)
        for method in pivot.columns:
            if method == REFERENCE:
                continue
            pair = pivot[[REFERENCE, method]].dropna()
            diff = (pair[REFERENCE] - pair[method]).values
            nonzero = diff[diff != 0]
            try:
                p = wilcoxon(pair[REFERENCE], pair[method]).pvalue
            except ValueError:
                p = np.nan
            r = 0.0
            if len(nonzero):
                ranks = rankdata(np.abs(nonzero))
                r = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
            rows.append({"metric": metric, "vs": method, "win_rate_%": 100 * (diff > 0).mean(),
                         "r": r, "p": p, "n": len(pair)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].fillna(1.0).values)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    for dataset in ("brisc", "voc"):
        try:
            df = pd.read_csv(f"{args.results_dir}/ablation_{dataset}.csv")
        except FileNotFoundError:
            continue
        n_images = df[df.method == REFERENCE].shape[0]
        print("\n" + "=" * 80 + f"\n{dataset.upper()}  (n = {n_images})\n" + "=" * 80)

        print("\n-- Table 3: design axes, deepest layer only --")
        print(summary(df, ["Grad-CAM", "SVD1 (raw sign)", "SVD1 (sign-corrected)",
                           "SVD-Fusion CAM (single layer)"])[["lift_median", "PG", "IoU50", "n"]].round(2))

        print("\n-- Table 4: fusion x weighting --")
        print(summary(df, ["SVD-Grad-CAM", "SVD-Grad-CAM (fused)",
                           "SVD-Fusion CAM (single layer)", REFERENCE])[["lift_median", "PG", "IoU50", "n"]].round(2))

        print("\n-- Appendix C: IoU at three thresholds, full split --")
        print(summary(df)[["IoU30", "IoU50", "IoU70", "n"]].round(1))
        print("\n-- Appendix C: IoU at three thresholds, targets below 10% of the image --")
        print(summary(df[df.area < 0.10])[["IoU30", "IoU50", "IoU70", "n"]].round(1))

        print("\n-- Appendix D: selected k versus k = 5 --")
        print(summary(df, [REFERENCE, "SVD-Fusion CAM (multi-scale, k=5)"])[["lift_median", "PG", "IoU50"]].round(2))
        try:
            energy = pd.read_csv(f"{args.results_dir}/energy_{dataset}.csv")
            print("\n-- Appendix D: retained gradient energy eta_k, median [IQR] --")
            for column in [c for c in energy.columns if "_eta" in c]:
                q = energy[column].quantile([0.25, 0.5, 0.75]).values
                print(f"  {column:32s} {q[1]:.3f} [{q[0]:.3f}-{q[2]:.3f}]")
        except FileNotFoundError:
            pass

        print("\n-- Paired tests against the multi-scale variant --")
        tests = paired_tests(df)
        print(tests.to_string(index=False, formatters={
            "win_rate_%": "{:.1f}".format, "r": "{:+.3f}".format,
            "p": "{:.1e}".format, "p_holm": "{:.1e}".format}))


if __name__ == "__main__":
    main()
