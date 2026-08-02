"""Generate the analytical figures from summary CSVs.

Unlike ``make_figures.py`` (which renders qualitative heatmap grids and needs the
model), this script draws the quantitative trend figures directly from the CSVs
produced by the evaluation pipeline. It needs no GPU and runs in seconds.

Figures produced (when the corresponding CSV is present):
    crossover   paired multi-scale minus single-layer difference vs target size
    size_law    fusion gain in pointing game vs target size
    sanity      cascading weight-randomisation correlation by stage
    per_size    median lift of every method in each target-size bin

Usage
-----
    python -m scripts.make_analysis_figures \
        --crossover results/crossover_voc.csv \
        --sanity results/sanity_brisc.csv \
        --per-size results/persize_voc.csv results/persize_brisc.csv \
        --out-dir figures/
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        kwargs = {"dpi": 200} if ext == "png" else {}
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", **kwargs)
    print(f"  wrote {name}.pdf and {name}.png")


def figure_crossover(path: str, out_dir: str) -> None:
    """Paired multi-scale minus single-layer difference, by target size.

    Smaller targets are on the left. The pointing-game curve is rescaled for
    display. EBPG and IoU cross zero as targets grow; the pointing game does not.
    The x-axis is not inverted, so smaller targets sit on the left throughout.
    """
    d = pd.read_csv(path).sort_values("mean_area")
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, style, label, scale in [
            ("Δ_ebpg", "-o", "EBPG", 1.0),
            ("Δ_iou",  "-^", "IoU", 1.0),
            ("Δ_pg",   "-s", r"Pointing game ($\div 5$)", 0.2)]:
        if col in d.columns:
            ax.plot(d["mean_area"], d[col] * scale, style,
                    linewidth=2.2, markersize=7, label=label)
    ax.axhline(0, color="crimson", ls="--", linewidth=1.4, label="no difference")
    ax.set_xscale("log")
    ax.set_xlabel("mean target area (% of image, log scale)")
    ax.set_ylabel(r"paired $\Delta$  (multi-scale $-$ single layer)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "FIG_crossover")


def figure_size_law(path: str, out_dir: str) -> None:
    """Fusion gain in pointing game as a function of target size.

    This quantity is defined within the method (multi-scale vs single-layer) and
    is independent of the baselines.
    """
    d = pd.read_csv(path)
    voc = d[d["dataset"] == "VOC"].sort_values("mean_area") if "dataset" in d.columns else d
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(voc["mean_area"], voc["pg_gain_pts"], "-o", lw=2.5, ms=8, label="VOC")
    if "dataset" in d.columns:
        br = d[d["dataset"] == "BRISC"]
        if len(br):
            ax.scatter(br["mean_area"], br["pg_gain_pts"], s=200, marker="*",
                       color="crimson", zorder=5, label="BRISC")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("mean target area (% of image, log scale)")
    ax.set_ylabel("PG gain: multi-scale $-$ single layer (points)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "FIG_size_law")


def figure_sanity(path: str, out_dir: str) -> None:
    """Cascading weight-randomisation correlation by stage.

    A faithful explanation should degrade as the model is randomised; methods
    whose curves stay high behave as edge detectors.
    """
    d = pd.read_csv(path, index_col=0)
    short = {"SVD-Fusion CAM (proposed)": "Ours (multi-scale)",
             "SVD-Fusion CAM (single layer)": "Ours (single layer)",
             "EigenCAM": "Eigen-CAM"}
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for col in d.columns:
        label = short.get(col, col)
        ours = "Ours" in label
        ax.plot(d.index, d[col], "-o" if ours else "--s",
                linewidth=2.8 if ours else 1.4,
                markersize=7 if ours else 5, label=label)
    ax.axhline(0.5, color="gray", ls=":", linewidth=1)
    ax.set_ylabel(r"$|\rho_{\mathrm{Spearman}}|$ vs original map")
    ax.set_xlabel("cascading randomisation stage")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "FIG_sanity")


def figure_per_size(paths: list, out_dir: str) -> None:
    """Median lift of every method in each disjoint target-size bin.

    Takes one or two per-size CSVs with columns ``bin``, ``method`` and
    ``lift``; with two, they are drawn as side-by-side panels. This is the
    appendix figure of the paper.
    """
    method_order = [
        ("SVD-Fusion CAM (proposed)", "Ours (multi-scale)"),
        ("SVD-Fusion CAM (single layer)", "Ours (single layer)"),
        ("HiResCAM", "HiResCAM"),
        ("SVD-Grad-CAM", "SVD-Grad-CAM"),
        ("Grad-CAM", "Grad-CAM"),
        ("Grad-CAM++", "Grad-CAM++"),
        ("LayerCAM", "LayerCAM"),
        ("XGrad-CAM", "XGrad-CAM"),
        ("Eigen-CAM", "Eigen-CAM"),
    ]
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    tables = []
    for path in paths:
        frame = pd.read_csv(path)
        table = frame.pivot(index="method", columns="bin", values="lift")
        # keep bins in ascending order of the lower edge
        order = sorted(table.columns,
                       key=lambda b: float(str(b).split("–")[0].strip("%")))
        tables.append((table[order], order, os.path.basename(path)))

    widths = [max(2, len(order)) for _, order, _ in tables]
    fig, axes = plt.subplots(1, len(tables), figsize=(2.6 * sum(widths), 4.6),
                             gridspec_kw={"width_ratios": widths},
                             squeeze=False)
    for ax, (table, order, title) in zip(axes[0], tables):
        n = len(method_order)
        x = range(len(order))
        width = 0.9 / n
        for i, ((key, _), colour) in enumerate(zip(method_order, colours)):
            if key not in table.index:
                continue
            values = [table.loc[key, b] for b in order]
            ax.bar([xi + (i - (n - 1) / 2) * width for xi in x],
                   values, width, color=colour)
        ax.axhline(1.0, color="0.45", linestyle=":", linewidth=1.0, zorder=0)
        ax.set_xticks(list(x))
        ax.set_xticklabels(order)
        ax.set_xlabel("target size bin")
        ax.set_ylabel(r"median lift (EBPG $\div$ chance)")
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours]
    labels = [label for _, label in method_order]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    _save(fig, out_dir, "FIG_per_size")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossover", help="crossover CSV with Δ_ebpg, Δ_iou, Δ_pg")
    parser.add_argument("--size-law", help="size-law CSV with mean_area, pg_gain_pts")
    parser.add_argument("--sanity", help="sanity CSV, stages as rows, methods as columns")
    parser.add_argument("--per-size", nargs="*",
                        help="one or two per-size CSVs with bin, method, lift")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    print("Generating analytical figures...")
    if args.crossover:
        figure_crossover(args.crossover, args.out_dir)
    if args.size_law:
        figure_size_law(args.size_law, args.out_dir)
    if args.sanity:
        figure_sanity(args.sanity, args.out_dir)
    if args.per_size:
        figure_per_size(args.per_size, args.out_dir)
    if not any([args.crossover, args.size_law, args.sanity, args.per_size]):
        parser.error("provide at least one of --crossover, --size-law, "
                     "--sanity, --per-size")


if __name__ == "__main__":
    main()
