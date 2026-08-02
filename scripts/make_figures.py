"""Generate the qualitative comparison figure for a dataset.

Produces a publication-quality grid: rows are example images, columns are
methods. The ground-truth region is drawn as a contour on every panel and the
per-panel EBPG is printed, so the figure is semi-quantitative rather than purely
visual.

Sample selection is explicit and reported, so that the selection rule can be
stated in the caption. Two rules are available:

    margin      cases where the proposed method exceeds the strongest baseline by
                the largest margin in lift (makes the comparison legible)
    percentile  cases spanning the proposed method's own performance distribution

Usage
-----
    python -m scripts.make_figures --config configs/brisc.yaml \
        --results results/raw_brisc.csv --out figures/qualitative_brisc.pdf
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf

from svdfusion import cams, fusion, metrics
from svdfusion.data import load_brisc_sample, load_voc_sample
from svdfusion.gradients import GradientExtractor

# Columns shown in the figure, in order. HiResCAM is kept because it is the
# strongest baseline; the two weakest baselines are omitted to keep panels large.
FIGURE_COLUMNS = [
    "SVD-Fusion CAM (proposed)",
    "SVD-Fusion CAM (single layer)",
    "HiResCAM",
    "SVD-Grad-CAM",
    "Grad-CAM",
    "Grad-CAM++",
    "LayerCAM",
]
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_WIDTH = 2
HEAT_ALPHA = 0.45


def _resize(cam, image_size):
    return cv2.resize(cam, (image_size[1], image_size[0]))


def _draw_contour(image, mask):
    out = image.copy()
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(out, contours, -1, CONTOUR_COLOR, CONTOUR_WIDTH)
    return out


def _overlay(rgb, cam, mask):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = (heat * HEAT_ALPHA + rgb * (1 - HEAT_ALPHA)).astype(np.uint8)
    return _draw_contour(blended, mask)


def select_samples(results, ref_method, mode, n_rows, rows_per_class,
                   is_brisc, max_area):
    """Choose example images and report the selection to stdout."""
    if "lift" not in results.columns:
        results = results.assign(
            lift=results["ebpg"] / results["area"].clip(lower=1e-8))

    pivot = results.pivot_table(index="image", columns="method", values="lift")
    variants = [c for c in pivot.columns if "SVD-Fusion" in c]
    baselines = [c for c in pivot.columns if c not in variants]

    stats = pd.DataFrame({
        "lift": pivot[ref_method],
        "best_baseline": pivot[baselines].max(axis=1),
    }).dropna()
    stats["margin"] = stats["lift"] - stats["best_baseline"]

    meta_cols = ["area"] + (["class"] if "class" in results.columns else [])
    meta = results[results["method"] == ref_method].set_index("image")[meta_cols]
    stats = stats.join(meta, how="inner")

    picked = []

    def take(subset, count, group):
        if mode == "margin":
            chosen = subset.sort_values("margin", ascending=False).head(count)
            for image_id, row in chosen.iterrows():
                picked.append({"id": str(image_id), "group": group,
                               "margin": row["margin"], "lift": row["lift"],
                               "best": row["best_baseline"]})
        else:
            ordered = subset.sort_values("lift").reset_index()
            for q in (0.50, 0.80, 0.95)[:count]:
                row = ordered.iloc[int(q * (len(ordered) - 1))]
                picked.append({"id": str(row["image"]), "group": group,
                               "margin": row["margin"], "lift": row["lift"],
                               "best": row["best_baseline"]})

    if is_brisc:
        for cls in ("glioma", "meningioma", "pituitary"):
            take(stats[stats["class"] == cls], rows_per_class, cls)
    else:
        subset = stats[stats["area"] < max_area]
        if len(subset) < n_rows:
            subset = stats
        take(subset, n_rows, "object")

    print(f"Selection mode: {mode}")
    for p in picked:
        print(f"  {p['group']:<12s} lift={p['lift']:.2f} "
              f"best baseline={p['best']:.2f} margin={p['margin']:+.2f}  {p['id']}")
    return picked


def compute_maps(sample, extractor, layers, deep_idx, k, scheme, image_size):
    feats, grads = extractor(sample.model_input, sample.class_index)
    feat_deep, grad_deep = feats[deep_idx], grads[deep_idx]
    per_layer = [
        cams.svd_fusion(f, g, k)
        for name, f, g in zip(extractor.layer_names, feats, grads)
        if name in layers
    ]
    maps = {
        "SVD-Fusion CAM (proposed)": fusion.fuse(per_layer, image_size, scheme),
        "SVD-Fusion CAM (single layer)": _resize(
            cams.svd_fusion(feat_deep, grad_deep, k), image_size),
    }
    for name, fn in cams.BASELINE_METHODS.items():
        maps[name] = _resize(fn(feat_deep, grad_deep), image_size)
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True, help="per-image CSV")
    parser.add_argument("--out", required=True, help="output figure path (.pdf)")
    parser.add_argument("--mode", default="margin", choices=["margin", "percentile"])
    parser.add_argument("--n-rows-voc", type=int, default=6)
    parser.add_argument("--rows-per-class", type=int, default=1)
    parser.add_argument("--max-area-voc", type=float, default=0.20)
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    image_size = tuple(config["image_size"])
    layers = config["layers"]
    is_brisc = config["dataset"] == "brisc"
    ref = "SVD-Fusion CAM (proposed)"

    results = pd.read_csv(args.results)
    picked = select_samples(results, ref, args.mode, args.n_rows_voc,
                            args.rows_per_class, is_brisc, args.max_area_voc)

    model = tf.keras.models.load_model(config["model_path"], compile=False)
    extractor = GradientExtractor(model, layers + [config["deepest_layer"]])
    deep_idx = extractor.layer_names.index(config["deepest_layer"])

    rendered = []
    for item in picked:
        if is_brisc:
            sample = load_brisc_sample(
                item["id"], config["images_dir"], config["masks_dir"], image_size)
        else:
            sample = load_voc_sample(
                item["id"], config["voc_root"], model, image_size)
        if sample is None:
            continue
        maps = compute_maps(sample, extractor, layers, deep_idx,
                            config["rank_k"], config["fusion_scheme"], image_size)
        rendered.append((sample, maps, item))

    columns = [c for c in FIGURE_COLUMNS if c in rendered[0][1]]
    n_cols = 1 + len(columns)
    fig, axes = plt.subplots(
        len(rendered), n_cols, figsize=(2.1 * n_cols, 2.4 * len(rendered)))
    axes = np.atleast_2d(axes)

    for r, (sample, maps, item) in enumerate(rendered):
        axes[r, 0].imshow(_draw_contour(sample.display_rgb, sample.mask))
        axes[r, 0].set_ylabel(
            f"{sample.label}\n{100 * sample.mask.mean():.1f}% of image\n"
            f"[+{item['margin']:.1f} over best baseline]",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=52)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
        if r == 0:
            axes[r, 0].set_title("Input\n(ground truth)", fontsize=9)

        for c, name in enumerate(columns, start=1):
            ax = axes[r, c]
            ax.imshow(_overlay(sample.display_rgb, maps[name], sample.mask))
            ax.axis("off")
            if r == 0:
                ax.set_title(name.replace(" CAM ", " CAM\n"), fontsize=9)
            e = metrics.ebpg(maps[name], sample.mask)
            ax.text(0.5, -0.06, f"EBPG {100 * e:.1f}%",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
