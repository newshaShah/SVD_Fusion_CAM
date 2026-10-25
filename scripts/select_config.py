"""Select the fusion configuration on the training split.

Reproduces the configuration search of the paper (Sec. 4.3, Tables 2 and 5).
Every combination of resolution levels, fusion weighting and rank ``k`` listed
under ``search`` in the config file is scored on images of the *training* split,
by the mean of lift (median), pointing game and IoU after min-max normalising
each across configurations. The best configuration is then fixed and used
unchanged on the evaluation split, so nothing is tuned on reported data.

Usage
-----
    python -m scripts.select_config --config configs/brisc.yaml
"""

from __future__ import annotations

import argparse
import itertools
import os

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

import tensorflow as tf

from svdfusion import cams, fusion, metrics
from svdfusion.data import load_brisc_sample
from svdfusion.gradients import GradientExtractor

SCHEMES = ["uniform", "depth", "inverse-depth"]


def candidate_configs(search: dict):
    """Yield ``(name, layers)`` pairs, layers ordered shallow to deep."""
    if "combos" in search:
        for name, layers in search["combos"].items():
            yield name, list(layers)
        return
    levels = search["levels"]                       # shallow to deep
    for r in range(1, len(levels) + 1):
        for subset in itertools.combinations(levels, r):
            yield "+".join(subset), list(subset)


def _image_ids(images_dir: str, per_class: int) -> list[str]:
    every = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    def token(name):
        for part in os.path.splitext(name)[0].split("_"):
            if part in ("gl", "me", "pi", "no"):
                return part
        return None

    ids = []
    for tok in ("gl", "me", "pi"):
        ids.extend([f for f in every if token(f) == tok][:per_class])
    return ids


def search(config: dict, per_class: int) -> pd.DataFrame:
    """Per-image metrics for every candidate configuration."""
    if config["dataset"] != "brisc":
        raise NotImplementedError("the search script covers BRISC only")
    spec = config["search"]
    image_size = tuple(config["image_size"])
    combos = list(candidate_configs(spec))
    k_values = spec.get("k_values", [1, 2, 3])
    layers = list(dict.fromkeys(l for _, ls in combos for l in ls))

    model = tf.keras.models.load_model(config["model_path"], compile=False)
    extractor = GradientExtractor(model, layers)

    rows = []
    for image_id in tqdm(_image_ids(spec["images_dir"], per_class), desc="search"):
        sample = load_brisc_sample(image_id, spec["images_dir"],
                                   spec["masks_dir"], image_size)
        if sample is None:
            continue
        feats, grads = extractor(sample.model_input, sample.class_index)
        area = float(sample.mask.mean())
        for k in k_values:
            per_layer = {
                name: cv2.resize(cams.svd_fusion(f, g, k),
                                 (image_size[1], image_size[0]))
                for name, f, g in zip(extractor.layer_names, feats, grads)
            }
            for combo, combo_layers in combos:
                schemes = ["uniform"] if len(combo_layers) == 1 else SCHEMES
                for scheme in schemes:
                    cam = fusion.fuse([per_layer[l] for l in combo_layers],
                                      image_size, scheme)
                    e = metrics.ebpg(cam, sample.mask)
                    if e is None:
                        continue
                    tag = f"{combo} | k={k}"
                    if len(combo_layers) > 1:
                        tag += f" | {scheme}"
                    rows.append({"image": image_id, "class": sample.label,
                                 "config": tag, "area": area, "ebpg": e,
                                 "lift": e / max(area, 1e-8),
                                 "pg": metrics.pointing_game(cam, sample.mask),
                                 "iou": metrics.iou(cam, sample.mask)})
    return pd.DataFrame(rows)


def rank(raw: pd.DataFrame) -> pd.DataFrame:
    """Composite score: mean of min-max normalised lift, PG and IoU."""
    table = raw.groupby("config").agg(
        lift_median=("lift", "median"),
        PG=("pg", lambda s: 100 * s.mean()),
        IoU=("iou", lambda s: 100 * s.mean()),
        n=("ebpg", "count"),
    )
    for col in ("lift_median", "PG", "IoU"):
        lo, hi = table[col].min(), table[col].max()
        table[col + "_n"] = (table[col] - lo) / (hi - lo + 1e-12)
    table["score"] = table[["lift_median_n", "PG_n", "IoU_n"]].mean(axis=1)
    return table.sort_values("score", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--per-class", type=int, default=250,
                        help="training images per tumour class (paper: 250)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    raw = search(config, args.per_class)
    table = rank(raw)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = config["dataset"]
    raw.to_csv(f"{args.out_dir}/search_{tag}_raw.csv", index=False)
    table.round(4).to_csv(f"{args.out_dir}/search_{tag}.csv")

    print("\nTop configurations on the training split\n")
    print(table[["lift_median", "PG", "IoU", "score", "n"]]
          .head(15).round(3).to_string())
    print(f"\nselected: {table.index[0]}")


if __name__ == "__main__":
    main()
