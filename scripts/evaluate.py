"""Evaluate all CAM methods on a dataset and write per-image results.

Runs the seven baselines and the two proposed variants on every image of the
configured dataset, computing localisation metrics for each, and writes a tidy
CSV with one row per (image, method). Because CAM evaluation is deterministic,
re-running reproduces the numbers in the paper's tables exactly.

Usage
-----
    python -m scripts.evaluate --config configs/brisc.yaml
    python -m scripts.evaluate --config configs/voc.yaml
"""

from __future__ import annotations

import argparse
import os

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

import tensorflow as tf

from svdfusion import cams, fusion, metrics
from svdfusion.data import load_brisc_sample, load_voc_sample
from svdfusion.gradients import GradientExtractor


def _resize(cam, image_size):
    return cv2.resize(cam, (image_size[1], image_size[0]))


def _brisc_image_ids(images_dir):
    return sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png"))
    )


def _voc_image_ids(split_file):
    with open(split_file) as fh:
        return [line.strip() for line in fh if line.strip()]


def evaluate(config: dict) -> pd.DataFrame:
    """Evaluate every method on every image and return a tidy results frame."""
    image_size = tuple(config["image_size"])
    layers = config["layers"]
    deepest = config["deepest_layer"]
    k = config["rank_k"]
    scheme = config["fusion_scheme"]
    is_brisc = config["dataset"] == "brisc"

    model = tf.keras.models.load_model(config["model_path"], compile=False)
    extractor = GradientExtractor(model, layers + [deepest])
    deep_idx = extractor.layer_names.index(deepest)

    if is_brisc:
        image_ids = _brisc_image_ids(config["images_dir"])
    else:
        image_ids = _voc_image_ids(config["split_file"])
    # The paper reports VOC on the first 1500 images of the val list.
    if config.get("max_images"):
        image_ids = image_ids[:config["max_images"]]

    rows = []
    for image_id in tqdm(image_ids, desc=config["dataset"]):
        if is_brisc:
            sample = load_brisc_sample(
                image_id, config["images_dir"], config["masks_dir"], image_size)
        else:
            sample = load_voc_sample(
                image_id, config["voc_root"], model, image_size)
        if sample is None:
            continue

        feats, grads = extractor(sample.model_input, sample.class_index)
        feat_deep, grad_deep = feats[deep_idx], grads[deep_idx]
        area = float(sample.mask.mean())

        # Proposed: multi-scale fusion of the spatial-field method.
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
        # Baselines, all on the deepest layer.
        for name, fn in cams.BASELINE_METHODS.items():
            maps[name] = _resize(fn(feat_deep, grad_deep), image_size)

        for method, cam in maps.items():
            e = metrics.ebpg(cam, sample.mask)
            if e is None:
                continue
            row = {
                "image": image_id,
                "method": method,
                "class": sample.label,
                "area": area,
                "ebpg": e,
                "lift": metrics.lift(cam, sample.mask),
                "pg": metrics.pointing_game(cam, sample.mask),
                "iou": metrics.iou(cam, sample.mask),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    results = evaluate(config)
    out_path = config["output_csv"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results.to_csv(out_path, index=False)

    summary = (
        results.groupby("method")
        .agg(
            EBPG_median=("ebpg", lambda s: 100 * s.median()),
            lift_median=("lift", "median"),
            PG=("pg", lambda s: 100 * s.mean()),
            IoU=("iou", lambda s: 100 * s.mean()),
            n=("ebpg", "count"),
        )
        .sort_values("lift_median", ascending=False)
    )
    print(f"\nResults written to {out_path}\n")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
