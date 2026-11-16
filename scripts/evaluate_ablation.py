"""Extended localisation evaluation used for the ablation tables and appendices.

Runs once over the evaluation split of the configured dataset and produces, in
addition to the methods of ``scripts/evaluate.py``:

* the channel-only variants (``svd1`` with and without sign correction) and the
  fused variant of SVD-Grad-CAM, used in the design-axis and fusion ablations
  (Tables 3 and 4 of the paper);
* IoU at binarisation thresholds 0.3, 0.5 and 0.7 (Appendix C);
* the retained gradient energy eta_1 .. eta_5 of every selected layer and the
  proposed method evaluated with k = 5 at every layer (Appendix D).

Usage::

    python -m scripts.evaluate_ablation --config configs/brisc.yaml
    python -m scripts.evaluate_ablation --config configs/voc.yaml

Outputs ``results/ablation_<dataset>.csv`` (one row per image and method) and
``results/energy_<dataset>.csv`` (one row per image). Summarise them with
``python -m scripts.make_tables_ablation``.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import tensorflow as tf

from svdfusion import cams, fusion, metrics
from svdfusion.data import load_brisc_sample, load_voc_sample
from svdfusion.gradients import GradientExtractor

IOU_THRESHOLDS = (0.3, 0.5, 0.7)
MAX_RANK = 5


def retained_energy(grad: np.ndarray, kmax: int = MAX_RANK) -> np.ndarray:
    """Cumulative fraction of gradient energy retained by the leading k modes."""
    h, w, c = grad.shape
    s = np.linalg.svd(grad.reshape(h * w, c), compute_uv=False)
    energy = s ** 2
    total = energy.sum()
    out = np.ones(kmax)
    if total > 0:
        cumulative = np.cumsum(energy) / total
        out[: min(kmax, len(cumulative))] = cumulative[:kmax]
    return out


def _resize(cam: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(cam, (image_size[1], image_size[0]))


def _image_ids(config: dict) -> list[str]:
    if config["dataset"] == "brisc":
        return sorted(f for f in os.listdir(config["images_dir"])
                      if f.lower().endswith((".jpg", ".png")))
    with open(config["split_file"]) as fh:
        ids = [line.strip() for line in fh if line.strip()]
    return ids[: config.get("max_images") or len(ids)]


def evaluate(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_size = tuple(config["image_size"])
    layers, deepest = config["layers"], config["deepest_layer"]
    k, scheme = config["rank_k"], config["fusion_scheme"]
    is_brisc = config["dataset"] == "brisc"

    model = tf.keras.models.load_model(config["model_path"], compile=False)
    extractor = GradientExtractor(model, layers + [deepest])
    deep_idx = extractor.layer_names.index(deepest)
    fuse_idx = [i for i, name in enumerate(extractor.layer_names) if name in layers]

    rows, energy_rows = [], []
    for image_id in tqdm(_image_ids(config), desc=config["dataset"]):
        sample = (load_brisc_sample(image_id, config["images_dir"], config["masks_dir"], image_size)
                  if is_brisc else load_voc_sample(image_id, config["voc_root"], model, image_size))
        if sample is None:
            continue
        feats, grads = extractor(sample.model_input, sample.class_index)
        feat_deep, grad_deep = feats[deep_idx], grads[deep_idx]
        area = float(sample.mask.mean())

        energy_row = {"image": image_id, "class": sample.label, "area": area}
        for i in fuse_idx:
            for j, value in enumerate(retained_energy(grads[i]), start=1):
                energy_row[f"{extractor.layer_names[i]}_eta{j}"] = float(value)
        energy_rows.append(energy_row)

        maps = {
            "SVD-Fusion CAM (multi-scale)": fusion.fuse(
                [cams.svd_fusion(feats[i], grads[i], k) for i in fuse_idx], image_size, scheme),
            "SVD-Fusion CAM (single layer)": _resize(cams.svd_fusion(feat_deep, grad_deep, k), image_size),
            "SVD-Fusion CAM (multi-scale, k=5)": fusion.fuse(
                [cams.svd_fusion(feats[i], grads[i], MAX_RANK) for i in fuse_idx], image_size, scheme),
            "SVD1 (raw sign)": _resize(cams.svd1(feat_deep, grad_deep, sign_correct=False), image_size),
            "SVD1 (sign-corrected)": _resize(cams.svd1(feat_deep, grad_deep, sign_correct=True), image_size),
            "SVD-Grad-CAM (fused)": fusion.fuse(
                [cams.svd_grad_cam(feats[i], grads[i]) for i in fuse_idx], image_size, scheme),
        }
        for name, fn in cams.BASELINE_METHODS.items():
            maps[name] = _resize(fn(feat_deep, grad_deep), image_size)

        for method, cam in maps.items():
            e = metrics.ebpg(cam, sample.mask)
            if e is None:
                continue
            row = {"image": image_id, "method": method, "class": sample.label, "area": area,
                   "ebpg": e, "lift": metrics.lift(cam, sample.mask),
                   "pg": metrics.pointing_game(cam, sample.mask)}
            for thr in IOU_THRESHOLDS:
                row[f"iou{int(thr * 100)}"] = metrics.iou(cam, sample.mask, thr)
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(energy_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    results, energy = evaluate(config)
    os.makedirs(args.out_dir, exist_ok=True)
    results.to_csv(os.path.join(args.out_dir, f"ablation_{config['dataset']}.csv"), index=False)
    energy.to_csv(os.path.join(args.out_dir, f"energy_{config['dataset']}.csv"), index=False)
    print(f"wrote {len(results)} rows for {results.image.nunique()} images to {args.out_dir}/")


if __name__ == "__main__":
    main()
