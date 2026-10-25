"""Deletion and insertion curves for every CAM method.

Reproduces the faithfulness table of the paper (Table 11, Sec. 4.7). Pixels are removed
(deletion) or restored onto a blurred image (insertion) in order of the
saliency ranking, and the class score is averaged over the sweep. Deletion is
better when lower, insertion when higher.

Both perturbations operate on the *preprocessed* model input, so no second
preprocessing step is applied before the forward pass; applying one would place
the "removed" value in a different colour space and make the numbers
incomparable with the localisation results.

Usage
-----
    python -m scripts.faithfulness --config configs/brisc.yaml --n-images 180
    python -m scripts.faithfulness --config configs/voc.yaml --n-images 180
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from tqdm import tqdm

import tensorflow as tf

from svdfusion import cams, fusion
from svdfusion.data import load_brisc_sample, load_voc_sample
from svdfusion.gradients import GradientExtractor

REFERENCE = "SVD-Fusion CAM (proposed)"


def _resize(cam, image_size):
    return cv2.resize(cam, (image_size[1], image_size[0]))


def deletion_insertion(model, model_input, cam, class_index, steps=50):
    """Mean class score over the deletion and insertion sweeps.

    ``model_input`` is the preprocessed tensor of shape ``(1, H, W, 3)``.
    """
    image = model_input[0]
    height, width = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    order = np.argsort(-cam.flatten())
    per_step = int(np.ceil(height * width / steps))

    deleted, inserted = image.copy(), blurred.copy()
    del_batch, ins_batch = [deleted.copy()], [inserted.copy()]
    for step in range(steps):
        idx = order[step * per_step:(step + 1) * per_step]
        if len(idx):
            rows, cols = np.unravel_index(idx, (height, width))
            deleted[rows, cols, :] = 0.0
            inserted[rows, cols, :] = image[rows, cols, :]
        del_batch.append(deleted.copy())
        ins_batch.append(inserted.copy())

    d = model.predict(np.array(del_batch), verbose=0)[:, class_index]
    i = model.predict(np.array(ins_batch), verbose=0)[:, class_index]
    return float(d.mean()), float(i.mean())


def _image_ids(config, n_images):
    """Image ids in the same order used for the localisation tables.

    For BRISC the tumour classes are taken in turn so that the sample is
    balanced across them, matching the protocol described in the paper.
    """
    if config["dataset"] == "brisc":
        every = sorted(
            f for f in os.listdir(config["images_dir"])
            if f.lower().endswith((".jpg", ".png"))
        )

        def token(name):
            for part in os.path.splitext(name)[0].split("_"):
                if part in ("gl", "me", "pi", "no"):
                    return part
            return None

        per_class = max(1, n_images // 3)
        ids = []
        for tok in ("gl", "me", "pi"):
            ids.extend([f for f in every if token(f) == tok][:per_class])
        return ids

    with open(config["split_file"]) as fh:
        return [line.strip() for line in fh if line.strip()]


def run(config: dict, n_images: int, steps: int) -> pd.DataFrame:
    """Per-image deletion and insertion scores for every method."""
    image_size = tuple(config["image_size"])
    layers, deepest = config["layers"], config["deepest_layer"]
    k, scheme = config["rank_k"], config["fusion_scheme"]
    is_brisc = config["dataset"] == "brisc"

    model = tf.keras.models.load_model(config["model_path"], compile=False)
    extractor = GradientExtractor(model, layers + [deepest])
    deep_idx = extractor.layer_names.index(deepest)

    rows, done = [], 0
    progress = tqdm(total=n_images, desc=f"del/ins {config['dataset']}")
    for image_id in _image_ids(config, n_images):
        if done >= n_images:
            break
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

        per_layer = [
            cams.svd_fusion(f, g, k)
            for name, f, g in zip(extractor.layer_names, feats, grads)
            if name in layers
        ]
        maps = {
            REFERENCE: fusion.fuse(per_layer, image_size, scheme),
            "SVD-Fusion CAM (single layer)": _resize(
                cams.svd_fusion(feat_deep, grad_deep, k), image_size),
        }
        for name, fn in cams.BASELINE_METHODS.items():
            maps[name] = _resize(fn(feat_deep, grad_deep), image_size)

        for method, cam in maps.items():
            d, i = deletion_insertion(
                model, sample.model_input, cam, sample.class_index, steps)
            rows.append({"image": image_id, "method": method,
                         "deletion": d, "insertion": i})

        done += 1
        progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby("method")
        .agg(deletion_mean=("deletion", "mean"),
             deletion_median=("deletion", "median"),
             insertion_mean=("insertion", "mean"),
             insertion_median=("insertion", "median"),
             n=("deletion", "count"))
        .sort_values("insertion_mean", ascending=False)
    )


def paired(raw: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Wilcoxon signed-rank tests of the reference against every other method."""
    wide = raw.pivot(index="image", columns="method")
    rows = []
    for metric in ("deletion", "insertion"):
        block = wide[metric].dropna()
        ref = block[reference]
        for method in block.columns:
            if method == reference:
                continue
            other = block[method]
            try:
                _, p = wilcoxon(ref, other)
            except ValueError:          # identical vectors
                p = float("nan")
            better = ref < other if metric == "deletion" else ref > other
            rows.append({"metric": metric, "vs": method,
                         "mean_delta": float((other - ref).mean())
                         if metric == "deletion"
                         else float((ref - other).mean()),
                         "win_rate_pct": 100 * float(better.mean()),
                         "p": p, "n": len(ref)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--n-images", type=int, default=180,
                        help="number of images to sweep (paper uses 180)")
    parser.add_argument("--steps", type=int, default=50,
                        help="perturbation steps per curve (paper uses 50)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    raw = run(config, args.n_images, args.steps)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = config["dataset"]
    raw.to_csv(f"{args.out_dir}/faithfulness_{tag}_raw.csv", index=False)

    summary = summarise(raw)
    summary.to_csv(f"{args.out_dir}/faithfulness_{tag}.csv")
    print("\nDeletion (lower is better), insertion (higher is better)\n")
    print(summary.round(4).to_string())

    tests = paired(raw)
    tests.to_csv(f"{args.out_dir}/faithfulness_{tag}_paired.csv", index=False)
    print("\nPaired tests against the proposed method\n")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
