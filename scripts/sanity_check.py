"""Cascading weight randomisation (sanity check).

Reproduces the sanity table of the paper (Table 12, Sec. 4.7). Weights are randomised
from the classifier head backwards, one stage at a time and cumulatively, and
each method's map is compared against the map from the trained model by rank
correlation. A method whose correlation stays high once the classifier can no
longer classify anything is responding to image structure rather than to what
the model learned.

Lower is better: it means the explanation depends more on the learned weights.

The randomisation is repeated for several seeds (by default 0-4, as in the
paper) and the table reports the mean and standard deviation across them; a
single draw can shift the correlations of the gradient-based methods by more
than the differences between them.

Usage
-----
    python -m scripts.sanity_check --config configs/brisc.yaml --per-class 20
    python -m scripts.sanity_check --config configs/brisc.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from tqdm import tqdm

import tensorflow as tf

from svdfusion import cams, fusion
from svdfusion.data import load_brisc_sample, load_voc_sample
from svdfusion.gradients import GradientExtractor


def _resize(cam, image_size):
    return cv2.resize(cam, (image_size[1], image_size[0]))


def all_maps(feats, grads, layer_names, config, image_size):
    """Maps for the proposed method (fused) and every baseline (deepest layer)."""
    layers, deepest = config["layers"], config["deepest_layer"]
    k, scheme = config["rank_k"], config["fusion_scheme"]
    deep_idx = layer_names.index(deepest)
    feat_deep, grad_deep = feats[deep_idx], grads[deep_idx]

    per_layer = [
        cams.svd_fusion(f, g, k)
        for name, f, g in zip(layer_names, feats, grads)
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


def randomise(model, layer_names):
    """Replace the kernels of the named layers with fresh random weights."""
    for name in layer_names:
        try:
            layer = model.get_layer(name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        fresh = []
        for w in weights:
            if w.ndim >= 2:
                init = tf.keras.initializers.GlorotUniform()
                fresh.append(init(shape=w.shape).numpy().astype(w.dtype))
            else:
                fresh.append(np.zeros_like(w))
        layer.set_weights(fresh)


def _check_identical(extractor, copy_extractor, samples, tol=1e-5):
    """Raise if the untouched copy does not reproduce the trained model."""
    if not samples:
        raise RuntimeError("no usable images for the sanity check")
    sample = next(iter(samples.values()))
    feats_a, _ = extractor(sample.model_input, sample.class_index)
    feats_b, _ = copy_extractor(sample.model_input, sample.class_index)
    diff = max(float(np.abs(a - b).max()) for a, b in zip(feats_a, feats_b))
    print(f"  untouched copy, max feature difference: {diff:.2e}")
    if diff > tol:
        raise RuntimeError(
            f"model copy differs from the trained model (max diff {diff:.2e})")


def build_stages(model, blocks):
    """Randomisation stages, from the classifier head backwards."""
    names = [layer.name for layer in model.layers]

    def convs(prefix):
        return [n for n in names if n.startswith(prefix)
                and isinstance(model.get_layer(n), tf.keras.layers.Conv2D)]

    stages = [
        ("head", [layer.name for layer in model.layers
                  if isinstance(layer, tf.keras.layers.Dense)]),
        ("top_conv", [n for n in names if n == "top_conv"]),
    ]
    stages += [(b, convs(b)) for b in blocks]
    return [(name, layers) for name, layers in stages if layers]


def _image_ids(config, per_class):
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

        ids = []
        for tok in ("gl", "me", "pi"):
            ids.extend([f for f in every if token(f) == tok][:per_class])
        return ids

    with open(config["split_file"]) as fh:
        return [line.strip() for line in fh if line.strip()][:3 * per_class]


def run(config: dict, per_class: int, blocks, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Per-seed correlations in long format: seed, stage, method, rho."""
    image_size = tuple(config["image_size"])
    layers, deepest = config["layers"], config["deepest_layer"]
    is_brisc = config["dataset"] == "brisc"

    model = tf.keras.models.load_model(config["model_path"], compile=False)
    needed = list(dict.fromkeys(layers + [deepest]))

    def load(image_id):
        if is_brisc:
            return load_brisc_sample(
                image_id, config["images_dir"], config["masks_dir"], image_size)
        return load_voc_sample(image_id, config["voc_root"], model, image_size)

    # Reference maps from the trained model.
    extractor = GradientExtractor(model, needed)
    reference, samples = {}, {}
    for image_id in tqdm(_image_ids(config, per_class), desc="reference"):
        sample = load(image_id)
        if sample is None:
            continue
        feats, grads = extractor(sample.model_input, sample.class_index)
        reference[image_id] = all_maps(
            feats, grads, extractor.layer_names, config, image_size)
        samples[image_id] = sample

    rows = []
    for seed in seeds:
        tf.keras.utils.set_random_seed(seed)
        # The randomised copy is made by loading the model file a second time.
        # tf.keras.models.clone_model + set_weights does not give an identical
        # copy for this model (top_conv activations differed by up to ~20 before
        # any weight was touched), which inflated the correlations of every
        # method. Verify the copy before randomising anything.
        randomised = tf.keras.models.load_model(config["model_path"], compile=False)
        _check_identical(extractor, GradientExtractor(randomised, needed), samples)

        for stage_name, stage_layers in build_stages(model, blocks):
            randomise(randomised, stage_layers)      # cumulative, by design
            stage_extractor = GradientExtractor(randomised, needed)
            scores = {}
            for image_id, sample in samples.items():
                try:
                    feats, grads = stage_extractor(
                        sample.model_input, sample.class_index)
                    maps = all_maps(feats, grads, stage_extractor.layer_names,
                                    config, image_size)
                except Exception:
                    continue
                for method, cam in maps.items():
                    rho, _ = spearmanr(reference[image_id][method].flatten(),
                                       cam.flatten())
                    if not np.isnan(rho):
                        scores.setdefault(method, []).append(abs(rho))
            for method, values in scores.items():
                rows.append({"seed": seed, "stage": stage_name, "method": method,
                             "rho": float(np.mean(values)), "n": len(values)})
            print(f"  seed {seed}, done: {stage_name}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--per-class", type=int, default=20,
                        help="images per class (paper uses 20, i.e. n=60)")
    parser.add_argument("--blocks", nargs="*",
                        default=["block7", "block6", "block5", "block4", "block3"],
                        help="convolutional block prefixes, deepest first")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="seeds for the random weights (paper uses 0-4)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    raw = run(config, args.per_class, args.blocks, args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = config["dataset"]
    raw.to_csv(f"{args.out_dir}/sanity_{tag}_seeds.csv", index=False)
    order = list(dict.fromkeys(raw["stage"]))
    mean = raw.pivot_table(index="stage", columns="method", values="rho",
                           aggfunc="mean").reindex(order)
    std = raw.pivot_table(index="stage", columns="method", values="rho",
                          aggfunc="std").reindex(order)
    out = f"{args.out_dir}/sanity_{tag}.csv"
    mean.round(4).to_csv(out)
    std.round(4).to_csv(f"{args.out_dir}/sanity_{tag}_std.csv")
    print(f"\n|Spearman| against the trained model's map (lower is better), "
          f"mean over seeds {args.seeds}\n")
    print(mean.round(3).to_string())
    print("\nstandard deviation across seeds\n")
    print(std.round(3).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
