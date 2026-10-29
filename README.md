# SVD-Fusion CAM

Reference implementation of **SVD-Fusion CAM: Hierarchical Robust Visual
Explanations via Singular Value Decomposition**.

SVD-Fusion CAM is a class activation mapping method for convolutional networks.
Instead of averaging each channel's gradient map into a scalar weight, it applies
a singular value decomposition to the joint spatial–channel gradient matrix and
keeps the low-rank reconstruction as a *spatial weight field*. Maps from
different resolution levels are then fused, and a cell-coverage criterion
indicates when a higher-resolution level is needed.

This repository contains the method, the eight baselines it is compared with,
and the scripts behind the paper's localisation, faithfulness and sanity-check
results.

---

## Method in brief

For a target class and a convolutional layer, let `G ∈ R^{N×C}` be the gradient
matrix whose rows index the `N = H·W` spatial locations and whose columns index
the `C` channels. Grad-CAM reduces each column to its mean, which is a projection
onto the uniform spatial direction. SVD-Fusion CAM instead takes the rank-`k`
reconstruction of `G` and reshapes each column back to `H×W`, giving one spatial
weight map per channel:

```
Ĝ_k = Σ_{i≤k} σᵢ uᵢ vᵢᵀ
L   = ReLU( Σ_j reshape(Ĝ_k[:, j]) ⊙ A_j )
```

The rank-`k` reconstruction is the best rank-`k` approximation of `G` in the
Frobenius norm (Eckart–Young–Mirsky). Because it is built from the products
`uᵢvᵢᵀ`, it is invariant to the sign ambiguity of the singular vectors, unlike a
channel-weight-only variant, which needs an external sign convention.

**Layer selection.** The precision of an explanation is bounded by the number of
feature-map cells the target occupies, `n_l = ρ · H_l · W_l`. When the deepest
layer supplies too few cells (empirically fewer than about four), a
higher-resolution level is worth fusing in. The criterion depends only on the
feature-map size, so it is not tied to a particular backbone. In the paper the
exact levels, the fusion weighting and `k` were chosen by a search on the
training split (`scripts/select_config.py`).

---

## Installation

```bash
git clone https://github.com/<user>/svd-fusion-cam.git
cd svd-fusion-cam
pip install -r requirements.txt
pip install -e .
```

Python 3.9+ is required. The methods themselves depend only on NumPy and OpenCV;
TensorFlow is needed to extract gradients from a trained model and to run the
scripts.

---

## Quick start

```python
import cv2
import tensorflow as tf
from svdfusion import cams, fusion, metrics
from svdfusion.data import load_brisc_sample
from svdfusion.gradients import GradientExtractor

model = tf.keras.models.load_model("models/efficientnet_brisc.keras", compile=False)
extractor = GradientExtractor(model, ["block3c_project_conv", "top_conv"])

sample = load_brisc_sample("case.jpg", "images/", "masks/")
feats, grads = extractor(sample.model_input, sample.class_index)

# single layer: the raw map is at the layer's resolution (8x8 here),
# so resize it to the image size before computing metrics or overlaying
cam_single = cv2.resize(cams.svd_fusion(feats[-1], grads[-1], k=3), (240, 240))

# multi-scale fusion: fuse() upsamples internally
per_layer = [cams.svd_fusion(f, g, k=3) for f, g in zip(feats, grads)]
cam_multi = fusion.fuse(per_layer, output_size=(240, 240), scheme="uniform")

print(metrics.lift(cam_multi, sample.mask))
```

---

## Final configurations (paper, Table 2)

| Dataset | Backbone | Levels | Rank `k` | Fusion weights |
|---|---|---|---|---|
| BRISC 2025 | EfficientNet-B1, 240×240 | `block3c_project_conv` (30×30) + `top_conv` (8×8) | 3 | uniform (0.50 / 0.50) |
| PASCAL VOC 2012 | EfficientNet-B0, 224×224 | `block3b_project_conv` (28×28) + `top_conv` (7×7) | 1 | depth (0.33 / 0.67) |

These are stored in `configs/brisc.yaml` and `configs/voc.yaml`. All baselines
are computed on the deepest convolutional layer, `top_conv`.

---

## Reproducing the paper

Run every command from the repository root. Gradients and maps are computed
deterministically, so up to floating-point differences between TensorFlow
versions and hardware, these commands reproduce the reported numbers. The
sanity check draws random weights; it runs seeds 0-4 by default and reports the
mean and standard deviation across them, as in the paper (`--seeds` changes the
set).

```bash
# 0. (optional) train the classifiers; only the trained models are needed later
python -m scripts.train_brisc --data-dir data/BRISC/brisc2025/classification_task \
    --out models/efficientnet_brisc.keras
python -m scripts.train_voc --voc-root data/VOCdevkit/VOC2012 \
    --out models/efficientnet_voc.h5

# 1. (optional) configuration search on the BRISC training split (Tables 2 and 5)
python -m scripts.select_config --config configs/brisc.yaml --per-class 250

# 2. localisation for all ten methods (writes results/raw_*.csv)
python -m scripts.evaluate --config configs/brisc.yaml
python -m scripts.evaluate --config configs/voc.yaml

# 3. summary, paired-significance and target-size tables
python -m scripts.make_tables --results results/raw_brisc.csv
python -m scripts.make_tables --results results/raw_voc.csv

# 4. faithfulness (deletion / insertion) and the sanity check
python -m scripts.faithfulness --config configs/brisc.yaml --n-images 180
python -m scripts.faithfulness --config configs/voc.yaml --n-images 180
python -m scripts.sanity_check --config configs/brisc.yaml --per-class 20

# 5. qualitative comparison figures
python -m scripts.make_figures --config configs/brisc.yaml \
    --results results/raw_brisc.csv --out figures/qualitative_brisc.pdf
python -m scripts.make_figures --config configs/voc.yaml \
    --results results/raw_voc.csv --out figures/qualitative_voc.pdf
```

### Where each result comes from

| Paper item | Produced by |
|---|---|
| Table 1 — design space of channel weighting | `cams.grad_cam`, `cams.svd1`, `cams.svd_fusion` |
| Table 2 — selected configurations | `select_config.py` → `configs/*.yaml` |
| Table 3 — the two design axes | `cams.svd1` (raw / sign-corrected) vs `cams.svd_fusion` |
| Table 4 — fusion with scalar vs spatial weighting | `cams.svd_grad_cam` vs `cams.svd_fusion`, through `fusion.fuse` |
| Table 5 — sensitivity to `k` | `select_config.py` (rows for the selected levels) |
| Table 6 — localisation, BRISC | `evaluate.py` → `make_tables.py` |
| Table 7 — paired comparison, BRISC | `make_tables.py` |
| Table 8 — localisation, VOC | `evaluate.py` → `make_tables.py` |
| Table 9 — per-class results, BRISC | `evaluate.py` (grouped by the `class` column) |
| Table 10 — advantage by target size | `make_tables.py` |
| Table 11 — faithfulness | `faithfulness.py` |
| Table 12 — sanity check | `sanity_check.py` |
| Table A.14 — paired comparison, VOC | `make_tables.py` |
| Appendix B — lift by target-size bin | `figures/make_fig_per_size.py`, `make_analysis_figures.py --per-size` |
| Figure 1 — method overview | `figures/make_fig1_overview.py` |
| Qualitative figures | `make_figures.py` |

Tables 3 and 4 were computed with the functions listed above in exploratory
notebooks; no separate runner for them is included. The runtime measurements of
Table 13 are not included either, since timings depend on the hardware.

`scripts/verify_tables.py` compares the numbers printed in the paper with the
result CSVs of the original runs. It expects those CSVs (named as in the
original Colab runs, for example `T1_summary_BRISC.csv`), not the files written
by the scripts above, except for the sanity table, where it also accepts
`results/sanity_brisc.csv`.

### Note on the sanity check

An earlier version of `sanity_check.py` built the randomised copy of the model
with `tf.keras.models.clone_model` followed by `set_weights`. For the BRISC model
that copy was not identical to the trained model even before any weight was
randomised, which inflated the correlations. The copy is now made by loading the
model file a second time, and the script stops if the untouched copy differs
from the trained model. Table 12 of the paper reports the corrected run,
averaged over seeds 0-4. A single unseeded draw can move the correlations of the
gradient-based methods by more than the differences between them, which is why
the paper reports the mean and standard deviation.

---

## Repository layout

```
svd-fusion-cam/
├── svdfusion/            core library (no experiment code)
│   ├── cams.py           SVD-Fusion CAM, SVD1 and the eight baselines
│   ├── fusion.py         multi-scale fusion and the cell-coverage criterion
│   ├── metrics.py        EBPG, lift, pointing game, IoU
│   ├── data.py           BRISC and VOC loaders with aligned masks
│   └── gradients.py      one-pass activation and gradient extraction
├── scripts/
│   ├── download_data.sh  fetch both datasets
│   ├── train_brisc.py    train the BRISC classifier (EfficientNet-B1)
│   ├── train_voc.py      train the VOC classifier (EfficientNet-B0)
│   ├── select_config.py  configuration search on the training split
│   ├── evaluate.py       run all methods on a dataset → per-image CSV
│   ├── make_tables.py    summary, paired and target-size tables
│   ├── faithfulness.py   deletion / insertion curves + paired tests
│   ├── sanity_check.py   cascading weight randomisation
│   ├── verify_tables.py  check published numbers against the original CSVs
│   ├── make_figures.py   qualitative comparison figures (heatmap grids)
│   └── make_analysis_figures.py  crossover, size-law, sanity, per-size plots
├── figures/
│   ├── make_fig1_overview.py     method overview diagram
│   └── make_fig_per_size.py      lift by target-size bin
├── configs/
│   ├── brisc.yaml        final BRISC configuration and search grid
│   └── voc.yaml          final VOC configuration
└── requirements.txt
```

---

## Datasets

- **BRISC 2025**: brain-MRI tumour dataset,
  <https://www.kaggle.com/datasets/briscdataset/brisc2025>.
- **PASCAL VOC 2012**: from the challenge website.

A helper script downloads both (BRISC needs a Kaggle API token):

```bash
bash scripts/download_data.sh
```

Both are used under their original licences. The paths in `configs/*.yaml`
assume the layout the script creates (`data/BRISC`, `data/VOCdevkit`).

### Evaluation subsets

- **BRISC**: the classifier is trained on the classification split. Evaluation
  uses the segmentation split, which has pixel-level tumour masks: all 859
  tumour slices of its test part for reporting, and about 250 slices per tumour
  class from its training part for the configuration search. The no-tumour class
  has no target region and is skipped.
- **VOC**: the first 1500 images of the `val` list, of which 1401 remain after
  dropping images where the predicted class has no annotated box. The mask is the
  union of the predicted class's boxes. Both rules are applied by
  `configs/voc.yaml` and `load_voc_sample`.

---

## Method registry

`cams.BASELINE_METHODS` maps display names to callables for convenient iteration:

```python
for name, fn in cams.BASELINE_METHODS.items():
    saliency = fn(feat, grad)
```

The proposed method, `cams.svd_fusion`, and the intermediate variant `cams.svd1`
are kept separate because they take an extra argument (`k` and `sign_correct`
respectively).

---

## Citation

```bibtex
@article{shahbodaghkhan2026svdfusioncam,
  title  = {{SVD-Fusion CAM}: Hierarchical Robust Visual Explanations via
            Singular Value Decomposition},
  author = {Shahbodaghkhan, Newsha and Amirmazlaghani, Maryam and
            Rahmati, Mohammad and Rezghi, Mansoor},
  year   = {2026},
  note   = {Under review}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
