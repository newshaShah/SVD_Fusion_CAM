# SVD-Fusion CAM

Reference implementation of **SVD-Fusion CAM**, a class activation mapping method
for convolutional networks. Instead of averaging each channel's gradient map into
a scalar weight, it applies a singular value decomposition to the joint
spatial–channel gradient matrix and keeps the low-rank reconstruction as a
*spatial weight field*, then fuses maps across resolution levels chosen by a
cell-coverage criterion.

This repository reproduces every quantitative result and figure in the paper.

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

Two properties follow. The reconstruction is optimal in the Frobenius norm by the
Eckart–Young–Mirsky theorem, and because it is built from the products `uᵢvᵢᵀ` it
is invariant to the sign ambiguity of the singular vectors — unlike a
channel-weight-only variant, which collapses without an external sign convention.

**Layer selection.** The precision of an explanation is bounded by the number of
feature-map cells the target occupies, `n_l = ρ · H_l · W_l`. When the deepest
layer supplies too few cells (empirically fewer than about four), a
higher-resolution level is fused in. This criterion is architecture-independent
and requires no tuning.

---

## Installation

```bash
git clone https://github.com/<user>/svd-fusion-cam.git
cd svd-fusion-cam
pip install -r requirements.txt
```

Python 3.9+ is required. The methods themselves depend only on NumPy and OpenCV;
TensorFlow is needed to extract gradients from a trained model.

---

## Quick start

```python
import cv2
import tensorflow as tf
from svdfusion import cams, fusion, metrics
from svdfusion.data import load_brisc_sample
from svdfusion.gradients import GradientExtractor

model = tf.keras.models.load_model("efficientnet_brisc.keras", compile=False)
extractor = GradientExtractor(model, ["block3c_project_conv", "top_conv"])

sample = load_brisc_sample("case.jpg", "images/", "masks/")
feats, grads = extractor(sample.model_input, sample.class_index)

# single layer — the raw map is at the layer's resolution (e.g. 8x8),
# so resize it to the image size before computing metrics or overlaying
cam_single = cv2.resize(cams.svd_fusion(feats[-1], grads[-1], k=3), (240, 240))

# multi-scale fusion — fuse() upsamples internally, no manual resize needed
per_layer = [cams.svd_fusion(f, g, k=3) for f, g in zip(feats, grads)]
cam_multi = fusion.fuse(per_layer, output_size=(240, 240), scheme="uniform")

print(metrics.lift(cam_multi, sample.mask))
```

---

## Reproducing the paper

Evaluation is deterministic, so these commands reproduce the reported numbers
exactly. The trained classifiers can either be obtained separately or retrained
with the training scripts; training is optional and only the trained model is
needed for evaluation.

```bash
# 0. (optional) train the classifiers from scratch
python -m scripts.train_brisc --data-dir data/BRISC/brisc2025/classification_task \
    --out efficientnet_brisc.keras
python -m scripts.train_voc --voc-root data/VOCdevkit/VOC2012 \
    --out efficientnet_voc.h5

# 1. evaluate all nine methods (writes results/raw_brisc.csv)
python -m scripts.evaluate --config configs/brisc.yaml
python -m scripts.evaluate --config configs/voc.yaml

# 2. summary and paired-significance tables
python -m scripts.make_tables --results results/raw_brisc.csv
python -m scripts.make_tables --results results/raw_voc.csv

# 3. faithfulness (deletion / insertion) and the sanity check
python -m scripts.faithfulness --config configs/brisc.yaml --n-images 180
python -m scripts.faithfulness --config configs/voc.yaml --n-images 180
python -m scripts.sanity_check --config configs/brisc.yaml --per-class 20

# 4. qualitative comparison figures
python -m scripts.make_figures --config configs/brisc.yaml \
    --results results/raw_brisc.csv --out figures/qualitative_brisc.pdf
python -m scripts.make_figures --config configs/voc.yaml \
    --results results/raw_voc.csv --out figures/qualitative_voc.pdf
```

### Where each result comes from

| Paper item | Produced by |
|---|---|
| Table 1 — BRISC localisation | `evaluate.py` → `make_tables.py` |
| Table 2 — VOC localisation | `evaluate.py` → `make_tables.py` |
| Table 3 — selected configurations | `configs/brisc.yaml`, `configs/voc.yaml` |
| Table 4 — design-axis taxonomy | `cams.svd1` (raw / sign-corrected) vs `svd_fusion` |
| Table 5 — weighting vs fusion (2×2) | `svd_grad_cam` vs `svd_fusion`, single vs fused |
| Table 6 — paired comparison (BRISC) | `make_tables.py` |
| Per-class table | `evaluate.py` grouped by `class` column |
| Faithfulness (deletion / insertion) | `faithfulness.py` |
| Sanity check | `sanity_check.py` |
| Lift by target-size bin (appendix) | `make_analysis_figures.py --per-size` |
| Qualitative figures | `make_figures.py` |

---

## Repository layout

```
svd-fusion-cam/
├── svdfusion/            core library (no experiment code)
│   ├── cams.py           the nine CAM methods
│   ├── fusion.py         multi-scale fusion and the cell-coverage criterion
│   ├── metrics.py        EBPG, lift, pointing game, IoU
│   ├── data.py           BRISC and VOC loaders with aligned masks
│   └── gradients.py      one-pass activation and gradient extraction
├── scripts/
│   ├── download_data.sh  fetch both datasets
│   ├── train_brisc.py    train the BRISC classifier (EfficientNet-B1)
│   ├── train_voc.py      train the VOC classifier (EfficientNet-B0)
│   ├── evaluate.py       run all methods on a dataset → per-image CSV
│   ├── faithfulness.py   deletion / insertion curves + paired tests
│   ├── sanity_check.py   cascading weight randomisation
│   ├── make_tables.py    rebuild the summary and paired tables
│   ├── verify_tables.py  check every published number against its CSV
│   ├── make_figures.py   qualitative comparison figures (heatmap grids)
│   └── make_analysis_figures.py  crossover, size-law, sanity, per-size plots
├── figures/
│   ├── make_fig1_overview.py     method overview diagram
│   └── make_fig_per_size.py      lift by target-size bin
├── configs/
│   ├── brisc.yaml        final BRISC configuration
│   └── voc.yaml          final VOC configuration
└── requirements.txt
```

---

## Datasets

- **BRISC 2025** — brain-MRI tumour dataset,
  <https://www.kaggle.com/datasets/briscdataset/brisc2025>.
- **PASCAL VOC 2012** — from the challenge website.

A helper script downloads both (BRISC needs a Kaggle API token):

```bash
bash scripts/download_data.sh
```

Both are used under their original licences. After downloading, edit the paths in
`configs/*.yaml` to point at `data/BRISC` and `data/VOCdevkit`.

### Evaluation subsets

- **BRISC**: all 859 images of the segmentation test split that contain a tumour.
- **VOC**: of the 5823 images in the validation split, evaluation keeps only
  those where the model's predicted class has at least one bounding box in the
  image (about 1400 images). Images where the predicted class is absent yield an
  empty mask and are skipped, since there is no region to localise against. This
  filter is applied automatically in `load_voc_sample`.

---

## Method registry

`cams.BASELINE_METHODS` maps display names to callables for convenient iteration:

```python
for name, fn in cams.BASELINE_METHODS.items():
    saliency = fn(feat, grad)
```

The proposed methods, `cams.svd_fusion` and `cams.svd1`, are kept separate
because they take an extra argument (`k` and `sign_correct` respectively).

---

## Citation

```bibtex
@article{svdfusioncam,
  title  = {SVD-Fusion CAM: spectral gradient weighting with
            resolution-aware fusion},
  author = {Shahbodaghkhan, Newsha and others},
  year   = {2026},
  note   = {Under review}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
