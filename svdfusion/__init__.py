"""SVD-Fusion CAM: Hierarchical Robust Visual Explanations via SVD.

A saliency-map method for convolutional networks. Instead of averaging each
channel's gradient map into a scalar, it applies a singular value decomposition
to the joint spatial-channel gradient matrix and keeps the low-rank
reconstruction as a spatial weight field, then fuses maps across resolution
levels chosen by a cell-coverage criterion.

See the accompanying paper for the method and evaluation.
"""

from svdfusion import cams, data, fusion, gradients, metrics

__all__ = ["cams", "fusion", "metrics", "data", "gradients"]
__version__ = "1.0.0"
