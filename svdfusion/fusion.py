"""Multi-scale fusion and the cell-coverage criterion.

This module combines per-layer saliency maps into a single explanation and
provides the cell-coverage rule that decides *which* resolution levels are worth
fusing. The rule is a contribution of the paper (Section 3.4): the precision of
an explanation is bounded by the number of feature-map cells the target occupies,
so a higher-resolution level is only worth adding when the deepest layer supplies
too few cells.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["cell_coverage", "recommend_extra_level", "fuse"]

_EPS = 1e-8


def cell_coverage(area_fraction: float, grid_h: int, grid_w: int) -> float:
    """Number of feature-map cells a target occupies at a given layer.

    Implements ``n_l = rho * N_l = rho * H_l * W_l`` from the paper. A value of
    order one means the target is essentially sub-cell and cannot be localised
    from that layer regardless of how the channel weights are computed.

    Parameters
    ----------
    area_fraction : float
        Fraction of the image the target occupies, ``rho`` in ``[0, 1]``.
    grid_h, grid_w : int
        Spatial dimensions of the layer's feature map.

    Returns
    -------
    float
        Expected number of occupied cells.
    """
    return area_fraction * grid_h * grid_w


def recommend_extra_level(
    area_fraction: float,
    deepest_grid: tuple[int, int],
    threshold_cells: float = 4.0,
) -> bool:
    """Whether a higher-resolution level should be fused in.

    Returns True when the target occupies fewer than ``threshold_cells`` cells at
    the deepest layer, i.e. when that layer alone cannot resolve it. The default
    threshold of four cells is the empirical sign-change point reported in the
    paper (Section 4.6).
    """
    return cell_coverage(area_fraction, *deepest_grid) < threshold_cells


def _fusion_weights(n_levels: int, scheme: str) -> np.ndarray:
    """Normalised fusion weights for ``n_levels`` layers.

    Schemes
    -------
    uniform      : equal weight to every level.
    depth        : weight proportional to depth rank (deeper layers favoured).
    inverse-depth: weight proportional to reverse depth rank (shallower favoured).
    """
    if scheme == "uniform":
        raw = np.ones(n_levels, dtype=np.float64)
    elif scheme == "depth":
        raw = np.arange(1, n_levels + 1, dtype=np.float64)
    elif scheme == "inverse-depth":
        raw = np.arange(n_levels, 0, -1, dtype=np.float64)
    else:
        raise ValueError(f"unknown fusion scheme: {scheme!r}")
    return raw / raw.sum()


def fuse(
    layer_maps: list[np.ndarray],
    output_size: tuple[int, int],
    scheme: str = "uniform",
) -> np.ndarray:
    """Combine per-layer saliency maps into one explanation.

    Each map is upsampled to ``output_size`` and combined by a convex
    combination whose weights follow ``scheme`` (layers assumed ordered shallow
    to deep). The result is min-max normalised to ``[0, 1]``.

    Parameters
    ----------
    layer_maps : list of np.ndarray
        Per-layer saliency maps, ordered from shallow to deep.
    output_size : (int, int)
        Target ``(height, width)`` for upsampling.
    scheme : str
        One of ``"uniform"``, ``"depth"``, ``"inverse-depth"``.

    Returns
    -------
    np.ndarray, shape ``output_size``
        The fused explanation.
    """
    if len(layer_maps) == 1:
        resized = cv2.resize(layer_maps[0], (output_size[1], output_size[0]))
        return resized

    weights = _fusion_weights(len(layer_maps), scheme)
    fused = np.zeros(output_size, dtype=np.float64)
    for weight, cam in zip(weights, layer_maps):
        resized = cv2.resize(cam, (output_size[1], output_size[0]))
        fused += weight * resized

    fused = np.maximum(fused, 0.0)
    lo, hi = fused.min(), fused.max()
    if hi > lo:
        return (fused - lo) / (hi - lo)
    return fused / hi if hi > 0 else fused
