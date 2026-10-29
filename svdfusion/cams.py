"""Class Activation Mapping methods.

This module implements the proposed SVD-Fusion CAM together with the baseline
methods used in the paper. Every function takes the feature map and the gradient
of the target-class logit for a single convolutional layer and returns a 2-D
saliency map, normalised to ``[0, 1]``.

Shapes
------
feat : np.ndarray, shape (H, W, C)
    Activations of the layer.
grad : np.ndarray, shape (H, W, C)
    Gradient of the target logit with respect to ``feat``.

The proposed method (:func:`svd_fusion`) differs from all baselines in that it
keeps the low-rank reconstruction of the joint spatial-channel gradient matrix as
a *spatial field* rather than collapsing it to a per-channel scalar. See the
paper, Section 3, for the derivation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "grad_cam",
    "grad_cam_pp",
    "xgrad_cam",
    "hires_cam",
    "layer_cam",
    "eigen_cam",
    "svd_grad_cam",
    "pcg_cam",
    "svd1",
    "svd_fusion",
]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _relu_normalise(cam: np.ndarray) -> np.ndarray:
    """Apply ReLU and divide by the maximum, mapping the map into ``[0, 1]``."""
    cam = np.maximum(cam, 0.0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def _relu_minmax(cam: np.ndarray) -> np.ndarray:
    """Apply ReLU and min-max normalise. Used where weights may be negative."""
    cam = np.maximum(cam, 0.0)
    lo, hi = cam.min(), cam.max()
    if hi > lo:
        return (cam - lo) / (hi - lo)
    return cam / hi if hi > 0 else cam


# --------------------------------------------------------------------------- #
# Gradient-based baselines
# --------------------------------------------------------------------------- #
def grad_cam(feat: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017).

    Channel weights are the spatial average of the gradient. Equivalent to
    projecting each gradient map onto the uniform spatial direction.
    """
    weights = grad.mean(axis=(0, 1))
    return _relu_normalise((weights * feat).sum(axis=-1))


def grad_cam_pp(feat: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Grad-CAM++ (Chattopadhay et al., 2018).

    Weights the gradient by a higher-order coefficient to better handle multiple
    instances of a class.
    """
    grad_sq = grad ** 2
    grad_cu = grad ** 3
    denom = 2.0 * grad_sq + feat.sum(axis=(0, 1)) * grad_cu + 1e-7
    alpha = np.where(grad != 0.0, grad_sq / denom, 0.0)
    weights = (alpha * np.maximum(grad, 0.0)).sum(axis=(0, 1))
    return _relu_normalise((weights * feat).sum(axis=-1))


def xgrad_cam(feat: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """XGrad-CAM (Fu et al., 2020).

    Axiom-based weighting: the gradient is normalised by the channel's activation
    mass. The derivation assumes non-negative activations; on networks with
    signed pre-activations (for example Swish) the normalising term can become
    ill-conditioned. See the paper, Section 4.4. Channels with zero activation
    mass get zero weight, as in the implementation used for the paper.
    """
    mass = feat.sum(axis=(0, 1))
    safe = np.where(mass != 0, mass, 1.0)
    weights = np.where(mass != 0, (grad * feat).sum(axis=(0, 1)) / safe, 0.0)
    return _relu_normalise((weights * feat).sum(axis=-1))


def hires_cam(feat: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """HiResCAM (Draelos and Carin, 2020).

    Element-wise product of gradients and activations, summed over channels.
    Preserves per-location gradient sign before summation.
    """
    return _relu_normalise((grad * feat).sum(axis=-1))


def layer_cam(feat: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """LayerCAM (Jiang et al., 2021).

    Uses only the positive part of the gradient as an element-wise weight, which
    makes shallower layers usable.
    """
    return _relu_normalise((np.maximum(grad, 0.0) * feat).sum(axis=-1))


def eigen_cam(feat: np.ndarray, grad: np.ndarray | None = None) -> np.ndarray:
    """Eigen-CAM (Muhammad and Yeasin, 2020).

    Projects the *activations* onto their first principal component. Independent
    of the target class, hence class-agnostic; ``grad`` is accepted only for a
    uniform call signature and is ignored.
    """
    h, w, c = feat.shape
    flat = feat.reshape(h * w, c)
    try:
        _, eigvecs = np.linalg.eig(np.cov((flat - flat.mean(axis=0)).T))
        projection = flat @ eigvecs[:, 0].real
    except np.linalg.LinAlgError:
        projection = flat.mean(axis=1)
    return _relu_normalise(projection.reshape(h, w))


# --------------------------------------------------------------------------- #
# Decomposition-based methods
# --------------------------------------------------------------------------- #
def svd_grad_cam(feat: np.ndarray, grad: np.ndarray, n_components: int = 1) -> np.ndarray:
    """SVD-Grad-CAM (Thota and Vishwakarma, 2024).

    Applies an SVD to each channel's 2-D gradient map separately, reconstructs a
    denoised version, and then averages it into a scalar channel weight. The
    decomposition acts as a denoiser only; the spatial structure it recovers is
    discarded by the final averaging. Contrast with :func:`svd_fusion`.
    """
    _, _, channels = grad.shape
    weights = np.zeros(channels, dtype=np.float64)
    for k in range(channels):
        g_k = grad[:, :, k]
        if not np.any(g_k):
            continue
        try:
            u, s, vt = np.linalg.svd(g_k, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        keep = min(n_components, len(s))
        if keep > 0:
            weights[k] = ((u[:, :keep] * s[:keep]) @ vt[:keep, :]).mean()
    return _relu_minmax((weights * feat).sum(axis=-1))


def pcg_cam(feat: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """PCG-CAM (Huang et al., 2025), Eqs. 6-10 of the original paper.

    Each channel's gradient is centred over space, the leading eigenvector of
    the resulting ``HW x HW`` covariance (channels as samples) is found, and the
    uncentred gradients are projected onto it to give one weight per channel.
    The absolute value of the weighted sum removes the sign ambiguity of the
    eigenvector.

    Without the centring step the weights reduce exactly to ``v1`` (see
    :func:`svd1`), so PCG-CAM lies in the uniform-profile row of the taxonomy.
    """
    h, w, c = feat.shape
    g = grad.reshape(h * w, c).T                      # (C, HW)
    centred = g - g.mean(axis=1, keepdims=True)       # Eq. 6
    cov = centred.T @ centred / max(c - 1, 1)         # Eq. 7
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)        # Eq. 8
    except np.linalg.LinAlgError:
        return np.zeros((h, w), dtype=np.float32)
    weights = g @ eigvecs[:, np.argmax(eigvals)]      # Eq. 9
    return _relu_normalise(np.abs((weights * feat).sum(axis=-1)))   # Eq. 10


def svd1(feat: np.ndarray, grad: np.ndarray, sign_correct: bool = True) -> np.ndarray:
    """SVD1: channel weighting from the principal right singular vector.

    The channel weights are the first right singular vector ``v1`` of the joint
    gradient matrix. Because singular vectors are defined only up to a joint sign
    flip, ``v1`` is aligned with the mean-gradient direction when
    ``sign_correct`` is True; without this correction the map is negated on the
    large majority of images and the subsequent ReLU keeps the complement of the
    intended region (see the paper, Section 4.3).

    This is the intermediate cell of the taxonomy: v1 channel weights with a
    uniform spatial profile.
    """
    h, w, c = feat.shape
    try:
        _, _, vt = np.linalg.svd(grad.reshape(h * w, c), full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((h, w), dtype=np.float32)
    v1 = vt[0, :]
    if sign_correct:
        mean_grad = grad.mean(axis=(0, 1))
        if np.dot(v1, mean_grad) < 0:
            v1 = -v1
    return _relu_normalise((v1 * feat).sum(axis=-1))


def svd_fusion(feat: np.ndarray, grad: np.ndarray, k: int = 1) -> np.ndarray:
    """SVD-Fusion CAM, single layer (proposed).

    Arranges the gradients into the joint spatial-channel matrix
    ``G in R^{HW x C}``, takes its rank-``k`` reconstruction, reshapes each
    column back to the spatial grid to obtain one weight *map* per channel, and
    applies it element-wise to the corresponding feature map.

    Unlike :func:`svd_grad_cam`, the reconstruction is kept as a spatial field
    and never collapsed to a scalar. The rank-``k`` truncation is optimal in the
    Frobenius norm by the Eckart-Young-Mirsky theorem. Being built from the
    product ``u_i v_i^T``, the map is invariant to the sign ambiguity of the
    singular vectors, unlike :func:`svd1`.

    For multi-layer fusion, call this per layer and combine with
    :func:`svdfusion.fusion.fuse`.
    """
    h, w, c = feat.shape
    try:
        u, s, vt = np.linalg.svd(grad.reshape(h * w, c), full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((h, w), dtype=np.float32)
    keep = min(k, len(s))
    recon = (u[:, :keep] * s[:keep]) @ vt[:keep, :]
    weight_maps = recon.reshape(h, w, c)
    return _relu_normalise((weight_maps * feat).sum(axis=-1))


# --------------------------------------------------------------------------- #
# Registry for convenient iteration
# --------------------------------------------------------------------------- #
BASELINE_METHODS = {
    "Grad-CAM": grad_cam,
    "Grad-CAM++": grad_cam_pp,
    "XGrad-CAM": xgrad_cam,
    "HiResCAM": hires_cam,
    "LayerCAM": layer_cam,
    "Eigen-CAM": eigen_cam,
    "SVD-Grad-CAM": svd_grad_cam,
    "PCG-CAM": pcg_cam,
}
"""Baseline methods keyed by display name. Each maps ``(feat, grad) -> cam``."""
