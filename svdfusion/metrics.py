"""Localisation metrics for saliency maps.

All metrics take a saliency map in ``[0, 1]`` and a binary ground-truth mask of
the same spatial size. Energy-based pointing game and lift are the primary
metrics used in the paper; IoU and Dice require a threshold and are reported for
completeness.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ebpg", "lift", "pointing_game", "iou", "dice"]

_EPS = 1e-8


def ebpg(cam: np.ndarray, mask: np.ndarray) -> float | None:
    """Energy-based pointing game.

    Fraction of the map's total energy that falls inside the mask. Returns None
    for an all-zero map.
    """
    total = cam.sum()
    if total <= 0:
        return None
    return float(((cam / total) * mask).sum())


def lift(cam: np.ndarray, mask: np.ndarray) -> float | None:
    """EBPG divided by the mask's area fraction.

    A value of 1 corresponds to a uniform map (chance). Making the metric
    comparable across datasets whose targets differ greatly in size is the reason
    lift, rather than raw EBPG, is used as the headline metric.
    """
    e = ebpg(cam, mask)
    if e is None:
        return None
    area = float(mask.mean())
    return e / max(area, _EPS)


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> float:
    """1.0 if the map's maximum lies inside the mask, else 0.0.

    Note that after bilinear upsampling the maximum coincides with the centre of
    a feature-map cell, so for targets smaller than one cell this metric is
    largely determined by grid alignment (paper, Section 4.6).
    """
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    return 1.0 if mask[peak] > 0 else 0.0


def _binarise(cam: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a map at a fraction of its maximum."""
    return ((cam / (cam.max() + _EPS)) >= threshold).astype(np.uint8)


def iou(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float | None:
    """Intersection over union after thresholding the map."""
    binary = _binarise(cam, threshold)
    union = np.logical_or(binary, mask).sum()
    if union == 0:
        return None
    intersection = np.logical_and(binary, mask).sum()
    return float(intersection / union)


def dice(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float | None:
    """Dice coefficient after thresholding the map.

    Provided for convenience; the paper reports IoU only, since the two are
    monotonically related and reporting both adds no information.

    Related to IoU by ``dice = 2 * iou / (1 + iou)``, so it is systematically
    higher; this is a property of the metric, not an improvement.
    """
    binary = _binarise(cam, threshold)
    denom = binary.sum() + mask.sum()
    if denom == 0:
        return None
    intersection = np.logical_and(binary, mask).sum()
    return float(2.0 * intersection / denom)
