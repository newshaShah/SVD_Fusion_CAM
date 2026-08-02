"""Gradient extraction for gradient-based CAM methods.

Wraps a Keras model so that, for a chosen set of layers, one call returns the
activations and the gradient of the target-class logit with respect to each. A
single forward and backward pass is shared across all layers, matching the cost
model described in the paper (Section 3.5).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

__all__ = ["GradientExtractor"]


class GradientExtractor:
    """Return activations and target-logit gradients for selected layers.

    Parameters
    ----------
    model : tf.keras.Model
        The trained classifier.
    layer_names : list of str
        Layers whose activations and gradients are needed. The deepest layer
        should be included when it is the fusion target.

    Examples
    --------
    >>> extractor = GradientExtractor(model, ["block3c_project_conv", "top_conv"])
    >>> feats, grads = extractor(sample.model_input, sample.class_index)
    >>> cam = svd_fusion(feats[-1], grads[-1], k=3)
    """

    def __init__(self, model: tf.keras.Model, layer_names: list[str]) -> None:
        self.layer_names = list(dict.fromkeys(layer_names))  # de-duplicate, keep order
        outputs = [model.get_layer(n).output for n in self.layer_names]
        self._grad_model = tf.keras.Model(model.input, outputs + [model.output])

    def __call__(
        self, model_input: np.ndarray, class_index: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute per-layer activations and gradients.

        Returns
        -------
        feats : list of np.ndarray
            Activations, one array of shape ``(H, W, C)`` per requested layer.
        grads : list of np.ndarray
            Gradients of the target logit, same shapes as ``feats``. Layers whose
            gradient is None (disconnected) are returned as zeros.
        """
        x = tf.convert_to_tensor(model_input)
        with tf.GradientTape() as tape:
            outputs = self._grad_model(x)
            predictions = outputs[-1]
            if isinstance(predictions, list):
                predictions = predictions[0]
            score = predictions[:, class_index]

        layer_outputs = outputs[:-1]
        gradients = tape.gradient(score, layer_outputs)

        feats = [o[0].numpy() for o in layer_outputs]
        grads = [
            (g[0].numpy() if g is not None else np.zeros_like(o[0].numpy()))
            for g, o in zip(gradients, layer_outputs)
        ]
        return feats, grads
