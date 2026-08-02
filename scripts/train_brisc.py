"""Train the BRISC 2025 tumour classifier.

Fine-tunes an EfficientNet-B1 on the four-class BRISC classification split. The
brain-crop applied here is the same one used at evaluation time
(``svdfusion.data.brain_crop_box``), so that the explained model sees images
preprocessed identically to how it was trained.

The resulting model is the one explained in the paper; its held-out accuracy is
97.90%.

Usage
-----
    python -m scripts.train_brisc \
        --data-dir data/BRISC/brisc2025/classification_task \
        --out efficientnet_brisc.keras
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from svdfusion.data import brain_crop_box

IMAGE_SIZE = 240                       # EfficientNet-B1 input
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def _crop_and_resize(path: str) -> np.ndarray | None:
    """Apply the shared brain-crop and resize, matching evaluation."""
    image = cv2.imread(path)
    if image is None:
        return None
    box = brain_crop_box(image)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cropped = cv2.resize(image[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype("float32")
    return preprocess_input(rgb)


def _load_split(split_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load every cropped image in a split with its class index."""
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            x = _crop_and_resize(os.path.join(cls_dir, fname))
            if x is not None:
                images.append(x)
                labels.append(idx)
    return np.asarray(images), np.asarray(labels)


def build_model() -> tuple[Model, Model]:
    """EfficientNet-B1 backbone with a small classification head."""
    base = EfficientNetB1(
        weights="imagenet", include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(len(CLASSES), activation="softmax")(x)
    return Model(base.input, predictions), base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="classification_task directory with train/ and test/")
    parser.add_argument("--out", default="efficientnet_brisc.keras")
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=35)
    args = parser.parse_args()

    x_train, y_train = _load_split(os.path.join(args.data_dir, "train"))
    x_test, y_test = _load_split(os.path.join(args.data_dir, "test"))
    print(f"train: {len(x_train)}  test: {len(x_test)}")

    model, base = build_model()

    callbacks = [
        ModelCheckpoint(args.out, monitor="val_accuracy",
                        save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=3, min_lr=1e-7, mode="max"),
    ]

    # Stage 1: train the head with the backbone frozen.
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.2, epochs=args.epochs_head,
              batch_size=16, callbacks=callbacks)

    # Stage 2: unfreeze the backbone and fine-tune at a low learning rate.
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.2,
              epochs=args.epochs_finetune, batch_size=16, callbacks=callbacks)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test accuracy: {100 * acc:.2f}%")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
