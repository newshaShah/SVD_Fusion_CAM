"""Train the BRISC 2025 tumour classifier.

Fine-tunes an EfficientNet-B1 on the four-class BRISC classification split,
following the schedule used to train the model explained in the paper (held-out
accuracy 97.90%). The brain-crop applied here is the same one used at
evaluation time (``svdfusion.data.brain_crop_box``), so the explained model sees
images preprocessed as it was trained.

The cropped images are first written to ``--cache-dir`` as 240x240 JPEGs, one
folder per class, and then read with ``ImageDataGenerator``. Its
``validation_split`` holds out the last 20% of every class folder, so the
validation set covers all four classes.

Schedule
--------
1. Backbone frozen, head trained at lr 1e-3 (up to 30 epochs).
2. The last 30 layers of the backbone unfrozen, lr 5e-5 (up to 40 epochs).

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
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (
    BatchNormalization, Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from svdfusion.data import brain_crop_box

IMAGE_SIZE = 240                       # EfficientNet-B1 input
BATCH_SIZE = 16
# Alphabetical, which is the order flow_from_directory assigns indices in and
# the order svdfusion.data.BRISC_CLASSES expects.
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
UNFROZEN_LAYERS = 30


def crop_split(src_dir: str, dst_dir: str) -> None:
    """Crop every image of a split to the brain and save it at 240x240."""
    for cls in CLASSES:
        src = os.path.join(src_dir, cls)
        dst = os.path.join(dst_dir, cls)
        os.makedirs(dst, exist_ok=True)
        if not os.path.isdir(src):
            continue
        saved = 0
        for fname in sorted(os.listdir(src)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image = cv2.imread(os.path.join(src, fname))
            if image is None:
                continue
            box = brain_crop_box(image)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cropped = cv2.resize(image[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(os.path.join(dst, f"{saved}.jpg"), cropped)
            saved += 1
        print(f"  {os.path.basename(src_dir)}/{cls}: {saved} images")


def build_model() -> tuple[Model, Model]:
    """EfficientNet-B1 backbone with the classification head used in the paper."""
    base = EfficientNetB1(
        weights="imagenet", include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(CLASSES), activation="softmax")(x)
    return Model(base.input, predictions), base


def _callbacks(checkpoint, patience, lr_patience, min_lr):
    return [
        ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_accuracy", patience=patience,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=lr_patience, min_lr=min_lr),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="classification_task directory with train/ and test/")
    parser.add_argument("--cache-dir", default="data/BRISC-cropped",
                        help="where the cropped images are written")
    parser.add_argument("--out", default="efficientnet_brisc.keras")
    parser.add_argument("--epochs-head", type=int, default=30)
    parser.add_argument("--epochs-finetune", type=int, default=40)
    args = parser.parse_args()

    print("Cropping images ...")
    for split in ("train", "test"):
        crop_split(os.path.join(args.data_dir, split),
                   os.path.join(args.cache_dir, split))

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode="nearest",
    )
    flow = dict(target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
                class_mode="categorical", classes=CLASSES)
    train_dir = os.path.join(args.cache_dir, "train")
    train_data = train_gen.flow_from_directory(
        train_dir, subset="training", shuffle=True, **flow)
    valid_data = train_gen.flow_from_directory(
        train_dir, subset="validation", shuffle=True, **flow)
    test_data = ImageDataGenerator(
        preprocessing_function=preprocess_input).flow_from_directory(
        os.path.join(args.cache_dir, "test"), shuffle=False, **flow)

    model, base = build_model()

    # Stage 1: train the head with the backbone frozen.
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=valid_data, epochs=args.epochs_head,
              callbacks=_callbacks("brisc_stage1.keras", 10, 4, 1e-6))

    # Stage 2: unfreeze the last layers of the backbone and fine-tune.
    base.trainable = True
    for layer in base.layers[:-UNFROZEN_LAYERS]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=valid_data,
              epochs=args.epochs_finetune,
              callbacks=_callbacks("brisc_stage2.keras", 15, 5, 1e-7))

    _, acc = model.evaluate(test_data, verbose=0)
    print(f"test accuracy: {100 * acc:.2f}%")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
