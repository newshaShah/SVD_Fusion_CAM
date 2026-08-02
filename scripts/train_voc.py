"""Train the PASCAL VOC 2012 classifier.

Trains an EfficientNet-B0 for multi-label classification on VOC 2012. The model
is used only to produce a predicted class per image, which the explanation then
targets; localisation quality, not classification accuracy, is what the paper
evaluates.

Training uses a two-stage schedule: a frozen backbone with a trainable head,
followed by full fine-tuning.

Usage
-----
    python -m scripts.train_voc --voc-root data/VOCdevkit/VOC2012 \
        --out efficientnet_voc.h5
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMAGE_SIZE = 224                       # EfficientNet-B0 input
BATCH_SIZE = 16

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


def _multi_hot(annotation_path: str) -> np.ndarray:
    """Multi-hot label vector of the classes present in an annotation."""
    root = ET.parse(annotation_path).getroot()
    label = np.zeros(len(VOC_CLASSES), dtype="float32")
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in CLASS_TO_IDX:
            label[CLASS_TO_IDX[name]] = 1.0
    return label


def _load_split(voc_root: str, split: str) -> tuple[list[str], np.ndarray]:
    """Return image paths and multi-hot labels for a split."""
    ids_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
    with open(ids_file) as fh:
        ids = [line.strip() for line in fh if line.strip()]
    paths, labels = [], []
    for image_id in ids:
        paths.append(os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg"))
        labels.append(_multi_hot(
            os.path.join(voc_root, "Annotations", f"{image_id}.xml")))
    return paths, np.asarray(labels)


def _generator(paths, labels, augment=False):
    """Yield preprocessed image batches indefinitely."""
    n = len(paths)
    order = np.arange(n)
    while True:
        np.random.shuffle(order)
        for start in range(0, n, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            images = []
            for i in batch:
                img = Image.open(paths[i]).convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE))
                arr = np.array(img)
                if augment and np.random.random() > 0.5:
                    arr = np.fliplr(arr)
                images.append(preprocess_input(arr.astype("float32")))
            yield np.asarray(images), labels[batch]


def build_model() -> tuple[Model, Model]:
    """EfficientNet-B0 backbone with a sigmoid multi-label head."""
    base = EfficientNetB0(
        weights="imagenet", include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(len(VOC_CLASSES), activation="sigmoid")(x)
    return Model(base.input, predictions), base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", required=True,
                        help="VOCdevkit/VOC2012 directory")
    parser.add_argument("--out", default="efficientnet_voc.h5")
    parser.add_argument("--epochs-head", type=int, default=20)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    args = parser.parse_args()

    train_paths, train_labels = _load_split(args.voc_root, "train")
    val_paths, val_labels = _load_split(args.voc_root, "val")
    print(f"train: {len(train_paths)}  val: {len(val_paths)}")

    train_gen = _generator(train_paths, train_labels, augment=True)
    val_gen = _generator(val_paths, val_labels, augment=False)
    train_steps = len(train_paths) // BATCH_SIZE + 1
    val_steps = len(val_paths) // BATCH_SIZE + 1

    model, base = build_model()
    callbacks = [
        ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    # Stage 1: frozen backbone.
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(train_gen, steps_per_epoch=train_steps,
              validation_data=val_gen, validation_steps=val_steps,
              epochs=args.epochs_head, callbacks=callbacks)

    # Stage 2: full fine-tuning at a low learning rate.
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(train_gen, steps_per_epoch=train_steps,
              validation_data=val_gen, validation_steps=val_steps,
              epochs=args.epochs_finetune, callbacks=callbacks)

    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
