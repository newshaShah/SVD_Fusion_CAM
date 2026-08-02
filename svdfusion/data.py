"""Dataset loading for BRISC 2025 and PASCAL VOC 2012.

Both loaders return an image ready for the model together with an aligned binary
ground-truth mask. For BRISC the brain-crop applied during training is applied
here to both the image and the mask, so that annotations stay aligned; this is
the step that keeps evaluation faithful to the trained model.

The preprocessing must match training exactly, which is why the crop logic lives
in one place and is shared by training and evaluation.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

__all__ = ["Sample", "brain_crop_box", "load_brisc_sample", "load_voc_sample",
           "BRISC_CLASSES", "VOC_CLASSES"]


# BRISC label tokens embedded in filenames -> class name.
_BRISC_TOKENS = {"gl": "glioma", "me": "meningioma",
                 "pi": "pituitary", "no": "no_tumor"}
BRISC_CLASSES = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


@dataclass
class Sample:
    """A single evaluation example.

    Attributes
    ----------
    model_input : np.ndarray, shape (1, H, W, 3)
        Preprocessed image, ready for ``model.predict``.
    mask : np.ndarray, shape (H, W)
        Binary ground-truth mask, aligned to ``model_input``.
    class_index : int
        Index of the target class.
    display_rgb : np.ndarray, shape (H, W, 3), uint8
        The cropped, resized RGB image before preprocessing, for visualisation.
    label : str
        Human-readable class name.
    """

    model_input: np.ndarray
    mask: np.ndarray
    class_index: int
    display_rgb: np.ndarray
    label: str


# --------------------------------------------------------------------------- #
# BRISC
# --------------------------------------------------------------------------- #
def brain_crop_box(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of the brain region, as used during training.

    Thresholds the image, takes the largest external contour, and returns its
    extreme points. Returns None if no valid contour is found.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        if not len(contours):
            return None
        c = max(contours, key=cv2.contourArea)
        x1 = int(c[c[:, :, 0].argmin()][0][0])
        x2 = int(c[c[:, :, 0].argmax()][0][0])
        y1 = int(c[c[:, :, 1].argmin()][0][1])
        y2 = int(c[c[:, :, 1].argmax()][0][1])
        return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None
    except cv2.error:
        return None


def _brisc_class_token(filename: str) -> str | None:
    for part in os.path.splitext(filename)[0].split("_"):
        if part in _BRISC_TOKENS:
            return part
    return None


def load_brisc_sample(
    name: str,
    images_dir: str,
    masks_dir: str,
    image_size: tuple[int, int] = (240, 240),
) -> Sample | None:
    """Load one BRISC slice with its tumour mask.

    The brain-crop box is computed from the image and applied to *both* the image
    and the mask, then both are resized to ``image_size``. Returns None if the
    image, mask, or crop is unavailable, or if the mask is empty after cropping.
    """
    image = cv2.imread(os.path.join(images_dir, name))
    mask = cv2.imread(
        os.path.join(masks_dir, os.path.splitext(name)[0] + ".png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None or mask is None:
        return None
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    box = brain_crop_box(image)
    if box is None:
        return None
    x1, y1, x2, y2 = box

    image_c = cv2.resize(image[y1:y2, x1:x2], image_size,
                         interpolation=cv2.INTER_LINEAR)
    mask_c = cv2.resize(mask[y1:y2, x1:x2], image_size,
                        interpolation=cv2.INTER_NEAREST)
    mask_c = (mask_c > 0).astype(np.uint8)
    if mask_c.sum() == 0:
        return None

    rgb = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB).astype("float32")
    label = _BRISC_TOKENS[_brisc_class_token(name)]
    model_input = np.expand_dims(preprocess_input(rgb.copy()), axis=0)
    return Sample(model_input, mask_c, BRISC_CLASSES[label],
                  rgb.astype("uint8"), label)


# --------------------------------------------------------------------------- #
# PASCAL VOC
# --------------------------------------------------------------------------- #
def load_voc_sample(
    image_id: str,
    voc_root: str,
    model,
    image_size: tuple[int, int] = (224, 224),
) -> Sample | None:
    """Load one VOC image with a mask for the *predicted* class.

    The predicted class is used because the explanation targets the model's own
    decision. The mask is the union of the predicted class's bounding boxes.
    Returns None if no box of the predicted class is present.
    """
    ann_path = os.path.join(voc_root, "Annotations", f"{image_id}.xml")
    if not os.path.exists(ann_path):
        return None
    root = ET.parse(ann_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    image = cv2.imread(os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg"))
    if image is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(image, image_size),
                       cv2.COLOR_BGR2RGB).astype("float32")
    model_input = np.expand_dims(preprocess_input(rgb.copy()), axis=0)

    preds = model.predict(model_input, verbose=0)
    if isinstance(preds, list):
        preds = preds[0]
    class_index = int(np.argmax(preds[0]))
    label = VOC_CLASSES[class_index]

    mask = np.zeros(image_size, dtype=np.uint8)
    scale_w = image_size[1] / width
    scale_h = image_size[0] / height
    for obj in root.findall("object"):
        if obj.find("name").text != label:
            continue
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (
            int(float(box.find(t).text)) for t in ("xmin", "ymin", "xmax", "ymax")
        )
        x1, x2 = max(0, int(xmin * scale_w)), min(image_size[1], int(xmax * scale_w))
        y1, y2 = max(0, int(ymin * scale_h)), min(image_size[0], int(ymax * scale_h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    if mask.sum() == 0:
        return None

    return Sample(model_input, mask, class_index, rgb.astype("uint8"), label)
