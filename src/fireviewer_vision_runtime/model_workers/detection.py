"""Validated FireViewer detector preprocessing and output geometry.

The trained RT-DETR and D-FINE checkpoints were evaluated with a centered
768-by-768 black letterbox.  Applying their image processor directly to a
rectangular source changes the geometry and invalidates the published
benchmark.  This module is deliberately small so the same contract can be used
by the online worker and by offline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 768
FIREVIEWER_DETECTOR_LABELS = frozenset({"smoke_visible", "flame_visible"})


@dataclass(frozen=True, slots=True)
class LetterboxGeometry:
    scale: float
    pad_left: int
    pad_top: int
    original_width: int
    original_height: int


def center_letterbox(
    image: Image.Image,
    *,
    size: int = IMAGE_SIZE,
) -> tuple[Image.Image, LetterboxGeometry]:
    if size < 1:
        raise ValueError("letterbox size must be positive")
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    if width < 1 or height < 1:
        raise ValueError("source image dimensions must be positive")
    scale = size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        array,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return Image.fromarray(canvas), LetterboxGeometry(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        original_width=width,
        original_height=height,
    )


def unletterbox(
    result: dict[str, Any],
    geometry: LetterboxGeometry,
) -> dict[str, Any]:
    boxes = result["boxes"].detach().float().cpu().clone()
    scores = result["scores"].detach().float().cpu()
    labels = result["labels"].detach().long().cpu()
    if boxes.numel():
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry.pad_left) / geometry.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry.pad_top) / geometry.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, geometry.original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, geometry.original_height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    return {"boxes": boxes, "scores": scores, "labels": labels}


def is_fireviewer_detector(model: Any) -> bool:
    id2label = getattr(getattr(model, "config", None), "id2label", {})
    labels = {str(label) for label in id2label.values()}
    return FIREVIEWER_DETECTOR_LABELS.issubset(labels)
