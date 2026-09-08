"""Model-specific inference contracts shared by the production adapters."""

from fireviewer_vision_runtime.model_workers.detection import (
    FIREVIEWER_DETECTOR_LABELS,
    LetterboxGeometry,
    center_letterbox,
    unletterbox,
)

__all__ = [
    "FIREVIEWER_DETECTOR_LABELS",
    "LetterboxGeometry",
    "center_letterbox",
    "unletterbox",
]
