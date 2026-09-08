from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import Field

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    Detection,
    DetectionResultV1,
    EvidenceMedia,
    ProviderRun,
)
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth


class VisionImageLoader(Protocol):
    def load(self, media: EvidenceMedia) -> object: ...


class GroundingDinoConfig(StrictModel):
    prompts: tuple[str, ...] = (
        "wildfire smoke",
        "smoke plume",
        "visible fire",
        "flames",
    )
    box_threshold: float = Field(default=0.25, ge=0, le=1)
    text_threshold: float = Field(default=0.25, ge=0, le=1)
    nms_iou_threshold: float = Field(default=0.70, ge=0, le=1)
    max_detections: int = Field(default=128, ge=1, le=512)
    device: str = Field(default="cpu", min_length=1, max_length=64)


def _iou(left: Detection, right: Detection) -> float:
    left_x, left_y, left_right, left_bottom = left.bbox
    right_x, right_y, right_right, right_bottom = right.bbox
    width = max(0.0, min(left_right, right_right) - max(left_x, right_x))
    height = max(0.0, min(left_bottom, right_bottom) - max(left_y, right_y))
    intersection = width * height
    left_area = (left_right - left_x) * (left_bottom - left_y)
    right_area = (right_right - right_x) * (right_bottom - right_y)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _non_maximum_suppression(
    detections: list[Detection],
    *,
    iou_threshold: float,
    maximum: int,
) -> tuple[Detection, ...]:
    ordered = sorted(detections, key=lambda item: (-item.score, item.detection_id))
    selected: list[Detection] = []
    for candidate in ordered:
        if any(
            existing.detection_class == candidate.detection_class
            and _iou(existing, candidate) > iou_threshold
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= maximum:
            break
    return tuple(selected)


def _detection_class(label: str) -> Literal["fire", "smoke"] | None:
    normalized = label.casefold()
    if "smoke" in normalized:
        return "smoke"
    if "fire" in normalized or "flame" in normalized:
        return "fire"
    return None


class GroundingDinoVisionProvider:
    """Local zero-shot wildfire detector backed by a pinned Grounding DINO bundle."""

    def __init__(
        self,
        *,
        image_loader: VisionImageLoader,
        model_loader: Callable[[], tuple[Any, Any]],
        model_version: str,
        config: GroundingDinoConfig | None = None,
    ) -> None:
        self.image_loader = image_loader
        self.model_loader = model_loader
        self.config = config or GroundingDinoConfig()
        self.descriptor = ProviderDescriptor(
            provider_id="grounding-dino",
            provider_version="1.0.0",
            model_id="IDEA-Research/grounding-dino-tiny",
            model_version=model_version,
            config=self.config.model_dump(mode="json"),
            capabilities=("zero-shot-fire-smoke-detection",),
        )
        self._processor: Any | None = None
        self._model: Any | None = None

    def healthcheck(self) -> ProviderHealth:
        if self.config.device.startswith("cuda"):
            try:
                import torch
            except ImportError:
                return ProviderHealth(
                    status="unavailable",
                    checked_at=datetime.now(UTC),
                    reason_codes=("torch_unavailable",),
                )
            if not torch.cuda.is_available():
                return ProviderHealth(
                    status="unavailable",
                    checked_at=datetime.now(UTC),
                    reason_codes=("cuda_unavailable",),
                )
        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def detect(self, media: EvidenceMedia) -> DetectionResultV1:
        from PIL import Image

        started = perf_counter()
        loaded = self.image_loader.load(media)
        if not isinstance(loaded, Image.Image):
            raise TypeError("Grounding DINO inputs must be Pillow images")
        image = loaded.convert("RGB")
        processor, model = self._runtime()

        import torch

        inputs = processor(
            images=image,
            text=[list(self.config.prompts)],
            return_tensors="pt",
        ).to(self.config.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        boxes = processed["boxes"].detach().cpu().tolist()
        scores = processed["scores"].detach().cpu().tolist()
        labels = processed.get("text_labels")
        if labels is None:
            labels = processed.get("labels", ())
        detections: list[Detection] = []
        unmapped = False
        for box, score, label_value in zip(boxes, scores, labels, strict=True):
            label = str(label_value)
            detection_class = _detection_class(label)
            if detection_class is None:
                unmapped = True
                continue
            left, top, right, bottom = (
                max(0.0, min(1.0, float(box[0]) / image.width)),
                max(0.0, min(1.0, float(box[1]) / image.height)),
                max(0.0, min(1.0, float(box[2]) / image.width)),
                max(0.0, min(1.0, float(box[3]) / image.height)),
            )
            if left >= right or top >= bottom:
                unmapped = True
                continue
            identity = (
                f"{media.media_id}:{detection_class}:{left:.6f}:{top:.6f}:"
                f"{right:.6f}:{bottom:.6f}:{float(score):.6f}:{label}"
            )
            detections.append(
                Detection(
                    detection_id=f"DET-{sha256(identity.encode()).hexdigest()[:24]}",
                    detection_class=detection_class,
                    bbox=(left, top, right, bottom),
                    score=max(0.0, min(1.0, float(score))),
                    prompt=label,
                )
            )
        qualified = _non_maximum_suppression(
            detections,
            iou_threshold=self.config.nms_iou_threshold,
            maximum=self.config.max_detections,
        )
        classes = {item.detection_class for item in qualified}
        status: Literal["fire", "smoke", "fire_and_smoke", "none", "uncertain"]
        if classes == {"fire", "smoke"}:
            status = "fire_and_smoke"
        elif classes == {"fire"}:
            status = "fire"
        elif classes == {"smoke"}:
            status = "smoke"
        elif unmapped:
            status = "uncertain"
        else:
            status = "none"
        return DetectionResultV1(
            media_id=media.media_id,
            provider_run=ProviderRun(
                provider_id=self.descriptor.provider_id,
                provider_version=self.descriptor.provider_version,
                model_id=self.descriptor.model_id,
                model_version=self.descriptor.model_version,
                config=self.config.model_dump(mode="json"),
                input_hash=media.sha256,
                runtime_ms=int((perf_counter() - started) * 1_000),
                cost_usd=0,
                generated_at=datetime.now(UTC),
            ),
            detections=qualified,
            status=status,
            needs_human_review=unmapped,
        )

    def _runtime(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            processor, model = self.model_loader()
            self._processor = processor
            self._model = model.to(self.config.device)
            self._model.eval()
        return self._processor, self._model


__all__ = [
    "GroundingDinoConfig",
    "GroundingDinoVisionProvider",
    "VisionImageLoader",
]
