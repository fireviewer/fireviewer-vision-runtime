from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import Field

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    Detection,
    DetectionResultV1,
    EvidenceMedia,
    ProviderRun,
)
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth
from fireviewer_vision_runtime.mvp.vision.grounding_dino import VisionImageLoader

MODEL_ID = "mfranzon/fire-smoke-yolov8"
MODEL_REVISION = "f1c6426b069c1849cbf13b1ef5d2a260289286db"
MODEL_FILENAME = "fire_smoke_yolov8.pt"
MODEL_SHA256 = "ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16"


class YoloCpuConfig(StrictModel):
    threshold: float = Field(default=0.10, ge=0, le=1)
    input_size: Literal[1024] = 1024
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    max_detections: int = Field(default=128, ge=1, le=512)
    torch_threads: int = Field(default=4, ge=1, le=64)
    device: Literal["cpu"] = "cpu"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HuggingFaceYoloModelLoader:
    """Load one immutable YOLO fallback and reject any weight drift."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        torch_threads: int = 4,
    ) -> None:
        self.cache_dir = cache_dir
        self.token = token
        self.local_files_only = local_files_only
        self.torch_threads = torch_threads

    def __call__(self) -> Any:
        import torch
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        artifact = Path(
            hf_hub_download(
                MODEL_ID,
                MODEL_FILENAME,
                revision=MODEL_REVISION,
                token=self.token,
                cache_dir=cache_dir,
                local_files_only=self.local_files_only,
            )
        )
        artifact_hash = _sha256_file(artifact)
        if artifact_hash != MODEL_SHA256:
            raise RuntimeError(
                f"unexpected YOLO artifact SHA-256: {artifact_hash}; expected {MODEL_SHA256}"
            )
        torch.set_num_threads(self.torch_threads)
        return YOLO(str(artifact), task="detect")


def _mapped_class(label: str) -> Literal["fire", "smoke"] | None:
    normalized = label.casefold()
    if "smoke" in normalized:
        return "smoke"
    if "fire" in normalized or "flame" in normalized:
        return "fire"
    return None


class YoloCpuVisionProvider:
    """Temporary CPU fallback for fire/smoke filtering; never auto-publishes results."""

    def __init__(
        self,
        *,
        image_loader: VisionImageLoader,
        model_loader: Callable[[], Any],
        config: YoloCpuConfig | None = None,
    ) -> None:
        self.image_loader = image_loader
        self.model_loader = model_loader
        self.config = config or YoloCpuConfig()
        self.descriptor = ProviderDescriptor(
            provider_id="yolo-fire-smoke-cpu",
            provider_version="1.0.0",
            model_id=MODEL_ID,
            model_version=MODEL_REVISION,
            config=self.config.model_dump(mode="json"),
            capabilities=("fire-smoke-detection", "cpu-scale-to-zero"),
        )
        self._model: Any | None = None

    def healthcheck(self) -> ProviderHealth:
        try:
            import torch
            import ultralytics  # noqa: F401
        except ImportError:
            return ProviderHealth(
                status="unavailable",
                checked_at=datetime.now(UTC),
                reason_codes=("yolo_runtime_unavailable",),
            )
        if self.config.device != "cpu" or torch.get_num_threads() < 1:
            return ProviderHealth(
                status="unavailable",
                checked_at=datetime.now(UTC),
                reason_codes=("cpu_runtime_unavailable",),
            )
        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def detect(self, media: EvidenceMedia) -> DetectionResultV1:
        from PIL import Image

        started = perf_counter()
        loaded = self.image_loader.load(media)
        if not isinstance(loaded, Image.Image):
            raise TypeError("YOLO inputs must be Pillow images")
        image = loaded.convert("RGB")
        model = self._runtime()
        results = model.predict(
            source=image,
            device="cpu",
            imgsz=self.config.input_size,
            conf=self.config.threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("YOLO must return exactly one result per media item")
        result = results[0]
        names = result.names
        boxes = result.boxes.xyxy.float().cpu().tolist()
        scores = result.boxes.conf.float().cpu().tolist()
        labels = result.boxes.cls.cpu().tolist()
        width, height = image.size

        detections: list[Detection] = []
        malformed = False
        for box, score_value, label_value in zip(boxes, scores, labels, strict=True):
            raw_label = str(names[int(label_value)])
            detection_class = _mapped_class(raw_label)
            if detection_class is None:
                malformed = True
                continue
            left, top, right, bottom = (
                max(0.0, min(1.0, float(box[0]) / width)),
                max(0.0, min(1.0, float(box[1]) / height)),
                max(0.0, min(1.0, float(box[2]) / width)),
                max(0.0, min(1.0, float(box[3]) / height)),
            )
            if left >= right or top >= bottom:
                malformed = True
                continue
            score = max(0.0, min(1.0, float(score_value)))
            identity = (
                f"{media.media_id}:{detection_class}:{left:.6f}:{top:.6f}:"
                f"{right:.6f}:{bottom:.6f}:{score:.6f}:{MODEL_REVISION}"
            )
            detections.append(
                Detection(
                    detection_id=f"DET-{sha256(identity.encode()).hexdigest()[:24]}",
                    detection_class=detection_class,
                    bbox=(left, top, right, bottom),
                    score=score,
                    prompt=raw_label,
                )
            )
        detections.sort(key=lambda item: (-item.score, item.detection_id))
        classes = {item.detection_class for item in detections}
        status: Literal["fire", "smoke", "fire_and_smoke", "none", "uncertain"]
        if classes == {"fire", "smoke"}:
            status = "fire_and_smoke"
        elif classes == {"fire"}:
            status = "fire"
        elif classes == {"smoke"}:
            status = "smoke"
        elif malformed:
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
            detections=tuple(detections),
            status=status,
            needs_human_review=True,
        )

    def _runtime(self) -> Any:
        if self._model is None:
            self._model = self.model_loader()
        return self._model


__all__ = [
    "MODEL_FILENAME",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "HuggingFaceYoloModelLoader",
    "YoloCpuConfig",
    "YoloCpuVisionProvider",
]
