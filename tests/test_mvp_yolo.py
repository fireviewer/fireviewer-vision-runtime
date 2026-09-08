from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from fireviewer_contracts.mvp.contracts import EvidenceMedia
from fireviewer_vision_runtime.mvp.vision.yolo import (
    MODEL_ID,
    MODEL_REVISION,
    YoloCpuVisionProvider,
)


def _media() -> EvidenceMedia:
    return EvidenceMedia(
        media_id="MEDIA-YOLO-1",
        source_id="SOURCE-1",
        media_group_id="GROUP-1",
        origin_id="ORIGIN-1",
        kind="photo",
        sha256="a" * 64,
    )


class _ImageLoader:
    def load(self, media: EvidenceMedia) -> object:
        return Image.new("RGB", (100, 50), color=(20, 30, 40))


class _Model:
    def predict(self, **kwargs: object) -> list[object]:
        assert kwargs["device"] == "cpu"
        assert kwargs["imgsz"] == 1024
        assert kwargs["conf"] == 0.10
        boxes = SimpleNamespace(
            xyxy=torch.asarray(((25.0, 10.0, 75.0, 40.0), (0.0, 0.0, 20.0, 20.0))),
            conf=torch.asarray((0.91, 0.72)),
            cls=torch.asarray((1, 0)),
        )
        return [SimpleNamespace(names={0: "fire", 1: "smoke"}, boxes=boxes)]


def test_yolo_cpu_provider_normalizes_geometry_and_keeps_identity() -> None:
    provider = YoloCpuVisionProvider(
        image_loader=_ImageLoader(),
        model_loader=_Model,
    )
    result = provider.detect(_media())
    assert provider.descriptor.model_id == MODEL_ID
    assert provider.descriptor.model_version == MODEL_REVISION
    assert result.status == "fire_and_smoke"
    assert result.needs_human_review is True
    assert [item.detection_class for item in result.detections] == ["smoke", "fire"]
    assert result.detections[0].bbox == (0.25, 0.2, 0.75, 0.8)
    assert result.detections[1].bbox == (0.0, 0.0, 0.2, 0.4)
    assert result.provider_run.config["device"] == "cpu"
    assert result.provider_run.cost_usd == 0
