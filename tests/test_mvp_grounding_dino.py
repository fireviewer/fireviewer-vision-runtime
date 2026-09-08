from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image

from fireviewer_contracts.mvp.contracts import (
    Detection,
    DetectionResultV1,
    EventEvidenceV1,
    EvidenceMedia,
    ProviderRun,
)
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth
from fireviewer_vision_runtime.mvp.vision import (
    EventVisionRunner,
    GroundingDinoConfig,
    GroundingDinoVisionProvider,
    inspect_local_grounding_dino_bundle,
)


def _media(media_id: str = "MEDIA-1", digest: str = "a" * 64) -> EvidenceMedia:
    return EvidenceMedia(
        media_id=media_id,
        source_id="SOURCE-1",
        media_group_id="GROUP-1",
        origin_id="ORIGIN-1",
        kind="photo",
        sha256=digest,
    )


class _ImageLoader:
    def load(self, media: EvidenceMedia) -> object:
        return Image.new("RGB", (100, 50), color=(20, 30, 40))


class _Inputs(dict[str, torch.Tensor]):
    input_ids = torch.asarray(((1, 2, 3),))

    def to(self, device: str) -> _Inputs:
        return self


class _Processor:
    def __call__(self, **kwargs: object) -> _Inputs:
        return _Inputs(pixel_values=torch.zeros((1, 3, 8, 8)), input_ids=_Inputs.input_ids)

    def post_process_grounded_object_detection(
        self,
        outputs: object,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        return [
            {
                "boxes": torch.asarray(
                    (
                        (50.0, 25.0, 100.0, 50.0),
                        (0.0, 0.0, 50.0, 25.0),
                        (0.0, 0.0, 50.0, 25.0),
                    )
                ),
                "scores": torch.asarray((0.90, 0.80, 0.70)),
                "text_labels": ["flames", "smoke plume", "wildfire smoke"],
            }
        ]


class _Model:
    def to(self, device: str) -> _Model:
        return self

    def eval(self) -> _Model:
        return self

    def __call__(self, **kwargs: object) -> object:
        return object()


def test_grounding_dino_provider_normalizes_and_deduplicates_boxes() -> None:
    provider = GroundingDinoVisionProvider(
        image_loader=_ImageLoader(),
        model_loader=lambda: (_Processor(), _Model()),
        model_version="a" * 40,
        config=GroundingDinoConfig(device="cpu", nms_iou_threshold=0.5),
    )

    result = provider.detect(_media())

    assert result.status == "fire_and_smoke"
    assert len(result.detections) == 2
    assert [item.detection_class for item in result.detections] == ["fire", "smoke"]
    assert result.detections[0].bbox == (0.5, 0.5, 1.0, 1.0)
    assert result.provider_run.input_hash == "a" * 64
    assert result.provider_run.model_version == "a" * 40


class _EventProvider:
    descriptor = ProviderDescriptor(
        provider_id="grounding-dino",
        provider_version="1.0.0",
        model_id="IDEA-Research/grounding-dino-tiny",
        model_version="b" * 40,
        config={"device": "cpu"},
        capabilities=("zero-shot-fire-smoke-detection",),
    )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def detect(self, media: EvidenceMedia) -> DetectionResultV1:
        if media.media_id == "MEDIA-2":
            raise RuntimeError("fixture inference failure")
        return DetectionResultV1(
            media_id=media.media_id,
            provider_run=ProviderRun(
                provider_id=self.descriptor.provider_id,
                provider_version=self.descriptor.provider_version,
                model_id=self.descriptor.model_id,
                model_version=self.descriptor.model_version,
                config=self.descriptor.config,
                input_hash=media.sha256,
                runtime_ms=1,
                cost_usd=0,
                generated_at=datetime.now(UTC),
            ),
            detections=(
                Detection(
                    detection_id="DET-1",
                    detection_class="smoke",
                    bbox=(0.1, 0.1, 0.5, 0.5),
                    score=0.8,
                    prompt="smoke plume",
                ),
            ),
            status="smoke",
        )


def _event() -> EventEvidenceV1:
    return EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": "EVENT-VISION-1",
            "sources": [
                {
                    "source_id": "SOURCE-1",
                    "origin_id": "ORIGIN-1",
                    "publisher": "Fixture",
                    "retrieved_at": "2026-08-21T10:00:00Z",
                    "source_type": "witness",
                    "independence_weight": 1,
                }
            ],
            "media": [
                _media("MEDIA-1", "a" * 64).model_dump(mode="json"),
                _media("MEDIA-2", "b" * 64).model_dump(mode="json"),
            ],
        }
    )


def test_event_vision_runner_isolates_failures_and_replays_idempotently() -> None:
    runner = EventVisionRunner(provider=_EventProvider())

    first = runner.run(_event())
    replay = runner.run(first.evidence)

    assert len(first.artifacts) == 1
    assert len(first.evidence.visual_observations) == 1
    assert first.evidence.visual_observations[0].confidence == 0.8
    assert first.evidence.needs_human_review is True
    assert [(item.code, item.scope_id) for item in first.evidence.uncertainties] == [
        ("vision_inference_failed", "MEDIA-2")
    ]
    assert replay.evidence.visual_observations == first.evidence.visual_observations
    assert replay.evidence.uncertainties == first.evidence.uncertainties


def test_grounding_dino_bundle_manifest_qualifies_required_files(tmp_path: Path) -> None:
    required = (
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    )
    for filename in required:
        payload = b"fixture"
        if filename == "config.json":
            payload = json.dumps(
                {
                    "architectures": ["GroundingDinoForObjectDetection"],
                    "model_type": "grounding-dino",
                }
            ).encode()
        (tmp_path / filename).write_bytes(payload)

    manifest = inspect_local_grounding_dino_bundle(tmp_path, revision="c" * 40)

    assert manifest.revision == "c" * 40
    assert len(manifest.files) == 8
    assert all(item.byte_size > 0 for item in manifest.files)
