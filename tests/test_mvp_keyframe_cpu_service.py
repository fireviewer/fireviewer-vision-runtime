from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO

import httpx
from PIL import Image

from fireviewer_contracts.mvp.contracts import EventEvidenceV1, EvidenceMedia, EvidenceSource
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendDerivedKeyframeReceipt,
    BackendEvidenceMediaLocation,
    DurableEventEvidence,
)
from fireviewer_vision_runtime.mvp.vision.keyframe_cpu_service import KeyframeCpuWorker
from fireviewer_vision_runtime.mvp.vision.video_keyframes import (
    VideoKeyframeArtifact,
    VideoKeyframeExtractor,
    VideoKeyframeRun,
)


def _png(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 6), color=color).save(stream, format="PNG")
    return stream.getvalue()


def test_worker_persists_keyframes_before_yolo_and_advances_revision(monkeypatch) -> None:
    video_bytes = b"bounded-video-fixture"
    source = EvidenceSource(
        source_id="SOURCE-1",
        origin_id="ORIGIN-1",
        publisher="FireViewer test",
        retrieved_at=datetime.now(UTC),
        source_type="witness",
        independence_weight=1,
    )
    video = EvidenceMedia(
        media_id="VIDEO-1",
        source_id=source.source_id,
        media_group_id="GROUP-1",
        origin_id=source.origin_id,
        kind="video",
        sha256=sha256(video_bytes).hexdigest(),
    )
    event = EventEvidenceV1(event_id="EVENT-KEYFRAME-1", sources=(source,), media=(video,))
    durable = DurableEventEvidence(
        event=event,
        media_locations=(
            BackendEvidenceMediaLocation(
                media_id=video.media_id,
                working_file_url="https://api.example.test/video",
            ),
        ),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="a" * 64,
    )
    artifacts = tuple(
        VideoKeyframeArtifact(
            media=EvidenceMedia(
                media_id=f"KF-FRAME-{index}",
                source_id=source.source_id,
                media_group_id=video.media_group_id,
                origin_id=video.origin_id,
                kind="keyframe",
                sha256=sha256(content).hexdigest(),
                parent_media_id=video.media_id,
            ),
            frame_index=index,
            timestamp_seconds=float(index),
            format="png",
            data=content,
        )
        for index, content in enumerate((_png((255, 0, 0)), _png((80, 80, 80))))
    )

    class Repository:
        def read(self, event_id: str) -> DurableEventEvidence:
            assert event_id == event.event_id
            return durable

    class Publisher:
        def __init__(self) -> None:
            self.revisions: list[str] = []

        def publish(self, **kwargs) -> BackendDerivedKeyframeReceipt:
            self.revisions.append(str(kwargs["source_revision_sha256"]))
            revision = chr(ord("b") + len(self.revisions) - 1) * 64
            media = kwargs["media"]
            return BackendDerivedKeyframeReceipt(
                candidate_id=event.event_id,
                keyframe_id=media.media_id,
                source_revision_sha256=revision,
                receipt_sha256="f" * 64,
                replayed=False,
            )

    publisher = Publisher()
    monkeypatch.setattr(
        VideoKeyframeExtractor,
        "run",
        lambda _self, extraction_input: VideoKeyframeRun(
            evidence=extraction_input,
            artifacts=artifacts,
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer " + ("t" * 40)
        return httpx.Response(200, content=video_bytes)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = KeyframeCpuWorker(
            repository=Repository(),
            allowed_hosts=frozenset({"api.example.test"}),
            backend_origin="https://api.example.test",
            backend_token="t" * 40,
            publisher=publisher,
            materializer_client=client,
        ).run_candidate(event.event_id)

    assert publisher.revisions == ["a" * 64, "b" * 64]
    assert receipt.source_revision_sha256 == "c" * 64
    assert receipt.artifact_count == 2
    assert receipt.raw_keyframes_stored is True
    assert receipt.requires_durable_sink_before_yolo is False
