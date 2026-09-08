from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from fireviewer_contracts.mvp.contracts import EventEvidenceV1, EvidenceMedia
from fireviewer_vision_runtime.mvp.vision import (
    DecodedVideoFrame,
    DecodedVideoSample,
    OpenCvVideoFrameDecoder,
    VideoKeyframeConfig,
    VideoKeyframeExtractor,
)


class _Decoder:
    def __init__(self, sample: DecodedVideoSample | Exception) -> None:
        self.decoded = sample
        self.calls: list[tuple[str, float, int]] = []

    def sample(
        self,
        media: EvidenceMedia,
        *,
        interval_seconds: float,
        max_frames: int,
    ) -> DecodedVideoSample:
        self.calls.append((media.media_id, interval_seconds, max_frames))
        if isinstance(self.decoded, Exception):
            raise self.decoded
        return self.decoded


def _event() -> EventEvidenceV1:
    return EventEvidenceV1.model_validate(
        {
            "event_id": "EVENT-VIDEO-1",
            "sources": [
                {
                    "source_id": "SOURCE-1",
                    "origin_id": "ORIGIN-1",
                    "publisher": "Fixture",
                    "retrieved_at": "2026-08-22T10:00:00Z",
                    "source_type": "witness",
                    "independence_weight": 1,
                }
            ],
            "media": [
                {
                    "media_id": "VIDEO-1",
                    "source_id": "SOURCE-1",
                    "media_group_id": "GROUP-1",
                    "origin_id": "ORIGIN-1",
                    "kind": "video",
                    "sha256": "a" * 64,
                    "captured_at": "2026-08-22T10:00:00Z",
                }
            ],
        }
    )


def _frames(colors: tuple[int, ...]) -> DecodedVideoSample:
    return DecodedVideoSample(
        frames=tuple(
            DecodedVideoFrame(
                frame_index=index * 50,
                timestamp_seconds=float(index * 2),
                image=Image.new("RGB", (32, 24), color=(color, color, color)),
            )
            for index, color in enumerate(colors)
        )
    )


def test_video_keyframes_selects_bounded_unique_frames_and_is_idempotent() -> None:
    decoder = _Decoder(_frames((0, 30, 60, 90, 120, 150, 180, 210)))
    extractor = VideoKeyframeExtractor(
        decoder=decoder,
        config=VideoKeyframeConfig(
            minimum_keyframes=5,
            maximum_keyframes=6,
            scene_change_threshold=0.1,
            duplicate_threshold=0.01,
        ),
    )

    first = extractor.run(_event())
    replay = extractor.run(first.evidence)

    assert len(first.artifacts) == 6
    assert len(first.evidence.media) == 7
    assert decoder.calls == [("VIDEO-1", 2.0, 900)]
    assert replay.evidence == first.evidence
    assert replay.artifacts == ()
    for artifact in first.artifacts:
        assert artifact.media.kind == "keyframe"
        assert artifact.media.parent_media_id == "VIDEO-1"
        assert artifact.media.media_group_id == "GROUP-1"
        assert artifact.media.sha256 == sha256(artifact.data).hexdigest()
        assert artifact.media.captured_at == datetime(2026, 8, 22, 10, 0, tzinfo=UTC) + timedelta(
            seconds=artifact.timestamp_seconds
        )


def test_video_keyframes_reports_insufficient_unique_frames() -> None:
    extractor = VideoKeyframeExtractor(decoder=_Decoder(_frames((40, 40, 40, 40, 40, 40))))

    result = extractor.run(_event())

    assert len(result.artifacts) == 1
    assert [item.code for item in result.evidence.uncertainties] == [
        "video_insufficient_unique_frames"
    ]
    assert result.evidence.needs_human_review is True


def test_video_keyframes_isolates_decode_failure() -> None:
    extractor = VideoKeyframeExtractor(decoder=_Decoder(ValueError("bad fixture")))

    result = extractor.run(_event())

    assert result.artifacts == ()
    assert [item.code for item in result.evidence.uncertainties] == [
        "video_keyframe_extraction_failed"
    ]
    assert result.evidence.needs_human_review is True


def test_opencv_decoder_verifies_video_hash_and_samples(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    for value in range(10):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()

    video = _event().media[0].model_copy(update={"sha256": sha256(path.read_bytes()).hexdigest()})
    decoder = OpenCvVideoFrameDecoder(path_resolver=lambda _: path)

    sample = decoder.sample(video, interval_seconds=0.4, max_frames=10)

    assert [item.frame_index for item in sample.frames] == [0, 2, 4, 6, 8]
    assert sample.truncated is False
    with pytest.raises(ValueError, match="SHA-256"):
        decoder.sample(
            video.model_copy(update={"sha256": "b" * 64}),
            interval_seconds=0.4,
            max_frames=10,
        )
