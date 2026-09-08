from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import ClassVar, Literal, Protocol

from PIL import Image
from pydantic import Field, model_validator

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import EventEvidenceV1, EvidenceMedia, Uncertainty


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    frame_index: int
    timestamp_seconds: float
    image: Image.Image


@dataclass(frozen=True, slots=True)
class DecodedVideoSample:
    frames: tuple[DecodedVideoFrame, ...]
    truncated: bool = False


class VideoFrameDecoder(Protocol):
    def sample(
        self,
        media: EvidenceMedia,
        *,
        interval_seconds: float,
        max_frames: int,
    ) -> DecodedVideoSample: ...


class VideoKeyframeConfig(StrictModel):
    minimum_keyframes: int = Field(default=5, ge=1, le=15)
    maximum_keyframes: int = Field(default=15, ge=1, le=15)
    maximum_videos: int = Field(default=32, ge=1, le=128)
    sample_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    maximum_sampled_frames: int = Field(default=900, ge=5, le=10_000)
    signature_size: int = Field(default=32, ge=8, le=128)
    scene_change_threshold: float = Field(default=0.12, ge=0, le=1)
    duplicate_threshold: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def validate_limits(self) -> VideoKeyframeConfig:
        if self.minimum_keyframes > self.maximum_keyframes:
            raise ValueError("minimum_keyframes must not exceed maximum_keyframes")
        if self.duplicate_threshold > self.scene_change_threshold:
            raise ValueError("duplicate_threshold must not exceed scene_change_threshold")
        return self


@dataclass(frozen=True, slots=True)
class VideoKeyframeArtifact:
    media: EvidenceMedia
    frame_index: int
    timestamp_seconds: float
    format: Literal["png"]
    data: bytes


@dataclass(frozen=True, slots=True)
class VideoKeyframeRun:
    evidence: EventEvidenceV1
    artifacts: tuple[VideoKeyframeArtifact, ...]


class OpenCvVideoFrameDecoder:
    """Validate and sample a bounded set of frames from a locally materialized video."""

    def __init__(self, *, path_resolver: Callable[[EvidenceMedia], Path]) -> None:
        self.path_resolver = path_resolver

    def sample(
        self,
        media: EvidenceMedia,
        *,
        interval_seconds: float,
        max_frames: int,
    ) -> DecodedVideoSample:
        import cv2

        path = self.path_resolver(media)
        if not path.is_file():
            raise ValueError("video path does not reference a regular file")
        digest = sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != media.sha256:
            raise ValueError("video SHA-256 does not match EventEvidence")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"video cannot be opened: {path.name}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                raise ValueError("video frame rate is unavailable")
            step = max(1, round(fps * interval_seconds))
            indices = range(0, max(frame_count, 1), step)
            frames: list[DecodedVideoFrame] = []
            truncated = False
            for position, frame_index in enumerate(indices):
                if position >= max_frames:
                    truncated = True
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, pixels = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                frames.append(
                    DecodedVideoFrame(
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        image=Image.fromarray(rgb),
                    )
                )
            return DecodedVideoSample(frames=tuple(frames), truncated=truncated)
        finally:
            capture.release()


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.convert("RGB").save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _signature(image: Image.Image, size: int) -> bytes:
    thumbnail = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return thumbnail.tobytes()


def _distance(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("frame signatures must have the same non-zero length")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / (255 * len(left))


def _uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0 or count <= 0:
        return ()
    if count == 1:
        return (0,)
    return tuple(round(index * (length - 1) / (count - 1)) for index in range(count))


class VideoKeyframeExtractor:
    """Select 5-15 unique scene-aware keyframes and attach them to EventEvidence."""

    _MEDIA_UNCERTAINTIES: ClassVar[frozenset[str]] = frozenset(
        {
            "video_keyframe_extraction_failed",
            "video_insufficient_unique_frames",
            "video_sample_limit_applied",
        }
    )

    def __init__(
        self,
        *,
        decoder: VideoFrameDecoder,
        config: VideoKeyframeConfig | None = None,
    ) -> None:
        self.decoder = decoder
        self.config = config or VideoKeyframeConfig()

    def run(self, evidence: EventEvidenceV1) -> VideoKeyframeRun:
        existing_parent_ids = {
            media.parent_media_id for media in evidence.media if media.kind == "keyframe"
        }
        eligible = tuple(
            media
            for media in sorted(evidence.media, key=lambda item: item.media_id)
            if media.kind == "video" and media.media_id not in existing_parent_ids
        )
        selected_videos = eligible[: self.config.maximum_videos]
        selected_ids = {media.media_id for media in selected_videos}
        uncertainties = [
            item
            for item in evidence.uncertainties
            if not (
                item.scope_type == "media"
                and item.scope_id in selected_ids
                and item.code in self._MEDIA_UNCERTAINTIES
            )
        ]
        artifacts: list[VideoKeyframeArtifact] = []
        media_records = list(evidence.media)
        needs_human_review = evidence.needs_human_review

        if len(eligible) > len(selected_videos):
            needs_human_review = True
            uncertainties.append(
                self._uncertainty(
                    evidence.event_id,
                    "video_media_limit_applied",
                    "event",
                    evidence.event_id,
                    "The event exceeded the configured video extraction limit.",
                )
            )

        for video in selected_videos:
            try:
                sample = self.decoder.sample(
                    video,
                    interval_seconds=self.config.sample_interval_seconds,
                    max_frames=self.config.maximum_sampled_frames,
                )
                if len(sample.frames) > self.config.maximum_sampled_frames:
                    raise ValueError("video decoder exceeded the configured sample limit")
                chosen = self._select(sample.frames)
            except Exception as exc:
                needs_human_review = True
                uncertainties.append(
                    self._uncertainty(
                        evidence.event_id,
                        "video_keyframe_extraction_failed",
                        "media",
                        video.media_id,
                        f"Video keyframe extraction failed with {type(exc).__name__}.",
                    )
                )
                continue

            if sample.truncated:
                needs_human_review = True
                uncertainties.append(
                    self._uncertainty(
                        evidence.event_id,
                        "video_sample_limit_applied",
                        "media",
                        video.media_id,
                        "Video sampling reached the configured frame limit.",
                    )
                )
            if len(chosen) < self.config.minimum_keyframes:
                needs_human_review = True
                uncertainties.append(
                    self._uncertainty(
                        evidence.event_id,
                        "video_insufficient_unique_frames",
                        "media",
                        video.media_id,
                        (
                            f"Only {len(chosen)} unique keyframes remained after "
                            "scene selection and deduplication."
                        ),
                    )
                )

            for frame in chosen:
                data = _png_bytes(frame.image)
                digest = sha256(data).hexdigest()
                identity = f"{video.media_id}:{frame.frame_index}:{digest}"
                captured_at = (
                    None
                    if video.captured_at is None
                    else video.captured_at + timedelta(seconds=frame.timestamp_seconds)
                )
                keyframe = EvidenceMedia(
                    media_id=f"KF-{sha256(identity.encode()).hexdigest()[:24]}",
                    source_id=video.source_id,
                    media_group_id=video.media_group_id,
                    origin_id=video.origin_id,
                    kind="keyframe",
                    sha256=digest,
                    captured_at=captured_at,
                    parent_media_id=video.media_id,
                )
                media_records.append(keyframe)
                artifacts.append(
                    VideoKeyframeArtifact(
                        media=keyframe,
                        frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds,
                        format="png",
                        data=data,
                    )
                )

        updated = EventEvidenceV1.model_validate(
            evidence.model_copy(
                update={
                    "media": tuple(media_records),
                    "uncertainties": tuple(uncertainties),
                    "needs_human_review": needs_human_review,
                }
            )
        )
        return VideoKeyframeRun(evidence=updated, artifacts=tuple(artifacts))

    def _select(self, frames: tuple[DecodedVideoFrame, ...]) -> tuple[DecodedVideoFrame, ...]:
        if not frames:
            return ()
        if any(
            frame.frame_index < 0
            or not isfinite(frame.timestamp_seconds)
            or frame.timestamp_seconds < 0
            or frame.image.width <= 0
            or frame.image.height <= 0
            for frame in frames
        ):
            raise ValueError("decoded video frames contain invalid metadata")
        if len({frame.frame_index for frame in frames}) != len(frames):
            raise ValueError("decoded video frame indices must be unique")
        ordered = tuple(sorted(frames, key=lambda item: (item.timestamp_seconds, item.frame_index)))
        signatures = tuple(_signature(frame.image, self.config.signature_size) for frame in ordered)
        novelty = (
            1.0,
            *(
                _distance(signatures[index - 1], signatures[index])
                for index in range(1, len(signatures))
            ),
        )
        scene_indices = sorted(
            (
                index
                for index, score in enumerate(novelty)
                if score >= self.config.scene_change_threshold
            ),
            key=lambda index: (-novelty[index], index),
        )
        priority = [0, len(ordered) - 1]
        priority.extend(scene_indices)
        priority.extend(_uniform_indices(len(ordered), self.config.minimum_keyframes))
        priority.extend(sorted(range(len(ordered)), key=lambda index: (-novelty[index], index)))

        selected: list[int] = []
        considered: set[int] = set()
        for index in priority:
            if index in considered:
                continue
            considered.add(index)
            if any(
                _distance(signatures[index], signatures[other]) <= self.config.duplicate_threshold
                for other in selected
            ):
                continue
            selected.append(index)
            if len(selected) >= self.config.maximum_keyframes:
                break
        return tuple(ordered[index] for index in sorted(selected))

    @staticmethod
    def _uncertainty(
        event_id: str,
        code: str,
        scope_type: str,
        scope_id: str,
        description: str,
    ) -> Uncertainty:
        identity = f"{event_id}:{code}:{scope_type}:{scope_id}"
        return Uncertainty.model_validate(
            {
                "uncertainty_id": f"UNC-{sha256(identity.encode()).hexdigest()[:24]}",
                "code": code,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "description": description,
            }
        )


__all__ = [
    "DecodedVideoFrame",
    "DecodedVideoSample",
    "OpenCvVideoFrameDecoder",
    "VideoFrameDecoder",
    "VideoKeyframeArtifact",
    "VideoKeyframeConfig",
    "VideoKeyframeExtractor",
    "VideoKeyframeRun",
]
