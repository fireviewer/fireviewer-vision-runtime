from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from pydantic import Field

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    DetectionResultV1,
    EventEvidenceV1,
    Uncertainty,
    VisualObservation,
)
from fireviewer_contracts.mvp.providers import VisionProvider


class EventVisionConfig(StrictModel):
    eligible_kinds: tuple[Literal["photo", "keyframe"], ...] = ("photo", "keyframe")
    max_media: int = Field(default=256, ge=1, le=512)


@dataclass(frozen=True, slots=True)
class VisionArtifact:
    result_reference: str
    result: DetectionResultV1


@dataclass(frozen=True, slots=True)
class EventVisionRun:
    evidence: EventEvidenceV1
    artifacts: tuple[VisionArtifact, ...]


def vision_result_reference(provider: VisionProvider, media_id: str, input_hash: str) -> str:
    descriptor = provider.descriptor
    payload = json.dumps(
        {
            "config": descriptor.config,
            "input_hash": input_hash,
            "media_id": media_id,
            "model_id": descriptor.model_id,
            "model_version": descriptor.model_version,
            "provider_id": descriptor.provider_id,
            "provider_version": descriptor.provider_version,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"GDN-{sha256(payload).hexdigest()[:24]}"


class EventVisionRunner:
    """Run one interchangeable VisionProvider over event media with explicit abstention."""

    def __init__(
        self,
        *,
        provider: VisionProvider,
        config: EventVisionConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or EventVisionConfig()

    def run(self, evidence: EventEvidenceV1) -> EventVisionRun:
        eligible = tuple(
            sorted(
                (item for item in evidence.media if item.kind in self.config.eligible_kinds),
                key=lambda item: item.media_id,
            )
        )
        selected = eligible[: self.config.max_media]
        selected_ids = {item.media_id for item in selected}
        observations = [
            item
            for item in evidence.visual_observations
            if not (
                item.media_id in selected_ids
                and item.observation_type == "detection"
                and item.result_reference.startswith("GDN-")
            )
        ]
        uncertainties = [
            item
            for item in evidence.uncertainties
            if not (
                (
                    item.code == "vision_inference_failed"
                    and item.scope_type == "media"
                    and item.scope_id in selected_ids
                )
                or (
                    item.code == "vision_media_limit_applied"
                    and item.scope_type == "event"
                    and item.scope_id == evidence.event_id
                )
            )
        ]
        artifacts: list[VisionArtifact] = []
        needs_human_review = evidence.needs_human_review

        if len(eligible) > len(selected):
            needs_human_review = True
            uncertainties.append(
                self._uncertainty(
                    evidence.event_id,
                    "vision_media_limit_applied",
                    "event",
                    evidence.event_id,
                    "The event exceeded the configured VisionProvider media limit.",
                )
            )

        for media in selected:
            reference = vision_result_reference(self.provider, media.media_id, media.sha256)
            try:
                result = self.provider.detect(media)
                if result.media_id != media.media_id:
                    raise ValueError("VisionProvider returned a result for another media item")
            except Exception as exc:
                needs_human_review = True
                uncertainties.append(
                    self._uncertainty(
                        evidence.event_id,
                        "vision_inference_failed",
                        "media",
                        media.media_id,
                        f"Vision inference failed with {type(exc).__name__}.",
                    )
                )
                continue
            artifacts.append(VisionArtifact(result_reference=reference, result=result))
            observations.append(
                VisualObservation(
                    observation_id=f"OBS-{sha256(reference.encode()).hexdigest()[:24]}",
                    media_id=media.media_id,
                    observation_type="detection",
                    result_reference=reference,
                    confidence=max((item.score for item in result.detections), default=None),
                )
            )
            needs_human_review = needs_human_review or result.needs_human_review

        updated = EventEvidenceV1.model_validate(
            evidence.model_copy(
                update={
                    "visual_observations": tuple(observations),
                    "uncertainties": tuple(uncertainties),
                    "needs_human_review": needs_human_review,
                }
            )
        )
        return EventVisionRun(evidence=updated, artifacts=tuple(artifacts))

    @staticmethod
    def _uncertainty(
        event_id: str,
        code: str,
        scope_type: Literal["event", "media"],
        scope_id: str,
        description: str,
    ) -> Uncertainty:
        identity = f"{event_id}:{code}:{scope_type}:{scope_id}"
        return Uncertainty(
            uncertainty_id=f"UNC-{sha256(identity.encode()).hexdigest()[:24]}",
            code=code,
            scope_type=scope_type,
            scope_id=scope_id,
            description=description,
        )


__all__ = [
    "EventVisionConfig",
    "EventVisionRun",
    "EventVisionRunner",
    "VisionArtifact",
    "vision_result_reference",
]
