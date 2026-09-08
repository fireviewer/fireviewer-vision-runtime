from __future__ import annotations

from contextlib import suppress
from hashlib import sha256
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fireviewer_contracts.event_models import (
    EventPipelineInput,
    EvidenceAssetKind,
    PerceptionAnchor,
    PerceptionFailure,
    PhenomenonKind,
)
from fireviewer_contracts.model_registry import ModelSpec

EventPointSemantic = Literal[
    "active_fire_point",
    "visible_fire_front_point",
    "smoke_origin_point",
]
PerceptionPhenomenon = Literal[
    PhenomenonKind.ACTIVE_FIRE_POINT,
    PhenomenonKind.VISIBLE_FIRE_FRONT,
    PhenomenonKind.SMOKE_COLUMN_BASE,
]


class EventPerceptionPoint(BaseModel):
    """Closed pixel-only result returned by the event pointing adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_anchor: EventPointSemantic
    source_point_normalized: tuple[float, float]
    model_score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def point_is_normalized(self) -> EventPerceptionPoint:
        if any(value < 0 or value > 1 for value in self.source_point_normalized):
            raise ValueError("event perception points must be normalized")
        return self


class EventPerceptionAdapter(Protocol):
    spec: ModelSpec

    def load(self) -> None: ...

    def infer_event_image(
        self,
        *,
        evidence_asset_id: str,
        working_file_url: str,
    ) -> tuple[EventPerceptionPoint, ...]: ...

    def unload(self) -> None: ...


def event_has_working_urls(value: EventPipelineInput) -> bool:
    return any(asset.working_file_url is not None for asset in value.bundle.evidence_assets)


def validate_event_working_urls(
    value: EventPipelineInput,
    allowed_hosts: frozenset[str],
) -> tuple[PerceptionFailure, ...]:
    """Validate every private media URL against an exact HTTPS hostname allowlist."""

    failures: list[PerceptionFailure] = []
    normalized_hosts = frozenset(host.lower() for host in allowed_hosts)
    for asset in value.bundle.evidence_assets:
        if asset.working_file_url is None:
            continue
        parsed = urlsplit(str(asset.working_file_url))
        try:
            port = parsed.port
        except ValueError:
            port = -1
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.hostname is None
            or parsed.hostname.lower() not in normalized_hosts
            or port not in {None, 443}
        ):
            failures.append(
                PerceptionFailure(
                    evidence_asset_id=asset.evidence_asset_id,
                    reason_code="media_url_not_allowed",
                )
            )
    return tuple(failures)


def event_requires_image_inference(
    value: EventPipelineInput,
    url_failures: tuple[PerceptionFailure, ...],
) -> bool:
    invalid_assets = {
        failure.evidence_asset_id
        for failure in url_failures
        if failure.evidence_asset_id is not None
    }
    return any(
        asset.kind == EvidenceAssetKind.IMAGE
        and asset.working_file_url is not None
        and asset.evidence_asset_id not in invalid_assets
        for asset in value.bundle.evidence_assets
    )


def _phenomenon(semantic_anchor: EventPointSemantic) -> PerceptionPhenomenon:
    if semantic_anchor == "visible_fire_front_point":
        return PhenomenonKind.VISIBLE_FIRE_FRONT
    if semantic_anchor == "smoke_origin_point":
        return PhenomenonKind.SMOKE_COLUMN_BASE
    return PhenomenonKind.ACTIVE_FIRE_POINT


def _anchor_id(
    candidate_id: str,
    evidence_asset_id: str,
    semantic_anchor: str,
    index: int,
) -> str:
    digest = sha256(
        f"{candidate_id}\x1f{evidence_asset_id}\x1f{semantic_anchor}\x1f{index}".encode()
    ).hexdigest()[:24]
    return f"ANCHOR-{digest}"


def _failure(
    *,
    asset_id: str,
    reason_code: str,
    spec: ModelSpec | None,
) -> PerceptionFailure:
    return PerceptionFailure(
        evidence_asset_id=asset_id,
        reason_code=reason_code,
        model_id=spec.model_id if spec is not None else None,
        model_revision=spec.revision if spec is not None else None,
    )


def run_event_image_perception(
    value: EventPipelineInput,
    *,
    adapter: EventPerceptionAdapter | None,
    url_failures: tuple[PerceptionFailure, ...] = (),
    unavailable_reason_code: str = "fire_pointing_model_unavailable",
) -> tuple[EventPipelineInput, tuple[PerceptionFailure, ...]]:
    """Create pixel anchors for unanchored images without producing geography."""

    if value.perception_anchors:
        return value, url_failures

    spec = adapter.spec if adapter is not None else None
    failures = list(url_failures)
    invalid_assets = {
        failure.evidence_asset_id
        for failure in url_failures
        if failure.evidence_asset_id is not None
    }
    images = []
    for asset in value.bundle.evidence_assets:
        if asset.kind == EvidenceAssetKind.VIDEO:
            failures.append(
                _failure(
                    asset_id=asset.evidence_asset_id,
                    reason_code="video_frames_missing",
                    spec=spec,
                )
            )
            continue
        if asset.kind != EvidenceAssetKind.IMAGE:
            continue
        if asset.evidence_asset_id in invalid_assets:
            continue
        if asset.working_file_url is None:
            failures.append(
                _failure(
                    asset_id=asset.evidence_asset_id,
                    reason_code="image_working_file_url_missing",
                    spec=spec,
                )
            )
            continue
        images.append(asset)

    if not images:
        return value, tuple(failures)
    if adapter is None:
        failures.extend(
            _failure(
                asset_id=asset.evidence_asset_id,
                reason_code=unavailable_reason_code,
                spec=None,
            )
            for asset in images
        )
        return value, tuple(failures)

    anchors: list[PerceptionAnchor] = []
    try:
        adapter.load()
    except Exception:
        failures.extend(
            _failure(
                asset_id=asset.evidence_asset_id,
                reason_code="fire_pointing_model_unavailable",
                spec=adapter.spec,
            )
            for asset in images
        )
        with suppress(Exception):
            adapter.unload()
        return value, tuple(failures)

    release_failed = False
    try:
        for asset in images:
            try:
                raw_points = adapter.infer_event_image(
                    evidence_asset_id=asset.evidence_asset_id,
                    working_file_url=str(asset.working_file_url),
                )
                points = tuple(EventPerceptionPoint.model_validate(point) for point in raw_points)
            except Exception:
                failures.append(
                    _failure(
                        asset_id=asset.evidence_asset_id,
                        reason_code="fire_pointing_inference_failed",
                        spec=adapter.spec,
                    )
                )
                continue
            if not points:
                failures.append(
                    _failure(
                        asset_id=asset.evidence_asset_id,
                        reason_code="fire_pointing_no_anchor",
                        spec=adapter.spec,
                    )
                )
                continue
            for index, point in enumerate(points, start=1):
                anchors.append(
                    PerceptionAnchor(
                        anchor_id=_anchor_id(
                            value.bundle.candidate_id,
                            asset.evidence_asset_id,
                            point.semantic_anchor,
                            index,
                        ),
                        evidence_asset_id=asset.evidence_asset_id,
                        phenomenon=_phenomenon(point.semantic_anchor),
                        source_point_normalized=point.source_point_normalized,
                        model_id=adapter.spec.model_id,
                        model_revision=adapter.spec.revision,
                        model_score=point.model_score,
                    )
                )
    finally:
        try:
            adapter.unload()
        except Exception:
            release_failed = True

    if release_failed:
        anchors.clear()
        failures.extend(
            _failure(
                asset_id=asset.evidence_asset_id,
                reason_code="fire_pointing_model_release_failed",
                spec=adapter.spec,
            )
            for asset in images
        )

    enriched = value.model_copy(update={"perception_anchors": tuple(anchors)})
    return EventPipelineInput.model_validate(enriched), tuple(failures)
