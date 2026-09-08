from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from time import perf_counter
from typing import Any, Literal, Protocol, cast

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import StrictModel
from fireviewer_contracts.mvp.contracts import (
    Detection,
    DetectionResultV1,
    EvidenceMedia,
    ProviderRun,
)
from fireviewer_contracts.mvp.providers import ProviderDescriptor, ProviderHealth


class BedrockRuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


class BedrockImage(StrictModel):
    data: bytes = Field(min_length=1, max_length=25 * 1024 * 1024)
    format: Literal["png", "jpeg", "gif", "webp"]


class BedrockImageLoader(Protocol):
    def load(self, media: EvidenceMedia) -> BedrockImage: ...


class BedrockNovaVisionConfig(StrictModel):
    region_name: str = Field(default="eu-west-3", min_length=3, max_length=32)
    model_id: str = Field(default="eu.amazon.nova-2-lite-v1:0", min_length=3, max_length=255)
    model_version: str = Field(default="v1:0", min_length=1, max_length=64)
    prompts: tuple[str, ...] = (
        "wildfire smoke",
        "smoke plume",
        "visible fire",
        "flames",
    )
    max_output_tokens: int = Field(default=1_024, ge=128, le=4_096)
    max_detections: int = Field(default=128, ge=1, le=512)
    require_human_review: bool = True
    input_token_price_per_million_usd: float = Field(default=0.30, ge=0)
    output_token_price_per_million_usd: float = Field(default=2.50, ge=0)

    @model_validator(mode="after")
    def validate_prompts(self) -> BedrockNovaVisionConfig:
        if not self.prompts or len(self.prompts) != len(set(self.prompts)):
            raise ValueError("Nova prompts must be present and unique")
        return self


class _NovaDetection(StrictModel):
    detection_class: Literal["fire", "smoke"] = Field(alias="class")
    bbox: tuple[int, int, int, int]
    score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bbox(self) -> _NovaDetection:
        left, top, right, bottom = self.bbox
        if not 0 <= left < right <= 1_000 or not 0 <= top < bottom <= 1_000:
            raise ValueError("Nova bounding boxes must be ordered on the 0-1000 scale")
        return self


class _NovaResponse(StrictModel):
    detections: tuple[_NovaDetection, ...] = Field(default=(), max_length=512)


def _extract_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Nova response does not contain a JSON object")


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", ())
    chunks = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
    if not chunks:
        raise ValueError("Nova response does not contain text output")
    return "\n".join(str(chunk) for chunk in chunks)


def _cost_usd(response: dict[str, Any], config: BedrockNovaVisionConfig) -> float | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    million = Decimal("1000000")
    cost = (
        Decimal(input_tokens)
        * Decimal(str(config.input_token_price_per_million_usd))
        / million
        + Decimal(output_tokens)
        * Decimal(str(config.output_token_price_per_million_usd))
        / million
    )
    return float(cost)


class BedrockNovaVisionProvider:
    """Managed RGB VisionProvider using Amazon Nova 2 Lite through Bedrock."""

    def __init__(
        self,
        *,
        image_loader: BedrockImageLoader,
        client: BedrockRuntimeClient | None = None,
        config: BedrockNovaVisionConfig | None = None,
    ) -> None:
        self.config = config or BedrockNovaVisionConfig()
        self.image_loader = image_loader
        self.client = client or self._build_client()
        self.descriptor = ProviderDescriptor(
            provider_id="aws-bedrock-nova-2-lite",
            provider_version="1.0.0",
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            config=self.config.model_dump(mode="json"),
            capabilities=("managed-fire-smoke-grounding", "managed-image-understanding"),
        )

    def _build_client(self) -> BedrockRuntimeClient:
        import boto3

        session = boto3.Session(region_name=self.config.region_name)
        return cast(BedrockRuntimeClient, session.client("bedrock-runtime"))

    def healthcheck(self) -> ProviderHealth:
        if not callable(getattr(self.client, "converse", None)):
            return ProviderHealth(
                status="unavailable",
                checked_at=datetime.now(UTC),
                reason_codes=("bedrock_converse_unavailable",),
            )
        return ProviderHealth(status="healthy", checked_at=datetime.now(UTC))

    def detect(self, media: EvidenceMedia) -> DetectionResultV1:
        started = perf_counter()
        image = self.image_loader.load(media)
        response = self.client.converse(
            modelId=self.config.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image.format,
                                "source": {"bytes": image.data},
                            }
                        },
                        {"text": self._prompt()},
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": self.config.max_output_tokens,
                "temperature": 0,
            },
        )
        provider_run = ProviderRun(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.provider_version,
            model_id=self.descriptor.model_id,
            model_version=self.descriptor.model_version,
            config=self.descriptor.config,
            input_hash=media.sha256,
            runtime_ms=int((perf_counter() - started) * 1_000),
            cost_usd=_cost_usd(response, self.config),
            generated_at=datetime.now(UTC),
        )
        try:
            parsed = _NovaResponse.model_validate(_extract_json_object(_response_text(response)))
        except (TypeError, ValueError):
            return DetectionResultV1(
                media_id=media.media_id,
                provider_run=provider_run,
                status="uncertain",
                needs_human_review=True,
            )

        detections: list[Detection] = []
        for item in parsed.detections[: self.config.max_detections]:
            normalized = (
                item.bbox[0] / 1_000,
                item.bbox[1] / 1_000,
                item.bbox[2] / 1_000,
                item.bbox[3] / 1_000,
            )
            identity = (
                f"{media.media_id}:{item.detection_class}:"
                f"{':'.join(str(value) for value in item.bbox)}:{item.score:.6f}"
            )
            detections.append(
                Detection(
                    detection_id=f"DET-{sha256(identity.encode()).hexdigest()[:24]}",
                    detection_class=item.detection_class,
                    bbox=normalized,
                    score=item.score,
                    prompt=item.detection_class,
                )
            )

        classes = {item.detection_class for item in detections}
        status: Literal["fire", "smoke", "fire_and_smoke", "none", "uncertain"]
        if classes == {"fire", "smoke"}:
            status = "fire_and_smoke"
        elif classes == {"fire"}:
            status = "fire"
        elif classes == {"smoke"}:
            status = "smoke"
        else:
            status = "none"
        return DetectionResultV1(
            media_id=media.media_id,
            provider_run=provider_run,
            detections=tuple(detections),
            status=status,
            needs_human_review=self.config.require_human_review,
        )

    def _prompt(self) -> str:
        targets = ", ".join(self.config.prompts)
        return (
            "Analyze this public RGB image for wildfire evidence. "
            f"Detect only these targets: {targets}. "
            "Reject clouds, fog, mist, haze, dust, steam, chimney smoke, industrial smoke, "
            "sun glare and backlight unless visible wildfire evidence is present. "
            "Return exactly one JSON object with this shape: "
            '{"detections":[{"class":"fire|smoke","bbox":[x1,y1,x2,y2],'
            '"score":0.0}]}. '
            "Coordinates must be ordered integers from 0 to 1000. "
            "Use an empty detections list when no qualifying target is visible."
        )


__all__ = [
    "BedrockImage",
    "BedrockImageLoader",
    "BedrockNovaVisionConfig",
    "BedrockNovaVisionProvider",
    "BedrockRuntimeClient",
]
