"""Bounded CPU keyframe extraction from durable backend video evidence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from hashlib import sha256
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field, SecretStr

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.contracts import EvidenceMedia
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceAdapter,
    AzureBackendEventEvidenceConfig,
    BackendDerivedKeyframeReceipt,
    BackendKeyframeEvidencePublisher,
    EventEvidenceRepository,
)
from fireviewer_vision_runtime.mvp.vision.video_keyframes import (
    OpenCvVideoFrameDecoder,
    VideoKeyframeConfig,
    VideoKeyframeExtractor,
)

_MAX_REQUEST_BYTES = 4 * 1_024


class KeyframeExtractionRequest(StrictModel):
    candidate_id: SafeIdentifierV2


class KeyframeArtifactTicket(StrictModel):
    media_id: SafeIdentifierV2
    parent_media_id: SafeIdentifierV2
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0, allow_inf_nan=False)
    sha256: Sha256HexV2
    size_bytes: int = Field(gt=0)


class KeyframeExtractionReceipt(StrictModel):
    candidate_id: SafeIdentifierV2
    source_revision_sha256: Sha256HexV2
    video_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)
    keyframes: tuple[KeyframeArtifactTicket, ...] = Field(default=(), max_length=480)
    raw_keyframes_stored: bool = False
    requires_durable_sink_before_yolo: bool = True


class VideoDownloadError(RuntimeError):
    pass


class KeyframeEvidencePublisher(Protocol):
    def publish(
        self,
        *,
        candidate_id: str,
        source_revision_sha256: str,
        media: EvidenceMedia,
        frame_index: int,
        timestamp_seconds: float,
        content: bytes,
    ) -> BackendDerivedKeyframeReceipt: ...


class HttpEvidenceVideoMaterializer:
    def __init__(
        self,
        *,
        locations: Mapping[str, str],
        allowed_hosts: frozenset[str],
        bearer_tokens_by_origin: Mapping[str, str] | None = None,
        maximum_bytes: int = 512 * 1_024 * 1_024,
        client: httpx.Client | None = None,
    ) -> None:
        if not allowed_hosts:
            raise ValueError("at least one video evidence host must be allowlisted")
        self._locations = dict(locations)
        self._allowed_hosts = allowed_hosts
        self._bearer_tokens_by_origin = {
            origin.rstrip("/").casefold(): token
            for origin, token in (bearer_tokens_by_origin or {}).items()
        }
        self._maximum_bytes = maximum_bytes
        self._client = client

    def materialize(self, media: EvidenceMedia, destination: Path) -> Path:
        raw_url = self._locations.get(media.media_id)
        if raw_url is None:
            raise VideoDownloadError("missing working URL for video evidence")
        parsed = urlsplit(raw_url)
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme != "https"
            or hostname not in self._allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise VideoDownloadError("video evidence URL is outside the HTTPS host allowlist")
        origin = f"https://{parsed.netloc.casefold()}"
        headers = {
            "Accept": "video/*,application/octet-stream",
            "User-Agent": "FireViewer-Keyframes/1.0 (+https://fireviewer.org)",
        }
        token = self._bearer_tokens_by_origin.get(origin)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        owned_client = self._client is None
        client = self._client or httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(120.0, connect=10.0),
            trust_env=False,
        )
        digest = sha256()
        size = 0
        try:
            with client.stream("GET", raw_url, headers=headers) as response:
                if response.is_redirect:
                    raise VideoDownloadError("video evidence redirects are not allowed")
                response.raise_for_status()
                declared = response.headers.get("Content-Length")
                if declared and declared.isdecimal() and int(declared) > self._maximum_bytes:
                    raise VideoDownloadError("video evidence exceeds the download limit")
                with destination.open("xb") as stream:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self._maximum_bytes:
                            raise VideoDownloadError(
                                "video evidence exceeds the download limit"
                            )
                        digest.update(chunk)
                        stream.write(chunk)
        except httpx.HTTPError as exc:
            raise VideoDownloadError("video evidence download failed") from exc
        finally:
            if owned_client:
                client.close()
        if size <= 0 or digest.hexdigest() != media.sha256:
            destination.unlink(missing_ok=True)
            raise VideoDownloadError("video evidence SHA-256 differs from EventEvidence")
        return destination


class KeyframeCpuWorker:
    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        allowed_hosts: frozenset[str],
        backend_origin: str,
        backend_token: str,
        publisher: KeyframeEvidencePublisher,
        maximum_video_bytes: int = 512 * 1_024 * 1_024,
        materializer_client: httpx.Client | None = None,
        config: VideoKeyframeConfig | None = None,
    ) -> None:
        self._repository = repository
        self._allowed_hosts = allowed_hosts
        self._backend_origin = backend_origin.rstrip("/")
        self._backend_token = backend_token
        self._publisher = publisher
        self._maximum_video_bytes = maximum_video_bytes
        self._materializer_client = materializer_client
        self._config = config

    def run_candidate(self, candidate_id: str) -> KeyframeExtractionReceipt:
        durable = self._repository.read(candidate_id)
        videos = tuple(item for item in durable.event.media if item.kind == "video")
        if not videos:
            return KeyframeExtractionReceipt(
                candidate_id=candidate_id,
                source_revision_sha256=durable.source_revision_sha256,
                video_count=0,
                artifact_count=0,
                raw_keyframes_stored=True,
                requires_durable_sink_before_yolo=False,
            )
        locations = {item.media_id: item.working_file_url for item in durable.media_locations}
        materializer = HttpEvidenceVideoMaterializer(
            locations=locations,
            allowed_hosts=self._allowed_hosts,
            bearer_tokens_by_origin={self._backend_origin: self._backend_token},
            maximum_bytes=self._maximum_video_bytes,
            client=self._materializer_client,
        )
        with ExitStack() as stack:
            temporary = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            paths: dict[str, Path] = {}
            for index, video in enumerate(videos):
                paths[video.media_id] = materializer.materialize(
                    video, temporary / f"video-{index:03d}.bin"
                )
            existing_keyframe_ids = {
                item.media_id for item in durable.event.media if item.kind == "keyframe"
            }
            extraction_input = durable.event.model_copy(
                update={
                    "media": tuple(
                        item for item in durable.event.media if item.kind != "keyframe"
                    ),
                    "visual_observations": tuple(
                        item
                        for item in durable.event.visual_observations
                        if item.media_id not in existing_keyframe_ids
                    ),
                }
            )
            run = VideoKeyframeExtractor(
                decoder=OpenCvVideoFrameDecoder(
                    path_resolver=lambda media: paths[media.media_id]
                ),
                config=self._config,
            ).run(extraction_input)
            revision = durable.source_revision_sha256
            tickets: list[KeyframeArtifactTicket] = []
            for artifact in run.artifacts:
                receipt = self._publisher.publish(
                    candidate_id=candidate_id,
                    source_revision_sha256=revision,
                    media=artifact.media,
                    frame_index=artifact.frame_index,
                    timestamp_seconds=artifact.timestamp_seconds,
                    content=artifact.data,
                )
                revision = receipt.source_revision_sha256
                tickets.append(
                    KeyframeArtifactTicket(
                    media_id=artifact.media.media_id,
                    parent_media_id=str(artifact.media.parent_media_id),
                    frame_index=artifact.frame_index,
                    timestamp_seconds=artifact.timestamp_seconds,
                    sha256=artifact.media.sha256,
                    size_bytes=len(artifact.data),
                )
                )
        return KeyframeExtractionReceipt(
            candidate_id=candidate_id,
            source_revision_sha256=revision,
            video_count=len(videos),
            artifact_count=len(tickets),
            keyframes=tuple(tickets),
            raw_keyframes_stored=True,
            requires_durable_sink_before_yolo=False,
        )


class KeyframeCpuServiceSettings(StrictModel):
    host: str = Field(default="0.0.0.0", min_length=1, max_length=255)  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65_535)
    auth_token: SecretStr = Field(min_length=32, max_length=4_096)
    allowed_hosts: frozenset[str] = Field(min_length=1, max_length=64)
    maximum_video_bytes: int = Field(
        default=512 * 1_024 * 1_024,
        ge=1_024 * 1_024,
        le=2 * 1_024 * 1_024 * 1_024,
    )
    backend: AzureBackendEventEvidenceConfig

    @classmethod
    def from_env(cls) -> KeyframeCpuServiceSettings:
        backend_url = os.environ["FIREVIEWER_BACKEND_BASE_URL"].rstrip("/")
        backend_host = (urlsplit(backend_url).hostname or "").casefold()
        allowed_hosts = frozenset(
            {
                backend_host,
                *(
                    item.strip().casefold()
                    for item in os.getenv("FIREVIEWER_KEYFRAME_ALLOWED_HOSTS", "").split(",")
                    if item.strip()
                ),
            }
        )
        return cls(
            host=os.getenv("FIREVIEWER_KEYFRAME_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.getenv("PORT", "8080")),
            auth_token=SecretStr(os.environ["FIREVIEWER_KEYFRAME_TOKEN"]),
            allowed_hosts=allowed_hosts,
            maximum_video_bytes=int(
                os.getenv(
                    "FIREVIEWER_KEYFRAME_MAX_VIDEO_BYTES",
                    str(512 * 1_024 * 1_024),
                )
            ),
            backend=AzureBackendEventEvidenceConfig(
                base_url=backend_url,
                bearer_token=SecretStr(os.environ["FIREVIEWER_BACKEND_TOKEN"]),
                timeout_seconds=float(os.getenv("FIREVIEWER_BACKEND_TIMEOUT_SECONDS", "20")),
            ),
        )


class KeyframeCpuService:
    def __init__(
        self,
        *,
        settings: KeyframeCpuServiceSettings,
        worker: KeyframeCpuWorker | None = None,
    ) -> None:
        self.settings = settings
        self.worker = worker or KeyframeCpuWorker(
            repository=AzureBackendEventEvidenceAdapter(settings.backend),
            allowed_hosts=settings.allowed_hosts,
            backend_origin=settings.backend.base_url,
            backend_token=settings.backend.bearer_token.get_secret_value(),
            publisher=BackendKeyframeEvidencePublisher(settings.backend),
            maximum_video_bytes=settings.maximum_video_bytes,
        )

    def authorize(self, header: str | None) -> bool:
        expected = f"Bearer {self.settings.auth_token.get_secret_value()}"
        return header is not None and compare_digest(header, expected)


def _handler_for(service: KeyframeCpuService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FireViewerKeyframesCpu/1.0"

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            import cv2

            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "runtime": "linux-cpu",
                    "opencv_version": cv2.__version__,
                    "raw_keyframes_stored": True,
                    "durable_sink_connected": True,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/event-evidence/keyframes":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not service.authorize(self.headers.get("Authorization")):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= _MAX_REQUEST_BYTES:
                    raise ValueError("request_size_invalid")
                request = KeyframeExtractionRequest.model_validate_json(
                    self.rfile.read(length)
                )
                result = service.worker.run_candidate(request.candidate_id)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__, "detail": str(exc)[:1_000]},
                )
                return
            self._json(HTTPStatus.OK, result.model_dump(mode="json"))

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def create_keyframe_cpu_server(service: KeyframeCpuService) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (service.settings.host, service.settings.port),
        _handler_for(service),
    )


def main() -> None:
    settings = KeyframeCpuServiceSettings.from_env()
    server = create_keyframe_cpu_server(KeyframeCpuService(settings=settings))
    server.serve_forever()


if __name__ == "__main__":
    main()


__all__ = [
    "HttpEvidenceVideoMaterializer",
    "KeyframeArtifactTicket",
    "KeyframeCpuService",
    "KeyframeCpuServiceSettings",
    "KeyframeCpuWorker",
    "KeyframeExtractionReceipt",
    "KeyframeExtractionRequest",
    "VideoDownloadError",
    "create_keyframe_cpu_server",
]
