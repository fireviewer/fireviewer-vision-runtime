"""Official Prithvi BurnScars inference adapter.

The adapter deliberately accepts only the explicit FireViewer six-band
GeoTIFF contract. RGB previews, thermal products and hotspots remain useful to
the rest of the satellite pipeline but never enter this model. This adapter is
an opportunistic burned-area enhancement, not the satellite-operation gate.
"""

from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from fireviewer_contracts.contracts import (
    SourceAnnotationV2,
    SpatialProposalV2,
    WorkerBatchItemV2,
    WorkerInputV2,
)
from fireviewer_contracts.client.media_fetcher import MediaFetcher
from fireviewer_vision_runtime.memory_manager import release_cuda_memory
from fireviewer_contracts.model_registry import ModelSpec, resolve_cached_snapshot
from fireviewer_vision_runtime.v2_burned_area import CANONICAL_BURNED_AREA_BANDS

_CHECKPOINT_NAME = "Prithvi_EO_V2_300M_BurnScars.pt"
_CONFIG_NAME = "burn_scars_config.yaml"


def _write_offline_inference_config(source: Path, target: Path) -> None:
    """Derive a no-network config while preserving the qualified task checkpoint."""

    import yaml

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("official Prithvi BurnScars config is not a mapping")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("official Prithvi BurnScars config has no model section")
    init_args = model.get("init_args")
    if not isinstance(init_args, dict):
        raise RuntimeError("official Prithvi BurnScars config has no model init args")
    model_args = init_args.get("model_args")
    if not isinstance(model_args, dict) or model_args.get("backbone") != "prithvi_eo_v2_300":
        raise RuntimeError("official Prithvi BurnScars config declares an unexpected backbone")
    model_args["backbone_pretrained"] = False
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _cuda_autocast_dtype(torch_module: Any, device: Any) -> Any | None:
    """Select an autocast dtype supported by the active CUDA device.

    T4 GPUs don't provide native BF16 execution, so they must use FP16. Newer
    accelerators keep BF16 when PyTorch reports native support.
    """

    if device.type != "cuda":
        return None
    supports_bf16 = getattr(torch_module.cuda, "is_bf16_supported", None)
    if callable(supports_bf16) and bool(supports_bf16()):
        return torch_module.bfloat16
    return torch_module.float16


def _default_tile_batch_size(torch_module: Any, device: Any) -> int:
    """Keep the first T4 deployment conservative without slowing larger GPUs."""

    if device.type != "cuda":
        return 1
    try:
        total_memory = int(torch_module.cuda.get_device_properties(device).total_memory)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 1
    return 1 if total_memory <= 20 * 1024**3 else 4


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20].upper()
    return f"{prefix}-{digest}"


def _closed_polygon(points: list[list[float]]) -> list[list[float]] | None:
    if len(points) < 3:
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    return points if len(points) >= 4 else None


class PrithviBurnedAreaAdapter:
    """Sequential GPU adapter for the immutable official BurnScars checkpoint."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        cache_root: Path,
        fetcher: MediaFetcher,
    ) -> None:
        self.spec = spec
        self.cache_root = cache_root
        self.fetcher = fetcher
        self.inference_model: Any | None = None

    def load(self) -> None:
        from terratorch.cli_tools import LightningInferenceModel

        snapshot = resolve_cached_snapshot(self.spec, self.cache_root)
        checkpoint = snapshot / _CHECKPOINT_NAME
        config = snapshot / _CONFIG_NAME
        if not checkpoint.is_file() or not config.is_file():
            raise RuntimeError("official Prithvi BurnScars snapshot is incomplete")
        with tempfile.TemporaryDirectory(prefix="fireviewer-prithvi-config-") as directory:
            offline_config = Path(directory) / _CONFIG_NAME
            _write_offline_inference_config(config, offline_config)
            self.inference_model = LightningInferenceModel.from_config(
                str(offline_config),
                str(checkpoint),
            )
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_model.model.to(device)
        self.inference_model.model.eval()

    def unload(self) -> None:
        self.inference_model = None
        release_cuda_memory()

    def infer(
        self,
        batch: WorkerInputV2,
    ) -> tuple[
        dict[str, tuple[SourceAnnotationV2, ...]],
        dict[str, tuple[SpatialProposalV2, ...]],
    ]:
        if self.inference_model is None:
            raise RuntimeError("official Prithvi BurnScars model is not loaded")
        annotations: dict[str, tuple[SourceAnnotationV2, ...]] = {}
        proposals: dict[str, tuple[SpatialProposalV2, ...]] = {}
        for item in batch.items:
            if not self._applicable(item):
                continue
            product_annotations, product_proposals = self._infer_product(batch, item)
            annotations[item.input_id] = product_annotations
            proposals[item.input_id] = product_proposals
        return annotations, proposals

    @staticmethod
    def _applicable(item: WorkerBatchItemV2) -> bool:
        return (
            item.media_type.value == "satellite_image"
            and item.working_file_url is not None
            and item.satellite is not None
            and tuple(item.satellite.bands) == CANONICAL_BURNED_AREA_BANDS
        )

    def _infer_product(
        self,
        batch: WorkerInputV2,
        item: WorkerBatchItemV2,
    ) -> tuple[tuple[SourceAnnotationV2, ...], tuple[SpatialProposalV2, ...]]:
        import cv2
        import numpy as np
        import rasterio
        from rasterio.warp import transform_geom

        satellite = item.satellite
        reference = batch.reference_bundle
        if satellite is None or item.working_file_url is None or reference is None:
            return (), ()
        max_cloud = float(os.getenv("FW_PRITHVI_MAX_CLOUD_PERCENT", "80"))
        if satellite.cloud_cover_percent is not None and satellite.cloud_cover_percent > max_cloud:
            return (), ()

        with (
            self.fetcher.download(str(item.working_file_url)) as raster_path,
            rasterio.open(raster_path) as dataset,
        ):
            if dataset.count != len(CANONICAL_BURNED_AREA_BANDS):
                raise ValueError("Prithvi input must contain exactly six raster bands")
            if dataset.crs is None:
                raise ValueError("Prithvi input GeoTIFF has no CRS")
            if (
                dataset.width != satellite.raster_width_px
                or dataset.height != satellite.raster_height_px
            ):
                raise ValueError("Prithvi raster dimensions differ from signed metadata")
            declared_transform = tuple(float(value) for value in satellite.geotransform)
            actual_transform = tuple(float(value) for value in dataset.transform.to_gdal())
            if any(
                abs(actual - declared) > 1e-9
                for actual, declared in zip(actual_transform, declared_transform, strict=True)
            ):
                raise ValueError("Prithvi GeoTIFF transform differs from signed metadata")
            descriptions = tuple(value for value in dataset.descriptions if value)
            if descriptions and descriptions != CANONICAL_BURNED_AREA_BANDS:
                raise ValueError("Prithvi GeoTIFF band descriptions are not canonical")
            reflectance = dataset.read(out_dtype="float32")
            transform = dataset.transform
            crs = dataset.crs
            width = dataset.width
            height = dataset.height

        if float(np.nanmean(reflectance)) > 1:
            reflectance = reflectance / 10_000.0
        reflectance = np.nan_to_num(reflectance, nan=0.0, posinf=0.0, neginf=0.0)
        prediction, confidence = self._predict(reflectance)
        if not np.any(prediction):
            return (), ()

        minimum_pixels = int(os.getenv("FW_PRITHVI_MIN_COMPONENT_PIXELS", "16"))
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            prediction.astype(np.uint8),
            connectivity=8,
        )
        cleaned = np.zeros_like(prediction, dtype=np.uint8)
        for label in range(1, component_count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_pixels:
                cleaned[labels == label] = 1
        if not np.any(cleaned):
            return (), ()

        contours, _hierarchy = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:64]
        normalized_polygons: list[list[list[list[float]]]] = []
        projected_polygons: list[list[list[list[float]]]] = []
        for contour in contours:
            epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
            approximated = cv2.approxPolyDP(contour, epsilon, True)
            pixels = [(float(point[0][0]), float(point[0][1])) for point in approximated]
            normalized = _closed_polygon(
                [
                    [
                        min(1.0, max(0.0, x / max(1, width - 1))),
                        min(1.0, max(0.0, y / max(1, height - 1))),
                    ]
                    for x, y in pixels
                ]
            )
            source_coordinates = _closed_polygon(
                [[float(value) for value in transform * (x, y)] for x, y in pixels]
            )
            if normalized is None or source_coordinates is None:
                continue
            source_geometry: dict[str, object] = {
                "type": "Polygon",
                "coordinates": [source_coordinates],
            }
            wgs84_geometry = transform_geom(
                crs,
                "EPSG:4326",
                source_geometry,
                precision=7,
            )
            raw_coordinates = wgs84_geometry.get("coordinates")
            if not isinstance(raw_coordinates, list):
                continue
            normalized_polygons.append([normalized])
            projected_polygons.append(raw_coordinates)
        if not projected_polygons:
            return (), ()

        normalized_geometry: dict[str, object] = {
            "type": "MultiPolygon",
            "coordinates": normalized_polygons,
        }
        projected_geometry: dict[str, object] = {
            "type": "MultiPolygon",
            "coordinates": projected_polygons,
        }
        annotation_id = _stable_id("SA", item.input_id, satellite.product_id, "burned-area")
        annotation = SourceAnnotationV2(
            annotation_id=annotation_id,
            evidence_id=item.input_id,
            evidence_kind="satellite_image",
            semantic_anchor="burned_area_polygon",
            source_geometry_normalized=normalized_geometry,
            model_score=confidence,
        )
        proposal = SpatialProposalV2(
            proposal_id=_stable_id("SP", annotation_id, "satellite-geotransform"),
            annotation_id=annotation_id,
            status="projected_geometry",
            proposal_kind="burned_area_polygon",
            observed_at=satellite.acquired_at,
            geometry_origin="SATELLITE_GEOTRANSFORM",
            geometry_geojson=projected_geometry,
            horizontal_accuracy_m=satellite.resolution_m,
            reference_bundle_sha256=reference.manifest_sha256,
            uncertainty_codes=("burned_area_model_proposal",),
        )
        return (annotation,), (proposal,)

    def _predict(self, reflectance: Any) -> tuple[Any, float]:
        import numpy as np
        import torch
        import torch.nn.functional as functional

        if self.inference_model is None:
            raise RuntimeError("official Prithvi BurnScars model is not loaded")
        model = self.inference_model.model
        datamodule = self.inference_model.datamodule
        tile_size = 512
        height, width = reflectance.shape[-2:]
        pad_height = (tile_size - height % tile_size) % tile_size
        pad_width = (tile_size - width % tile_size) % tile_size
        padded = np.pad(
            reflectance,
            ((0, 0), (0, pad_height), (0, pad_width)),
            mode="reflect",
        )
        tiles: list[tuple[int, int, Any]] = []
        for y in range(0, padded.shape[1], tile_size):
            for x in range(0, padded.shape[2], tile_size):
                patch = padded[:, y : y + tile_size, x : x + tile_size]
                transformed = datamodule.test_transform(image=patch.transpose(1, 2, 0))
                transformed["image"] = transformed["image"].unsqueeze(0)
                tiles.append((y, x, datamodule.aug(transformed)["image"]))

        device = next(model.parameters()).device
        default_batch_size = _default_tile_batch_size(torch, device)
        batch_size = max(
            1,
            int(os.getenv("FW_PRITHVI_TILE_BATCH_SIZE", str(default_batch_size))),
        )
        autocast_dtype = _cuda_autocast_dtype(torch, device)
        mask = np.zeros(padded.shape[1:], dtype=np.uint8)
        positive_confidences: list[float] = []
        for offset in range(0, len(tiles), batch_size):
            group = tiles[offset : offset + batch_size]
            tensor = torch.cat([tile[2] for tile in group], dim=0).to(device)
            autocast_enabled = device.type == "cuda"
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled and autocast_dtype is not None,
                ),
            ):
                raw_output = model(tensor)
                logits = raw_output.output
                logits = functional.interpolate(
                    logits.float(),
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
                probabilities = torch.softmax(logits, dim=1)
                predicted = probabilities.argmax(dim=1)
            for index, (y, x, _tensor) in enumerate(group):
                tile_mask = predicted[index].detach().cpu().numpy().astype(np.uint8)
                mask[y : y + tile_size, x : x + tile_size] = tile_mask
                positive = probabilities[index, 1][predicted[index] == 1]
                if positive.numel():
                    positive_confidences.append(float(positive.mean().detach().cpu()))
        confidence = float(np.mean(positive_confidences)) if positive_confidences else 0.0
        return mask[:height, :width], min(1.0, max(0.0, confidence))
