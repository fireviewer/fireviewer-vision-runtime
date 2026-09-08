from __future__ import annotations

from fireviewer_contracts.resources import schema_path

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fireviewer_contracts.contracts import WorkerInputV2
from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_vision_runtime.prithvi_burned_area import (
    PrithviBurnedAreaAdapter,
    _cuda_autocast_dtype,
    _default_tile_batch_size,
    _write_offline_inference_config,
)
from fireviewer_vision_runtime.v2_burned_area import run_burned_area_stage

EXAMPLE = schema_path('agent-worker/v2/examples/valid-input.json')


def _satellite_batch(
    *,
    bands: list[str],
    working_file_url: str = "https://media.internal/sentinel/six-band.tif",
    raster_width_px: int = 1024,
    raster_height_px: int = 768,
    geotransform: list[float] | None = None,
    bbox_wgs84: list[float] | None = None,
) -> WorkerInputV2:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["batch_type"] = "satellite_media"
    payload["items"] = [
        {
            "input_id": "SENTINEL-2-INPUT",
            "media_type": "satellite_image",
            "working_file_url": working_file_url,
            "provenance": {
                "source_key": "SENTINEL-2-L2A",
                "source_reference_url": "https://dataspace.copernicus.eu/",
                "license_identifier": "COPERNICUS-DATA",
                "attribution": "Contains modified Copernicus Sentinel data",
                "trust": "institutional",
            },
            "satellite": {
                "product_id": "S2B-31UDP-20260713-L2A",
                "provider": "Copernicus Sentinel-2",
                "acquired_at": "2026-07-13T10:51:00Z",
                "crs": "EPSG:4326",
                "raster_width_px": raster_width_px,
                "raster_height_px": raster_height_px,
                "geotransform": geotransform or [2.46, 0.00025, 0.0, 48.44, 0.0, -0.00025],
                "bbox_wgs84": bbox_wgs84 or [2.46, 48.34, 2.72, 48.44],
                "resolution_m": 20,
                "bands": bands,
                "cloud_cover_percent": 12.5,
            },
        }
    ]
    return WorkerInputV2.model_validate(payload)


class _AbstainingAdapter:
    spec = ModelSpec(
        role="burned_area",
        model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        revision="a3f2c410e45b8ac7417976614528a872f024d831",
    )

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def infer(self, batch: WorkerInputV2):
        assert self.loaded is True
        assert batch.items[0].satellite is not None
        return {}, {}

    def unload(self) -> None:
        self.unloaded = True


class _LocalRasterFetcher:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def download(self, _url: str):
        yield self.path


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeCuda:
    def __init__(self, *, bf16: bool, memory_gib: int) -> None:
        self._bf16 = bf16
        self._memory = memory_gib * 1024**3

    def is_bf16_supported(self) -> bool:
        return self._bf16

    def get_device_properties(self, _device: object) -> object:
        return type("Properties", (), {"total_memory": self._memory})()


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, *, bf16: bool, memory_gib: int) -> None:
        self.cuda = _FakeCuda(bf16=bf16, memory_gib=memory_gib)


def test_prithvi_uses_fp16_and_single_tile_batch_on_t4() -> None:
    torch_module = _FakeTorch(bf16=False, memory_gib=16)
    device = _FakeDevice("cuda")

    assert _cuda_autocast_dtype(torch_module, device) == "float16"
    assert _default_tile_batch_size(torch_module, device) == 1


def test_prithvi_keeps_bf16_and_larger_batch_on_large_gpu() -> None:
    torch_module = _FakeTorch(bf16=True, memory_gib=48)
    device = _FakeDevice("cuda")

    assert _cuda_autocast_dtype(torch_module, device) == "bfloat16"
    assert _default_tile_batch_size(torch_module, device) == 4


def test_prithvi_derives_an_offline_config_without_changing_the_source(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    target = tmp_path / "offline.yaml"
    source.write_text(
        "model:\n"
        "  init_args:\n"
        "    model_args:\n"
        "      backbone: prithvi_eo_v2_300\n"
        "      backbone_pretrained: true\n",
        encoding="utf-8",
    )

    _write_offline_inference_config(source, target)

    assert "backbone_pretrained: true" in source.read_text(encoding="utf-8")
    assert "backbone_pretrained: false" in target.read_text(encoding="utf-8")


def test_burned_area_skips_rgb_without_requesting_a_model() -> None:
    execution = run_burned_area_stage(
        _satellite_batch(bands=["RED", "GREEN", "BLUE"]),
        adapter=None,
        sequence=7,
    )

    assert execution.stage_trace.status == "skipped"
    assert execution.stage_trace.preflight.decision == "not_applicable"
    assert execution.stage_trace.preflight.reason_codes == ("no_compatible_multispectral_product",)
    assert execution.stage_trace.preflight.downstream_possible is True
    assert execution.model_run is None


def test_burned_area_requires_the_runtime_for_a_compatible_product() -> None:
    execution = run_burned_area_stage(
        _satellite_batch(
            bands=[
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "SWIR_1",
                "SWIR_2",
            ]
        ),
        adapter=None,
        sequence=7,
    )

    assert execution.stage_trace.status == "skipped"
    assert execution.stage_trace.preflight.decision == "human_review"
    assert execution.stage_trace.preflight.reason_codes == ("burned_area_model_unavailable",)


def test_burned_area_records_an_explicit_model_abstention() -> None:
    adapter = _AbstainingAdapter()
    execution = run_burned_area_stage(
        _satellite_batch(
            bands=[
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "SWIR_1",
                "SWIR_2",
            ]
        ),
        adapter=adapter,
        sequence=7,
    )

    assert adapter.loaded is True
    assert adapter.unloaded is True
    assert execution.stage_trace.status == "succeeded"
    assert execution.stage_trace.postflight is not None
    assert execution.stage_trace.postflight.reason_codes == ("burned_area_model_abstained",)
    assert execution.model_run is not None
    assert execution.model_run.status == "succeeded"


def test_prithvi_adapter_projects_a_mask_through_the_signed_geotiff(
    tmp_path: Path,
) -> None:
    import rasterio
    from rasterio.transform import from_origin

    raster_path = tmp_path / "six-band.tif"
    transform = from_origin(2.5, 48.49, 0.01, 0.01)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=20,
        height=10,
        count=6,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(np.zeros((6, 10, 20), dtype=np.uint16))
        for index, description in enumerate(
            ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"),
            start=1,
        ):
            dataset.set_band_description(index, description)

    adapter = PrithviBurnedAreaAdapter(
        _AbstainingAdapter.spec,
        cache_root=tmp_path,
        fetcher=_LocalRasterFetcher(raster_path),  # type: ignore[arg-type]
    )
    adapter.inference_model = object()
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 4:16] = 1
    adapter._predict = lambda _reflectance: (mask, 0.91)  # type: ignore[method-assign]
    annotations, proposals = adapter.infer(
        _satellite_batch(
            bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            raster_width_px=20,
            raster_height_px=10,
            geotransform=[2.5, 0.01, 0.0, 48.49, 0.0, -0.01],
            bbox_wgs84=[2.5, 48.39, 2.7, 48.49],
        )
    )

    assert len(annotations["SENTINEL-2-INPUT"]) == 1
    proposal = proposals["SENTINEL-2-INPUT"][0]
    assert proposal.proposal_kind == "burned_area_polygon"
    assert proposal.geometry_origin == "SATELLITE_GEOTRANSFORM"
    assert proposal.geometry_geojson is not None
    assert proposal.geometry_geojson["type"] == "MultiPolygon"
    assert proposal.horizontal_accuracy_m == 20
