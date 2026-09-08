from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_vision_runtime.model_workers.burnscar import (
    DeprecatedBurnScarModelError,
    require_promotable_burnscar_model,
)
from fireviewer_vision_runtime.model_workers.detection import (
    LetterboxGeometry,
    center_letterbox,
)
from fireviewer_vision_runtime.model_workers.pointing import parse_pointing_response
from fireviewer_vision_runtime.transformers_adapters import RTDETRAdapter


def test_center_letterbox_matches_the_validated_4_by_3_geometry() -> None:
    image = Image.new("RGB", (1_024, 768), color=(10, 20, 30))

    canvas, geometry = center_letterbox(image)

    assert canvas.size == (768, 768)
    assert geometry == LetterboxGeometry(
        scale=0.75,
        pad_left=0,
        pad_top=96,
        original_width=1_024,
        original_height=768,
    )


def test_local_detector_keeps_fp32_weights(monkeypatch, tmp_path: Path) -> None:
    received_dtypes: list[str] = []

    class FakeModel:
        config = SimpleNamespace(id2label={0: "smoke_visible", 1: "flame_visible"})

        def to(self, _device: str) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    def from_pretrained(*_args: Any, **kwargs: Any) -> FakeModel:
        received_dtypes.append(kwargs["dtype"])
        return FakeModel()

    torch = SimpleNamespace(float16="float16", float32="float32")
    transformers = SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
        AutoModelForObjectDetection=SimpleNamespace(from_pretrained=from_pretrained),
    )
    monkeypatch.setattr(
        "fireviewer_vision_runtime.transformers_adapters._torch_runtime",
        lambda: (torch, transformers),
    )
    monkeypatch.setattr(
        "fireviewer_vision_runtime.transformers_adapters.resolve_cached_snapshot",
        lambda _spec, _cache_root: tmp_path,
    )
    adapter = RTDETRAdapter(
        ModelSpec(
            role="fire_detection",
            model_id=str(tmp_path),
            revision="sha256:" + ("0" * 64),
            source="local",
        ),
        cache_root=tmp_path,
        fetcher=object(),  # type: ignore[arg-type]
    )

    adapter.load()

    assert received_dtypes == ["float32"]


def test_pointing_parser_accepts_only_normalized_closed_predictions() -> None:
    predictions = parse_pointing_response(
        json.dumps(
            {
                "points": [
                    {"kind": "flame_point", "x": 0.25, "y": 0.75},
                    {"kind": "smoke_origin", "x": 0.5, "y": 0.6},
                ]
            }
        )
    )

    assert [(point.kind, point.x, point.y) for point in predictions] == [
        ("flame_point", 0.25, 0.75),
        ("smoke_origin", 0.5, 0.6),
    ]
    with pytest.raises(ValueError, match="exactly"):
        parse_pointing_response('{"points":[],"latitude":44.75,"longitude":5.37}')
    with pytest.raises(ValueError, match="normalized"):
        parse_pointing_response('{"points":[{"kind":"flame_point","x":1.5,"y":0.5}]}')


def test_deprecated_prithvi_checkpoint_is_blocked_before_loading() -> None:
    with pytest.raises(DeprecatedBurnScarModelError, match="deprecated"):
        require_promotable_burnscar_model("fireviewer/prithvi-burnscars-firewarning-v1-deprecated")

    require_promotable_burnscar_model("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars")
