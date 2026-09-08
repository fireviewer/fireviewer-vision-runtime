from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from fireviewer_contracts.contracts import BatchItem
from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_vision_runtime.transformers_adapters import (
    MolmoPointAdapter,
    WhisperAdapter,
    _bounded_image,
    _qwen_memory_limits,
)


def test_qwen_memory_limits_reserve_vram_for_activations(monkeypatch) -> None:
    monkeypatch.setenv("FW_QWEN_GPU_MEMORY_GIB", "17")
    monkeypatch.setenv("FW_QWEN_CPU_MEMORY_GIB", "48")

    assert _qwen_memory_limits() == {0: "17GiB", "cpu": "48GiB"}


def test_qwen_image_is_resized_to_the_pixel_budget() -> None:
    image = Image.new("RGB", (2_000, 1_000))

    resized = _bounded_image(image, max_pixels=500_000)

    assert resized.size[0] * resized.size[1] <= 500_000
    assert resized.size[0] == 2 * resized.size[1]
    resized.close()


class FakeWhisperFetcher:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path

    @contextmanager
    def download(self, _url: str):
        yield self.audio_path


def test_whisper_pipeline_is_built_once_per_loaded_adapter(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    pipeline_calls: list[dict[str, Any]] = []
    inference_calls: list[str] = []

    class FakeModel:
        dtype = "float16"

        def to(self, _device: str):
            return self

    processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())

    def build_pipeline(_task: str, **kwargs: Any):
        pipeline_calls.append(kwargs)

        def infer(path: str, **_options: Any):
            inference_calls.append(path)
            return {"language": "fr", "chunks": []}

        return infer

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: processor),
        AutoModelForSpeechSeq2Seq=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeModel()
        ),
        pipeline=build_pipeline,
    )
    torch = SimpleNamespace(float16="float16")
    monkeypatch.setattr(
        "fireviewer_vision_runtime.transformers_adapters._torch_runtime",
        lambda: (torch, transformers),
    )
    monkeypatch.setattr(
        "fireviewer_vision_runtime.transformers_adapters.resolve_cached_snapshot",
        lambda _spec, _cache_root: tmp_path,
    )
    adapter = WhisperAdapter(
        ModelSpec(
            role="asr",
            model_id="openai/whisper-large-v3",
            revision="0" * 40,
        ),
        cache_root=tmp_path,
        fetcher=FakeWhisperFetcher(audio_path),  # type: ignore[arg-type]
    )
    item = BatchItem.model_validate(
        {
            "input_id": "AUDIO-1",
            "media_type": "audio",
            "audio_url": "https://media.internal/audio.wav",
        }
    )

    adapter.load()
    adapter.infer((item,), {})
    adapter.infer((item,), {})

    assert len(pipeline_calls) == 1
    assert inference_calls == [str(audio_path), str(audio_path)]

    adapter.unload()

    assert adapter.pipeline is None
    assert adapter.model is None
    assert adapter.processor is None


def test_molmopoint_decodes_only_generated_point_tokens(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    class FakeTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str):
            calls.setdefault("devices", []).append(device)
            return self

        def size(self, dimension: int) -> int:
            assert dimension == 1
            return 3

        def __getitem__(self, key: Any):
            calls["generated_slice"] = key
            return FakeTensor("generated_tokens")

    class FakeProcessor:
        def apply_chat_template(self, messages: Any, **options: Any):
            calls["messages"] = messages
            calls["template_options"] = options
            return {
                "input_ids": FakeTensor("input_ids"),
                "pixel_values": FakeTensor("pixel_values"),
                "metadata": {
                    "token_pooling": "pooling",
                    "subpatch_mapping": "mapping",
                    "image_sizes": "sizes",
                },
            }

        def post_process_image_text_to_text(self, tokens: FakeTensor, **options: Any):
            calls["decoded_tokens"] = tokens.name
            calls["decode_options"] = options
            return ["point-tokens"]

    class FakeModel:
        def build_logit_processor_from_inputs(self, inputs: dict[str, Any]):
            calls["logits_inputs"] = tuple(sorted(inputs))
            return "logits"

        def generate(self, **options: Any):
            calls["generate_options"] = options
            return FakeTensor("full_generation")

        def extract_image_points(self, text: str, pooling: str, mapping: str, sizes: str):
            calls["extract"] = (text, pooling, mapping, sizes)
            return [[1, 0, 50.0, 25.0]]

    @contextmanager
    def autocast(_device: str, *, dtype: str):
        calls["autocast"] = dtype
        yield

    @contextmanager
    def inference_mode():
        yield

    torch = SimpleNamespace(
        bfloat16="bfloat16",
        inference_mode=inference_mode,
        autocast=autocast,
    )
    monkeypatch.setattr(
        "fireviewer_vision_runtime.transformers_adapters._torch_runtime",
        lambda: (torch, object()),
    )
    adapter = MolmoPointAdapter(
        ModelSpec(
            role="fire_pointing",
            model_id="tests/fire-pointing-fixture",
            revision="0000000000000000000000000000000000000000",
        ),
        cache_root=tmp_path,
        fetcher=object(),  # type: ignore[arg-type]
    )
    adapter.processor = FakeProcessor()
    adapter.model = FakeModel()
    image = Image.new("RGB", (100, 50))

    points = adapter._point(image=image, prompt="Point to visible flames")

    assert points == [(0.5, 0.5)]
    assert calls["decoded_tokens"] == "generated_tokens"
    assert calls["decode_options"] == {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    assert calls["generate_options"]["max_new_tokens"] == 200
    assert calls["generate_options"]["logits_processor"] == "logits"
    assert calls["extract"] == ("point-tokens", "pooling", "mapping", "sizes")
    image.close()


def test_molmopoint_requires_extracted_frames_for_video() -> None:
    video = SimpleNamespace(
        input_id="VIDEO-1",
        media_type=SimpleNamespace(value="video"),
        working_file_url="https://media.internal/video.mp4",
        frames=(),
    )
    image = SimpleNamespace(
        input_id="IMAGE-1",
        media_type=SimpleNamespace(value="image"),
        working_file_url="https://media.internal/image.jpg",
        frames=(),
    )

    assert MolmoPointAdapter._sources(video, frozenset()) == []
    assert MolmoPointAdapter._sources(image, frozenset()) == [
        ("IMAGE-1", "https://media.internal/image.jpg", "image")
    ]


def test_molmopoint_event_bridge_reuses_pixel_queries_without_geographic_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "event.jpg"
    Image.new("RGB", (20, 10), color="red").save(image_path)
    fetcher = FakeWhisperFetcher(image_path)
    adapter = MolmoPointAdapter(
        ModelSpec(
            role="fire_pointing",
            model_id="tests/fire-pointing-fixture",
            revision="0000000000000000000000000000000000000000",
        ),
        cache_root=tmp_path,
        fetcher=fetcher,  # type: ignore[arg-type]
    )
    prompts: list[str] = []

    def fake_point(*, image: Any, prompt: str) -> list[tuple[float, float]]:
        assert image.mode == "RGB"
        prompts.append(prompt)
        return [(0.25, 0.75)]

    monkeypatch.setattr(adapter, "_point", fake_point)

    points = adapter.infer_event_image(
        evidence_asset_id="ASSET-1",
        working_file_url="https://media.internal/event.jpg",
    )

    assert [point.semantic_anchor for point in points] == [
        "active_fire_point",
        "visible_fire_front_point",
        "smoke_origin_point",
    ]
    assert all(point.source_point_normalized == (0.25, 0.75) for point in points)
    assert len(prompts) == 3
    assert all(not hasattr(point, "geometry_geojson") for point in points)
