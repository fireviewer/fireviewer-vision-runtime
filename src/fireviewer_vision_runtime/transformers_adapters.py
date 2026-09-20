from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fireviewer_vision_runtime.adapters import ItemPatch, ModelAdapter, ModelOutputError
from fireviewer_vision_runtime.consensus import (
    ConsensusJudgeVerdict,
    JudgeCandidate,
    PipelineRole,
)
from fireviewer_contracts.contracts import (
    BatchItem,
    ExplicitLiteral,
    FactualObservation,
    PixelRegion,
    SourceAnnotationV2,
    Transcript,
    TranscriptSegment,
    VisualEvidenceSelection,
    WorkerInput,
    WorkerInputV2,
    WorkerOutput,
)
from fireviewer_contracts.client.media_fetcher import MediaFetcher
from fireviewer_contracts.model_registry import ModelSpec, resolve_cached_snapshot
from fireviewer_vision_runtime.model_workers.detection import (
    IMAGE_SIZE,
    center_letterbox,
    is_fireviewer_detector,
    unletterbox,
)

if TYPE_CHECKING:
    from fireviewer_vision_runtime.event_perception import EventPerceptionPoint, EventPointSemantic


def _torch_runtime() -> tuple[Any, Any]:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production worker")
    return torch, transformers


def _qwen_memory_limits() -> dict[int | str, str]:
    gpu_gib = int(os.getenv("FW_QWEN_GPU_MEMORY_GIB", "44"))
    cpu_gib = int(os.getenv("FW_QWEN_CPU_MEMORY_GIB", "48"))
    if gpu_gib < 1 or cpu_gib < 1:
        raise RuntimeError("Qwen GPU and CPU memory budgets must be positive")
    return {0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"}


def _bounded_image(image: Any, *, max_pixels: int) -> Any:
    from PIL import Image

    if max_pixels < 1:
        raise RuntimeError("Qwen image pixel budget must be positive")
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    image.close()
    return resized


class _BaseAdapter:
    def __init__(self, spec: ModelSpec, *, cache_root: Path, fetcher: MediaFetcher) -> None:
        self.spec = spec
        self.cache_root = cache_root
        self.fetcher = fetcher
        self.model: Any = None
        self.processor: Any = None

    @property
    def model_path(self) -> Path:
        return resolve_cached_snapshot(self.spec, self.cache_root)

    def unload(self) -> None:
        self.model = None
        self.processor = None


class WhisperAdapter(_BaseAdapter):
    def __init__(self, spec: ModelSpec, *, cache_root: Path, fetcher: MediaFetcher) -> None:
        super().__init__(spec, cache_root=cache_root, fetcher=fetcher)
        self.pipeline: Any = None

    def load(self) -> None:
        torch, transformers = _torch_runtime()
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.pipeline = transformers.pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=self.model.dtype,
            device=0,
        )

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        if self.pipeline is None:
            raise RuntimeError("Whisper adapter is not loaded")
        patches: dict[str, ItemPatch] = {}
        for item in items:
            if item.audio_url is None:
                continue
            with self.fetcher.download(str(item.audio_url)) as audio_path:
                raw = self.pipeline(
                    str(audio_path),
                    return_timestamps=True,
                    generate_kwargs={"task": "transcribe"},
                )
            segments = []
            for index, chunk in enumerate(raw.get("chunks", []), start=1):
                timestamps = chunk.get("timestamp") or (0.0, 0.0)
                start = float(timestamps[0] or 0.0)
                end = float(timestamps[1] or start + 0.001)
                segments.append(
                    TranscriptSegment(
                        segment_id=f"{item.input_id}:audio:{index:04d}",
                        start_s=start,
                        end_s=max(end, start + 0.001),
                        text=str(chunk.get("text", "")).strip(),
                    )
                )
            patches[item.input_id] = ItemPatch(
                transcript=Transcript(language=raw.get("language"), segments=tuple(segments))
            )
        return patches

    def unload(self) -> None:
        self.pipeline = None
        super().unload()


class RTDETRAdapter(_BaseAdapter):
    ALLOWED_LABELS = frozenset(
        {
            "smoke_visible",
            "flame_visible",
            "firefighting_aircraft_visible",
            "fire_response_vehicle_visible",
            # The opt-in Apache-2.0 COCO baseline is used only to rank views and
            # expose generic objects during capability tests. Generic targets
            # are never relabelled as firefighting resources.
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
        }
    )

    def load(self) -> None:
        torch, transformers = _torch_runtime()
        self.processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        # FireViewer checkpoints were validated with FP32 weights and BF16
        # autocast, including the public immutable repositories. Permanently
        # converting their weights to FP16 reproduces the invalid historical
        # benchmark contract.
        is_fireviewer_checkpoint = self.spec.source == "local" or self.spec.model_id.startswith(
            "fireviewer/"
        )
        dtype = torch.float32 if is_fireviewer_checkpoint else torch.float16
        self.model = transformers.AutoModelForObjectDetection.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.model.eval()

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        from PIL import Image

        torch, _ = _torch_runtime()
        patches: dict[str, ItemPatch] = {}
        for item in items:
            sources = [(frame.frame_id, str(frame.working_file_url)) for frame in item.frames]
            if not sources and item.working_file_url is not None:
                sources = [(item.input_id, str(item.working_file_url))]
            regions: list[PixelRegion] = []
            scores_by_evidence: dict[str, float] = {}
            for evidence_id, url in sources:
                with self.fetcher.download(url) as image_path, Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    if is_fireviewer_detector(self.model):
                        canvas, geometry = center_letterbox(rgb)
                        inputs = self.processor(
                            images=canvas,
                            do_resize=False,
                            do_pad=False,
                            return_tensors="pt",
                        )
                        inputs = {key: value.to("cuda") for key, value in inputs.items()}
                        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
                        with torch.inference_mode(), autocast:
                            outputs = self.model(**inputs)
                        canvas_predictions = self.processor.post_process_object_detection(
                            outputs,
                            threshold=0.25,
                            target_sizes=torch.tensor(
                                [[IMAGE_SIZE, IMAGE_SIZE]],
                                device="cuda",
                            ),
                        )[0]
                        predictions = unletterbox(canvas_predictions, geometry)
                    else:
                        inputs = self.processor(images=rgb, return_tensors="pt").to("cuda")
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                            outputs = self.model(**inputs)
                        predictions = self.processor.post_process_object_detection(
                            outputs, threshold=0.25, target_sizes=[(height, width)]
                        )[0]
                for index, (score, label_id, box) in enumerate(
                    zip(
                        predictions["scores"],
                        predictions["labels"],
                        predictions["boxes"],
                        strict=True,
                    ),
                    start=1,
                ):
                    label = str(self.model.config.id2label[int(label_id)])
                    if label not in self.ALLOWED_LABELS:
                        continue
                    x1, y1, x2, y2 = (float(value) for value in box.tolist())
                    regions.append(
                        PixelRegion(
                            region_id=f"{evidence_id}:det:{index:04d}",
                            evidence_id=evidence_id,
                            label=label,
                            bbox_normalized=(x1 / width, y1 / height, x2 / width, y2 / height),
                            task="fire_detection",
                            model_score=float(score),
                        )
                    )
                    scores_by_evidence[evidence_id] = max(
                        scores_by_evidence.get(evidence_id, 0.0), float(score)
                    )
            selected_ids = self._select_sources(
                [evidence_id for evidence_id, _ in sources], scores_by_evidence, limit=8
            )
            selections = tuple(
                VisualEvidenceSelection(
                    evidence_id=evidence_id,
                    selected_for_grounding=evidence_id in selected_ids,
                    selection_reason=(
                        "single_image"
                        if len(sources) == 1
                        else "target_detection"
                        if evidence_id in selected_ids and evidence_id in scores_by_evidence
                        else "temporal_coverage"
                        if evidence_id in selected_ids
                        else "capacity_limit"
                    ),
                    max_detection_score=scores_by_evidence.get(evidence_id),
                )
                for evidence_id, _ in sources
            )
            patches[item.input_id] = ItemPatch(
                pixel_regions=tuple(regions), visual_evidence_selection=selections
            )
        return patches

    @staticmethod
    def _select_sources(
        evidence_ids: list[str], scores_by_evidence: Mapping[str, float], *, limit: int
    ) -> frozenset[str]:
        if len(evidence_ids) <= limit:
            return frozenset(evidence_ids)
        # RT-DETR prioritizes views; two contextual views remain reserved because a frame
        # without a target can still contain text, landmarks, or localization evidence.
        target_budget = max(limit - 2, 0)
        positions = {evidence_id: index for index, evidence_id in enumerate(evidence_ids)}
        ranked_targets = sorted(
            (evidence_id for evidence_id in evidence_ids if evidence_id in scores_by_evidence),
            key=lambda evidence_id: (-scores_by_evidence[evidence_id], positions[evidence_id]),
        )
        selected = set(ranked_targets[:target_budget])
        remaining = [evidence_id for evidence_id in evidence_ids if evidence_id not in selected]
        slots = limit - len(selected)
        if slots >= len(remaining):
            selected.update(remaining)
        elif slots == 1:
            selected.add(remaining[len(remaining) // 2])
        elif slots > 1:
            indexes = {
                round(position * (len(remaining) - 1) / (slots - 1)) for position in range(slots)
            }
            selected.update(remaining[index] for index in indexes)
        return frozenset(selected)


class FlorenceAdapter(_BaseAdapter):
    def load(self) -> None:
        torch, transformers = _torch_runtime()
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = transformers.Florence2ForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        from PIL import Image

        torch, _ = _torch_runtime()
        patches: dict[str, ItemPatch] = {}
        prompt = (
            "<CAPTION_TO_PHRASE_GROUNDING>smoke. flame. aircraft. "
            "emergency vehicle. road. building."
        )
        for item in items:
            existing = list(accumulated[item.input_id].pixel_regions or ())
            selected = {
                entry.evidence_id
                for entry in (accumulated[item.input_id].visual_evidence_selection or ())
                if entry.selected_for_grounding
            }
            sources = [
                (frame.frame_id, str(frame.working_file_url))
                for frame in item.frames
                if frame.frame_id in selected
            ]
            if not sources and item.working_file_url is not None:
                sources = [(item.input_id, str(item.working_file_url))]
            for evidence_id, url in sources:
                with self.fetcher.download(url) as image_path, Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    inputs = self.processor(text=prompt, images=rgb, return_tensors="pt")
                    inputs = {key: value.to(device="cuda", dtype=torch.float16) if value.is_floating_point() else value.to("cuda") for key, value in inputs.items()}
                    with torch.inference_mode():
                        generated = self.model.generate(
                            **inputs, max_new_tokens=256, do_sample=False
                        )
                    text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
                    parsed = self.processor.post_process_generation(
                        text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=rgb.size
                    ).get("<CAPTION_TO_PHRASE_GROUNDING>", {})
                    width, height = rgb.size
                for index, (label, box) in enumerate(
                    zip(parsed.get("labels", []), parsed.get("bboxes", []), strict=True), start=1
                ):
                    x1, y1, x2, y2 = (float(value) for value in box)
                    existing.append(
                        PixelRegion(
                            region_id=f"{evidence_id}:ground:{index:04d}",
                            evidence_id=evidence_id,
                            label=str(label)[:128],
                            bbox_normalized=(x1 / width, y1 / height, x2 / width, y2 / height),
                            task="phrase_grounding",
                        )
                    )
            patches[item.input_id] = ItemPatch(pixel_regions=tuple(existing))
        return patches


class MolmoPointAdapter(_BaseAdapter):
    """Dedicated pixel-point worker; geographic projection remains downstream."""

    QUERIES: tuple[tuple[EventPointSemantic, str], ...] = (
        (
            "active_fire_point",
            "Point to every directly visible active flame. Do not point to smoke alone.",
        ),
        (
            "visible_fire_front_point",
            "Point to representative locations along each directly visible active flame front.",
        ),
        (
            "smoke_origin_point",
            "Point to the visible terrain origin of each smoke column. Abstain if hidden.",
        ),
    )

    def load(self) -> None:
        torch, transformers = _torch_runtime()
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            padding_side="left",
        )
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        self.model.eval()

    @staticmethod
    def _annotation_id(
        input_id: str,
        evidence_id: str,
        semantic_anchor: str,
        index: int,
    ) -> str:
        digest = sha256(
            f"{input_id}\x1f{evidence_id}\x1f{semantic_anchor}\x1f{index}".encode()
        ).hexdigest()[:24]
        return f"ANN-{digest}"

    @staticmethod
    def _sources(
        item: Any,
        selected: frozenset[str],
    ) -> list[tuple[str, str, Literal["image", "frame"]]]:
        sources: list[tuple[str, str, Literal["image", "frame"]]] = [
            (str(frame.frame_id), str(frame.working_file_url), "frame")
            for frame in item.frames
            if not selected or frame.frame_id in selected
        ]
        if not sources and item.media_type.value == "image" and item.working_file_url is not None:
            sources = [(str(item.input_id), str(item.working_file_url), "image")]
        return sources[:8]

    def _point(
        self,
        *,
        image: Any,
        prompt: str,
    ) -> list[tuple[float, float]]:
        if self.model is None or self.processor is None:
            raise RuntimeError("MolmoPoint adapter is not loaded")
        torch, _ = _torch_runtime()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = encoded.pop("metadata")
        encoded = {
            key: value.to("cuda") if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        logits_processor = self.model.build_logit_processor_from_inputs(encoded)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.model.generate(
                **encoded,
                logits_processor=logits_processor,
                max_new_tokens=200,
            )
        generated_tokens = generated[:, encoded["input_ids"].size(1) :]
        generated_text = self.processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if isinstance(generated_text, list):
            generated_text = generated_text[0]
        points = self.model.extract_image_points(
            generated_text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )
        width, height = image.size
        normalized: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 4:
                continue
            x = min(max(float(point[-2]) / width, 0.0), 1.0)
            y = min(max(float(point[-1]) / height, 0.0), 1.0)
            normalized.append((x, y))
        return normalized[:16]

    def infer_event_image(
        self,
        *,
        evidence_asset_id: str,
        working_file_url: str,
    ) -> tuple[EventPerceptionPoint, ...]:
        """Run the existing MolmoPoint queries for one private event image."""

        from PIL import Image

        from fireviewer_vision_runtime.event_perception import EventPerceptionPoint

        del evidence_asset_id  # The caller owns the evidence-to-anchor association.
        points: list[EventPerceptionPoint] = []
        with self.fetcher.download(working_file_url) as image_path, Image.open(image_path) as image:
            rgb = image.convert("RGB")
            try:
                for semantic_anchor, query in self.QUERIES:
                    for x, y in self._point(image=rgb, prompt=query):
                        points.append(
                            EventPerceptionPoint(
                                semantic_anchor=semantic_anchor,
                                source_point_normalized=(x, y),
                            )
                        )
            finally:
                rgb.close()
        return tuple(points)

    def infer(
        self,
        batch: WorkerInputV2,
        legacy: WorkerOutput,
    ) -> dict[str, tuple[SourceAnnotationV2, ...]]:
        from PIL import Image

        legacy_by_id = {item.input_id: item for item in legacy.items}
        annotations_by_input: dict[str, tuple[SourceAnnotationV2, ...]] = {}
        for item in batch.items:
            if item.media_type.value not in {"image", "video"}:
                continue
            legacy_item = legacy_by_id.get(item.input_id)
            selected = frozenset(
                selection.evidence_id
                for selection in (
                    legacy_item.visual_evidence_selection if legacy_item is not None else ()
                )
                if selection.selected_for_grounding
            )
            annotations: list[SourceAnnotationV2] = []
            for evidence_id, url, evidence_kind in self._sources(item, selected):
                with self.fetcher.download(url) as image_path, Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    try:
                        for semantic_anchor, query in self.QUERIES:
                            for index, (x, y) in enumerate(
                                self._point(image=rgb, prompt=query),
                                start=1,
                            ):
                                annotations.append(
                                    SourceAnnotationV2(
                                        annotation_id=self._annotation_id(
                                            item.input_id,
                                            evidence_id,
                                            semantic_anchor,
                                            index,
                                        ),
                                        evidence_id=evidence_id,
                                        evidence_kind=evidence_kind,
                                        semantic_anchor=semantic_anchor,
                                        source_point_normalized=(x, y),
                                    )
                                )
                    finally:
                        rgb.close()
            annotations_by_input[item.input_id] = tuple(annotations[:64])
        return annotations_by_input


class QwenAdapter(_BaseAdapter):
    SYSTEM_PROMPT = """Extract only directly visible, explicitly written, or explicitly
spoken facts.
Return one JSON object with exactly these arrays: observations, explicit_places, explicit_times.
Every entry must contain evidence_kind and evidence_id. Never infer a geographic position,
forecast, propagation, threatened area, probability, or missing fact. Unknown means omitted.
Observation fields: type, evidence_kind, evidence_id, optional region_id, description, certainty.
For a situation report, use a precise type when the source explicitly states one: fire_progression,
burned_area, evacuation_count, evacuation_instruction, shelter_opening, personnel_engaged,
ground_resources_engaged, aircraft_engaged, casualty_or_damage, access_restriction,
service_disruption, air_quality_alert, public_relief_and_donations, or public_instruction.
Keep every number, unit, status and time in description. Do not merge contradictory values.
source_context describes provenance, not truth. A source_confidence=lead statement may be extracted
for the private draft only as a reported claim; it must not be rewritten as confirmed. A media
license controls republication, not whether the media may be privately analysed.
declared_observation contains the contributor's unverified statement and declared time/location.
It may support a metadata-attributed reported claim, but never a camera pose or inferred fire point.
Place/time fields: literal, evidence_kind, evidence_id. evidence_kind MUST be one of image, frame, transcript_segment, article_text, metadata. certainty MUST be directly_visible, explicitly_written, or explicitly_spoken. Use input_id for article_text evidence_id. JSON only."""

    def load(self) -> None:
        torch, transformers = _torch_runtime()
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Flash Attention 2 is required; an SDPA fallback is forbidden"
            ) from exc

        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError(
                "Flash Attention 2 requires an Ampere, Ada, Hopper, or newer NVIDIA GPU"
            )
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = transformers.AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            max_memory=_qwen_memory_limits(),
        )

    @staticmethod
    def _parse(
        text: str,
    ) -> tuple[
        tuple[FactualObservation, ...],
        tuple[ExplicitLiteral, ...],
        tuple[ExplicitLiteral, ...],
    ]:
        clean = text.strip()
        if clean.startswith("```json\n") and clean.endswith("\n```"):
            clean = clean[8:-4]
        payload = json.loads(clean)
        if not isinstance(payload, dict) or set(payload) != {
            "observations",
            "explicit_places",
            "explicit_times",
        }:
            raise ValueError("Qwen response must use the exact closed object shape")
        return (
            tuple(FactualObservation.model_validate(value) for value in payload["observations"]),
            tuple(ExplicitLiteral.model_validate(value) for value in payload["explicit_places"]),
            tuple(ExplicitLiteral.model_validate(value) for value in payload["explicit_times"]),
        )

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        from PIL import Image

        torch, _ = _torch_runtime()
        patches: dict[str, ItemPatch] = {}
        for item in items:
            opened: list[Any] = []
            contexts: list[Any] = []
            try:
                selected = {
                    entry.evidence_id
                    for entry in (accumulated[item.input_id].visual_evidence_selection or ())
                    if entry.selected_for_grounding
                }
                sources = [
                    (frame.frame_id, str(frame.working_file_url))
                    for frame in item.frames
                    if frame.frame_id in selected
                ]
                if not sources and item.working_file_url is not None:
                    sources = [(item.input_id, str(item.working_file_url))]
                max_pixels = int(os.getenv("FW_QWEN_VL_MAX_PIXELS", "1048576"))
                total_pixels = int(os.getenv("FW_QWEN_VL_TOTAL_PIXELS", "4194304"))
                per_image_pixels = min(
                    max_pixels,
                    max(1, total_pixels // max(1, len(sources))),
                )
                content: list[dict[str, Any]] = []
                for evidence_id, url in sources:
                    context = self.fetcher.download(url)
                    path = context.__enter__()
                    contexts.append(context)
                    with Image.open(path) as source_image:
                        image = source_image.convert("RGB")
                    image = _bounded_image(image, max_pixels=per_image_pixels)
                    opened.append(image)
                    content.extend(
                        [
                            {"type": "text", "text": f"Evidence image id: {evidence_id}"},
                            {"type": "image", "image": image},
                        ]
                    )
                transcript = accumulated[item.input_id].transcript
                context_payload = {
                    "article_text": item.article_text,
                    "input_id": item.input_id,
                    "text_evidence_id": item.input_id,
                    "source_context": (
                        item.source_context.model_dump(mode="json", exclude_none=True)
                        if item.source_context
                        else None
                    ),
                    "transcript": transcript.model_dump(mode="json") if transcript else None,
                    "pixel_regions": [
                        region.model_dump(mode="json")
                        for region in (accumulated[item.input_id].pixel_regions or ())
                    ],
                    "correction": correction,
                }
                if correction:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "The previous response was rejected by deterministic validation. "
                                "Return a corrected object using only the exact allowed fields and "
                                "existing evidence identifiers. Remove speculative statements."
                            ),
                        }
                    )
                content.append(
                    {"type": "text", "text": json.dumps(context_payload, ensure_ascii=False)}
                )
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs, max_new_tokens=1536, do_sample=False, temperature=None
                    )
                trimmed = generated[:, inputs.input_ids.shape[1] :]
                response = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                try:
                    observations, places, times = self._parse(response)
                except (TypeError, ValueError) as exc:
                    raise ModelOutputError("Qwen returned invalid JSON or fields") from exc
                patches[item.input_id] = ItemPatch(
                    factual_observations=observations,
                    explicit_places=places,
                    explicit_times=times,
                )
            finally:
                for image in opened:
                    image.close()
                for context in reversed(contexts):
                    context.__exit__(None, None, None)
        return patches


class QwenConsensusJudgeAdapter(QwenAdapter):
    SYSTEM_PROMPT = """You are the final FireWarning candidate adjudicator. A deterministic
comparator found a contradiction between model outputs for one stage. Inspect the supplied source
evidence and every candidate payload. Select a candidate only when its output is directly supported
by the evidence and respects the stage boundary. Never average incompatible outputs, invent a third
answer, infer a hidden location, or prefer a candidate because of its model name or rank. If source
coverage is incomplete, the evidence is ambiguous, or neither candidate is defensible, abstain.
Return JSON only with exactly: selected_candidate_id (string or null), confidence (0..1), and
reason_codes (one or more short snake_case identifiers)."""

    @staticmethod
    def _parse_verdict(
        text: str,
        *,
        candidate_ids: frozenset[str],
    ) -> ConsensusJudgeVerdict:
        payload = json.loads(text.strip())
        if not isinstance(payload, dict) or set(payload) != {
            "selected_candidate_id",
            "confidence",
            "reason_codes",
        }:
            raise ValueError("judge response must use the exact closed object shape")
        selected = payload["selected_candidate_id"]
        if selected is not None and selected not in candidate_ids:
            raise ValueError("judge selected an unknown candidate")
        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("judge confidence must be numeric")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("judge confidence must be between zero and one")
        reason_codes = payload["reason_codes"]
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or len(reason_codes) > 8
            or any(
                not isinstance(reason, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,63}", reason)
                for reason in reason_codes
            )
            or len(reason_codes) != len(set(reason_codes))
        ):
            raise ValueError("judge reason codes are invalid")
        normalized_payload: dict[str, object] = {
            "selected_candidate_id": selected,
            "confidence": confidence,
            "reason_codes": reason_codes,
        }
        return ConsensusJudgeVerdict(
            selected_candidate_id=selected,
            confidence=confidence,
            reason_codes=tuple(reason_codes),
            output_payload=normalized_payload,
        )

    def adjudicate(
        self,
        *,
        batch: WorkerInput,
        stage_role: PipelineRole,
        candidates: Sequence[JudgeCandidate],
        comparison_payload: Mapping[str, object],
        correction: bool = False,
    ) -> ConsensusJudgeVerdict:
        from PIL import Image

        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen consensus judge is not loaded")
        torch, _ = _torch_runtime()
        sources: list[tuple[str, str]] = []
        for item in batch.items:
            if item.frames:
                sources.extend(
                    (frame.frame_id, str(frame.working_file_url)) for frame in item.frames
                )
            elif item.working_file_url is not None and item.media_type in {
                "image",
                "satellite_image",
            }:
                sources.append((item.input_id, str(item.working_file_url)))
        maximum_images = int(os.getenv("FW_QWEN_JUDGE_MAX_IMAGES", "8"))
        if maximum_images < 1:
            raise RuntimeError("Qwen judge image limit must be positive")
        selected_sources = sources
        if len(sources) > maximum_images:
            if maximum_images == 1:
                selected_sources = [sources[len(sources) // 2]]
            else:
                indexes = {
                    round(position * (len(sources) - 1) / (maximum_images - 1))
                    for position in range(maximum_images)
                }
                selected_sources = [sources[index] for index in sorted(indexes)]

        opened: list[Any] = []
        contexts: list[Any] = []
        try:
            total_pixels = int(os.getenv("FW_QWEN_JUDGE_TOTAL_PIXELS", "4194304"))
            per_image_pixels = max(1, total_pixels // max(1, len(selected_sources)))
            content: list[dict[str, Any]] = []
            for evidence_id, url in selected_sources:
                context = self.fetcher.download(url)
                path = context.__enter__()
                contexts.append(context)
                with Image.open(path) as source_image:
                    image = source_image.convert("RGB")
                image = _bounded_image(image, max_pixels=per_image_pixels)
                opened.append(image)
                content.extend(
                    [
                        {"type": "text", "text": f"Evidence image id: {evidence_id}"},
                        {"type": "image", "image": image},
                    ]
                )
            judge_context = {
                "stage_role": stage_role,
                "source_image_count": len(sources),
                "provided_image_count": len(selected_sources),
                "source_coverage_complete": len(sources) == len(selected_sources),
                "items": [
                    {
                        "input_id": item.input_id,
                        "media_type": item.media_type,
                        "article_text": item.article_text,
                        "source_context": (
                            item.source_context.model_dump(mode="json", exclude_none=True)
                            if item.source_context is not None
                            else None
                        ),
                    }
                    for item in batch.items
                ],
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "output_payload": candidate.output_payload,
                    }
                    for candidate in candidates
                ],
                "deterministic_comparison": comparison_payload,
                "correction": correction,
            }
            if correction:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "The previous verdict violated the closed JSON contract. Return only "
                            "the exact object. Abstain instead of guessing."
                        ),
                    }
                )
            content.append({"type": "text", "text": json.dumps(judge_context, ensure_ascii=False)})
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=192,
                    do_sample=False,
                    temperature=None,
                )
            trimmed = generated[:, inputs.input_ids.shape[1] :]
            response = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            try:
                return self._parse_verdict(
                    response,
                    candidate_ids=frozenset(candidate.candidate_id for candidate in candidates),
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ModelOutputError("Qwen judge returned an invalid verdict") from exc
        finally:
            for image in opened:
                image.close()
            for context in reversed(contexts):
                context.__exit__(None, None, None)


class QwenTextConsensusJudgeAdapter(_BaseAdapter):
    """A40-compatible final judge for structured outputs and verbatim source text.

    Qwen3-14B is text-only. It may adjudicate sourced facts, but a visual or audio
    contradiction is forced to abstain because this adapter cannot inspect raw pixels
    or waveforms. The invocation and its abstention remain in the audit trace.
    """

    SYSTEM_PROMPT = """You are the final FireWarning candidate adjudicator. A deterministic
comparator found a contradiction. Use only the supplied candidate payloads, comparison metrics and
verbatim source text. Select a candidate only when one payload is directly and unambiguously
supported. Never average candidates, invent a third answer, infer a hidden location, or trust a
model name, rank or confidence score. If the raw evidence needed to decide is unavailable, abstain.
Return JSON only with exactly: selected_candidate_id (string or null), confidence (0..1), and
reason_codes (one or more short snake_case identifiers)."""

    def load(self) -> None:
        torch, transformers = _torch_runtime()
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Flash Attention 2 is required; an SDPA fallback is forbidden"
            ) from exc
        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError(
                "Flash Attention 2 requires an Ampere, Ada, Hopper, or newer NVIDIA GPU"
            )
        self.processor = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            max_memory=_qwen_memory_limits(),
        )

    def adjudicate(
        self,
        *,
        batch: WorkerInput,
        stage_role: PipelineRole,
        candidates: Sequence[JudgeCandidate],
        comparison_payload: Mapping[str, object],
        correction: bool = False,
    ) -> ConsensusJudgeVerdict:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen text consensus judge is not loaded")
        torch, _ = _torch_runtime()
        source_text = [
            {"input_id": item.input_id, "article_text": item.article_text}
            for item in batch.items
            if item.article_text
        ]
        context = {
            "stage_role": stage_role,
            "raw_visual_or_audio_evidence_available": False,
            "verbatim_source_text": source_text,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "output_payload": candidate.output_payload,
                }
                for candidate in candidates
            ],
            "deterministic_comparison": comparison_payload,
            "correction": correction,
        }
        correction_text = (
            " The previous response violated the closed JSON contract; abstain instead of guessing."
            if correction
            else ""
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False) + correction_text,
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                temperature=None,
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        candidate_ids = frozenset(candidate.candidate_id for candidate in candidates)
        try:
            verdict = QwenConsensusJudgeAdapter._parse_verdict(
                response,
                candidate_ids=candidate_ids,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelOutputError("Qwen text judge returned an invalid verdict") from exc

        directly_judgeable = stage_role == "multimodal_extraction" and bool(source_text)
        if verdict.selected_candidate_id is None or directly_judgeable:
            return verdict
        reason_codes = tuple(
            dict.fromkeys(
                (
                    "raw_evidence_unavailable_to_text_judge",
                    *verdict.reason_codes,
                )
            )
        )
        output_payload: dict[str, object] = {
            "selected_candidate_id": None,
            "confidence": 0.0,
            "reason_codes": list(reason_codes),
        }
        return ConsensusJudgeVerdict(
            selected_candidate_id=None,
            confidence=0.0,
            reason_codes=reason_codes,
            output_payload=output_payload,
        )


class TransformersAdapterFactory:
    def __init__(
        self,
        *,
        cache_root: Path,
        allowed_hosts: frozenset[str],
        max_download_bytes: int,
        max_cache_bytes: int | None = None,
        fetcher: MediaFetcher | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.fetcher = fetcher or MediaFetcher(
            allowed_hosts=allowed_hosts,
            max_bytes=max_download_bytes,
            max_cache_bytes=max_cache_bytes,
        )

    @contextmanager
    def job_scope(self) -> Iterator[None]:
        with self.fetcher.batch_scope():
            yield

    def create(self, spec: ModelSpec) -> ModelAdapter:
        adapter_type: type[_BaseAdapter]
        if spec.role == "asr":
            adapter_type = WhisperAdapter
        elif spec.role == "fire_detection":
            adapter_type = RTDETRAdapter
        elif spec.role == "visual_grounding":
            adapter_type = FlorenceAdapter
        else:
            adapter_type = QwenAdapter
        return adapter_type(spec, cache_root=self.cache_root, fetcher=self.fetcher)

    def create_fire_pointing(self, spec: ModelSpec) -> MolmoPointAdapter:
        if spec.role != "fire_pointing":
            raise ValueError("fire pointing factory requires a fire_pointing model")
        return MolmoPointAdapter(
            spec,
            cache_root=self.cache_root,
            fetcher=self.fetcher,
        )

    def create_burned_area(self, spec: ModelSpec) -> object:
        if spec.role != "burned_area":
            raise ValueError("burned area factory requires a burned_area model")
        from fireviewer_vision_runtime.prithvi_burned_area import PrithviBurnedAreaAdapter

        return PrithviBurnedAreaAdapter(
            spec,
            cache_root=self.cache_root,
            fetcher=self.fetcher,
        )

    def create_consensus_judge(
        self,
        spec: ModelSpec,
    ) -> QwenConsensusJudgeAdapter | QwenTextConsensusJudgeAdapter:
        if spec.role != "consensus_judge":
            raise ValueError("consensus judge factory requires a consensus_judge model")
        if spec.model_id == "prism-ml/Ternary-Bonsai-2-27B-gguf":
            from fireviewer_vision_runtime.bonsai_judge import BonsaiConsensusJudgeAdapter
            return BonsaiConsensusJudgeAdapter(spec, cache_root=self.cache_root, fetcher=self.fetcher)
        adapter_type: type[QwenConsensusJudgeAdapter | QwenTextConsensusJudgeAdapter]
        adapter_type = (
            QwenTextConsensusJudgeAdapter
            if spec.model_id == "Qwen/Qwen3-14B"
            else QwenConsensusJudgeAdapter
        )
        return adapter_type(
            spec,
            cache_root=self.cache_root,
            fetcher=self.fetcher,
        )
