from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from fireviewer_vision_runtime.consensus import JudgeCandidate
from fireviewer_contracts.contracts import WorkerInput
from fireviewer_contracts.model_registry import CONSENSUS_JUDGE
from fireviewer_vision_runtime.transformers_adapters import (
    QwenTextConsensusJudgeAdapter,
    TransformersAdapterFactory,
)


class _FakeInputs(dict[str, object]):
    input_ids = SimpleNamespace(shape=(1, 3))

    def to(self, _device: str) -> _FakeInputs:
        return self


class _FakeGenerated:
    def __getitem__(self, _key: object) -> object:
        return object()


class _FakeProcessor:
    def apply_chat_template(self, *_args: object, **_kwargs: object) -> _FakeInputs:
        return _FakeInputs()

    def batch_decode(self, *_args: object, **_kwargs: object) -> list[str]:
        return [
            json.dumps(
                {
                    "selected_candidate_id": "visual.candidate_a",
                    "confidence": 0.99,
                    "reason_codes": ["candidate_a_claimed_support"],
                }
            )
        ]


class _FakeModel:
    device = "cuda"

    def generate(self, **_kwargs: object) -> _FakeGenerated:
        return _FakeGenerated()


def test_factory_uses_text_only_a40_judge_for_qwen3_14b(tmp_path: Path) -> None:
    factory = TransformersAdapterFactory(
        cache_root=tmp_path,
        allowed_hosts=frozenset({"media.internal"}),
        max_download_bytes=1024,
    )

    judge = factory.create_consensus_judge(CONSENSUS_JUDGE)

    assert isinstance(judge, QwenTextConsensusJudgeAdapter)


def test_text_judge_cannot_promote_a_visual_disagreement(monkeypatch, tmp_path: Path) -> None:
    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr(
        "fireviewer_vision_runtime.transformers_adapters._torch_runtime",
        lambda: (fake_torch, object()),
    )
    judge = QwenTextConsensusJudgeAdapter(
        CONSENSUS_JUDGE,
        cache_root=tmp_path,
        fetcher=SimpleNamespace(),
    )
    judge.processor = _FakeProcessor()
    judge.model = _FakeModel()
    batch = WorkerInput.model_validate(
        {
            "batch_id": "BATCH-VISUAL-CONFLICT",
            "batch_type": "user_media",
            "priority": "user_deadline",
            "items": [
                {
                    "input_id": "INPUT-1",
                    "media_type": "image",
                    "working_file_url": "https://media.internal/image.jpg",
                }
            ],
        }
    )

    verdict = judge.adjudicate(
        batch=batch,
        stage_role="visual_grounding",
        candidates=(
            JudgeCandidate(
                candidate_id="visual.candidate_a",
                model_id="org/a",
                revision="a" * 40,
                output_payload={"regions": [{"x": 0.1, "y": 0.2}]},
            ),
            JudgeCandidate(
                candidate_id="visual.candidate_b",
                model_id="org/b",
                revision="b" * 40,
                output_payload={"regions": [{"x": 0.8, "y": 0.7}]},
            ),
        ),
        comparison_payload={"agreement_score": 0.0},
    )

    assert verdict.selected_candidate_id is None
    assert verdict.confidence == 0.0
    assert verdict.reason_codes[0] == "raw_evidence_unavailable_to_text_judge"
