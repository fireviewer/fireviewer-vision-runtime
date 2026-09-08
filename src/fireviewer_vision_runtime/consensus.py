from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import fmean
from typing import Literal, Protocol, runtime_checkable

from fireviewer_contracts.contracts import ItemResult, PixelRegion, WorkerInput
from fireviewer_contracts.model_registry import (
    ConsensusFailureDecision,
    ConsensusStrategy,
    ModelGroupSpec,
    ModelSpec,
)

PipelineRole = Literal[
    "asr",
    "fire_detection",
    "visual_grounding",
    "multimodal_extraction",
]
ConsensusDecision = Literal[
    "pass",
    "repair",
    "adjudicated",
    "abstain",
    "human_review",
]


@dataclass(frozen=True, slots=True)
class SuccessfulCandidate:
    candidate_id: str
    results: Mapping[str, ItemResult]
    repaired: bool


@dataclass(frozen=True, slots=True)
class ConsensusEvaluation:
    decision: ConsensusDecision
    selected_candidate_id: str | None
    agreement_score: float | None
    reason_codes: tuple[str, ...]
    downstream_allowed: bool
    comparison_payload: dict[str, object]
    requires_adjudication: bool = False


@dataclass(frozen=True, slots=True)
class JudgeCandidate:
    candidate_id: str
    model_id: str
    revision: str
    output_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ConsensusJudgeVerdict:
    selected_candidate_id: str | None
    confidence: float
    reason_codes: tuple[str, ...]
    output_payload: dict[str, object]


class ConsensusJudge(Protocol):
    spec: ModelSpec

    def load(self) -> None: ...

    def adjudicate(
        self,
        *,
        batch: WorkerInput,
        stage_role: PipelineRole,
        candidates: Sequence[JudgeCandidate],
        comparison_payload: Mapping[str, object],
        correction: bool = False,
    ) -> ConsensusJudgeVerdict: ...

    def unload(self) -> None: ...


@runtime_checkable
class ConsensusJudgeFactory(Protocol):
    def create_consensus_judge(self, spec: ModelSpec) -> ConsensusJudge: ...


def _consensus_failure_decision(
    decision: ConsensusFailureDecision,
) -> ConsensusDecision:
    if decision == ConsensusFailureDecision.ABSTAIN:
        return "abstain"
    return "human_review"


def apply_adjudication(
    evaluation: ConsensusEvaluation,
    verdict: ConsensusJudgeVerdict,
    *,
    candidate_ids: Sequence[str],
    confidence_threshold: float,
    failure_decision: ConsensusFailureDecision,
) -> ConsensusEvaluation:
    if not evaluation.requires_adjudication:
        raise ValueError("adjudication can only resolve a candidate contradiction")
    selected_candidate_id = verdict.selected_candidate_id
    accepted = selected_candidate_id in candidate_ids and verdict.confidence >= confidence_threshold
    comparison_payload = {
        **evaluation.comparison_payload,
        "adjudication": verdict.output_payload,
        "adjudication_confidence_threshold": confidence_threshold,
    }
    if accepted:
        return ConsensusEvaluation(
            decision="adjudicated",
            selected_candidate_id=selected_candidate_id,
            agreement_score=evaluation.agreement_score,
            reason_codes=("candidate_disagreement_adjudicated", *verdict.reason_codes),
            downstream_allowed=True,
            comparison_payload=comparison_payload,
        )
    return ConsensusEvaluation(
        decision=failure_decision.value,
        selected_candidate_id=None,
        agreement_score=evaluation.agreement_score,
        reason_codes=("adjudicator_abstained", *verdict.reason_codes),
        downstream_allowed=False,
        comparison_payload=comparison_payload,
    )


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _bbox_iou(left: PixelRegion, right: PixelRegion) -> float:
    left_x1, left_y1, left_x2, left_y2 = left.bbox_normalized
    right_x1, right_y1, right_x2, right_y2 = right.bbox_normalized
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _region_similarity(
    left: Sequence[PixelRegion],
    right: Sequence[PixelRegion],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    unmatched = set(range(len(right)))
    matched_iou = 0.0
    for left_region in left:
        matches = [
            (index, _bbox_iou(left_region, right[index]))
            for index in unmatched
            if (
                left_region.evidence_id == right[index].evidence_id
                and left_region.label == right[index].label
                and left_region.task == right[index].task
            )
        ]
        if not matches:
            continue
        index, score = max(matches, key=lambda item: item[1])
        if score <= 0:
            continue
        unmatched.remove(index)
        matched_iou += score
    return min(1.0, (2 * matched_iou) / (len(left) + len(right)))


def _transcript_similarity(left: ItemResult, right: ItemResult) -> float:
    left_text = _normalized_text(" ".join(segment.text for segment in left.transcript.segments))
    right_text = _normalized_text(" ".join(segment.text for segment in right.transcript.segments))
    if not left_text and not right_text:
        return 1.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _extraction_similarity(left: ItemResult, right: ItemResult) -> float:
    def observations(result: ItemResult) -> set[str]:
        return {
            "|".join(
                (
                    observation.type,
                    observation.evidence_kind,
                    observation.evidence_id,
                    _normalized_text(observation.description),
                    observation.certainty,
                )
            )
            for observation in result.factual_observations
        }

    def literals(result: ItemResult, field: str) -> set[str]:
        return {
            "|".join(
                (
                    literal.evidence_kind,
                    literal.evidence_id,
                    _normalized_text(literal.literal),
                )
            )
            for literal in getattr(result, field)
        }

    return fmean(
        (
            _jaccard(observations(left), observations(right)),
            _jaccard(literals(left, "explicit_places"), literals(right, "explicit_places")),
            _jaccard(literals(left, "explicit_times"), literals(right, "explicit_times")),
        )
    )


def compare_candidate_results(
    role: PipelineRole,
    left: Mapping[str, ItemResult],
    right: Mapping[str, ItemResult],
) -> float:
    """Compare only the deliverables owned by the current stage."""

    if set(left) != set(right):
        return 0.0
    scores: list[float] = []
    for input_id in sorted(left):
        left_item = left[input_id]
        right_item = right[input_id]
        if role == "asr":
            scores.append(_transcript_similarity(left_item, right_item))
        elif role == "fire_detection":
            scores.append(
                _region_similarity(
                    tuple(
                        region
                        for region in left_item.pixel_regions
                        if region.task == "fire_detection"
                    ),
                    tuple(
                        region
                        for region in right_item.pixel_regions
                        if region.task == "fire_detection"
                    ),
                )
            )
        elif role == "visual_grounding":
            scores.append(
                _region_similarity(
                    tuple(
                        region
                        for region in left_item.pixel_regions
                        if region.task in {"phrase_grounding", "ocr"}
                    ),
                    tuple(
                        region
                        for region in right_item.pixel_regions
                        if region.task in {"phrase_grounding", "ocr"}
                    ),
                )
            )
        else:
            scores.append(_extraction_similarity(left_item, right_item))
    return fmean(scores) if scores else 1.0


def candidate_requires_challenge(
    role: PipelineRole,
    results: Mapping[str, ItemResult],
) -> bool:
    if role == "asr":
        return not any(result.transcript.segments for result in results.values())
    if role == "fire_detection":
        has_visual = any(result.visual_evidence_selection for result in results.values())
        has_detection = any(
            region.task == "fire_detection"
            for result in results.values()
            for region in result.pixel_regions
        )
        return has_visual and not has_detection
    if role == "visual_grounding":
        has_selected = any(
            selection.selected_for_grounding
            for result in results.values()
            for selection in result.visual_evidence_selection
        )
        has_grounding = any(
            region.task in {"phrase_grounding", "ocr"}
            for result in results.values()
            for region in result.pixel_regions
        )
        return has_selected and not has_grounding
    has_context = any(
        result.transcript.segments or result.pixel_regions for result in results.values()
    )
    has_facts = any(
        result.factual_observations or result.explicit_places or result.explicit_times
        for result in results.values()
    )
    return has_context and not has_facts


def evaluate_consensus(
    role: PipelineRole,
    group: ModelGroupSpec,
    successful: Sequence[SuccessfulCandidate],
) -> ConsensusEvaluation:
    candidate_ids = [candidate.candidate_id for candidate in successful]
    base_payload: dict[str, object] = {
        "schema_version": "candidate-consensus-v1",
        "candidate_ids": candidate_ids,
        "minimum_successful": group.minimum_successful,
        "minimum_agreeing": group.minimum_agreeing,
        "agreement_threshold": group.agreement_threshold,
        "pairwise_scores": {},
    }
    if len(successful) < group.minimum_successful:
        failure_decision = _consensus_failure_decision(group.disagreement_decision)
        return ConsensusEvaluation(
            decision=failure_decision,
            selected_candidate_id=None,
            agreement_score=None,
            reason_codes=("insufficient_successful_candidates",),
            downstream_allowed=False,
            comparison_payload=base_payload,
        )

    if group.strategy == ConsensusStrategy.CASCADE:
        rank_by_id = {candidate.candidate_id: candidate.rank for candidate in group.candidates}
        admissible = sorted(
            (
                candidate
                for candidate in successful
                if not candidate_requires_challenge(role, candidate.results)
            ),
            key=lambda candidate: rank_by_id[candidate.candidate_id],
        )
        if not admissible:
            cascade_failure_decision = _consensus_failure_decision(group.disagreement_decision)
            return ConsensusEvaluation(
                decision=cascade_failure_decision,
                selected_candidate_id=None,
                agreement_score=None,
                reason_codes=("candidate_quality_insufficient",),
                downstream_allowed=False,
                comparison_payload=base_payload,
            )
        selected = admissible[0]
        cascade_pairwise = {
            f"{selected.candidate_id}|{candidate.candidate_id}": round(
                compare_candidate_results(
                    role,
                    selected.results,
                    candidate.results,
                ),
                6,
            )
            for candidate in admissible[1:]
        }
        agreeing = [selected.candidate_id]
        agreeing.extend(
            candidate.candidate_id
            for candidate in admissible[1:]
            if cascade_pairwise[f"{selected.candidate_id}|{candidate.candidate_id}"]
            >= group.agreement_threshold
        )
        agreement_score = fmean(cascade_pairwise.values()) if cascade_pairwise else 1.0
        comparison_payload = {
            **base_payload,
            "pairwise_scores": cascade_pairwise,
            "admissible_candidate_ids": [candidate.candidate_id for candidate in admissible],
            "agreeing_candidate_ids": agreeing,
            "selected_candidate_id": selected.candidate_id,
            "agreement_score": round(agreement_score, 6),
        }
        if len(agreeing) < group.minimum_agreeing:
            quorum_failure_decision = _consensus_failure_decision(group.disagreement_decision)
            return ConsensusEvaluation(
                decision=quorum_failure_decision,
                selected_candidate_id=None,
                agreement_score=agreement_score,
                reason_codes=("candidate_disagreement",),
                downstream_allowed=False,
                comparison_payload=comparison_payload,
                requires_adjudication=True,
            )
        cascade_success_decision: ConsensusDecision = "repair" if selected.repaired else "pass"
        fallback_selected = rank_by_id[selected.candidate_id] > min(rank_by_id.values())
        return ConsensusEvaluation(
            decision=cascade_success_decision,
            selected_candidate_id=selected.candidate_id,
            agreement_score=agreement_score,
            reason_codes=(
                "cascade_fallback_selected" if fallback_selected else "cascade_primary_selected",
            ),
            downstream_allowed=True,
            comparison_payload=comparison_payload,
        )

    if group.strategy == ConsensusStrategy.SINGLE_WITH_RULES or len(successful) == 1:
        selected = successful[0]
        single_success_decision: ConsensusDecision = "repair" if selected.repaired else "pass"
        return ConsensusEvaluation(
            decision=single_success_decision,
            selected_candidate_id=selected.candidate_id,
            agreement_score=1.0,
            reason_codes=(
                "single_candidate_repaired" if selected.repaired else "single_candidate_valid",
            ),
            downstream_allowed=True,
            comparison_payload={
                **base_payload,
                "selected_candidate_id": selected.candidate_id,
                "agreement_score": 1.0,
            },
        )

    pairwise: dict[str, float] = {}
    scores_by_candidate: dict[str, list[float]] = {
        candidate.candidate_id: [] for candidate in successful
    }
    for index, left in enumerate(successful):
        for right in successful[index + 1 :]:
            score = compare_candidate_results(role, left.results, right.results)
            pairwise[f"{left.candidate_id}|{right.candidate_id}"] = round(score, 6)
            scores_by_candidate[left.candidate_id].append(score)
            scores_by_candidate[right.candidate_id].append(score)
    mean_scores = {
        candidate_id: fmean(scores) if scores else 1.0
        for candidate_id, scores in scores_by_candidate.items()
    }
    rank_by_id = {candidate.candidate_id: candidate.rank for candidate in group.candidates}
    selected = max(
        successful,
        key=lambda candidate: (
            mean_scores[candidate.candidate_id],
            -rank_by_id[candidate.candidate_id],
        ),
    )
    agreeing = [
        candidate.candidate_id
        for candidate in successful
        if candidate.candidate_id == selected.candidate_id
        or compare_candidate_results(role, selected.results, candidate.results)
        >= group.agreement_threshold
    ]
    agreement_score = mean_scores[selected.candidate_id]
    comparison_payload = {
        **base_payload,
        "pairwise_scores": pairwise,
        "mean_scores": {
            candidate_id: round(score, 6) for candidate_id, score in sorted(mean_scores.items())
        },
        "agreeing_candidate_ids": agreeing,
        "selected_candidate_id": selected.candidate_id,
        "agreement_score": round(agreement_score, 6),
    }
    if len(agreeing) < group.minimum_agreeing:
        aggregate_failure_decision = _consensus_failure_decision(group.disagreement_decision)
        return ConsensusEvaluation(
            decision=aggregate_failure_decision,
            selected_candidate_id=None,
            agreement_score=agreement_score,
            reason_codes=("candidate_disagreement",),
            downstream_allowed=False,
            comparison_payload=comparison_payload,
            requires_adjudication=True,
        )
    aggregate_success_decision: ConsensusDecision = "repair" if selected.repaired else "pass"
    return ConsensusEvaluation(
        decision=aggregate_success_decision,
        selected_candidate_id=selected.candidate_id,
        agreement_score=agreement_score,
        reason_codes=("candidate_quorum_reached",),
        downstream_allowed=True,
        comparison_payload=comparison_payload,
    )
