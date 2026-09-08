"""Native V2 fire/smoke pointing stage.

The legacy media stages still select and ground useful views.  This module adds
the dedicated pointing contract after those stages without changing the
public-contribution or admin-review workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal, Protocol

from fireviewer_contracts.contracts import (
    SourceAnnotationV2,
    WorkerInputV2,
    WorkerModelRunV2,
    WorkerOutput,
    WorkerStageAttemptV2,
    WorkerStageGateV2,
    WorkerStageTraceV2,
)
from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_contracts.stage_contracts import StageRole, load_stage_contract_registry


def _now() -> datetime:
    return datetime.now(UTC)


class FirePointingAdapter(Protocol):
    spec: ModelSpec

    def load(self) -> None: ...

    def infer(
        self,
        batch: WorkerInputV2,
        legacy: WorkerOutput,
    ) -> dict[str, tuple[SourceAnnotationV2, ...]]: ...

    def unload(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FirePointingExecution:
    annotations_by_input: dict[str, tuple[SourceAnnotationV2, ...]]
    stage_trace: WorkerStageTraceV2
    model_run: WorkerModelRunV2 | None


def _gate(
    *,
    phase: Literal["preflight", "postflight"],
    decision: Literal["pass", "abstain", "human_review", "not_applicable"],
    reason: str,
    available: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
    downstream_possible: bool,
) -> WorkerStageGateV2:
    return WorkerStageGateV2(
        phase=phase,
        decision=decision,
        reason_codes=(reason,),
        available_capabilities=available,
        missing_capabilities=missing,
        downstream_possible=downstream_possible,
    )


def run_fire_pointing_stage(
    batch: WorkerInputV2,
    legacy: WorkerOutput,
    *,
    adapter: FirePointingAdapter | None,
    sequence: int,
) -> FirePointingExecution:
    """Run the active pointing adapter once, or record an explicit degraded fallback."""

    contract = load_stage_contract_registry()[StageRole.FIRE_POINTING]
    applicable = any(
        item.working_file_url is not None or bool(item.frames)
        for item in batch.items
        if item.media_type.value in {"image", "video"}
    )
    if not applicable:
        return FirePointingExecution(
            annotations_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="fire_pointing",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="skipped",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="not_applicable",
                    reason="no_applicable_input",
                    downstream_possible=True,
                ),
            ),
            model_run=None,
        )
    if adapter is None:
        return FirePointingExecution(
            annotations_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="fire_pointing",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="skipped",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="human_review",
                    reason="fire_pointing_model_unavailable",
                    available=("selected_visual",),
                    missing=("fire_pointing_model",),
                    downstream_possible=True,
                ),
            ),
            model_run=None,
        )

    started_at = _now()
    started = perf_counter()
    load_ms = 0
    inference_ms = 0
    try:
        load_started = perf_counter()
        adapter.load()
        load_ms = round((perf_counter() - load_started) * 1_000)
        infer_started = perf_counter()
        annotations = adapter.infer(batch, legacy)
        inference_ms = round((perf_counter() - infer_started) * 1_000)
    except Exception:
        finished_at = _now()
        elapsed_ms = round((perf_counter() - started) * 1_000)
        return FirePointingExecution(
            annotations_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="fire_pointing",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="failed",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="pass",
                    reason="requirements_satisfied",
                    available=("selected_visual", "fire_pointing_model"),
                    downstream_possible=True,
                ),
                postflight=_gate(
                    phase="postflight",
                    decision="human_review",
                    reason="fire_pointing_runtime_failed",
                    available=("selected_visual",),
                    missing=("fire_point_pixel",),
                    downstream_possible=True,
                ),
                attempts=(
                    WorkerStageAttemptV2(
                        attempt=1,
                        kind="initial",
                        status="failed",
                        started_at=started_at,
                        finished_at=finished_at,
                        inference_ms=elapsed_ms,
                        error_code="fire_pointing_runtime_failed",
                    ),
                ),
            ),
            model_run=WorkerModelRunV2(
                model_role="fire_pointing",
                model_id=adapter.spec.model_id,
                revision=adapter.spec.revision,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                load_ms=load_ms,
                inference_ms=inference_ms,
                error_code="fire_pointing_runtime_failed",
            ),
        )
    finally:
        adapter.unload()

    finished_at = _now()
    elapsed_ms = round((perf_counter() - started) * 1_000)
    has_points = any(annotations.values())
    capability = "fire_point_pixel" if has_points else "explicit_abstention"
    return FirePointingExecution(
        annotations_by_input=annotations,
        stage_trace=WorkerStageTraceV2(
            stage_role="fire_pointing",
            contract_id=contract.contract_id,
            sequence=sequence,
            status="succeeded",
            retryable=False,
            preflight=_gate(
                phase="preflight",
                decision="pass",
                reason="requirements_satisfied",
                available=("selected_visual", "fire_pointing_model"),
                downstream_possible=True,
            ),
            postflight=_gate(
                phase="postflight",
                decision="pass",
                reason="output_contract_satisfied",
                available=(capability,),
                downstream_possible=True,
            ),
            attempts=(
                WorkerStageAttemptV2(
                    attempt=1,
                    kind="initial",
                    status="succeeded",
                    started_at=started_at,
                    finished_at=finished_at,
                    inference_ms=elapsed_ms,
                ),
            ),
        ),
        model_run=WorkerModelRunV2(
            model_role="fire_pointing",
            model_id=adapter.spec.model_id,
            revision=adapter.spec.revision,
            status="succeeded",
            started_at=started_at,
            finished_at=finished_at,
            load_ms=load_ms,
            inference_ms=inference_ms,
        ),
    )
