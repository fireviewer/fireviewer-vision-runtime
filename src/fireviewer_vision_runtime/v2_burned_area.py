"""Native V2 burned-area stage for georeferenced multispectral products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal, Protocol

from fireviewer_contracts.contracts import (
    SourceAnnotationV2,
    SpatialProposalV2,
    WorkerInputV2,
    WorkerModelRunV2,
    WorkerStageAttemptV2,
    WorkerStageGateV2,
    WorkerStageTraceV2,
)
from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_contracts.stage_contracts import StageRole, load_stage_contract_registry

CANONICAL_BURNED_AREA_BANDS = (
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
)


def _now() -> datetime:
    return datetime.now(UTC)


class BurnedAreaAdapter(Protocol):
    spec: ModelSpec

    def load(self) -> None: ...

    def infer(
        self,
        batch: WorkerInputV2,
    ) -> tuple[
        dict[str, tuple[SourceAnnotationV2, ...]],
        dict[str, tuple[SpatialProposalV2, ...]],
    ]: ...

    def unload(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BurnedAreaExecution:
    annotations_by_input: dict[str, tuple[SourceAnnotationV2, ...]]
    proposals_by_input: dict[str, tuple[SpatialProposalV2, ...]]
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


def _has_compatible_product(batch: WorkerInputV2) -> bool:
    return any(
        item.media_type.value == "satellite_image"
        and item.working_file_url is not None
        and item.satellite is not None
        and tuple(item.satellite.bands) == CANONICAL_BURNED_AREA_BANDS
        for item in batch.items
    )


def run_burned_area_stage(
    batch: WorkerInputV2,
    *,
    adapter: BurnedAreaAdapter | None,
    sequence: int,
) -> BurnedAreaExecution:
    """Run official Prithvi BurnScars only on its explicit six-band contract."""

    contract = load_stage_contract_registry()[StageRole.BURNED_AREA]
    if not _has_compatible_product(batch):
        return BurnedAreaExecution(
            annotations_by_input={},
            proposals_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="burned_area",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="skipped",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="not_applicable",
                    reason="no_compatible_multispectral_product",
                    missing=("satellite_multispectral",),
                    downstream_possible=True,
                ),
            ),
            model_run=None,
        )
    if batch.reference_bundle is None:
        return BurnedAreaExecution(
            annotations_by_input={},
            proposals_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="burned_area",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="skipped",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="human_review",
                    reason="satellite_reference_manifest_missing",
                    available=("satellite_multispectral",),
                    missing=("reference_bundle",),
                    downstream_possible=True,
                ),
            ),
            model_run=None,
        )
    if adapter is None:
        return BurnedAreaExecution(
            annotations_by_input={},
            proposals_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="burned_area",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="skipped",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="human_review",
                    reason="burned_area_model_unavailable",
                    available=("satellite_multispectral",),
                    missing=("burned_area_model",),
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
        annotations, proposals = adapter.infer(batch)
        inference_ms = round((perf_counter() - infer_started) * 1_000)
    except Exception:
        finished_at = _now()
        elapsed_ms = round((perf_counter() - started) * 1_000)
        return BurnedAreaExecution(
            annotations_by_input={},
            proposals_by_input={},
            stage_trace=WorkerStageTraceV2(
                stage_role="burned_area",
                contract_id=contract.contract_id,
                sequence=sequence,
                status="failed",
                retryable=False,
                preflight=_gate(
                    phase="preflight",
                    decision="pass",
                    reason="requirements_satisfied",
                    available=("satellite_multispectral", "burned_area_model"),
                    downstream_possible=True,
                ),
                postflight=_gate(
                    phase="postflight",
                    decision="human_review",
                    reason="burned_area_runtime_failed",
                    available=("satellite_multispectral",),
                    missing=("burned_area_geometry",),
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
                        error_code="burned_area_runtime_failed",
                    ),
                ),
            ),
            model_run=WorkerModelRunV2(
                model_role="burned_area",
                model_id=adapter.spec.model_id,
                revision=adapter.spec.revision,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                load_ms=load_ms,
                inference_ms=inference_ms,
                error_code="burned_area_runtime_failed",
            ),
        )
    finally:
        adapter.unload()

    finished_at = _now()
    elapsed_ms = round((perf_counter() - started) * 1_000)
    has_geometry = any(proposals.values())
    return BurnedAreaExecution(
        annotations_by_input=annotations,
        proposals_by_input=proposals,
        stage_trace=WorkerStageTraceV2(
            stage_role="burned_area",
            contract_id=contract.contract_id,
            sequence=sequence,
            status="succeeded",
            retryable=False,
            preflight=_gate(
                phase="preflight",
                decision="pass",
                reason="requirements_satisfied",
                available=("satellite_multispectral", "burned_area_model"),
                downstream_possible=True,
            ),
            postflight=_gate(
                phase="postflight",
                decision="pass",
                reason=(
                    "output_contract_satisfied"
                    if has_geometry
                    else "burned_area_model_abstained"
                ),
                available=(
                    ("burned_area_geometry",) if has_geometry else ("explicit_abstention",)
                ),
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
            model_role="burned_area",
            model_id=adapter.spec.model_id,
            revision=adapter.spec.revision,
            status="succeeded",
            started_at=started_at,
            finished_at=finished_at,
            load_ms=load_ms,
            inference_ms=inference_ms,
        ),
    )
