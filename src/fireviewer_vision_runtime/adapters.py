from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fireviewer_contracts.contracts import (
    BatchItem,
    ExplicitLiteral,
    FactualObservation,
    PixelRegion,
    Transcript,
    VisualEvidenceSelection,
)
from fireviewer_contracts.model_registry import ModelRole, ModelSpec


class ModelOutputError(ValueError):
    """The model answered, but its response could not satisfy the closed contract."""


@dataclass(frozen=True, slots=True)
class ItemPatch:
    transcript: Transcript | None = None
    pixel_regions: tuple[PixelRegion, ...] | None = None
    visual_evidence_selection: tuple[VisualEvidenceSelection, ...] | None = None
    factual_observations: tuple[FactualObservation, ...] | None = None
    explicit_places: tuple[ExplicitLiteral, ...] | None = None
    explicit_times: tuple[ExplicitLiteral, ...] | None = None


class ModelAdapter(Protocol):
    spec: ModelSpec

    def load(self) -> None: ...

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]: ...

    def unload(self) -> None: ...


class AdapterFactory(Protocol):
    def create(self, spec: ModelSpec) -> ModelAdapter: ...


@runtime_checkable
class ScopedAdapterFactory(Protocol):
    """Optional per-job resource scope implemented by production factories."""

    def job_scope(self) -> AbstractContextManager[None]: ...


@contextmanager
def adapter_factory_job_scope(factory: AdapterFactory) -> Iterator[None]:
    """Open production-only resources without burdening lightweight test factories."""

    if isinstance(factory, ScopedAdapterFactory):
        with factory.job_scope():
            yield
        return
    yield


@dataclass(slots=True)
class UnavailableAdapter:
    """Explicit failure used when a production model integration is not installed."""

    spec: ModelSpec

    def load(self) -> None:
        raise RuntimeError(f"runtime adapter unavailable for {self.spec.role}")

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        raise RuntimeError(f"runtime adapter unavailable for {self.spec.role}")

    def unload(self) -> None:
        return None


@dataclass(slots=True)
class UnavailableAdapterFactory:
    created: list[ModelRole] = field(default_factory=list)

    def create(self, spec: ModelSpec) -> ModelAdapter:
        self.created.append(spec.role)
        return UnavailableAdapter(spec)
