"""Fail-closed contract for burned-area models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEPRECATED_FIREVIEWER_MODEL_ID = (
    "fireviewer/prithvi-burnscars-firewarning-v1-deprecated"
)
OFFICIAL_PRITHVI_MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"


class DeprecatedBurnScarModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BurnScarInputContract:
    bands: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
    chip_size: int = 512
    output_class: str = "burned_area"


def require_promotable_burnscar_model(model_id: str | Path) -> None:
    normalized = str(model_id).replace("\\", "/").lower()
    if "prithvi-burnscars-firewarning-v1-deprecated" in normalized:
        raise DeprecatedBurnScarModelError(
            "the FireViewer Prithvi checkpoint is deprecated after an HLS regression; "
            "use the pinned official baseline until a replacement passes the independent test"
        )
