"""Compatibility parser for the former JSON pointing export.

The native production binding now lives in ``v2_pointing.py`` and
``transformers_adapters.MolmoPointAdapter`` because MolmoPoint emits point
tokens plus preprocessing metadata rather than JSON.  This parser remains for
older offline artifacts only and is not used by the V2 runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

PointKind = Literal["flame_point", "visible_front_point", "smoke_origin"]

SYSTEM_PROMPT = """Locate directly visible wildfire evidence in the image.
Return JSON only with exactly one key, points. Each point has kind, x and y.
kind is flame_point, visible_front_point, or smoke_origin. x and y are normalized
image coordinates in [0,1]. Use smoke_origin only at the visible terrain origin
of a smoke column. Do not infer geographic coordinates. If no defensible point is
visible, return {"points":[]}."""


@dataclass(frozen=True, slots=True)
class PointingPrediction:
    kind: PointKind
    x: float
    y: float


def parse_pointing_response(
    text: str,
    *,
    maximum_points: int = 16,
) -> tuple[PointingPrediction, ...]:
    if maximum_points < 1:
        raise ValueError("maximum_points must be positive")
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"points"}:
        raise ValueError("pointing response must contain exactly the points key")
    raw_points = payload["points"]
    if not isinstance(raw_points, list) or len(raw_points) > maximum_points:
        raise ValueError("pointing response contains an invalid point list")
    predictions: list[PointingPrediction] = []
    allowed = {"flame_point", "visible_front_point", "smoke_origin"}
    for raw in raw_points:
        if not isinstance(raw, dict) or set(raw) != {"kind", "x", "y"}:
            raise ValueError("each pointing prediction must contain kind, x and y")
        kind = raw["kind"]
        x = raw["x"]
        y = raw["y"]
        if kind not in allowed:
            raise ValueError("pointing prediction kind is unsupported")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int | float)
            or not isinstance(y, int | float)
            or not 0 <= float(x) <= 1
            or not 0 <= float(y) <= 1
        ):
            raise ValueError("pointing coordinates must be normalized numbers")
        predictions.append(PointingPrediction(kind=kind, x=float(x), y=float(y)))
    return tuple(predictions)
