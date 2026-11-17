"""Shared data types and numeric helpers."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Compliance(str, Enum):
    """How trustworthy a number is."""

    STANDARD = "standard"            # implements a published standard (BS.1770-4, EBU Tech 3342)
    APPROXIMATION = "approximation"  # well-defined method with known error bounds
    DERIVED = "derived"              # simple arithmetic on other measurements, definition stated
    HEURISTIC = "heuristic"          # indicator, not a measurement of a physical quantity
    NOT_AVAILABLE = "not_available"  # could not be computed reliably


@dataclass
class Measurement:
    """A single reported number together with its unit, method and limitations."""

    value: float | None
    unit: str
    method: str
    compliance: Compliance
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


def amp_to_db(amplitude: float | np.ndarray) -> float | np.ndarray:
    """Linear amplitude -> dB (20·log10). Zero maps to -inf."""
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(amplitude)


def power_to_db(power: float | np.ndarray) -> float | np.ndarray:
    """Power / mean-square -> dB (10·log10). Zero maps to -inf."""
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(power)


def fmt(value: float | None, digits: int = 1, unit: str = "") -> str:
    """Human-readable number; non-finite or missing values become 'n/a'."""
    if value is None or not math.isfinite(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return f"{text} {unit}".strip()


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy / enums to JSON-safe values.
    Non-finite floats become None (JSON has no inf/nan)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
