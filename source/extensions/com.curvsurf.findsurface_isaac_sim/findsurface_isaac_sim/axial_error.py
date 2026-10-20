"""Application Gaussian ray-displacement models, separate from Isaac Sim noise."""

from __future__ import annotations

import math
import numpy as np


AXIAL_ERROR_MODELS = ("constant", "distance")


def validate_distance_error(model: str, slope: float, intercept: float) -> tuple[str, float, float]:
    if model not in AXIAL_ERROR_MODELS:
        raise ValueError("axial_error_model must be constant or distance")
    a, b = float(slope), float(intercept)
    if not all(math.isfinite(value) for value in (a, b)) or a < 0:
        raise ValueError("distance axial error coefficients must be finite; slope must be non-negative")
    return model, a, b


def distance_error_sigmas_m(ranges_m: np.ndarray, slope_m_per_m: float, intercept_m: float) -> np.ndarray:
    """Standard deviations at pre-noise sensor-to-point ranges (not optical Z)."""
    return np.maximum(0.0, slope_m_per_m * ranges_m + intercept_m)


def migrate_distance_error_sensor(sensor: dict, fallback_slope: float = 0.0) -> None:
    """Convert legacy 1 m references once, before merging with new defaults."""
    legacy = sensor.pop("axial_error_sigma_at_1m_m", None)
    if legacy is not None and "axial_error_intercept_m" not in sensor:
        sensor["axial_error_intercept_m"] = float(legacy) - float(
            sensor.get("axial_error_slope_m_per_m", fallback_slope))
