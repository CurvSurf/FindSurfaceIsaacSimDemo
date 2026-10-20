"""Temporary renderer/depth controls used to isolate sensor artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum
from .axial_error import validate_distance_error


class AntiAliasingMode(IntEnum):
    """Values accepted by ``/rtx/post/aa/op`` in Isaac Sim 6.0.1."""

    NONE = 0
    TAA = 1
    FXAA = 2
    DLSS = 3
    RTXAA = 4

    @property
    def label(self) -> str:
        return {
            AntiAliasingMode.NONE: "None",
            AntiAliasingMode.TAA: "TAA",
            AntiAliasingMode.FXAA: "FXAA",
            AntiAliasingMode.DLSS: "DLSS",
            AntiAliasingMode.RTXAA: "RTXAA",
        }[self]


ANTI_ALIASING_LABELS = tuple(mode.label for mode in AntiAliasingMode)


@dataclass(frozen=True)
class DepthSensorDiagnostics:
    """Isaac Sim depth controls plus opt-in application axial error."""

    disparity_noise_enabled: bool = True
    outlier_removal_enabled: bool = True
    axial_error_enabled: bool = False
    axial_error_sigma_m: float = 0.0
    disparity_noise_mean_px: float = 0.25
    disparity_noise_sigma_px: float = 0.25
    axial_error_model: str = "constant"
    axial_error_slope_m_per_m: float = 0.0
    axial_error_intercept_m: float = 0.0015

    def __post_init__(self) -> None:
        object.__setattr__(self, "disparity_noise_enabled", bool(self.disparity_noise_enabled))
        for name in ("disparity_noise_mean_px", "disparity_noise_sigma_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -10.0 <= value <= 10.0:
                raise ValueError(f"{name} must be finite and between -10 and 10 pixels")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "outlier_removal_enabled", bool(self.outlier_removal_enabled))
        object.__setattr__(self, "axial_error_enabled", bool(self.axial_error_enabled))
        sigma = float(self.axial_error_sigma_m)
        if not 0.0 <= sigma <= 0.1:
            raise ValueError("axial_error_sigma_m must be between 0 and 0.1 metres")
        object.__setattr__(self, "axial_error_sigma_m", sigma)
        model, slope, intercept = validate_distance_error(
            self.axial_error_model, self.axial_error_slope_m_per_m, self.axial_error_intercept_m)
        if slope > 0.1 or not -0.1 <= intercept <= 0.1:
            raise ValueError("distance axial error requires a in [0, 0.1] and b in [-0.1, 0.1]")
        object.__setattr__(self, "axial_error_model", model)
        object.__setattr__(self, "axial_error_slope_m_per_m", slope)
        object.__setattr__(self, "axial_error_intercept_m", intercept)

    @property
    def effective_axial_error_sigma_m(self) -> float:
        """Return the application error applied to frames, or zero when disabled."""

        return self.axial_error_sigma_m if self.axial_error_enabled and self.axial_error_model == "constant" else 0.0

    @property
    def effective_distance_error_parameters(self) -> tuple[float, float]:
        if self.axial_error_enabled and self.axial_error_model == "distance":
            return self.axial_error_slope_m_per_m, self.axial_error_intercept_m
        return 0.0, 0.0

    @property
    def effective_disparity_noise_parameters_px(self) -> tuple[float, float]:
        """Return the official noiseMean/noiseSigma values sent to Isaac Sim."""

        if not self.disparity_noise_enabled:
            return 0.0, 0.0
        return self.disparity_noise_mean_px, self.disparity_noise_sigma_px


def diagnostics_for_reset(current, target, scope):
    """Respect the Settings/Scanner ownership boundary without changing units."""
    if scope == "all":
        return target
    fields = (("axial_error_enabled", "axial_error_sigma_m", "axial_error_model",
               "axial_error_slope_m_per_m", "axial_error_intercept_m") if scope == "scanner" else
              ("disparity_noise_enabled", "outlier_removal_enabled",
               "disparity_noise_mean_px", "disparity_noise_sigma_px") if scope is None else ())
    return replace(current, **{name: getattr(target, name) for name in fields})


__all__ = ["ANTI_ALIASING_LABELS", "AntiAliasingMode", "DepthSensorDiagnostics", "diagnostics_for_reset"]
