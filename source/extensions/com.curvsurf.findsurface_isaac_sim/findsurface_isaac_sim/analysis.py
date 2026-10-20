"""One-pass histogram and heatmap aggregation for analytic surface distances."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .metrics import SurfaceMatchResult


MIN_HISTOGRAM_BINS = 23
MAX_HISTOGRAM_BINS = 47
DEFAULT_HISTOGRAM_BINS = 35
SIGMA_EPSILON_M = 1e-12
MIN_DISPLAY_SCALE_M = 1e-6


@dataclass(frozen=True)
class HistogramBin:
    start_m: float
    end_m: float
    center_m: float
    count: int
    height_fraction: float
    source_percent: float
    sigma_offset: float


@dataclass(frozen=True)
class HistogramResult:
    available: bool
    unavailable_reason: str | None = None
    source_count: int = 0
    used_count: int = 0
    excluded_count: int = 0
    bin_count: int = 0
    negative_count: int = 0
    zero_count: int = 0
    positive_count: int = 0
    max_bin_count: int = 0
    minimum_m: float | None = None
    maximum_m: float | None = None
    mean_m: float | None = None
    sigma_m: float | None = None
    source_sigma_m: float | None = None
    range_min_m: float | None = None
    range_max_m: float | None = None
    mean_fraction: float | None = None
    zero_fraction: float | None = None
    bins: tuple[HistogramBin, ...] = ()

    def __post_init__(self) -> None:
        if self.available:
            if self.unavailable_reason is not None:
                raise ValueError("available histogram cannot carry an unavailable reason")
            if self.source_count <= 0 or self.used_count + self.excluded_count != self.source_count:
                raise ValueError("histogram counts do not reconcile")
            if self.bin_count != len(self.bins):
                raise ValueError("histogram bin count does not match bins")
            numeric = (
                self.minimum_m,
                self.maximum_m,
                self.mean_m,
                self.sigma_m,
                self.source_sigma_m,
                self.range_min_m,
                self.range_max_m,
                self.mean_fraction,
                self.zero_fraction,
            )
            if any(value is None or not math.isfinite(value) for value in numeric):
                raise ValueError("available histogram requires finite summary values")
        elif not self.unavailable_reason:
            raise ValueError("unavailable histogram requires a reason")


@dataclass(frozen=True)
class HeatmapResult:
    available: bool
    unavailable_reason: str | None = None
    source_count: int = 0
    scale_m: float | None = None
    minimum_m: float | None = None
    maximum_m: float | None = None
    clamped_count: int = 0
    coordinates: tuple[float, ...] = ()
    colormap: str = "sequential-zero-error"

    def __post_init__(self) -> None:
        if self.available:
            if self.unavailable_reason is not None:
                raise ValueError("available heatmap cannot carry an unavailable reason")
            if self.source_count <= 0 or self.source_count != len(self.coordinates):
                raise ValueError("heatmap requires one coordinate per source sample")
            if self.scale_m is None or not math.isfinite(self.scale_m) or self.scale_m <= 0.0:
                raise ValueError("available heatmap requires a positive finite scale")
            if self.minimum_m is None or self.maximum_m is None:
                raise ValueError("available heatmap requires source bounds")
            if any(not 0.0 <= value <= 1.0 for value in self.coordinates):
                raise ValueError("heatmap coordinates must be clamped to [0, 1]")
        elif not self.unavailable_reason:
            raise ValueError("unavailable heatmap requires a reason")


@dataclass(frozen=True)
class MatchAnalysis:
    match: SurfaceMatchResult
    histogram: HistogramResult
    heatmap: HeatmapResult


def build_histogram(
    signed_distances_m: Iterable[float],
    *,
    bin_count: int = DEFAULT_HISTOGRAM_BINS,
    measurement_accuracy_m: float = 0.0,
    unfiltered_rms_m: float = 0.0,
) -> HistogramResult:
    """Build the upstream-compatible one-pass zero-centered 3-sigma histogram."""

    values = _finite_values(signed_distances_m)
    if values is None:
        return _unavailable_histogram("non-finite-distance")
    if not values:
        return _unavailable_histogram("empty-distance-set")
    return _build_histogram_from_values(
        values,
        bin_count=bin_count,
        measurement_accuracy_m=measurement_accuracy_m,
        unfiltered_rms_m=unfiltered_rms_m,
    )


def _build_histogram_from_values(
    values: tuple[float, ...],
    *,
    bin_count: int,
    measurement_accuracy_m: float,
    unfiltered_rms_m: float,
) -> HistogramResult:
    if not _finite_nonnegative(measurement_accuracy_m) or not _finite_nonnegative(unfiltered_rms_m):
        return _unavailable_histogram("invalid-fallback-scale")

    normalized_bins = _normalized_bin_count(bin_count)
    samples = np.asarray(values, dtype=np.float64)
    source_count = samples.size
    source_mean = float(np.mean(samples))
    source_variance = float(np.mean(np.square(samples - source_mean)))
    source_sigma = math.sqrt(max(0.0, source_variance))
    sigma = source_sigma
    if sigma <= SIGMA_EPSILON_M:
        sigma = max(measurement_accuracy_m, unfiltered_rms_m, MIN_DISPLAY_SCALE_M)
    half_range = 3.0 * sigma
    range_min = -half_range
    range_max = half_range

    used = samples[(samples >= range_min) & (samples <= range_max)]
    used_count = used.size
    excluded_count = source_count - used_count
    mean = float(np.mean(used)) if used_count else 0.0
    minimum = float(np.min(used)) if used_count else 0.0
    maximum = float(np.max(used)) if used_count else 0.0
    zero_epsilon = max(abs(maximum - minimum) * 1e-12, 1e-12) if used_count else 1e-12
    negative_count = int(np.count_nonzero(used < -zero_epsilon))
    positive_count = int(np.count_nonzero(used > zero_epsilon))
    zero_count = used_count - negative_count - positive_count

    width = (range_max - range_min) / normalized_bins
    indices = np.floor((used - range_min) / width).astype(np.int64)
    indices = np.clip(indices, 0, normalized_bins - 1)
    counts = np.bincount(indices, minlength=normalized_bins)
    max_bin_count = int(np.max(counts))
    bins = []
    for index, raw_count in enumerate(counts):
        count = int(raw_count)
        start = range_min + width * index
        end = start + width
        center = start + width * 0.5
        bins.append(
            HistogramBin(
                start,
                end,
                center,
                count,
                count / max_bin_count if max_bin_count else 0.0,
                count * 100.0 / source_count,
                center / sigma,
            )
        )
    mean_fraction = _clamp((mean - range_min) / (range_max - range_min), 0.0, 1.0)
    zero_fraction = _clamp((0.0 - range_min) / (range_max - range_min), 0.0, 1.0)
    return HistogramResult(
        available=True,
        source_count=source_count,
        used_count=used_count,
        excluded_count=excluded_count,
        bin_count=normalized_bins,
        negative_count=negative_count,
        zero_count=zero_count,
        positive_count=positive_count,
        max_bin_count=max_bin_count,
        minimum_m=minimum,
        maximum_m=maximum,
        mean_m=mean,
        sigma_m=sigma,
        source_sigma_m=source_sigma,
        range_min_m=range_min,
        range_max_m=range_max,
        mean_fraction=mean_fraction,
        zero_fraction=zero_fraction,
        bins=tuple(bins),
    )


def build_heatmap(
    distances_m: Iterable[float],
    *,
    scale_m: float | None = None,
    measurement_accuracy_m: float = 0.0,
    unfiltered_rms_m: float = 0.0,
) -> HeatmapResult:
    """Map non-negative match distances to renderer-ready sequential coordinates."""

    values = _finite_values(distances_m)
    if values is None:
        return _unavailable_heatmap("non-finite-distance")
    if not values:
        return _unavailable_heatmap("empty-distance-set")
    return _build_heatmap_from_values(
        values,
        scale_m=scale_m,
        measurement_accuracy_m=measurement_accuracy_m,
        unfiltered_rms_m=unfiltered_rms_m,
    )


def build_signed_heatmap(
    signed_distances_m: Iterable[float],
    *,
    scale_m: float | None = None,
    measurement_accuracy_m: float = 0.0,
    unfiltered_rms_m: float = 0.0,
    colormap: str = "diverging-signed-inlier",
) -> HeatmapResult:
    """Map demo-1 signed inlier residuals to blue/yellow/red coordinates.

    ``-scale`` maps to 0 (negative/inside blue), zero maps to 0.5
    (surface yellow), and ``+scale`` maps to 1 (positive/outside red).
    """

    values = _finite_values(signed_distances_m)
    if values is None:
        return _unavailable_heatmap("non-finite-distance")
    if not values:
        return _unavailable_heatmap("empty-distance-set")
    return _build_signed_heatmap_from_values(
        values,
        scale_m=scale_m,
        measurement_accuracy_m=measurement_accuracy_m,
        unfiltered_rms_m=unfiltered_rms_m,
        colormap=colormap,
    )


def _build_heatmap_from_values(
    values: tuple[float, ...],
    *,
    scale_m: float | None,
    measurement_accuracy_m: float,
    unfiltered_rms_m: float,
) -> HeatmapResult:
    samples = np.asarray(values, dtype=np.float64)
    if np.any(samples < 0.0):
        return _unavailable_heatmap("negative-match-distance")
    if not _finite_nonnegative(measurement_accuracy_m) or not _finite_nonnegative(unfiltered_rms_m):
        return _unavailable_heatmap("invalid-fallback-scale")
    if scale_m is None:
        scale = max(float(np.max(samples)), measurement_accuracy_m, unfiltered_rms_m, MIN_DISPLAY_SCALE_M)
    else:
        try:
            scale = float(scale_m)
        except (TypeError, ValueError):
            return _unavailable_heatmap("invalid-scale")
        if not math.isfinite(scale) or scale <= 0.0:
            return _unavailable_heatmap("invalid-scale")
    coordinates_array = np.clip(samples / scale, 0.0, 1.0)
    coordinates = tuple(float(value) for value in coordinates_array)
    return HeatmapResult(
        available=True,
        source_count=len(values),
        scale_m=scale,
        minimum_m=float(np.min(samples)),
        maximum_m=float(np.max(samples)),
        clamped_count=int(np.count_nonzero(samples > scale)),
        coordinates=coordinates,
    )


def _build_signed_heatmap_from_values(
    values: tuple[float, ...],
    *,
    scale_m: float | None,
    measurement_accuracy_m: float,
    unfiltered_rms_m: float,
    colormap: str = "diverging-signed-inlier",
) -> HeatmapResult:
    if not _finite_nonnegative(measurement_accuracy_m) or not _finite_nonnegative(unfiltered_rms_m):
        return _unavailable_heatmap("invalid-fallback-scale")
    samples = np.asarray(values, dtype=np.float64)
    if scale_m is None:
        mean = float(np.mean(samples))
        variance = float(np.mean(np.square(samples - mean)))
        scale = math.sqrt(max(0.0, variance))
        if scale <= SIGMA_EPSILON_M:
            scale = max(measurement_accuracy_m, unfiltered_rms_m, MIN_DISPLAY_SCALE_M)
    else:
        try:
            scale = float(scale_m)
        except (TypeError, ValueError):
            return _unavailable_heatmap("invalid-scale")
        if not math.isfinite(scale) or scale <= 0.0:
            return _unavailable_heatmap("invalid-scale")
    coordinates_array = np.clip(0.5 + samples / (2.0 * scale), 0.0, 1.0)
    coordinates = tuple(float(value) for value in coordinates_array)
    return HeatmapResult(
        available=True,
        source_count=len(values),
        scale_m=scale,
        minimum_m=float(np.min(samples)),
        maximum_m=float(np.max(samples)),
        clamped_count=int(np.count_nonzero(np.abs(samples) > scale)),
        coordinates=coordinates,
        colormap=str(colormap),
    )


def build_histogram_heatmap(
    distances_m: Iterable[float],
    *,
    bin_count: int = DEFAULT_HISTOGRAM_BINS,
    measurement_accuracy_m: float = 0.0,
    unfiltered_rms_m: float = 0.0,
    heatmap_scale_m: float | None = None,
) -> tuple[HistogramResult, HeatmapResult]:
    """Build both views from one immutable, finite distance sample packet.

    Histogram and heatmap are two presentations of the same surface-distance
    calculation.  Materializing and validating that iterable here prevents a
    worker from traversing a large point-derived sequence twice.
    """

    values = _finite_values(distances_m)
    if values is None:
        reason = "non-finite-distance"
        return _unavailable_histogram(reason), _unavailable_heatmap(reason)
    if not values:
        reason = "empty-distance-set"
        return _unavailable_histogram(reason), _unavailable_heatmap(reason)
    return (
        _build_histogram_from_values(
            values,
            bin_count=bin_count,
            measurement_accuracy_m=measurement_accuracy_m,
            unfiltered_rms_m=unfiltered_rms_m,
        ),
        _build_heatmap_from_values(
            values,
            scale_m=heatmap_scale_m,
            measurement_accuracy_m=measurement_accuracy_m,
            unfiltered_rms_m=unfiltered_rms_m,
        ),
    )


def build_signed_histogram_heatmap(
    signed_distances_m: Iterable[float],
    *,
    bin_count: int = DEFAULT_HISTOGRAM_BINS,
    measurement_accuracy_m: float = 0.0,
    unfiltered_rms_m: float | None = None,
    heatmap_scale_m: float | None = None,
    heatmap_colormap: str = "diverging-signed-inlier",
) -> tuple[HistogramResult, HeatmapResult]:
    """Build demo 1's histogram and point colors from one signed population."""

    values = _finite_values(signed_distances_m)
    if values is None:
        reason = "non-finite-distance"
        return _unavailable_histogram(reason), _unavailable_heatmap(reason)
    if not values:
        reason = "empty-distance-set"
        return _unavailable_histogram(reason), _unavailable_heatmap(reason)
    residual_rms = (
        float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))
        if unfiltered_rms_m is None
        else unfiltered_rms_m
    )
    histogram = _build_histogram_from_values(
        values,
        bin_count=bin_count,
        measurement_accuracy_m=measurement_accuracy_m,
        unfiltered_rms_m=residual_rms,
    )
    if not histogram.available:
        return histogram, _unavailable_heatmap(histogram.unavailable_reason or "histogram-unavailable")
    if (histogram.source_sigma_m or 0.0) <= SIGMA_EPSILON_M:
        # Demo 1's point-color path uses the raw population sigma and returns
        # the zero/surface color when it collapses.  The histogram still uses
        # its accuracy/RMS fallback range, so retain that positive scale only
        # for the legend and histogram bar colors.
        heatmap = HeatmapResult(
            available=True,
            source_count=len(values),
            scale_m=histogram.sigma_m,
            minimum_m=float(np.min(np.asarray(values, dtype=np.float64))),
            maximum_m=float(np.max(np.asarray(values, dtype=np.float64))),
            clamped_count=0,
            coordinates=(0.5,) * len(values),
            colormap="diverging-signed-inlier",
        )
    else:
        heatmap = _build_signed_heatmap_from_values(
            values,
            scale_m=(
                histogram.source_sigma_m
                if heatmap_scale_m is None
                else heatmap_scale_m
            ),
            measurement_accuracy_m=measurement_accuracy_m,
            unfiltered_rms_m=residual_rms,
            colormap=heatmap_colormap,
        )
    if heatmap_scale_m is not None and (histogram.source_sigma_m or 0.0) <= SIGMA_EPSILON_M:
        heatmap = _build_signed_heatmap_from_values(
            values,
            scale_m=heatmap_scale_m,
            measurement_accuracy_m=measurement_accuracy_m,
            unfiltered_rms_m=residual_rms,
            colormap=heatmap_colormap,
        )
    return histogram, heatmap


def aggregate_match(
    match: SurfaceMatchResult,
    *,
    bin_count: int = DEFAULT_HISTOGRAM_BINS,
    measurement_accuracy_m: float = 0.0,
    heatmap_scale_m: float | None = None,
    signed_inlier_distances_m: Iterable[float] | None = None,
) -> MatchAnalysis:
    """Aggregate presentation data without replacing or filtering the source match."""

    if not match.available:
        reason = f"match-unavailable:{match.unavailable_reason}"
        return MatchAnalysis(
            match,
            _unavailable_histogram(reason),
            _unavailable_heatmap(reason),
        )
    if signed_inlier_distances_m is None:
        histogram, heatmap = build_histogram_heatmap(
            match.distances_m,
            bin_count=bin_count,
            measurement_accuracy_m=measurement_accuracy_m,
            unfiltered_rms_m=match.rms_m or 0.0,
            heatmap_scale_m=heatmap_scale_m,
        )
    else:
        histogram, heatmap = build_signed_histogram_heatmap(
            signed_inlier_distances_m,
            bin_count=bin_count,
            measurement_accuracy_m=measurement_accuracy_m,
        )
    return MatchAnalysis(match, histogram, heatmap)


def _normalized_bin_count(value: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = DEFAULT_HISTOGRAM_BINS
    if result <= 0:
        result = DEFAULT_HISTOGRAM_BINS
    result = max(MIN_HISTOGRAM_BINS, min(MAX_HISTOGRAM_BINS, result))
    if result % 2 == 0:
        result += 1 if result < MAX_HISTOGRAM_BINS else -1
    return result


def _finite_values(values: Iterable[float]) -> tuple[float, ...] | None:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def _finite_nonnegative(value: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0.0


def _unavailable_histogram(reason: str) -> HistogramResult:
    return HistogramResult(False, reason)


def _unavailable_heatmap(reason: str) -> HeatmapResult:
    return HeatmapResult(False, reason)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


__all__ = [
    "DEFAULT_HISTOGRAM_BINS",
    "MAX_HISTOGRAM_BINS",
    "MIN_HISTOGRAM_BINS",
    "HistogramBin",
    "HistogramResult",
    "HeatmapResult",
    "MatchAnalysis",
    "aggregate_match",
    "build_heatmap",
    "build_histogram",
    "build_histogram_heatmap",
    "build_signed_heatmap",
    "build_signed_histogram_heatmap",
]
