"""Toolkit-neutral detected-result overlay built from the shared analysis packet."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .analysis import HeatmapResult, HistogramResult
from .analysis_pipeline import (
    AnalysisPresentation,
    ResidualBasis,
    SurfaceSideScale,
    SurfaceSideScaleMode,
    detected_scene_object,
)
from .domain import CapturedSurface, DetectionResult, GeometryMode
from .object_orientation import orientation_axes
from .geometry_comparison import ComparisonRow, comparison_rows


@dataclass(frozen=True)
class HistogramBarPresentation:
    count: int
    height_fraction: float
    heat_coordinate: float


@dataclass(frozen=True)
class HistogramOverlayPresentation:
    bars: tuple[HistogramBarPresentation, ...]
    source_count: int
    used_count: int
    excluded_count: int
    maximum_count: int
    range_min_m: float
    range_max_m: float
    mean_fraction: float
    zero_fraction: float
    mean_m: float = 0.0
    sigma_m: float = 0.0


@dataclass(frozen=True)
class ResidualHistogramPresentation:
    basis: ResidualBasis
    histogram: HistogramOverlayPresentation


@dataclass(frozen=True)
class ResultOverlayPresentation:
    visible: bool = False
    title: str = "Preview"
    result_type: GeometryMode | None = None
    rows: tuple[tuple[str, str], ...] = ()
    histogram: HistogramOverlayPresentation | None = None
    histograms: tuple[ResidualHistogramPresentation, ...] = ()
    comparison: tuple[ComparisonRow, ...] = ()
    histogram_reserved: bool = True


def build_result_overlay(
    result: DetectionResult | None,
    analysis: AnalysisPresentation | None,
    *,
    title: str = "Preview",
    pending: bool = False,
    error: str | None = None,
    reserve_histogram: bool = True,
) -> ResultOverlayPresentation:
    """Project one successful result and its immutable analysis into UI data.

    No distances are recalculated here: the histogram bars and heatmap color
    coordinates both come from the same ``AnalysisPresentation`` produced by
    the analysis worker.
    ``pending`` remains a compatible caller hint; pending analysis has no transient text.
    """

    if result is None:
        return ResultOverlayPresentation()
    return replace(_build_result_info_overlay(
        result, analysis, inlier_count=len(result.inlier_indices), fit_rms_m=result.rms_error_m,
        title=title, error=error,
    ), histogram_reserved=bool(reserve_histogram))


def _build_result_info_overlay(
    result: DetectionResult | CapturedSurface,
    analysis: AnalysisPresentation | None,
    *,
    inlier_count: int,
    fit_rms_m: float,
    title: str,
    error: str | None = None,
) -> ResultOverlayPresentation:
    detected = detected_scene_object(result.geometry_type, result.parameters)
    rows = [
        ("Seed Index", f"{result.request_key.seed_index:,}"),
        ("Inliers", f"{inlier_count:,} pts"),
        ("Fit RMS", _metres(fit_rms_m)),
        ("Pos", _vector(detected.pose_world_m.position_m)),
    ]
    if result.geometry_type is not GeometryMode.SPHERE:
        axis = orientation_axes(detected.pose_world_m.orientation_xyzw)[2]
        rows.append(("Normal" if result.geometry_type is GeometryMode.PLANE else "Axis", _vector(axis, unit=False)))
    rows.append(("Size", _size_summary(detected.geometry_type, detected.parameter_dict)))

    matching = analysis is not None and analysis.key.request_key == result.request_key
    histogram_presentation = None
    if matching and analysis is not None and analysis.available:
        rows.append(("Match RMS", _metres(analysis.analysis.match.rms_m or 0.0)))
        heatmap = analysis.analysis.heatmap
        histogram = analysis.analysis.histogram
        if heatmap.available:
            scale_m = (heatmap.scale_m or 0.0)
            rows.append((
                "Heatmap",
                f"-{scale_m:.4f} to +{scale_m:.4f} m · {heatmap.clamped_count:,} clamped",
            ))
        if histogram.available:
            rows.append((
                "Histogram",
                f"{histogram.used_count:,}/{histogram.source_count:,} pts · "
                f"{histogram.excluded_count:,} excluded",
            ))
            histogram_presentation = _histogram_presentation(histogram, heatmap)
    elif matching and analysis is not None:
        rows.append((
            "Analysis",
            f"Unavailable: {analysis.analysis.match.unavailable_reason or 'surface-unavailable'}",
        ))
    elif error:
        rows.append(("Analysis", f"Unavailable: {error}"))

    return ResultOverlayPresentation(
        True,
        str(title),
        result.geometry_type,
        tuple(rows),
        histogram_presentation,
    )


def build_capture_result_overlay(
    surface: CapturedSurface,
    analysis: AnalysisPresentation,
    *,
    surface_side_scale: SurfaceSideScale = SurfaceSideScale(),
) -> ResultOverlayPresentation:
    """Project an immutable capture through the same geometry-info path as preview."""

    overlay = _build_result_info_overlay(
        surface, analysis,
        inlier_count=surface.inlier_points_world_m.shape[0], fit_rms_m=surface.fit_rms_error_m,
        title=f"{surface.geometry_type.value} · {surface.capture_id}",
    )
    histograms = []
    for basis in (
        ResidualBasis.DETECTED,
        ResidualBasis.DESIGNED,
        ResidualBasis.SURFACE_SIDE,
    ):
        residual = analysis.residual(basis, surface_side_scale)
        if residual is None or not residual.histogram.available:
            continue
        histograms.append(ResidualHistogramPresentation(
            basis,
            _histogram_presentation(residual.histogram, residual.heatmap),
        ))
    rows = []
    for name, value in overlay.rows:
        rows.append((f"Detected {name}" if name in {"Heatmap", "Histogram"} else name, value))
    designed = analysis.residual(ResidualBasis.DESIGNED)
    if designed is not None and designed.available:
        heatmap = designed.heatmap
        histogram = designed.histogram
        rows.extend((
            (
                "Original Heatmap",
                f"-{(heatmap.scale_m or 0.0):.4f} to +"
                f"{(heatmap.scale_m or 0.0):.4f} m · "
                f"{heatmap.clamped_count:,} clamped",
            ),
            (
                "Original Histogram",
                f"{histogram.used_count:,}/{histogram.source_count:,} pts · "
                f"{histogram.excluded_count:,} excluded",
            ),
        ))
    surface_side = analysis.residual(ResidualBasis.SURFACE_SIDE, surface_side_scale)
    if surface_side is not None and surface_side.available:
        heatmap = surface_side.heatmap
        histogram = surface_side.histogram
        rows.extend((
            (
                "Surface Side Heatmap",
                f"{_surface_side_scale_label(surface_side_scale)} · "
                f"-{(heatmap.scale_m or 0.0):.4f} to +"
                f"{(heatmap.scale_m or 0.0):.4f} m · "
                f"{heatmap.clamped_count:,} clamped",
            ),
            (
                "Surface Side Histogram",
                f"{histogram.used_count:,}/{histogram.source_count:,} pts · "
                f"{histogram.excluded_count:,} excluded",
            ),
        ))
    rows = [(name, value) for name, value in rows if name not in {
        "Detected Histogram", "Original Histogram", "Surface Side Histogram",
    }]
    comparison = comparison_rows(detected_scene_object(surface.geometry_type, surface.parameters),
                                 analysis.designed_snapshot)
    return replace(overlay, rows=tuple(rows), histograms=tuple(histograms), comparison=comparison)


def _histogram_presentation(
    histogram: HistogramResult,
    heatmap: HeatmapResult,
) -> HistogramOverlayPresentation:
    scale = heatmap.scale_m if heatmap.available else None
    bars = tuple(
        HistogramBarPresentation(
            item.count,
            item.height_fraction,
            0.5 if scale is None else _clamp(0.5 + item.center_m / (2.0 * scale)),
        )
        for item in histogram.bins
    )
    return HistogramOverlayPresentation(
        bars,
        histogram.source_count,
        histogram.used_count,
        histogram.excluded_count,
        histogram.max_bin_count,
        (histogram.range_min_m or 0.0),
        (histogram.range_max_m or 0.0),
        histogram.mean_fraction or 0.0,
        histogram.zero_fraction or 0.0,
        histogram.mean_m or 0.0,
        histogram.source_sigma_m or 0.0,
    )


def _surface_side_scale_label(scale: SurfaceSideScale) -> str:
    if scale.mode is SurfaceSideScaleMode.FIXED:
        return f"Fixed ±{scale.fixed_scale_m:.4f} m"
    if scale.mode is SurfaceSideScaleMode.SIGMA:
        return f"Sigma ±{scale.sigma_multiplier:.1f} sigma"
    return "Auto full range"


def _size_summary(mode: GeometryMode, values: dict[str, float]) -> str:
    metres = lambda name: values[name]
    if mode is GeometryMode.PLANE:
        return f"{metres('width_m'):.4f} × {metres('height_m'):.4f} m"
    if mode is GeometryMode.SPHERE:
        return f"R {metres('radius_m'):.4f} m"
    if mode is GeometryMode.CYLINDER:
        return f"R {metres('radius_m'):.4f} · L {metres('length_m'):.4f} m"
    if mode is GeometryMode.CONE:
        return (
            f"R {metres('bottom_radius_m'):.4f}/{metres('top_radius_m'):.4f} · "
            f"L {metres('length_m'):.4f} m"
        )
    if mode is GeometryMode.TORUS:
        return (
            f"R {metres('major_radius_m'):.4f} · r {metres('minor_radius_m'):.4f} m · "
            f"{values['sweep_angle_deg']:.1f}°"
        )
    return "N/A"


def _metres(value_m: float) -> str:
    return f"{float(value_m):.4f} m"


def _vector(values, *, unit=True) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values) + (") m" if unit else ")")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "HistogramBarPresentation",
    "HistogramOverlayPresentation",
    "ResidualHistogramPresentation",
    "ResultOverlayPresentation",
    "build_capture_result_overlay",
    "build_result_overlay",
]
