"""Generation-aware preview/capture match analysis and presentation packets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
import math
import threading
import time
import traceback
from typing import Iterable, Mapping, Sequence

import numpy as np

from .spatial_math import cross_vec3
from .analysis import (
    HeatmapResult,
    HistogramResult,
    MatchAnalysis,
    MIN_DISPLAY_SCALE_M,
    SIGMA_EPSILON_M,
    aggregate_match,
    build_signed_heatmap,
    build_signed_histogram_heatmap,
)
from .domain import (
    CapturedSurface,
    DetectionRequestKey,
    DetectionResult,
    GenerationSnapshot,
    GeometryMode,
    Pose,
    ScanFrame,
    immutable_float32_points,
    thaw,
)
from .geometry import build_mesh
from .metrics import SurfaceMatchResult, compare_surfaces, signed_surface_distances
from .render_layers import (
    MeshLayerPacket,
    PointLayerPacket,
    RenderLayer,
    build_inlier_layer,
    build_mesh_layer,
    recolor_inlier_layer,
)
from .scene import SceneObject


@dataclass(frozen=True)
class AnalysisSubject:
    subject_id: str
    request_key: DetectionRequestKey
    generations: GenerationSnapshot
    geometry_type: GeometryMode
    parameters: tuple
    inlier_points_world_m: np.ndarray
    measurement_accuracy_m: float
    include_designed_residual: bool = False

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id cannot be empty")
        object.__setattr__(self, "geometry_type", GeometryMode(self.geometry_type))
        object.__setattr__(self, "inlier_points_world_m", immutable_float32_points(self.inlier_points_world_m))
        accuracy = float(self.measurement_accuracy_m)
        if not math.isfinite(accuracy) or accuracy <= 0.0:
            raise ValueError("measurement_accuracy_m must be finite and positive")
        object.__setattr__(self, "measurement_accuracy_m", accuracy)

    @classmethod
    def from_preview(
        cls,
        result: DetectionResult,
        frame: ScanFrame,
        *,
        measurement_accuracy_m: float,
    ) -> "AnalysisSubject":
        if result.request_key.scan_generation != frame.generation or result.request_key.scan_sequence != frame.sequence:
            raise ValueError("preview result does not belong to the supplied frame")
        if any(index >= frame.compacted_count for index in result.inlier_indices):
            raise ValueError("preview result contains an out-of-range inlier index")
        return cls(
            "preview",
            result.request_key,
            result.generations,
            result.geometry_type,
            result.parameters,
            frame.points_world_m[list(result.inlier_indices)],
            measurement_accuracy_m,
        )

    @classmethod
    def from_capture(cls, surface: CapturedSurface) -> "AnalysisSubject":
        return cls(
            surface.capture_id,
            surface.request_key,
            surface.generations,
            surface.geometry_type,
            surface.parameters,
            surface.inlier_points_world_m,
            surface.detection_options.measurement_accuracy_m,
            True,
        )


@dataclass(frozen=True)
class AnalysisKey:
    subject_id: str
    request_key: DetectionRequestKey
    generations: GenerationSnapshot
    scene_generation: int
    designed_signature: tuple | None
    include_mesh_layers: bool = True
    include_designed_residual: bool = False


@dataclass(frozen=True)
class AnalysisRequest:
    key: AnalysisKey
    subject: AnalysisSubject
    designed: SceneObject | None
    detected: SceneObject | None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class MetricRow:
    name: str
    value: float
    display_value: float
    display_unit: str


@dataclass(frozen=True)
class HeatmapLegend:
    colormap: str
    minimum_m: float
    maximum_m: float
    scale_m: float
    clamped_count: int


class ResidualBasis(str, Enum):
    DETECTED = "detected"
    DESIGNED = "designed"
    SURFACE_SIDE = "surface-side"


class SurfaceSideScaleMode(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"
    SIGMA = "sigma"


@dataclass(frozen=True)
class SurfaceSideScale:
    """Global colour saturation policy for the signed surface-side residual."""

    mode: SurfaceSideScaleMode = SurfaceSideScaleMode.SIGMA
    fixed_scale_m: float = 0.01
    sigma_multiplier: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SurfaceSideScaleMode(self.mode))
        fixed = float(self.fixed_scale_m)
        multiplier = float(self.sigma_multiplier)
        if not math.isfinite(fixed) or not 1e-6 <= fixed <= 100.0:
            raise ValueError("surface-side fixed scale must be between 1e-6 and 100 metres")
        if not math.isfinite(multiplier) or not 0.1 <= multiplier <= 10.0:
            raise ValueError("surface-side sigma multiplier must be between 0.1 and 10")
        object.__setattr__(self, "fixed_scale_m", fixed)
        object.__setattr__(self, "sigma_multiplier", multiplier)

    def resolve_scale_m(
        self,
        signed_distances_m: Sequence[float],
        histogram: HistogramResult | None = None,
    ) -> float:
        if self.mode is SurfaceSideScaleMode.FIXED:
            return self.fixed_scale_m
        if self.mode is SurfaceSideScaleMode.SIGMA:
            sigma = 0.0 if histogram is None else float(histogram.source_sigma_m or 0.0)
            if sigma <= SIGMA_EPSILON_M:
                sigma = max(
                    0.0 if histogram is None else float(histogram.sigma_m or 0.0),
                    MIN_DISPLAY_SCALE_M,
                )
            return max(self.sigma_multiplier * sigma, MIN_DISPLAY_SCALE_M)
        maximum = max((abs(float(value)) for value in signed_distances_m), default=0.0)
        return max(maximum, MIN_DISPLAY_SCALE_M)


class _SurfaceSideColorMemo:
    """One derived color packet, owned and released with its source residual."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.policy: SurfaceSideScale | None = None
        self.result: ResidualPresentation | None = None


@dataclass(frozen=True)
class ResidualPresentation:
    basis: ResidualBasis
    histogram: HistogramResult
    heatmap: HeatmapResult
    heatmap_legend: HeatmapLegend | None = None
    inlier_layer: PointLayerPacket | None = None
    signed_distances_m: tuple[float, ...] = ()
    _color_memo: _SurfaceSideColorMemo = field(
        default_factory=_SurfaceSideColorMemo, init=False, repr=False, compare=False,
    )

    @property
    def available(self) -> bool:
        return self.histogram.available and self.heatmap.available


@dataclass(frozen=True)
class AnalysisPresentation:
    key: AnalysisKey
    analysis: MatchAnalysis
    metrics: tuple[MetricRow, ...] = ()
    heatmap_legend: HeatmapLegend | None = None
    mesh_layers: tuple[MeshLayerPacket, ...] = ()
    inlier_layer: PointLayerPacket | None = None
    elapsed_s: float = 0.0
    residuals: tuple[ResidualPresentation, ...] = ()
    designed_snapshot: SceneObject | None = None
    detected_snapshot: SceneObject | None = None

    @property
    def available(self) -> bool:
        return self.analysis.match.available

    def residual(
        self,
        basis: ResidualBasis | str,
        surface_side_scale: SurfaceSideScale | None = None,
    ) -> ResidualPresentation | None:
        basis = ResidualBasis(basis)
        for presentation in self.residuals:
            if presentation.basis is basis:
                if basis is ResidualBasis.SURFACE_SIDE and surface_side_scale is not None:
                    return scale_surface_side_residual(presentation, surface_side_scale)
                return presentation
        # Compatibility for packets constructed before dual capture residuals.
        if basis is ResidualBasis.DETECTED:
            return ResidualPresentation(
                basis,
                self.analysis.histogram,
                self.analysis.heatmap,
                self.heatmap_legend,
                self.inlier_layer,
            )
        return None


def make_analysis_request(
    subject: AnalysisSubject,
    designed_objects: Iterable[SceneObject],
    *,
    scene_generation: int,
    include_mesh_layers: bool = True,
) -> AnalysisRequest:
    """Resolve the unique, enabled same-type design object and freeze a cache key."""

    candidates = tuple(item for item in designed_objects if item.geometry_type is subject.geometry_type)
    designed = candidates[0] if len(candidates) == 1 else None
    reason = None
    if not candidates:
        reason = "matching-object-missing"
    elif len(candidates) > 1:
        reason = "matching-object-ambiguous"
    elif not designed.enabled:
        reason = "matching-object-disabled"
    detected = None
    if reason is None:
        try:
            detected = detected_scene_object(subject.geometry_type, subject.parameters)
        except (KeyError, TypeError, ValueError):
            reason = "invalid-detected-parameters"
    signature = _surface_signature(designed) if designed is not None else None
    key = AnalysisKey(
        subject.subject_id,
        subject.request_key,
        subject.generations,
        int(scene_generation),
        signature,
        bool(include_mesh_layers),
        bool(subject.include_designed_residual),
    )
    return AnalysisRequest(key, subject, designed, detected, reason)


def evaluate_analysis(request: AnalysisRequest) -> AnalysisPresentation:
    """Perform analytic metrics, histogram, heatmap, and render packing off the UI thread."""

    started = time.perf_counter()
    if request.unavailable_reason is not None or request.designed is None or request.detected is None:
        match = SurfaceMatchResult(False, None, 0, unavailable_reason=request.unavailable_reason or "surface-unavailable")
    else:
        match = compare_surfaces(request.designed, request.detected, request.subject.inlier_points_world_m)
    signed_residuals = (
        ()
        if request.detected is None
        else signed_surface_distances(request.detected, request.subject.inlier_points_world_m)
    )
    analysis = aggregate_match(
        match,
        measurement_accuracy_m=request.subject.measurement_accuracy_m,
        signed_inlier_distances_m=signed_residuals,
    )
    metrics: tuple[MetricRow, ...] = ()
    legend = None
    meshes: tuple[MeshLayerPacket, ...] = ()
    inliers = None
    residuals: tuple[ResidualPresentation, ...] = ()
    if match.available:
        metrics = (_metric_row("rms_m", match.rms_m or 0.0),) + tuple(
            _metric_row(name, value) for name, value in match.parameter_deltas
        )
        heatmap = analysis.heatmap
        legend = _heatmap_legend(heatmap)
        if request.key.include_mesh_layers:
            meshes = (
                build_mesh_layer(
                    RenderLayer.DESIGNED_MESH,
                    (build_mesh(request.designed).transformed(request.designed.pose_world_m),),
                ),
                build_mesh_layer(
                    RenderLayer.DETECTED_MESH,
                    (build_mesh(request.detected).transformed(request.detected.pose_world_m),),
                    sphere_pole_center_world_m=(
                        request.detected.pose_world_m.position_m if request.detected.geometry_type is GeometryMode.SPHERE else None
                    ),
                ),
            )
        inliers = build_inlier_layer(request.subject.inlier_points_world_m, heatmap)
        residuals = (
            ResidualPresentation(
                ResidualBasis.DETECTED,
                analysis.histogram,
                analysis.heatmap,
                legend,
                inliers,
                tuple(float(value) for value in signed_residuals),
            ),
        )
        if request.subject.include_designed_residual:
            designed_signed_residuals = signed_surface_distances(
                request.designed,
                request.subject.inlier_points_world_m,
            )
            designed_histogram, designed_heatmap = build_signed_histogram_heatmap(
                designed_signed_residuals,
                measurement_accuracy_m=request.subject.measurement_accuracy_m,
            )
            residuals += (ResidualPresentation(
                ResidualBasis.DESIGNED,
                designed_histogram,
                designed_heatmap,
                _heatmap_legend(designed_heatmap),
                recolor_inlier_layer(inliers, designed_heatmap),
                tuple(float(value) for value in designed_signed_residuals),
            ),)
        surface_side_values = match.surface_side_distances_m
        side_histogram, side_heatmap = build_signed_histogram_heatmap(
            surface_side_values,
            measurement_accuracy_m=request.subject.measurement_accuracy_m,
            heatmap_colormap="diverging-signed-inlier",
        )
        side_residual = ResidualPresentation(
            ResidualBasis.SURFACE_SIDE,
            side_histogram,
            side_heatmap,
            _heatmap_legend(side_heatmap),
            recolor_inlier_layer(inliers, side_heatmap),
            tuple(float(value) for value in surface_side_values),
        )
        residuals += (scale_surface_side_residual(side_residual, SurfaceSideScale()),)
    return AnalysisPresentation(
        request.key,
        analysis,
        metrics,
        legend,
        meshes,
        inliers,
        time.perf_counter() - started,
        residuals,
        request.designed,
        request.detected,
    )


def scale_surface_side_residual(
    presentation: ResidualPresentation,
    scale_policy: SurfaceSideScale,
) -> ResidualPresentation:
    """Reuse the last policy's colors without repeating distance or color analysis."""

    if presentation.basis is not ResidualBasis.SURFACE_SIDE:
        return presentation
    if not presentation.signed_distances_m or presentation.inlier_layer is None:
        return presentation
    memo = presentation._color_memo
    with memo.lock:
        if memo.policy == scale_policy:
            return presentation if memo.result is None else memo.result
        result = _recolor_surface_side_residual(presentation, scale_policy)
        # Do not retain the source itself: avoid a source -> memo -> source cycle.
        memo.result = None if result is presentation else result
        memo.policy = scale_policy
        return result


def _recolor_surface_side_residual(
    presentation: ResidualPresentation,
    scale_policy: SurfaceSideScale,
) -> ResidualPresentation:
    scale_m = scale_policy.resolve_scale_m(
        presentation.signed_distances_m,
        presentation.histogram,
    )
    if (
        presentation.heatmap.available
        and presentation.heatmap.scale_m == scale_m
        and presentation.heatmap.colormap == "diverging-signed-inlier"
    ):
        return presentation
    heatmap = build_signed_heatmap(
        presentation.signed_distances_m,
        scale_m=scale_m,
        colormap="diverging-signed-inlier",
    )
    return replace(
        presentation,
        heatmap=heatmap,
        heatmap_legend=_heatmap_legend(heatmap),
        inlier_layer=recolor_inlier_layer(presentation.inlier_layer, heatmap),
    )


def _heatmap_legend(heatmap: HeatmapResult) -> HeatmapLegend | None:
    if not heatmap.available:
        return None
    return HeatmapLegend(
        heatmap.colormap,
        (heatmap.minimum_m or 0.0),
        (heatmap.maximum_m or 0.0),
        (heatmap.scale_m or 0.0),
        heatmap.clamped_count,
    )


def detected_scene_object(
    geometry_type: GeometryMode | str,
    frozen_parameters: tuple | Mapping[str, object],
) -> SceneObject:
    """Convert FindSurface's canonical native output frame to a SceneObject."""

    mode = GeometryMode(geometry_type)
    values = thaw(frozen_parameters) if isinstance(frozen_parameters, tuple) else dict(frozen_parameters)
    origin = _vector(values, "origin")
    if mode is GeometryMode.SPHERE:
        return SceneObject(mode, True, Pose(origin), {"radius_m": _number(values, "radius")})
    if mode in (GeometryMode.CYLINDER, GeometryMode.CONE):
        # FindSurface axis_frame uses Y for the finite axis; SceneObject/mesh uses local Z.
        native_x = _vector(values, "x_axis")
        native_y = _vector(values, "y_axis")
        native_z = _vector(values, "z_axis")
        pose = Pose(origin, _quaternion_from_basis(native_x, tuple(-value for value in native_z), native_y))
    else:
        pose = Pose(origin, _quaternion_from_axes(values))
    if mode is GeometryMode.PLANE:
        parameters = {"width_m": _number(values, "width"), "height_m": _number(values, "height")}
    elif mode is GeometryMode.CYLINDER:
        parameters = {"radius_m": _number(values, "radius"), "length_m": _number(values, "height")}
    elif mode is GeometryMode.CONE:
        parameters = {
            "top_radius_m": _number(values, "top_radius"),
            "bottom_radius_m": _number(values, "bottom_radius"),
            "length_m": _number(values, "height"),
        }
    elif mode is GeometryMode.TORUS:
        parameters = {
            "major_radius_m": _number(values, "major_radius"),
            "minor_radius_m": _number(values, "tube_radius"),
            "sweep_start_deg": 0.0,
            "sweep_angle_deg": math.degrees(_number(values, "sweep_angle")),
        }
    else:
        raise ValueError("AUTO cannot be converted to an analytic surface")
    return SceneObject(mode, True, pose, parameters)


class AnalysisSubmit(str, Enum):
    STARTED = "STARTED"
    QUEUED = "QUEUED"
    REPLACED = "REPLACED"


class AnalysisCompletionStatus(str, Enum):
    PUBLISHED = "PUBLISHED"
    STALE = "STALE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AnalysisCompletion:
    key: AnalysisKey
    status: AnalysisCompletionStatus
    presentation: AnalysisPresentation | None
    error: str | None
    cache_hit: bool
    worker_thread_id: int


@dataclass(frozen=True)
class AnalysisTelemetry:
    submitted: int
    completed: int
    stale: int
    failed: int
    cache_hits: int
    pending_replacements: int
    mailbox_overwrites: int


class SerializedAnalysisScheduler:
    """Single-worker latest-only analysis with a bounded generation-keyed cache."""

    def __init__(self, *, cache_capacity: int = 16) -> None:
        if cache_capacity <= 0:
            raise ValueError("cache_capacity must be positive")
        self._condition = threading.Condition()
        self._cache_capacity = int(cache_capacity)
        self._cache: OrderedDict[AnalysisKey, AnalysisPresentation] = OrderedDict()
        self._current_key: AnalysisKey | None = None
        self._pending: AnalysisRequest | None = None
        self._active = False
        self._closed = False
        self._mailbox: AnalysisCompletion | None = None
        self._counts = {name: 0 for name in AnalysisTelemetry.__dataclass_fields__}
        self._thread = threading.Thread(target=self._run, name="findsurface-analysis", daemon=True)
        self._thread.start()

    def update_current(self, key: AnalysisKey | None) -> None:
        with self._condition:
            self._current_key = key

    def submit(self, request: AnalysisRequest) -> AnalysisSubmit:
        with self._condition:
            if self._closed:
                raise RuntimeError("analysis scheduler is closed")
            self._counts["submitted"] += 1
            if self._active or self._pending is not None:
                status = AnalysisSubmit.REPLACED if self._pending is not None else AnalysisSubmit.QUEUED
                if self._pending is not None:
                    self._counts["pending_replacements"] += 1
            else:
                status = AnalysisSubmit.STARTED
            self._pending = request
            self._condition.notify()
            return status

    def drain_completion(self) -> AnalysisCompletion | None:
        with self._condition:
            completion = self._mailbox
            self._mailbox = None
            return completion

    @property
    def telemetry(self) -> AnalysisTelemetry:
        with self._condition:
            return AnalysisTelemetry(**self._counts)

    def close(self, timeout_s: float | None = None) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("analysis worker did not stop")

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                self._active = True
                cached = self._cache.get(request.key)
            cache_hit = cached is not None
            try:
                presentation = cached or evaluate_analysis(request)
                error = None
            except Exception:
                presentation = None
                error = traceback.format_exc()
            with self._condition:
                self._active = False
                if presentation is not None and not cache_hit:
                    self._cache[request.key] = presentation
                    self._cache.move_to_end(request.key)
                    while len(self._cache) > self._cache_capacity:
                        self._cache.popitem(last=False)
                if cache_hit:
                    self._counts["cache_hits"] += 1
                if error is not None:
                    status = AnalysisCompletionStatus.FAILED
                    self._counts["failed"] += 1
                elif request.key != self._current_key:
                    status = AnalysisCompletionStatus.STALE
                    presentation = None
                    self._counts["stale"] += 1
                else:
                    status = AnalysisCompletionStatus.PUBLISHED
                self._counts["completed"] += 1
                completion = AnalysisCompletion(
                    request.key, status, presentation, error, cache_hit, threading.get_ident()
                )
                if self._mailbox is not None:
                    self._counts["mailbox_overwrites"] += 1
                self._mailbox = completion
                self._condition.notify_all()


def _surface_signature(surface: SceneObject) -> tuple:
    return (
        surface.geometry_type.value,
        surface.enabled,
        surface.pose_world_m.position_m,
        surface.pose_world_m.orientation_xyzw,
        surface.parameters,
    )


def _metric_row(name: str, value: float) -> MetricRow:
    if name.endswith("_m"):
        return MetricRow(name, value, value, "m")
    if name.endswith("_rad"):
        return MetricRow(name, value, math.degrees(value), "deg")
    return MetricRow(name, value, value, "")


def _number(values: Mapping[str, object], name: str) -> float:
    value = float(values[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(values: Mapping[str, object], name: str) -> tuple[float, float, float]:
    raw = values[name]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _quaternion_from_axes(values: Mapping[str, object]) -> tuple[float, float, float, float]:
    return _quaternion_from_basis(
        _vector(values, "x_axis"), _vector(values, "y_axis"), _vector(values, "z_axis")
    )


def _quaternion_from_basis(
    x_raw: Sequence[float], y_raw: Sequence[float], supplied_z_raw: Sequence[float]
) -> tuple[float, float, float, float]:
    x_axis = _unit(x_raw)
    y_axis = _unit(tuple(y_raw[index] - _dot(y_raw, x_axis) * x_axis[index] for index in range(3)))
    z_axis = cross_vec3(x_axis, y_axis)
    supplied_z = _unit(supplied_z_raw)
    if _dot(z_axis, supplied_z) < 0.999:
        raise ValueError("detected axes must form a right-handed orthonormal frame")
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        size = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((m21 - m12) / size, (m02 - m20) / size, (m10 - m01) / size, 0.25 * size)
    elif m00 > m11 and m00 > m22:
        size = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (0.25 * size, (m01 + m10) / size, (m02 + m20) / size, (m21 - m12) / size)
    elif m11 > m22:
        size = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m01 + m10) / size, 0.25 * size, (m12 + m21) / size, (m02 - m20) / size)
    else:
        size = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m02 + m20) / size, (m12 + m21) / size, 0.25 * size, (m10 - m01) / size)
    return quaternion


def _unit(value: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(value, value))
    if length <= 1e-12:
        raise ValueError("axis cannot be zero")
    return tuple(component / length for component in value)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


__all__ = [
    "AnalysisCompletion", "AnalysisCompletionStatus", "AnalysisKey", "AnalysisPresentation",
    "AnalysisRequest", "AnalysisSubject", "AnalysisSubmit", "AnalysisTelemetry", "HeatmapLegend",
    "MetricRow", "ResidualBasis", "ResidualPresentation", "SerializedAnalysisScheduler",
    "SurfaceSideScale", "SurfaceSideScaleMode", "detected_scene_object", "evaluate_analysis",
    "make_analysis_request", "scale_surface_side_residual",
]
