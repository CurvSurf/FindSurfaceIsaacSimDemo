"""Renderer-independent analytic nearest-foot and surface match metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .spatial_math import cross_vec3, rotate_vector_xyzw
from .domain import GeometryMode, Pose
from .scene import SceneObject


Vec3 = tuple[float, float, float]
_SINGULAR_EPSILON = 1e-12
SURFACE_SIDE_BOUNDARY_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class SurfaceMatchResult:
    """Canonical metre/radian metric result; histogram filtering is not applied."""

    available: bool
    rms_m: float | None
    sample_count: int
    distances_m: tuple[float, ...] = ()
    surface_side_distances_m: tuple[float, ...] = ()
    parameter_deltas: tuple[tuple[str, float], ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        if self.available:
            if self.rms_m is None or not math.isfinite(self.rms_m) or self.rms_m < 0.0:
                raise ValueError("available result requires a finite non-negative RMS")
            if self.sample_count == 0 or len(self.distances_m) != self.sample_count:
                raise ValueError("available result requires one distance per sample")
            if (
                self.surface_side_distances_m
                and len(self.surface_side_distances_m) != self.sample_count
            ):
                raise ValueError("surface-side result requires one distance per sample")
            if self.unavailable_reason is not None:
                raise ValueError("available result cannot carry an unavailable reason")
        else:
            if (
                self.rms_m is not None
                or self.distances_m
                or self.surface_side_distances_m
                or self.parameter_deltas
            ):
                raise ValueError("unavailable result cannot carry numeric measurements")
            if not self.unavailable_reason:
                raise ValueError("unavailable result requires a reason")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.distances_m):
            raise ValueError("distances_m must be finite and non-negative")
        if not all(math.isfinite(value) for value in self.surface_side_distances_m):
            raise ValueError("surface_side_distances_m must be finite")
        if not all(math.isfinite(value) for _, value in self.parameter_deltas):
            raise ValueError("parameter deltas must be finite")

    @property
    def delta_dict(self) -> dict[str, float]:
        return dict(self.parameter_deltas)


def nearest_foot(surface: SceneObject, point_world_m: Sequence[float]) -> Vec3:
    """Return the nearest point on the closed finite analytic surface in world metres."""

    point = _vec3(point_world_m)
    local = _world_to_local(surface.pose_world_m, point)
    values = surface.parameter_dict
    mode = surface.geometry_type
    if mode is GeometryMode.PLANE:
        foot = (
            _clamp(local[0], -values["width_m"] * 0.5, values["width_m"] * 0.5),
            _clamp(local[1], -values["height_m"] * 0.5, values["height_m"] * 0.5),
            0.0,
        )
    elif mode is GeometryMode.SPHERE:
        foot = _sphere_foot(local, values["radius_m"])
    elif mode is GeometryMode.CYLINDER:
        foot = _frustum_foot(
            local, values["radius_m"], values["radius_m"], values["length_m"]
        )
    elif mode is GeometryMode.CONE:
        foot = _frustum_foot(
            local,
            values["bottom_radius_m"],
            values["top_radius_m"],
            values["length_m"],
        )
    elif mode is GeometryMode.TORUS:
        foot = _torus_foot(
            local,
            values["major_radius_m"],
            values["minor_radius_m"],
            math.radians(values["sweep_start_deg"]),
            math.radians(values["sweep_start_deg"] + values["sweep_angle_deg"]),
        )
    else:
        raise ValueError(f"unsupported geometry type: {mode.value}")
    return _local_to_world(surface.pose_world_m, foot)


def compare_surfaces(
    designed: SceneObject,
    detected: SceneObject,
    inlier_points_world_m: Iterable[Sequence[float]],
) -> SurfaceMatchResult:
    """Compare two same-type surfaces at the same unfiltered inlier samples."""

    if designed.geometry_type is not detected.geometry_type:
        return _unavailable("geometry-type-mismatch")
    if not designed.enabled or not detected.enabled:
        return _unavailable("disabled-surface")
    try:
        points = _point_array(inlier_points_world_m)
    except (TypeError, ValueError):
        return _unavailable("invalid-inlier-point")
    if points.shape[0] == 0:
        return _unavailable("empty-inlier-set")
    try:
        designed_feet = _nearest_feet(designed, points)
        detected_feet = _nearest_feet(detected, points)
        distances_array = np.linalg.norm(designed_feet - detected_feet, axis=1)
        distances = tuple(float(value) for value in distances_array)
        surface_side = _signed_surface_side_from_feet(
            designed,
            detected_feet,
            distances_array,
            boundary_tolerance_m=SURFACE_SIDE_BOUNDARY_TOLERANCE_M,
        )
        rms = float(np.sqrt(np.mean(np.square(distances_array))))
        deltas = parameter_deltas(designed, detected)
    except (ArithmeticError, ValueError):
        return _unavailable("nearest-foot-undefined")
    return SurfaceMatchResult(
        True,
        rms,
        points.shape[0],
        distances,
        surface_side,
        deltas,
    )


def surface_side_distances(
    designed: SceneObject,
    detected: SceneObject,
    inlier_points_world_m: Iterable[Sequence[float]],
    *,
    boundary_tolerance_m: float = SURFACE_SIDE_BOUNDARY_TOLERANCE_M,
) -> tuple[float, ...]:
    """Return signed distances between the two nearest surface feet.

    Magnitude is ``||foot(designed, p) - foot(detected, p)||`` in metres.
    The sign classifies the detected foot against the designed finite solid:
    interior/negative-normal is negative, exterior/positive-normal is positive,
    and coincident feet within ``boundary_tolerance_m`` are exactly zero.
    """

    if designed.geometry_type is not detected.geometry_type:
        raise ValueError("surface-side distances require matching geometry types")
    if not designed.enabled or not detected.enabled:
        raise ValueError("surface-side distances require enabled surfaces")
    tolerance = float(boundary_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary_tolerance_m must be finite and non-negative")
    points = _point_array(inlier_points_world_m)
    if points.shape[0] == 0:
        return ()
    designed_feet = _nearest_feet(designed, points)
    detected_feet = _nearest_feet(detected, points)
    magnitudes = np.linalg.norm(designed_feet - detected_feet, axis=1)
    return _signed_surface_side_from_feet(
        designed,
        detected_feet,
        magnitudes,
        boundary_tolerance_m=tolerance,
    )


def _signed_surface_side_from_feet(
    designed: SceneObject,
    detected_feet_world_m: np.ndarray,
    magnitudes_m: np.ndarray,
    *,
    boundary_tolerance_m: float,
) -> tuple[float, ...]:
    local = _world_to_local_points(designed.pose_world_m, detected_feet_world_m)
    values = designed.parameter_dict
    mode = designed.geometry_type
    if mode is GeometryMode.PLANE:
        inside = local[:, 2] < 0.0
    elif mode is GeometryMode.SPHERE:
        inside = np.linalg.norm(local, axis=1) <= values["radius_m"]
    elif mode in (GeometryMode.CYLINDER, GeometryMode.CONE):
        half = values["length_m"] * 0.5
        within_caps = (local[:, 2] >= -half) & (local[:, 2] <= half)
        radial = np.hypot(local[:, 0], local[:, 1])
        if mode is GeometryMode.CYLINDER:
            radius = np.full(local.shape[0], values["radius_m"], dtype=np.float64)
        else:
            fraction = np.clip((local[:, 2] + half) / values["length_m"], 0.0, 1.0)
            radius = values["bottom_radius_m"] + fraction * (
                values["top_radius_m"] - values["bottom_radius_m"]
            )
        inside = within_caps & (radial <= radius)
    elif mode is GeometryMode.TORUS:
        radial = np.hypot(local[:, 0], local[:, 1])
        inside_tube = (
            np.hypot(radial - values["major_radius_m"], local[:, 2])
            <= values["minor_radius_m"]
        )
        sweep = math.radians(values["sweep_angle_deg"])
        if sweep >= math.tau - 1e-10:
            inside_sweep = np.ones(local.shape[0], dtype=bool)
        else:
            start = math.radians(values["sweep_start_deg"])
            angle = np.arctan2(local[:, 1], local[:, 0])
            relative = np.mod(angle - start, math.tau)
            inside_sweep = relative <= sweep + 1e-12
        inside = inside_tube & inside_sweep
    else:
        raise ValueError(f"unsupported geometry type: {mode.value}")

    signed = np.where(inside, -magnitudes_m, magnitudes_m)
    signed[magnitudes_m <= boundary_tolerance_m] = 0.0
    return tuple(float(value) for value in signed)


def signed_surface_distance(surface: SceneObject, point_world_m: Sequence[float]) -> float:
    """Return demo-1-compatible signed distance to the detected analytic surface.

    Negative values are on the inner/negative-normal side, positive values are
    on the outer/positive-normal side, and zero lies on the fitted surface.
    Cylinder/cone end caps and torus sweep caps intentionally do not participate:
    demo 1 colors FindSurface inliers against the fitted infinite side surface.
    """

    local = _world_to_local(surface.pose_world_m, _vec3(point_world_m))
    values = surface.parameter_dict
    mode = surface.geometry_type
    if mode is GeometryMode.PLANE:
        return local[2]
    if mode is GeometryMode.SPHERE:
        return _length(local) - values["radius_m"]
    radial = math.hypot(local[0], local[1])
    if mode is GeometryMode.CYLINDER:
        return radial - values["radius_m"]
    if mode is GeometryMode.CONE:
        length = values["length_m"]
        slope = (values["top_radius_m"] - values["bottom_radius_m"]) / length
        axial_from_bottom = local[2] + length * 0.5
        radius_at_axial = values["bottom_radius_m"] + slope * axial_from_bottom
        return (radial - radius_at_axial) / math.sqrt(slope * slope + 1.0)
    if mode is GeometryMode.TORUS:
        return (
            math.hypot(radial - values["major_radius_m"], local[2])
            - values["minor_radius_m"]
        )
    raise ValueError(f"unsupported geometry type: {mode.value}")


def signed_surface_distances(
    surface: SceneObject,
    points_world_m: Iterable[Sequence[float]],
) -> tuple[float, ...]:
    """Evaluate one immutable signed residual per inlier point in one NumPy batch.

    The analysis scheduler runs this function on its dedicated worker.  Keeping
    the point-wise arithmetic inside NumPy avoids a long GIL-held Python loop
    that would otherwise starve Kit's UI thread for large torus inlier sets.
    """

    points = _point_array(points_world_m)
    if points.shape[0] == 0:
        return ()
    local = _world_to_local_points(surface.pose_world_m, points)
    values = surface.parameter_dict
    mode = surface.geometry_type
    if mode is GeometryMode.PLANE:
        distances = local[:, 2]
    elif mode is GeometryMode.SPHERE:
        distances = np.linalg.norm(local, axis=1) - values["radius_m"]
    else:
        radial = np.hypot(local[:, 0], local[:, 1])
        if mode is GeometryMode.CYLINDER:
            distances = radial - values["radius_m"]
        elif mode is GeometryMode.CONE:
            length = values["length_m"]
            slope = (values["top_radius_m"] - values["bottom_radius_m"]) / length
            axial_from_bottom = local[:, 2] + length * 0.5
            radius_at_axial = values["bottom_radius_m"] + slope * axial_from_bottom
            distances = (radial - radius_at_axial) / math.sqrt(slope * slope + 1.0)
        elif mode is GeometryMode.TORUS:
            distances = (
                np.hypot(radial - values["major_radius_m"], local[:, 2])
                - values["minor_radius_m"]
            )
        else:
            raise ValueError(f"unsupported geometry type: {mode.value}")
    return tuple(float(value) for value in distances)


def _point_array(points_world_m: Iterable[Sequence[float]]) -> np.ndarray:
    """Normalize an arbitrary point iterable to finite float64 ``[N, 3]`` data."""

    if isinstance(points_world_m, np.ndarray):
        points = np.asarray(points_world_m, dtype=np.float64)
    else:
        points = np.asarray(tuple(points_world_m), dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite [N, 3] array")
    return np.ascontiguousarray(points)


def _nearest_feet(surface: SceneObject, points_world_m: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of :func:`nearest_foot` for immutable analysis batches."""

    local = _world_to_local_points(surface.pose_world_m, points_world_m)
    values = surface.parameter_dict
    mode = surface.geometry_type
    if mode is GeometryMode.PLANE:
        feet = local.copy()
        feet[:, 0] = np.clip(feet[:, 0], -values["width_m"] * 0.5, values["width_m"] * 0.5)
        feet[:, 1] = np.clip(feet[:, 1], -values["height_m"] * 0.5, values["height_m"] * 0.5)
        feet[:, 2] = 0.0
    elif mode is GeometryMode.SPHERE:
        lengths = np.linalg.norm(local, axis=1)
        directions = np.zeros_like(local)
        directions[:, 0] = 1.0
        regular = lengths > _SINGULAR_EPSILON
        directions[regular] = local[regular] / lengths[regular, None]
        feet = directions * values["radius_m"]
    elif mode is GeometryMode.CYLINDER:
        feet = _frustum_feet(
            local, values["radius_m"], values["radius_m"], values["length_m"]
        )
    elif mode is GeometryMode.CONE:
        feet = _frustum_feet(
            local,
            values["bottom_radius_m"],
            values["top_radius_m"],
            values["length_m"],
        )
    elif mode is GeometryMode.TORUS:
        feet = _torus_feet(
            local,
            values["major_radius_m"],
            values["minor_radius_m"],
            math.radians(values["sweep_start_deg"]),
            math.radians(values["sweep_start_deg"] + values["sweep_angle_deg"]),
        )
    else:
        raise ValueError(f"unsupported geometry type: {mode.value}")
    return _local_to_world_points(surface.pose_world_m, feet)


def _frustum_feet(
    points: np.ndarray,
    bottom_radius: float,
    top_radius: float,
    length: float,
) -> np.ndarray:
    radial = np.hypot(points[:, 0], points[:, 1])
    directions = np.zeros((points.shape[0], 2), dtype=np.float64)
    directions[:, 0] = 1.0
    regular = radial > _SINGULAR_EPSILON
    directions[regular] = points[regular, :2] / radial[regular, None]
    half = length * 0.5

    segment = np.array((top_radius - bottom_radius, length), dtype=np.float64)
    relative = np.column_stack((radial - bottom_radius, points[:, 2] + half))
    fractions = np.clip((relative @ segment) / float(segment @ segment), 0.0, 1.0)
    side_radial = bottom_radius + fractions * segment[0]
    side_z = -half + fractions * length
    candidates = [
        np.column_stack((directions * side_radial[:, None], side_z)),
    ]
    if bottom_radius > 0.0:
        cap_radial = np.minimum(radial, bottom_radius)
        candidates.append(
            np.column_stack((directions * cap_radial[:, None], np.full(points.shape[0], -half)))
        )
    if top_radius > 0.0:
        cap_radial = np.minimum(radial, top_radius)
        candidates.append(
            np.column_stack((directions * cap_radial[:, None], np.full(points.shape[0], half)))
        )
    return _nearest_candidates(points, candidates)


def _torus_feet(
    points: np.ndarray,
    major: float,
    minor: float,
    start: float,
    end: float,
) -> np.ndarray:
    radial = np.hypot(points[:, 0], points[:, 1])
    alpha = np.where(radial > _SINGULAR_EPSILON, np.arctan2(points[:, 1], points[:, 0]), start)
    periodic = start + np.mod(alpha - start, math.tau)
    periodic_valid = periodic <= end + 1e-12

    candidates = [
        _torus_body_feet(points, major, minor, np.full(points.shape[0], start)),
        _torus_body_feet(points, major, minor, np.full(points.shape[0], end)),
        _torus_body_feet(points, major, minor, periodic),
    ]
    validities = [
        np.ones(points.shape[0], dtype=bool),
        np.ones(points.shape[0], dtype=bool),
        periodic_valid,
    ]
    if end - start < math.tau - 1e-10:
        candidates.extend((
            _torus_cap_feet(points, major, minor, start),
            _torus_cap_feet(points, major, minor, end),
        ))
        validities.extend((
            np.ones(points.shape[0], dtype=bool),
            np.ones(points.shape[0], dtype=bool),
        ))
    return _nearest_candidates(points, candidates, validities)


def _torus_body_feet(
    points: np.ndarray,
    major: float,
    minor: float,
    theta: np.ndarray,
) -> np.ndarray:
    directions = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(theta.shape[0])))
    centers = directions * major
    radial_component = np.einsum("ij,ij->i", points, directions) - major
    vertical_component = points[:, 2].copy()
    section_distance = np.hypot(radial_component, vertical_component)
    singular = section_distance <= _SINGULAR_EPSILON
    radial_component[singular] = 1.0
    vertical_component[singular] = 0.0
    section_distance[singular] = 1.0
    scale = minor / section_distance
    return centers + np.column_stack((
        directions[:, 0] * radial_component * scale,
        directions[:, 1] * radial_component * scale,
        vertical_component * scale,
    ))


def _torus_cap_feet(
    points: np.ndarray,
    major: float,
    minor: float,
    theta: float,
) -> np.ndarray:
    direction = np.array((math.cos(theta), math.sin(theta), 0.0), dtype=np.float64)
    center = direction * major
    delta = points - center
    radial_component = delta @ direction
    vertical_component = delta[:, 2]
    disk_distance = np.hypot(radial_component, vertical_component)
    scales = np.ones(points.shape[0], dtype=np.float64)
    outside = disk_distance > minor
    scales[outside] = minor / disk_distance[outside]
    return center + np.column_stack((
        direction[0] * radial_component * scales,
        direction[1] * radial_component * scales,
        vertical_component * scales,
    ))


def _nearest_candidates(
    points: np.ndarray,
    candidates: Sequence[np.ndarray],
    validities: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    packet = np.stack(candidates, axis=0)
    squared = np.sum(np.square(packet - points[None, :, :]), axis=2)
    if validities is not None:
        squared = np.where(np.stack(validities, axis=0), squared, np.inf)
    selected = np.argmin(squared, axis=0)
    return packet[selected, np.arange(points.shape[0])]


def _rotation_matrix(pose: Pose) -> np.ndarray:
    x, y, z, w = pose.orientation_xyzw
    return np.array((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


def _world_to_local_points(pose: Pose, points: np.ndarray) -> np.ndarray:
    return (points - np.asarray(pose.position_m, dtype=np.float64)) @ _rotation_matrix(pose)


def _local_to_world_points(pose: Pose, points: np.ndarray) -> np.ndarray:
    return points @ _rotation_matrix(pose).T + np.asarray(pose.position_m, dtype=np.float64)


def parameter_deltas(
    designed: SceneObject, detected: SceneObject
) -> tuple[tuple[str, float], ...]:
    """Return diagnostic deltas as detected-minus-designed or non-negative separations."""

    if designed.geometry_type is not detected.geometry_type:
        raise ValueError("parameter deltas require matching geometry types")
    mode = designed.geometry_type
    left = designed.parameter_dict
    right = detected.parameter_dict
    center_delta = _distance(designed.pose_world_m.position_m, detected.pose_world_m.position_m)
    designed_axis = _axis(designed.pose_world_m)
    detected_axis = _axis(detected.pose_world_m)
    axis_angle = _undirected_angle(designed_axis, detected_axis)
    if mode is GeometryMode.PLANE:
        return (
            ("width_delta_m", right["width_m"] - left["width_m"]),
            ("height_delta_m", right["height_m"] - left["height_m"]),
            ("center_delta_m", center_delta),
            ("normal_angle_rad", axis_angle),
        )
    if mode is GeometryMode.SPHERE:
        return (
            ("center_delta_m", center_delta),
            ("radius_delta_m", right["radius_m"] - left["radius_m"]),
        )
    if mode is GeometryMode.CYLINDER:
        return (
            ("axis_angle_rad", axis_angle),
            ("center_axis_distance_m", _axis_line_distance(designed, detected)),
            ("radius_delta_m", right["radius_m"] - left["radius_m"]),
            ("length_delta_m", right["length_m"] - left["length_m"]),
        )
    if mode is GeometryMode.CONE:
        designed_bottom, designed_top = _end_centers(designed)
        detected_bottom, detected_top = _end_centers(detected)
        return (
            ("axis_angle_rad", _directed_angle(designed_axis, detected_axis)),
            ("top_center_delta_m", _distance(designed_top, detected_top)),
            ("bottom_center_delta_m", _distance(designed_bottom, detected_bottom)),
            ("top_radius_delta_m", right["top_radius_m"] - left["top_radius_m"]),
            ("bottom_radius_delta_m", right["bottom_radius_m"] - left["bottom_radius_m"]),
            ("length_delta_m", right["length_m"] - left["length_m"]),
        )
    return (
        ("axis_angle_rad", axis_angle),
        ("center_delta_m", center_delta),
        ("major_radius_delta_m", right["major_radius_m"] - left["major_radius_m"]),
        ("minor_radius_delta_m", right["minor_radius_m"] - left["minor_radius_m"]),
        (
            "sweep_start_angle_delta_rad",
            _direction_angle(
                _sweep_direction(designed, "sweep_start_deg"),
                _sweep_direction(detected, "sweep_start_deg"),
            ),
        ),
        (
            "sweep_end_angle_delta_rad",
            _direction_angle(_sweep_end_direction(designed), _sweep_end_direction(detected)),
        ),
        (
            "sweep_extent_delta_rad",
            math.radians(right["sweep_angle_deg"] - left["sweep_angle_deg"]),
        ),
    )


def _sphere_foot(point: Vec3, radius: float) -> Vec3:
    length = _length(point)
    direction = (1.0, 0.0, 0.0) if length <= _SINGULAR_EPSILON else _scale(point, 1.0 / length)
    return _scale(direction, radius)


def _frustum_foot(point: Vec3, bottom_radius: float, top_radius: float, length: float) -> Vec3:
    x, y, z = point
    radial = math.hypot(x, y)
    direction = (1.0, 0.0) if radial <= _SINGULAR_EPSILON else (x / radial, y / radial)
    half = length * 0.5
    candidates: list[Vec3] = []
    side_radial, side_z = _segment_foot_2d(
        (radial, z), (bottom_radius, -half), (top_radius, half)
    )
    candidates.append((direction[0] * side_radial, direction[1] * side_radial, side_z))
    if bottom_radius > 0.0:
        cap_radial = min(radial, bottom_radius)
        candidates.append((direction[0] * cap_radial, direction[1] * cap_radial, -half))
    if top_radius > 0.0:
        cap_radial = min(radial, top_radius)
        candidates.append((direction[0] * cap_radial, direction[1] * cap_radial, half))
    return _nearest_candidate(point, candidates)


def _torus_foot(point: Vec3, major: float, minor: float, start: float, end: float) -> Vec3:
    x, y, z = point
    radial = math.hypot(x, y)
    alpha = math.atan2(y, x) if radial > _SINGULAR_EPSILON else start
    angles = [start, end]
    _append_periodic_angles(angles, alpha, start, end)
    candidates = [_torus_body_foot(point, major, minor, theta) for theta in angles]
    if end - start < math.tau - 1e-10:
        candidates.append(_torus_cap_foot(point, major, minor, start))
        candidates.append(_torus_cap_foot(point, major, minor, end))
    return _nearest_candidate(point, candidates)


def _torus_body_foot(point: Vec3, major: float, minor: float, theta: float) -> Vec3:
    radial_direction = (math.cos(theta), math.sin(theta), 0.0)
    center = _scale(radial_direction, major)
    radial_component = _dot(point, radial_direction) - major
    vertical_component = point[2]
    section_distance = math.hypot(radial_component, vertical_component)
    if section_distance <= _SINGULAR_EPSILON:
        radial_component, vertical_component, section_distance = 1.0, 0.0, 1.0
    return _add(
        center,
        (
            radial_direction[0] * minor * radial_component / section_distance,
            radial_direction[1] * minor * radial_component / section_distance,
            minor * vertical_component / section_distance,
        ),
    )


def _torus_cap_foot(point: Vec3, major: float, minor: float, theta: float) -> Vec3:
    radial_direction = (math.cos(theta), math.sin(theta), 0.0)
    center = _scale(radial_direction, major)
    delta = _sub(point, center)
    radial_component = _dot(delta, radial_direction)
    vertical_component = delta[2]
    disk_distance = math.hypot(radial_component, vertical_component)
    scale = 1.0 if disk_distance <= minor else minor / disk_distance
    return _add(
        center,
        (
            radial_direction[0] * radial_component * scale,
            radial_direction[1] * radial_component * scale,
            vertical_component * scale,
        ),
    )


def _append_periodic_angles(values: list[float], base: float, start: float, end: float) -> None:
    first = math.ceil((start - base) / math.tau - 1e-12)
    last = math.floor((end - base) / math.tau + 1e-12)
    for period in range(first, last + 1):
        candidate = base + period * math.tau
        if start - 1e-12 <= candidate <= end + 1e-12:
            values.append(candidate)


def _segment_foot_2d(point, start, end):
    delta = (end[0] - start[0], end[1] - start[1])
    length_squared = delta[0] * delta[0] + delta[1] * delta[1]
    fraction = (
        (point[0] - start[0]) * delta[0] + (point[1] - start[1]) * delta[1]
    ) / length_squared
    fraction = _clamp(fraction, 0.0, 1.0)
    return start[0] + fraction * delta[0], start[1] + fraction * delta[1]


def _nearest_candidate(point: Vec3, candidates: Sequence[Vec3]) -> Vec3:
    return min(candidates, key=lambda candidate: _distance_squared(point, candidate))


def _axis_line_distance(designed: SceneObject, detected: SceneObject) -> float:
    left_axis = _axis(designed.pose_world_m)
    right_axis = _axis(detected.pose_world_m)
    center_delta = _sub(detected.pose_world_m.position_m, designed.pose_world_m.position_m)
    cross = cross_vec3(left_axis, right_axis)
    cross_length = _length(cross)
    if cross_length <= _SINGULAR_EPSILON:
        return _length(cross_vec3(center_delta, left_axis))
    return abs(_dot(center_delta, _scale(cross, 1.0 / cross_length)))


def _end_centers(surface: SceneObject) -> tuple[Vec3, Vec3]:
    half = surface.parameter_dict["length_m"] * 0.5
    return (
        _local_to_world(surface.pose_world_m, (0.0, 0.0, -half)),
        _local_to_world(surface.pose_world_m, (0.0, 0.0, half)),
    )


def _sweep_direction(surface: SceneObject, parameter: str) -> Vec3:
    angle = math.radians(surface.parameter_dict[parameter])
    return rotate_vector_xyzw(surface.pose_world_m.orientation_xyzw, (math.cos(angle), math.sin(angle), 0.0))


def _sweep_end_direction(surface: SceneObject) -> Vec3:
    values = surface.parameter_dict
    angle = math.radians(values["sweep_start_deg"] + values["sweep_angle_deg"])
    return rotate_vector_xyzw(surface.pose_world_m.orientation_xyzw, (math.cos(angle), math.sin(angle), 0.0))


def _axis(pose: Pose) -> Vec3:
    return rotate_vector_xyzw(pose.orientation_xyzw, (0.0, 0.0, 1.0))


def _undirected_angle(left: Vec3, right: Vec3) -> float:
    return math.acos(_clamp(abs(_dot(left, right)), -1.0, 1.0))


def _directed_angle(left: Vec3, right: Vec3) -> float:
    return math.acos(_clamp(_dot(left, right), -1.0, 1.0))


def _direction_angle(left: Vec3, right: Vec3) -> float:
    return _directed_angle(left, right)


def _world_to_local(pose: Pose, point: Vec3) -> Vec3:
    x, y, z, w = pose.orientation_xyzw
    return rotate_vector_xyzw((-x, -y, -z, w), _sub(point, pose.position_m))


def _local_to_world(pose: Pose, point: Vec3) -> Vec3:
    return _add(rotate_vector_xyzw(pose.orientation_xyzw, point), pose.position_m)


def _vec3(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("point must contain exactly three values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError("point must contain only finite values")
    return result  # type: ignore[return-value]


def _unavailable(reason: str) -> SurfaceMatchResult:
    return SurfaceMatchResult(False, None, 0, unavailable_reason=reason)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _distance(left: Vec3, right: Vec3) -> float:
    return math.sqrt(_distance_squared(left, right))


def _distance_squared(left: Vec3, right: Vec3) -> float:
    return math.fsum((left[index] - right[index]) ** 2 for index in range(3))


def _length(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _dot(left: Vec3, right: Vec3) -> float:
    return math.fsum(left[index] * right[index] for index in range(3))


def _add(left: Vec3, right: Vec3) -> Vec3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _sub(left: Vec3, right: Vec3) -> Vec3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _scale(value: Vec3, scale: float) -> Vec3:
    return tuple(component * scale for component in value)  # type: ignore[return-value]


__all__ = [
    "SURFACE_SIDE_BOUNDARY_TOLERANCE_M",
    "SurfaceMatchResult",
    "compare_surfaces",
    "nearest_foot",
    "parameter_deltas",
    "signed_surface_distance",
    "signed_surface_distances",
    "surface_side_distances",
]
