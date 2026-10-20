"""Immutable perspective-camera math shared by OpenGL, input, and picking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Callable

from .spatial_math import cross_vec3, quat_multiply_xyzw as quat_multiply, rotate_vector_unit_xyzw
from .domain import Pose
from .transactions import CommandScope


EPSILON = 1.0e-9
Vec3 = tuple[float, float, float]


class CameraRole(str, Enum):
    SCANNER = "scanner"
    SCENE = "scene"


class OrbitMode(str, Enum):
    TURNTABLE = "turntable"
    TRACKBALL = "trackball"


class ScenePreset(str, Enum):
    POSITIVE_X = "+X"
    POSITIVE_Y = "+Y"
    POSITIVE_Z = "+Z"


@dataclass(frozen=True)
class CameraIntrinsics:
    vertical_fov_degrees: float = 45.0
    aspect_ratio: float = 16.0 / 9.0
    near_m: float = 0.01
    far_m: float = 1000.0
    orthographic_height_m: float | None = None

    def __post_init__(self) -> None:
        if self.orthographic_height_m is not None:
            extent = float(self.orthographic_height_m)
            if not math.isfinite(extent) or extent <= 0:
                raise ValueError("orthographic height must be finite and positive")
            object.__setattr__(self, "orthographic_height_m", extent)
        values = tuple(float(value) for value in (
            self.vertical_fov_degrees, self.aspect_ratio, self.near_m, self.far_m
        ))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if not 1.0 <= values[0] <= 160.0:
            raise ValueError("vertical_fov_degrees must be in [1, 160]")
        if values[1] <= 0.0 or values[2] <= 0.0 or values[3] <= values[2]:
            raise ValueError("aspect and clip planes must be positive and ordered")
        object.__setattr__(self, "vertical_fov_degrees", values[0])
        object.__setattr__(self, "aspect_ratio", values[1])
        object.__setattr__(self, "near_m", values[2])
        object.__setattr__(self, "far_m", values[3])

    @property
    def horizontal_fov_radians(self) -> float:
        half_y = math.radians(self.vertical_fov_degrees) * 0.5
        return 2.0 * math.atan(math.tan(half_y) * self.aspect_ratio)

    @property
    def vertical_fov_radians(self) -> float:
        return math.radians(self.vertical_fov_degrees)

    @classmethod
    def from_centered_pinhole(
        cls,
        width_px: int,
        height_px: int,
        focal_x_px: float,
        focal_y_px: float,
        *,
        near_m: float = 0.01,
        far_m: float = 1000.0,
    ) -> "CameraIntrinsics":
        """Preserve a centered sensor's X/Y focal calibration in the GL projection."""

        width = int(width_px)
        height = int(height_px)
        focal_x = float(focal_x_px)
        focal_y = float(focal_y_px)
        if width <= 0 or height <= 0:
            raise ValueError("pinhole resolution must be positive")
        if not all(math.isfinite(value) and value > 0.0 for value in (focal_x, focal_y)):
            raise ValueError("pinhole focal lengths must be finite and positive")
        vertical_fov = math.degrees(2.0 * math.atan(height / (2.0 * focal_y)))
        projection_aspect = (width / focal_x) / (height / focal_y)
        return cls(vertical_fov, projection_aspect, near_m, far_m)


@dataclass(frozen=True)
class CameraSnapshot:
    role: CameraRole
    pose_world: Pose
    intrinsics: CameraIntrinsics
    target_world_m: Vec3 = (0.0, 0.0, 0.0)
    orbit_mode: OrbitMode = OrbitMode.TRACKBALL

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", CameraRole(self.role))
        if self.role is CameraRole.SCANNER and self.intrinsics.orthographic_height_m is not None:
            raise ValueError("scanner must remain perspective")
        object.__setattr__(self, "orbit_mode", OrbitMode(self.orbit_mode))
        object.__setattr__(self, "target_world_m", _vec3(self.target_world_m))

    @property
    def forward(self) -> Vec3:
        return _normalize(rotate_vector_unit_xyzw(self.pose_world.orientation_xyzw, (0.0, 0.0, -1.0)))

    @property
    def right(self) -> Vec3:
        return _normalize(rotate_vector_unit_xyzw(self.pose_world.orientation_xyzw, (1.0, 0.0, 0.0)))

    @property
    def up(self) -> Vec3:
        return _normalize(rotate_vector_unit_xyzw(self.pose_world.orientation_xyzw, (0.0, 1.0, 0.0)))


@dataclass(frozen=True)
class CameraSnapshotCommand:
    """One durable camera gesture represented by immutable before/after values."""

    name: str
    sink: Callable[[CameraSnapshot], None]
    before: CameraSnapshot
    after: CameraSnapshot
    scope: CommandScope = CommandScope.DURABLE

    def apply(self) -> None:
        self.sink(self.after)

    def undo(self) -> None:
        self.sink(self.before)


def aligned_orbit_target(pose_world: Pose, reference_world_m: Vec3) -> Vec3:
    """Derive a legacy orbit target without changing the stored view direction.

    Older app-default files stored only pose.  Using the scene bounds center
    directly as their target makes trackball orbit jump whenever that center is
    not exactly on the stored camera forward ray.  Preserve the approximate
    orbit distance to the reference while projecting the target onto that ray.
    """

    reference = _vec3(reference_world_m)
    position = pose_world.position_m
    distance = _length(_sub(reference, position))
    if distance <= EPSILON:
        distance = 1.0
    forward = _normalize(rotate_vector_unit_xyzw(
        pose_world.orientation_xyzw, (0.0, 0.0, -1.0),
    ))
    return _add(position, _mul(forward, distance))


def initial_camera(
    role: CameraRole,
    center_world_m: Vec3,
    radius_m: float,
    intrinsics: CameraIntrinsics,
    *,
    margin: float = 1.05,
) -> CameraSnapshot:
    """Fit a bounding sphere using the role-specific FOV rule from the spec."""

    role = CameraRole(role)
    center = _vec3(center_world_m)
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    half_angles = (intrinsics.horizontal_fov_radians * 0.5, intrinsics.vertical_fov_radians * 0.5)
    # Scene guarantees the sphere in the narrower axis. Scanner follows the
    # explicit wider-axis fit contract because its calibrated frame is primary.
    fit_half_angle = min(half_angles) if role is CameraRole.SCENE else max(half_angles)
    distance = radius * margin / max(math.sin(fit_half_angle), EPSILON)
    direction_from_target = _normalize((1.0, 1.0, 1.0)) if role is CameraRole.SCENE else (0.0, 0.0, 1.0)
    position = _add(center, _mul(direction_from_target, distance))
    orientation = look_orientation(_mul(direction_from_target, -1.0), (0.0, 1.0, 0.0))
    near_m = max(1.0e-4, distance - radius * margin * 1.25)
    far_m = max(near_m + 1.0, distance + radius * margin * 4.0)
    return CameraSnapshot(
        role,
        Pose(position, orientation),
        replace(intrinsics, near_m=near_m, far_m=far_m),
        center,
    )


def scene_preset(camera: CameraSnapshot, preset: ScenePreset) -> CameraSnapshot:
    if camera.role is not CameraRole.SCENE:
        raise ValueError("scene presets apply only to the scene camera")
    preset = ScenePreset(preset)
    direction, up = {
        ScenePreset.POSITIVE_X: ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ScenePreset.POSITIVE_Y: ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        ScenePreset.POSITIVE_Z: ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    }[preset]
    distance = _length(_sub(camera.pose_world.position_m, camera.target_world_m))
    position = _sub(camera.target_world_m, _mul(direction, distance))
    return replace(camera, pose_world=Pose(position, look_orientation(direction, up)))


def look_orientation(forward: Vec3, up_hint: Vec3) -> tuple[float, float, float, float]:
    forward = _normalize(forward)
    right = _normalize(cross_vec3(forward, _normalize(up_hint)))
    up = _normalize(cross_vec3(right, forward))
    # Columns are local +X, +Y, +Z; camera forward is local -Z.
    return _quat_from_matrix((
        right[0], up[0], -forward[0],
        right[1], up[1], -forward[1],
        right[2], up[2], -forward[2],
    ))


def view_matrix(camera: CameraSnapshot) -> tuple[float, ...]:
    right, up, backward = camera.right, camera.up, _mul(camera.forward, -1.0)
    position = camera.pose_world.position_m
    return (
        right[0], up[0], backward[0], 0.0,
        right[1], up[1], backward[1], 0.0,
        right[2], up[2], backward[2], 0.0,
        -_dot(right, position), -_dot(up, position), -_dot(backward, position), 1.0,
    )


def projection_matrix(intrinsics: CameraIntrinsics) -> tuple[float, ...]:
    if intrinsics.orthographic_height_m is not None:
        from .orthographic import projection_matrix as orthographic_matrix
        return orthographic_matrix(intrinsics.orthographic_height_m, intrinsics.aspect_ratio,
                                   intrinsics.near_m, intrinsics.far_m)
    near, far = intrinsics.near_m, intrinsics.far_m
    scale_y = 1.0 / math.tan(intrinsics.vertical_fov_radians * 0.5)
    scale_x = scale_y / intrinsics.aspect_ratio
    return (
        scale_x, 0.0, 0.0, 0.0,
        0.0, scale_y, 0.0, 0.0,
        0.0, 0.0, -(far + near) / (far - near), -1.0,
        0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0,
    )


def view_projection_matrix(camera: CameraSnapshot) -> tuple[float, ...]:
    return _mat4_multiply(projection_matrix(camera.intrinsics), view_matrix(camera))


def perspective_ray(
    camera: CameraSnapshot,
    logical_x: float,
    logical_y: float,
    logical_size: tuple[float, float],
) -> tuple[Vec3, Vec3]:
    """Construct a top-left-origin ray without rewriting scanner intrinsics."""

    if camera.intrinsics.orthographic_height_m is not None:
        from .orthographic import screen_ray
        return screen_ray(camera, logical_x, logical_y, logical_size,
                          camera.intrinsics.orthographic_height_m)

    width, height = (float(logical_size[0]), float(logical_size[1]))
    if width <= 0.0 or height <= 0.0:
        raise ValueError("logical_size must be positive")
    nx = 2.0 * float(logical_x) / width - 1.0
    ny = 1.0 - 2.0 * float(logical_y) / height
    tangent = math.tan(camera.intrinsics.vertical_fov_radians * 0.5)
    local = _normalize((nx * camera.intrinsics.aspect_ratio * tangent, ny * tangent, -1.0))
    return camera.pose_world.position_m, _normalize(rotate_vector_unit_xyzw(camera.pose_world.orientation_xyzw, local))


def with_viewport_aspect(camera: CameraSnapshot, aspect_ratio: float) -> CameraSnapshot:
    """Return the camera projection used for one live viewport presentation."""

    aspect = float(aspect_ratio)
    if not math.isfinite(aspect) or aspect <= 0.0:
        raise ValueError("viewport aspect ratio must be finite and positive")
    if camera.intrinsics.aspect_ratio == aspect:
        return camera
    return replace(camera, intrinsics=replace(camera.intrinsics, aspect_ratio=aspect))


def _mat4_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    # Column-major product for glUniformMatrix4fv(..., GL_FALSE, ...).
    return tuple(
        sum(left[row + inner * 4] * right[inner + column * 4] for inner in range(4))
        for column in range(4)
        for row in range(4)
    )


def _quat_from_matrix(matrix: tuple[float, ...]) -> tuple[float, float, float, float]:
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    magnitude = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / magnitude for value in quaternion)


def quat_axis_angle(axis: Vec3, radians: float) -> tuple[float, float, float, float]:
    axis = _normalize(axis)
    half = float(radians) * 0.5
    sine = math.sin(half)
    return axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half)


def _vec3(value) -> Vec3:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError("expected a finite vec3")
    return result


def _add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _mul(value: Vec3, scale: float) -> Vec3:
    return value[0] * scale, value[1] * scale, value[2] * scale


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(left * right for left, right in zip(a, b))


def _length(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: Vec3) -> Vec3:
    length = _length(value)
    if length <= EPSILON:
        raise ValueError("vector must have non-zero length")
    return _mul(value, 1.0 / length)


__all__ = (
    "CameraIntrinsics", "CameraRole", "CameraSnapshot", "CameraSnapshotCommand",
    "OrbitMode", "ScenePreset", "initial_camera", "look_orientation",
    "perspective_ray", "projection_matrix", "quat_axis_angle", "quat_multiply",
    "scene_preset", "view_matrix", "view_projection_matrix", "with_viewport_aspect",
)
