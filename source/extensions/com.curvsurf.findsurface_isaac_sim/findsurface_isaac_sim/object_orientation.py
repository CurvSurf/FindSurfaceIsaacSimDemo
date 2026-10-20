"""Pure orientation helpers for the editable as-designed object controls."""

from __future__ import annotations

import math

from .spatial_math import cross_vec3, quat_multiply_xyzw, rotate_vector_unit_xyzw


Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def orientation_axes(quaternion: Quaternion) -> tuple[Vec3, Vec3, Vec3]:
    """Return the object's local X/Y/Z axes expressed in world coordinates."""

    return tuple(_rotate(quaternion, axis) for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def corrected_orientation(
    axes: tuple[Vec3, Vec3, Vec3],
    previous: Quaternion,
    *,
    edited_axis: str,
) -> tuple[Quaternion, tuple[Vec3, Vec3, Vec3]]:
    """Normalize and right-handedly orthogonalize an edited world-space basis.

    The edited row is treated as the primary direction. Degenerate or parallel
    input falls back to the corresponding previous valid direction.
    """

    old_x, old_y, old_z = orientation_axes(previous)
    raw_x, raw_y, raw_z = (_vec3(value) for value in axes)
    if edited_axis == "x":
        x = _unit_or(raw_x, old_x)
        y = _orthogonal_or(raw_y, x, old_y, old_z)
        z = _normalize(cross_vec3(x, y))
        y = _normalize(cross_vec3(z, x))
    elif edited_axis == "y":
        y = _unit_or(raw_y, old_y)
        x = _orthogonal_or(raw_x, y, old_x, old_z)
        z = _normalize(cross_vec3(x, y))
        x = _normalize(cross_vec3(y, z))
    elif edited_axis == "z":
        z = _unit_or(raw_z, old_z)
        x = _orthogonal_or(raw_x, z, old_x, old_y)
        y = _normalize(cross_vec3(z, x))
        x = _normalize(cross_vec3(y, z))
    else:
        raise ValueError("edited_axis must be x, y, or z")
    result = (x, y, z)
    return _quaternion_from_axes(result), result


def rotate_in_view(
    orientation: Quaternion,
    *,
    camera_right: Vec3,
    camera_up: Vec3,
    drag_x: float,
    drag_y: float,
    radians_per_pad: float = math.pi,
) -> Quaternion:
    """Rotate an object in the screen-space direction of a 2-D pad drag."""

    yaw = _axis_angle(camera_up, float(drag_x) * radians_per_pad)
    pitch = _axis_angle(camera_right, float(drag_y) * radians_per_pad)
    return _normalized_quaternion(quat_multiply_xyzw(pitch, quat_multiply_xyzw(yaw, orientation)))


def screen_roll_radians(previous: tuple[float, float], current: tuple[float, float], center: tuple[float, float]) -> float:
    """Return the signed scene-view-style roll between two pointer positions."""

    a = (float(previous[0]) - center[0], center[1] - float(previous[1]))
    b = (float(current[0]) - center[0], center[1] - float(current[1]))
    if math.hypot(*a) <= 1.0e-9 or math.hypot(*b) <= 1.0e-9:
        return 0.0
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def roll_in_view(orientation: Quaternion, *, camera_forward: Vec3, radians: float) -> Quaternion:
    """Roll an object around the current view axis, matching scene-camera RMB drag."""

    return _normalized_quaternion(
        quat_multiply_xyzw(_axis_angle(camera_forward, float(radians)), orientation)
    )


def _quaternion_from_axes(axes: tuple[Vec3, Vec3, Vec3]) -> Quaternion:
    x, y, z = axes
    m00, m10, m20 = x
    m01, m11, m21 = y
    m02, m12, m22 = z
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    return _normalized_quaternion(q)


def _orthogonal_or(value: Vec3, primary: Vec3, *fallbacks: Vec3) -> Vec3:
    for candidate in (value, *fallbacks, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        projected = _sub(candidate, _scale(primary, _dot(candidate, primary)))
        if _length(projected) > 1.0e-8:
            return _normalize(projected)
    raise ValueError("could not construct an orthogonal direction")


def _unit_or(value: Vec3, fallback: Vec3) -> Vec3:
    return _normalize(value) if _length(value) > 1.0e-8 else _normalize(fallback)


def _axis_angle(axis: Vec3, radians: float) -> Quaternion:
    axis = _normalize(axis)
    half = radians * 0.5
    sine = math.sin(half)
    return axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half)


def _rotate(quaternion: Quaternion, vector: Vec3) -> Vec3:
    return rotate_vector_unit_xyzw(_normalized_quaternion(quaternion), vector)


def _normalized_quaternion(value: Quaternion) -> Quaternion:
    values = tuple(float(component) for component in value)
    magnitude = math.sqrt(sum(component * component for component in values))
    if not math.isfinite(magnitude) or magnitude <= 1.0e-12:
        raise ValueError("orientation quaternion must be finite and non-zero")
    return tuple(component / magnitude for component in values)


def _vec3(value) -> Vec3:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError("direction must contain three finite values")
    return result


def _sub(left: Vec3, right: Vec3) -> Vec3:
    return tuple(a - b for a, b in zip(left, right))


def _scale(value: Vec3, scale: float) -> Vec3:
    return tuple(component * scale for component in value)


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _length(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: Vec3) -> Vec3:
    length = _length(value)
    if not math.isfinite(length) or length <= 1.0e-12:
        raise ValueError("direction must be finite and non-zero")
    return _scale(value, 1.0 / length)
