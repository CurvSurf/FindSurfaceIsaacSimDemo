"""Right-handed scalar leaf math; policy and validation belong to callers."""

from __future__ import annotations


Vec3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def quat_multiply_xyzw(left: QuaternionXYZW, right: QuaternionXYZW) -> QuaternionXYZW:
    """Hamilton product left * right, without normalization.

    For unit inputs, composition applies right first, then left. Keep scalar
    evaluation order stable for mesh, camera and metric consumers.
    """
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quat_conjugate_xyzw(value: QuaternionXYZW) -> QuaternionXYZW:
    return -value[0], -value[1], -value[2], value[3]


def cross_vec3(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def rotate_vector_xyzw(rotation: QuaternionXYZW, value: Vec3) -> Vec3:
    """Evaluate q * (v,0) * conjugate(q), preserving the existing mesh formula.

    No validation, normalization or inverse division: non-unit q scales the
    vector by norm(q)**2. Do not replace this with the camera shortcut below.
    """
    vector = (value[0], value[1], value[2], 0.0)
    conjugate = quat_conjugate_xyzw(rotation)
    result = quat_multiply_xyzw(quat_multiply_xyzw(rotation, vector), conjugate)
    return result[0], result[1], result[2]


def rotate_vector_unit_xyzw(quaternion: QuaternionXYZW, vector: Vec3) -> Vec3:
    """Existing camera shortcut; caller owns the unit-quaternion precondition.

    Intentionally does not normalize or change non-unit input behavior. Its
    arithmetic/rounding differs from the conjugate-sandwich formula above.
    """
    qx, qy, qz, qw = quaternion
    qv = (qx, qy, qz)
    term = _scale_vec3(cross_vec3(qv, vector), 2.0)
    return _add_vec3(vector, _add_vec3(_scale_vec3(term, qw), cross_vec3(qv, term)))


def _add_vec3(left: Vec3, right: Vec3) -> Vec3:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _scale_vec3(value: Vec3, scale: float) -> Vec3:
    return value[0] * scale, value[1] * scale, value[2] * scale
