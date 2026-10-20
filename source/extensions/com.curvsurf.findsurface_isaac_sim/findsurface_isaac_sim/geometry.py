"""Deterministic analytic-object meshes shared by USD and OpenGL adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Mapping

from .spatial_math import rotate_vector_xyzw
from .domain import GeometryMode, Pose
from .scene import DEFAULT_OBJECT_SUBDIVISION, SceneObject


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class MeshResolution:
    plane_segments: int = DEFAULT_OBJECT_SUBDIVISION
    radial_segments: int = DEFAULT_OBJECT_SUBDIVISION
    sphere_latitude_segments: int = DEFAULT_OBJECT_SUBDIVISION // 2
    torus_major_segments: int = DEFAULT_OBJECT_SUBDIVISION
    torus_minor_segments: int = DEFAULT_OBJECT_SUBDIVISION // 2

    def __post_init__(self) -> None:
        if self.plane_segments < 1:
            raise ValueError("plane_segments must be at least 1")
        if self.radial_segments < 3:
            raise ValueError("radial_segments must be at least 3")
        if self.sphere_latitude_segments < 3:
            raise ValueError("sphere_latitude_segments must be at least 3")
        if self.torus_major_segments < 3 or self.torus_minor_segments < 3:
            raise ValueError("torus segment counts must be at least 3")


DEFAULT_MESH_RESOLUTION = MeshResolution()


def mesh_resolution_for_subdivision(subdivision: int) -> MeshResolution:
    """Map one user-facing quality value to each primitive's two-dimensional grid."""

    value = int(subdivision)
    return MeshResolution(
        plane_segments=value,
        radial_segments=value,
        sphere_latitude_segments=max(3, value // 2),
        torus_major_segments=value,
        torus_minor_segments=max(3, value // 2),
    )


@dataclass(frozen=True)
class MeshData:
    vertices: tuple[Vec3, ...]
    normals: tuple[Vec3, ...]
    triangle_indices: tuple[int, ...]
    edge_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.vertices or len(self.vertices) != len(self.normals):
            raise ValueError("mesh vertices and normals must be non-empty and aligned")
        if len(self.triangle_indices) % 3 or len(self.edge_indices) % 2:
            raise ValueError("mesh index buffers have invalid arity")
        limit = len(self.vertices)
        if any(index < 0 or index >= limit for index in self.triangle_indices + self.edge_indices):
            raise ValueError("mesh index is outside the vertex buffer")
        if not all(math.isfinite(value) for vector in self.vertices + self.normals for value in vector):
            raise ValueError("mesh values must be finite")
        for normal in self.normals:
            if abs(_length(normal) - 1.0) > 1e-9:
                raise ValueError("mesh normals must be unit length")

    @property
    def triangle_count(self) -> int:
        return len(self.triangle_indices) // 3

    @property
    def bounds(self) -> tuple[Vec3, Vec3]:
        return (
            tuple(min(point[axis] for point in self.vertices) for axis in range(3)),
            tuple(max(point[axis] for point in self.vertices) for axis in range(3)),
        )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            struct.pack(
                "<4I",
                len(self.vertices),
                len(self.normals),
                len(self.triangle_indices),
                len(self.edge_indices),
            )
        )
        for vector in self.vertices + self.normals:
            digest.update(struct.pack("<3d", *vector))
        for index in self.triangle_indices + self.edge_indices:
            digest.update(struct.pack("<I", index))
        return digest.hexdigest()

    def transformed(self, pose: Pose) -> "MeshData":
        return MeshData(
            tuple(_add(rotate_vector_xyzw(pose.orientation_xyzw, point), pose.position_m) for point in self.vertices),
            tuple(_normalize(rotate_vector_xyzw(pose.orientation_xyzw, normal)) for normal in self.normals),
            self.triangle_indices,
            self.edge_indices,
        )


def build_mesh(
    item: SceneObject, resolution: MeshResolution | None = None,
) -> MeshData:
    resolution = resolution or mesh_resolution_for_subdivision(item.subdivision)
    values = item.parameter_dict
    if item.geometry_type is GeometryMode.PLANE:
        return _plane(values["width_m"], values["height_m"], resolution.plane_segments)
    if item.geometry_type is GeometryMode.SPHERE:
        return _sphere(
            values["radius_m"],
            resolution.radial_segments,
            resolution.sphere_latitude_segments,
        )
    if item.geometry_type is GeometryMode.CYLINDER:
        return _frustum(
            values["radius_m"],
            values["radius_m"],
            values["length_m"],
            resolution.radial_segments,
            axial_divisions=3,
        )
    if item.geometry_type is GeometryMode.CONE:
        return _frustum(
            values["bottom_radius_m"],
            values["top_radius_m"],
            values["length_m"],
            resolution.radial_segments,
            axial_divisions=2,
        )
    return _torus(
        values["major_radius_m"],
        values["minor_radius_m"],
        values["sweep_start_deg"],
        values["sweep_angle_deg"],
        resolution.torus_major_segments,
        resolution.torus_minor_segments,
    )


def build_parameter_mesh(
    geometry_type: GeometryMode | str,
    parameters: Mapping[str, Any],
    resolution: MeshResolution | None = None,
) -> MeshData:
    """Build from normalized parameters for detected as well as designed geometry."""

    return build_mesh(SceneObject(geometry_type, True, Pose(), parameters), resolution)


def _plane(width: float, height: float, segments: int) -> MeshData:
    x, y = width * 0.5, height * 0.5
    stride = segments + 1
    vertices = tuple(
        (-x + width * column / segments, -y + height * row / segments, 0.0)
        for row in range(stride)
        for column in range(stride)
    )
    triangles: list[int] = []
    for row in range(segments):
        for column in range(segments):
            lower_left = row * stride + column
            lower_right = lower_left + 1
            upper_left = lower_left + stride
            upper_right = upper_left + 1
            triangles.extend((lower_left, lower_right, upper_right, lower_left, upper_right, upper_left))
    edges: list[int] = []
    for row in range(stride):
        for column in range(segments):
            start = row * stride + column
            edges.extend((start, start + 1))
    for column in range(stride):
        for row in range(segments):
            start = row * stride + column
            edges.extend((start, start + stride))
    return MeshData(vertices, ((0.0, 0.0, 1.0),) * len(vertices), tuple(triangles), tuple(edges))


def _sphere(radius: float, radial: int, latitudes: int) -> MeshData:
    vertices: list[Vec3] = [(0.0, 0.0, -radius)]
    normals: list[Vec3] = [(0.0, 0.0, -1.0)]
    for latitude_index in range(1, latitudes):
        latitude = -math.pi * 0.5 + math.pi * latitude_index / latitudes
        ring = radius * math.cos(latitude)
        z = radius * math.sin(latitude)
        for radial_index in range(radial):
            angle = math.tau * radial_index / radial
            normal = (math.cos(angle) * math.cos(latitude), math.sin(angle) * math.cos(latitude), math.sin(latitude))
            vertices.append((normal[0] * radius, normal[1] * radius, z))
            normals.append(_normalize(normal))
    top = len(vertices)
    vertices.append((0.0, 0.0, radius))
    normals.append((0.0, 0.0, 1.0))
    triangles: list[int] = []
    first_ring = 1
    for segment in range(radial):
        next_segment = (segment + 1) % radial
        triangles.extend((0, first_ring + next_segment, first_ring + segment))
    ring_count = latitudes - 1
    for ring in range(ring_count - 1):
        lower = 1 + ring * radial
        upper = lower + radial
        for segment in range(radial):
            next_segment = (segment + 1) % radial
            triangles.extend((lower + segment, lower + next_segment, upper + next_segment))
            triangles.extend((lower + segment, upper + next_segment, upper + segment))
    last_ring = 1 + (ring_count - 1) * radial
    for segment in range(radial):
        next_segment = (segment + 1) % radial
        triangles.extend((top, last_ring + segment, last_ring + next_segment))
    edges: list[int] = []
    for ring in range(ring_count):
        offset = 1 + ring * radial
        _append_ring_edges(edges, offset, radial)
    for segment in range(radial):
        edges.extend((0, first_ring + segment))
        for ring in range(ring_count - 1):
            edges.extend((1 + ring * radial + segment, 1 + (ring + 1) * radial + segment))
        edges.extend((last_ring + segment, top))
    return MeshData(tuple(vertices), tuple(normals), tuple(triangles), tuple(edges))


def _frustum(
    bottom_radius: float, top_radius: float, length: float, radial: int,
    *, axial_divisions: int,
) -> MeshData:
    half = length * 0.5
    slope = (top_radius - bottom_radius) / length
    vertices: list[Vec3] = []
    normals: list[Vec3] = []
    triangles: list[int] = []
    edges: list[int] = []

    if bottom_radius == 0.0:
        apex = len(vertices)
        vertices.append((0.0, 0.0, -half))
        normals.append((0.0, 0.0, -1.0))
        top_side = _append_side_ring(vertices, normals, top_radius, half, radial, slope)
        for segment in range(radial):
            nxt = (segment + 1) % radial
            triangles.extend((apex, top_side + nxt, top_side + segment))
            edges.extend((apex, top_side + segment))
        _append_ring_edges(edges, top_side, radial)
    else:
        bottom_side = _append_side_ring(vertices, normals, bottom_radius, -half, radial, slope)
        if top_radius > 0.0:
            top_side = _append_side_ring(vertices, normals, top_radius, half, radial, slope)
            for segment in range(radial):
                nxt = (segment + 1) % radial
                triangles.extend((bottom_side + segment, bottom_side + nxt, top_side + nxt))
                triangles.extend((bottom_side + segment, top_side + nxt, top_side + segment))
                edges.extend((bottom_side + segment, top_side + segment))
            _append_ring_edges(edges, top_side, radial)
        else:
            apex = len(vertices)
            vertices.append((0.0, 0.0, half))
            normals.append((0.0, 0.0, 1.0))
            for segment in range(radial):
                nxt = (segment + 1) % radial
                triangles.extend((bottom_side + segment, bottom_side + nxt, apex))
                edges.extend((bottom_side + segment, apex))
        _append_ring_edges(edges, bottom_side, radial)
    if bottom_radius > 0.0:
        _append_cap(vertices, normals, triangles, bottom_radius, -half, radial, top=False)
    if top_radius > 0.0:
        _append_cap(vertices, normals, triangles, top_radius, half, radial, top=True)
    for division in range(1, axial_divisions):
        fraction = division / axial_divisions
        ring = _append_side_ring(
            vertices, normals,
            bottom_radius + (top_radius - bottom_radius) * fraction,
            -half + length * fraction, radial, slope,
        )
        _append_ring_edges(edges, ring, radial)
    return MeshData(tuple(vertices), tuple(normals), tuple(triangles), tuple(edges))


def _append_side_ring(vertices, normals, radius, z, radial, slope) -> int:
    offset = len(vertices)
    for segment in range(radial):
        angle = math.tau * segment / radial
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        normals.append(_normalize((math.cos(angle), math.sin(angle), -slope)))
    return offset


def _append_cap(vertices, normals, triangles, radius, z, radial, *, top) -> None:
    normal = (0.0, 0.0, 1.0 if top else -1.0)
    center = len(vertices)
    vertices.append((0.0, 0.0, z))
    normals.append(normal)
    ring = len(vertices)
    for segment in range(radial):
        angle = math.tau * segment / radial
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        normals.append(normal)
    for segment in range(radial):
        nxt = (segment + 1) % radial
        triangles.extend((center, ring + segment, ring + nxt) if top else (center, ring + nxt, ring + segment))


def _torus(major_radius, minor_radius, start_deg, sweep_deg, major_segments, minor_segments) -> MeshData:
    full = abs(sweep_deg - 360.0) <= 1e-10
    segment_count = major_segments if full else max(1, math.ceil(major_segments * sweep_deg / 360.0))
    ring_count = segment_count if full else segment_count + 1
    vertices: list[Vec3] = []
    normals: list[Vec3] = []
    triangles: list[int] = []
    edges: list[int] = []
    for major_index in range(ring_count):
        fraction = major_index / segment_count
        theta = math.radians(start_deg + sweep_deg * fraction)
        for minor_index in range(minor_segments):
            phi = math.tau * minor_index / minor_segments
            normal = (math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), math.sin(phi))
            radius = major_radius + minor_radius * math.cos(phi)
            vertices.append((radius * math.cos(theta), radius * math.sin(theta), minor_radius * math.sin(phi)))
            normals.append(_normalize(normal))
    major_cells = ring_count if full else ring_count - 1
    for major_index in range(major_cells):
        next_major = (major_index + 1) % ring_count
        for minor_index in range(minor_segments):
            next_minor = (minor_index + 1) % minor_segments
            a = major_index * minor_segments + minor_index
            b = next_major * minor_segments + minor_index
            c = next_major * minor_segments + next_minor
            d = major_index * minor_segments + next_minor
            triangles.extend((a, b, c, a, c, d))
            edges.extend((a, b))
        _append_ring_edges(edges, major_index * minor_segments, minor_segments)
    if not full:
        _append_ring_edges(edges, (ring_count - 1) * minor_segments, minor_segments)
        _append_torus_cap(vertices, normals, triangles, major_radius, minor_radius, math.radians(start_deg), minor_segments, start=True)
        _append_torus_cap(
            vertices, normals, triangles, major_radius, minor_radius,
            math.radians(start_deg + sweep_deg), minor_segments, start=False,
        )
    return MeshData(tuple(vertices), tuple(normals), tuple(triangles), tuple(edges))


def _append_torus_cap(vertices, normals, triangles, major_radius, minor_radius, theta, segments, *, start):
    tangent = (-math.sin(theta), math.cos(theta), 0.0)
    normal = tuple(-value for value in tangent) if start else tangent
    center = len(vertices)
    vertices.append((major_radius * math.cos(theta), major_radius * math.sin(theta), 0.0))
    normals.append(normal)
    ring = len(vertices)
    for index in range(segments):
        phi = math.tau * index / segments
        radius = major_radius + minor_radius * math.cos(phi)
        vertices.append((radius * math.cos(theta), radius * math.sin(theta), minor_radius * math.sin(phi)))
        normals.append(normal)
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend((center, ring + index, ring + nxt) if start else (center, ring + nxt, ring + index))


def _append_ring_edges(edges: list[int], offset: int, count: int) -> None:
    for index in range(count):
        edges.extend((offset + index, offset + (index + 1) % count))


def _add(left, right):
    return tuple(left[index] + right[index] for index in range(3))


def _length(value):
    return math.sqrt(sum(component * component for component in value))


def _normalize(value):
    length = _length(value)
    if length <= 1e-15:
        raise ValueError("cannot normalize a zero vector")
    return tuple(component / length for component in value)


__all__ = [
    "DEFAULT_MESH_RESOLUTION",
    "MeshData",
    "MeshResolution",
    "build_mesh",
    "build_parameter_mesh",
    "mesh_resolution_for_subdivision",
]
