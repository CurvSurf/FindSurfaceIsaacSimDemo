"""Immutable renderer-layer packets and theme/style policy for OpenGL."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
from typing import Iterable, Sequence

import numpy as np

from .analysis import HeatmapResult
from .geometry import MeshData
from .domain import Pose, ScanFrame


Color = tuple[float, float, float, float]
IDENTITY_MATRIX = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


class RenderTheme(str, Enum):
    EDITOR_DARK = "editor-dark"


class RenderLayer(str, Enum):
    DESIGNED_MESH = "Mesh.D"
    DETECTED_MESH = "Mesh.B"
    SOURCE_POINTS = "Source"
    INLIER_POINTS = "Inlier"
    OUTLIER_POINTS = "Outlier"
    SEED = "Seed"
    PICKING_GUIDE = "PickingGuide"
    SEED_RADIUS = "SeedRadius"
    CAMERA_FRUSTUM = "CameraFrustum"
    CAMERA_FRUSTUM_FACE = "CameraFrustumFace"
    PREVIEW = "Preview"
    CAPTURE = "Capture"


MESH_LAYERS = frozenset(
    {
        RenderLayer.DESIGNED_MESH,
        RenderLayer.DETECTED_MESH,
        RenderLayer.PREVIEW,
        RenderLayer.CAPTURE,
        RenderLayer.CAMERA_FRUSTUM_FACE,
    }
)
TRANSLUCENT_MESH_LAYERS = frozenset(
    {
        RenderLayer.DESIGNED_MESH,
        RenderLayer.DETECTED_MESH,
        RenderLayer.PREVIEW,
        RenderLayer.CAPTURE,
    }
)
POINT_LAYERS = frozenset(
    {
        RenderLayer.SOURCE_POINTS,
        RenderLayer.INLIER_POINTS,
        RenderLayer.OUTLIER_POINTS,
        RenderLayer.SEED,
    }
)
LINE_LAYERS = frozenset({RenderLayer.PICKING_GUIDE, RenderLayer.SEED_RADIUS, RenderLayer.CAMERA_FRUSTUM})


def _color(value: Sequence[float], name: str) -> Color:
    if len(value) != 4:
        raise ValueError(f"{name} must contain four RGBA values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) and 0.0 <= component <= 1.0 for component in result):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class RenderStyle:
    theme: RenderTheme
    background: Color
    designed_face: Color
    designed_wire: Color
    detected_face: Color
    detected_wire: Color
    preview_face: Color
    preview_wire: Color
    capture_face: Color
    capture_wire: Color
    source_fill: Color
    outlier_fill: Color
    inlier_outline: Color
    heatmap_zero: Color
    heatmap_middle: Color
    heatmap_high: Color
    seed_fill: Color
    seed_fallback_fill: Color
    guide_color: Color
    frustum_face: Color
    frustum_edge: Color
    point_size_dip: float = 3.0
    inlier_size_dip: float = 5.0
    seed_size_dip: float = 11.0
    wire_width_dip: float = 1.5
    seed_radius_color: Color = (1.0, 1.0, 1.0, 0.9)

    def __post_init__(self) -> None:
        object.__setattr__(self, "theme", RenderTheme(self.theme))
        for name in (
            "background", "designed_face", "designed_wire", "detected_face",
            "detected_wire", "preview_face", "preview_wire", "capture_face",
            "capture_wire", "source_fill", "outlier_fill", "inlier_outline",
            "heatmap_zero", "heatmap_middle", "heatmap_high", "seed_fill", "seed_fallback_fill",
            "guide_color", "seed_radius_color",
            "frustum_face", "frustum_edge",
        ):
            object.__setattr__(self, name, _color(getattr(self, name), name))
        for name in ("point_size_dip", "inlier_size_dip", "seed_size_dip", "wire_width_dip"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


def style_for_theme(theme: RenderTheme | str) -> RenderStyle:
    theme = RenderTheme(theme)
    return RenderStyle(
        theme,
        (0.035, 0.05, 0.075, 1.0),
        (0.36, 0.72, 0.94, 0.25), (0.10, 0.43, 0.62, 1.0),
        (0.72, 0.96, 0.92, 0.25), (0.15, 1.0, 0.65, 1.0),
        (0.72, 0.96, 0.92, 0.20), (0.15, 1.0, 0.65, 1.0),
        (1.0, 0.72, 0.18, 0.22), (1.0, 0.78, 0.16, 1.0),
        (0.88, 0.94, 1.0, 1.0), (0.70, 0.78, 0.88, 1.0),
        (1.0, 0.96, 0.84, 1.0),
        (0.0, 0.22, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 0.02, 0.0, 1.0),
        (1.0, 0.03, 0.03, 1.0), (0.72, 0.18, 1.0, 1.0), (1.0, 0.62, 0.08, 1.0),
        (1.0, 0.62, 0.08, 0.10), (1.0, 0.62, 0.08, 1.0),
    )


def heatmap_color(style: RenderStyle, coordinate: float) -> Color:
    """Return the exact diverging colour uploaded to the point shader."""

    coordinate = max(0.0, min(1.0, float(coordinate)))
    if coordinate <= 0.5:
        start, end, amount = style.heatmap_zero, style.heatmap_middle, coordinate * 2.0
    else:
        start, end = style.heatmap_middle, style.heatmap_high
        amount = (coordinate - 0.5) * 2.0
    return tuple(
        start[index] + (end[index] - start[index]) * amount
        for index in range(4)
    )  # type: ignore[return-value]


def hover_emphasis_color(color: Sequence[float]) -> Color:
    """Brighten a transient geometry-card hover without changing alpha."""

    rgba = _color(color, "hover emphasis color")
    emphasized = tuple(component + (1.0 - component) * 0.24 for component in rgba[:3])
    return emphasized + (rgba[3],)  # type: ignore[return-value]


def hover_context_color(color: Sequence[float]) -> Color:
    """Dim non-hovered capture context without changing alpha."""

    rgba = _color(color, "hover context color")
    return tuple(component * 0.62 for component in rgba[:3]) + (rgba[3],)  # type: ignore[return-value]


@dataclass(frozen=True)
class LayerVisibility:
    enabled: frozenset[RenderLayer] = frozenset(RenderLayer)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", frozenset(RenderLayer(item) for item in self.enabled))

    def is_visible(self, layer: RenderLayer | str) -> bool:
        return RenderLayer(layer) in self.enabled

    def with_layer(self, layer: RenderLayer | str, visible: bool) -> "LayerVisibility":
        layer = RenderLayer(layer)
        enabled = set(self.enabled)
        (enabled.add if visible else enabled.discard)(layer)
        return LayerVisibility(frozenset(enabled))


def _points_bytes(values, name: str) -> tuple[bytes, int]:
    points = np.asarray(values, dtype=np.float32, order="C")
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite float32[N,3]")
    return points.tobytes(order="C"), int(points.shape[0])


def _indices_bytes(values: Iterable[int], arity: int, name: str) -> tuple[bytes, int]:
    result = tuple(int(value) for value in values)
    if len(result) % arity or any(value < 0 for value in result):
        raise ValueError(f"{name} indices have invalid arity or value")
    packed = array("I", result)
    return packed.tobytes(), len(result)


@dataclass(frozen=True)
class MeshGeometryPacket:
    """Immutable local-space buffers that can be shared by many draw packets."""

    vertices_f32: bytes
    vertex_count: int
    triangle_indices_u32: bytes
    triangle_index_count: int
    edge_indices_u32: bytes
    edge_index_count: int

    def __post_init__(self) -> None:
        if len(self.vertices_f32) != self.vertex_count * 12:
            raise ValueError("mesh vertex byte size mismatch")
        if len(self.triangle_indices_u32) != self.triangle_index_count * 4 or self.triangle_index_count % 3:
            raise ValueError("mesh triangle index byte size mismatch")
        if len(self.edge_indices_u32) != self.edge_index_count * 4 or self.edge_index_count % 2:
            raise ValueError("mesh edge index byte size mismatch")


@dataclass(frozen=True)
class MeshDrawPacket:
    """One independently visible draw using local geometry and a model matrix."""

    draw_id: str
    geometry: MeshGeometryPacket
    model_matrix: tuple[float, ...] = IDENTITY_MATRIX
    face_color: Color | None = None
    wire_color: Color | None = None
    sphere_pole_center_world_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        draw_id = str(self.draw_id).strip()
        if not draw_id:
            raise ValueError("mesh draw_id cannot be empty")
        matrix = tuple(float(value) for value in self.model_matrix)
        if len(matrix) != 16 or not all(math.isfinite(value) for value in matrix):
            raise ValueError("mesh model_matrix must be a finite 4x4 matrix")
        object.__setattr__(self, "draw_id", draw_id)
        object.__setattr__(self, "model_matrix", matrix)
        if self.sphere_pole_center_world_m is not None:
            center = tuple(float(value) for value in self.sphere_pole_center_world_m)
            if len(center) != 3 or not all(math.isfinite(value) for value in center):
                raise ValueError("sphere pole center must contain three finite world coordinates")
            object.__setattr__(self, "sphere_pole_center_world_m", center)
        if self.face_color is not None:
            object.__setattr__(self, "face_color", _color(self.face_color, "face_color"))
        if self.wire_color is not None:
            object.__setattr__(self, "wire_color", _color(self.wire_color, "wire_color"))


@dataclass(frozen=True)
class MeshLayerPacket:
    layer: RenderLayer
    draws: tuple[MeshDrawPacket, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", RenderLayer(self.layer))
        object.__setattr__(self, "draws", tuple(self.draws))
        if self.layer not in MESH_LAYERS:
            raise ValueError("mesh packet uses a non-mesh layer")
        draw_ids = tuple(draw.draw_id for draw in self.draws)
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("mesh draw_id values must be unique inside a layer")

    @property
    def vertex_count(self) -> int:
        return sum(draw.geometry.vertex_count for draw in self.draws)

    @property
    def triangle_index_count(self) -> int:
        return sum(draw.geometry.triangle_index_count for draw in self.draws)

    @property
    def edge_index_count(self) -> int:
        return sum(draw.geometry.edge_index_count for draw in self.draws)


@dataclass(frozen=True)
class PointLayerPacket:
    layer: RenderLayer
    positions_f32: bytes
    point_count: int
    heatmap_f32: bytes
    used_fallback: bool = False
    emphasis_f32: bytes = b""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", RenderLayer(self.layer))
        if self.layer not in POINT_LAYERS:
            raise ValueError("point packet uses a non-point layer")
        if len(self.positions_f32) != self.point_count * 12:
            raise ValueError("point position byte size mismatch")
        if len(self.heatmap_f32) != self.point_count * 4:
            raise ValueError("point heatmap byte size mismatch")
        emphasis = self.emphasis_f32 or (array("f", [0.0]) * self.point_count).tobytes()
        if len(emphasis) != self.point_count * 4:
            raise ValueError("point emphasis byte size mismatch")
        object.__setattr__(self, "emphasis_f32", emphasis)
        object.__setattr__(self, "used_fallback", bool(self.used_fallback))
        if self.used_fallback and self.layer is not RenderLayer.SEED:
            raise ValueError("fallback style is valid only for the seed layer")


@dataclass(frozen=True)
class LineLayerPacket:
    layer: RenderLayer
    vertices_f32: bytes
    vertex_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", RenderLayer(self.layer))
        if self.layer not in LINE_LAYERS:
            raise ValueError("line packet uses a non-line layer")
        if self.vertex_count % 2 or len(self.vertices_f32) != self.vertex_count * 12:
            raise ValueError("line vertex byte size mismatch")


def build_mesh_geometry(meshes: Iterable[MeshData]) -> MeshGeometryPacket:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    edges: list[int] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        triangles.extend(offset + index for index in mesh.triangle_indices)
        edges.extend(offset + index for index in mesh.edge_indices)
    vertex_bytes, vertex_count = _points_bytes(vertices, "mesh vertices")
    triangle_bytes, triangle_count = _indices_bytes(triangles, 3, "triangle")
    edge_bytes, edge_count = _indices_bytes(edges, 2, "edge")
    return MeshGeometryPacket(
        vertex_bytes, vertex_count,
        triangle_bytes, triangle_count,
        edge_bytes, edge_count,
    )


def build_mesh_layer(
    layer: RenderLayer | str, meshes: Iterable[MeshData], *,
    sphere_pole_center_world_m: tuple[float, float, float] | None = None,
) -> MeshLayerPacket:
    layer = RenderLayer(layer)
    return MeshLayerPacket(layer, (MeshDrawPacket(
        layer.value, build_mesh_geometry(meshes),
        sphere_pole_center_world_m=sphere_pole_center_world_m,
    ),))


def mesh_model_matrix(
    pose: Pose,
    scale: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[float, ...]:
    """Compose column-major translation * rotation * scale for OpenGL."""

    scale_xyz = tuple(float(value) for value in scale)
    if len(scale_xyz) != 3 or not all(math.isfinite(value) and value > 0.0 for value in scale_xyz):
        raise ValueError("mesh scale must contain three finite positive values")
    x, y, z, w = pose.orientation_xyzw
    sx, sy, sz = scale_xyz
    m00 = 1.0 - 2.0 * (y * y + z * z)
    m01 = 2.0 * (x * y - z * w)
    m02 = 2.0 * (x * z + y * w)
    m10 = 2.0 * (x * y + z * w)
    m11 = 1.0 - 2.0 * (x * x + z * z)
    m12 = 2.0 * (y * z - x * w)
    m20 = 2.0 * (x * z - y * w)
    m21 = 2.0 * (y * z + x * w)
    m22 = 1.0 - 2.0 * (x * x + y * y)
    tx, ty, tz = pose.position_m
    return (
        m00 * sx, m10 * sx, m20 * sx, 0.0,
        m01 * sy, m11 * sy, m21 * sy, 0.0,
        m02 * sz, m12 * sz, m22 * sz, 0.0,
        tx, ty, tz, 1.0,
    )


def flatten_mesh_layer(layer: MeshLayerPacket) -> MeshGeometryPacket:
    """Bake model matrices only for consumers such as the RTX USD adapter."""

    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    edges: list[int] = []
    for draw in layer.draws:
        geometry = draw.geometry
        points = np.frombuffer(geometry.vertices_f32, dtype=np.float32).reshape((-1, 3))
        matrix = draw.model_matrix
        offset = len(vertices)
        for point in points:
            px, py, pz = (float(value) for value in point)
            vertices.append((
                matrix[0] * px + matrix[4] * py + matrix[8] * pz + matrix[12],
                matrix[1] * px + matrix[5] * py + matrix[9] * pz + matrix[13],
                matrix[2] * px + matrix[6] * py + matrix[10] * pz + matrix[14],
            ))
        triangles.extend(
            offset + int(index)
            for index in np.frombuffer(geometry.triangle_indices_u32, dtype=np.uint32)
        )
        edges.extend(
            offset + int(index)
            for index in np.frombuffer(geometry.edge_indices_u32, dtype=np.uint32)
        )
    vertex_bytes, vertex_count = _points_bytes(vertices, "flattened mesh vertices")
    triangle_bytes, triangle_count = _indices_bytes(triangles, 3, "flattened triangle")
    edge_bytes, edge_count = _indices_bytes(edges, 2, "flattened edge")
    return MeshGeometryPacket(
        vertex_bytes, vertex_count,
        triangle_bytes, triangle_count,
        edge_bytes, edge_count,
    )


def build_point_layers(
    frame: ScanFrame,
    *,
    inlier_indices: Sequence[int] = (),
    heatmap: HeatmapResult | None = None,
) -> tuple[PointLayerPacket, ...]:
    points = frame.points_world_m
    source_bytes, source_count = _points_bytes(points, "source points")
    zero_source = array("f", [0.0]) * source_count
    source = PointLayerPacket(RenderLayer.SOURCE_POINTS, source_bytes, source_count, zero_source.tobytes())
    return (source,) + build_classified_point_layers(frame, inlier_indices=inlier_indices, heatmap=heatmap)


def _validated_inlier_indices(frame: ScanFrame, inlier_indices: Sequence[int]) -> tuple[int, ...]:
    if not inlier_indices:
        return ()
    indices = tuple(int(index) for index in inlier_indices)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= frame.compacted_count for index in indices):
        raise ValueError("inlier indices must be unique and inside the frame")
    return indices


def build_classified_point_layers(
    frame: ScanFrame,
    *,
    inlier_indices: Sequence[int] = (),
    heatmap: HeatmapResult | None = None,
) -> tuple[PointLayerPacket, ...]:
    """Pack only the inlier/outlier pair; preserve the legacy empty-index result."""

    indices = _validated_inlier_indices(frame, inlier_indices)
    if not indices:
        return ()
    if heatmap is None:
        # Signed residual heatmaps reserve the midpoint for points that have
        # not received their background analysis yet.  This keeps pending
        # inliers visible in the neutral surface color instead of falsely
        # presenting them as negative/inside samples.
        inlier_coordinates = (0.5,) * len(indices)
    elif not heatmap.available or len(heatmap.coordinates) != len(indices):
        raise ValueError("classified inliers require one available heatmap coordinate per index")
    else:
        inlier_coordinates = heatmap.coordinates
    inlier_bytes, inlier_count = _points_bytes(frame.points_world_m[list(indices)], "inlier points")
    inlier_heatmap = array("f", inlier_coordinates).tobytes()
    return (
        PointLayerPacket(RenderLayer.INLIER_POINTS, inlier_bytes, inlier_count, inlier_heatmap),
        _build_outlier_layer(frame, indices),
    )


def build_outlier_layer(frame: ScanFrame, *, inlier_indices: Sequence[int] = ()) -> PointLayerPacket:
    """Pack only the complement; inliers already have their own analysis packets."""

    return _build_outlier_layer(frame, _validated_inlier_indices(frame, inlier_indices))


def _build_outlier_layer(frame: ScanFrame, indices: tuple[int, ...]) -> PointLayerPacket:
    mask = np.ones(frame.compacted_count, dtype=bool)
    mask[list(indices)] = False
    outlier_bytes, outlier_count = _points_bytes(frame.points_world_m[mask], "outlier points")
    return PointLayerPacket(
        RenderLayer.OUTLIER_POINTS,
        outlier_bytes,
        outlier_count,
        (array("f", [0.0]) * outlier_count).tobytes(),
    )


def build_inlier_layer(
    points_world_m: Iterable[Sequence[float]],
    heatmap: HeatmapResult,
) -> PointLayerPacket:
    """Pack standalone preview/capture inliers without retaining a source frame."""

    # Analysis subjects already own immutable contiguous NumPy storage.  Do
    # not expand that array into thousands of Python row objects on the worker
    # before converting it straight back into an array in ``_points_bytes``.
    positions, count = _points_bytes(points_world_m, "inlier points")
    if not heatmap.available or len(heatmap.coordinates) != count:
        raise ValueError("inliers require one available heatmap coordinate per point")
    coordinates = array("f", heatmap.coordinates).tobytes()
    return PointLayerPacket(RenderLayer.INLIER_POINTS, positions, count, coordinates)


def recolor_inlier_layer(
    packet: PointLayerPacket,
    heatmap: HeatmapResult,
) -> PointLayerPacket:
    """Reuse immutable point positions while replacing only heatmap attributes."""

    if packet.layer is not RenderLayer.INLIER_POINTS:
        raise ValueError("only an inlier packet can be recolored")
    if not heatmap.available or len(heatmap.coordinates) != packet.point_count:
        raise ValueError("recolored inliers require one heatmap coordinate per point")
    return replace(
        packet,
        heatmap_f32=array("f", heatmap.coordinates).tobytes(),
    )


def emphasize_inlier_layer(packet: PointLayerPacket, emphasis: float = 1.0) -> PointLayerPacket:
    """Reuse capture inliers while marking pixels as bright, neutral, or dim."""

    if packet.layer is not RenderLayer.INLIER_POINTS:
        raise ValueError("only an inlier packet can be emphasized")
    value = max(-1.0, min(1.0, float(emphasis)))
    return replace(packet, emphasis_f32=(array("f", [value]) * packet.point_count).tobytes())


def build_pending_inlier_layer(
    points_world_m: Iterable[Sequence[float]],
) -> PointLayerPacket:
    """Pack inliers whose signed-residual analysis has not completed yet."""

    positions, count = _points_bytes(points_world_m, "pending inlier points")
    coordinates = (array("f", [0.5]) * count).tobytes()
    return PointLayerPacket(RenderLayer.INLIER_POINTS, positions, count, coordinates)


def merge_point_layers(
    layer: RenderLayer | str,
    packets: Iterable[PointLayerPacket | None],
) -> PointLayerPacket | None:
    """Merge same-role immutable point packets into one renderer layer."""

    layer = RenderLayer(layer)
    items = tuple(packet for packet in packets if packet is not None and packet.point_count)
    if not items:
        return None
    if any(packet.layer is not layer for packet in items):
        raise ValueError("merged point packets must use the requested layer")
    if any(packet.used_fallback for packet in items):
        raise ValueError("fallback seed packets cannot be merged")
    if len(items) == 1:
        return items[0]
    return PointLayerPacket(
        layer,
        b"".join(packet.positions_f32 for packet in items),
        sum(packet.point_count for packet in items),
        b"".join(packet.heatmap_f32 for packet in items),
        emphasis_f32=b"".join(packet.emphasis_f32 for packet in items),
    )


def build_seed_layer(
    point_world_m: Sequence[float], *, used_fallback: bool = False,
) -> PointLayerPacket:
    positions, count = _points_bytes((point_world_m,), "seed point")
    return PointLayerPacket(
        RenderLayer.SEED,
        positions,
        count,
        array("f", [0.0]).tobytes(),
        used_fallback=used_fallback,
    )


def build_line_layer(layer: RenderLayer | str, line_vertices: Iterable[Sequence[float]]) -> LineLayerPacket:
    vertices, count = _points_bytes(tuple(line_vertices), "line vertices")
    return LineLayerPacket(RenderLayer(layer), vertices, count)


@dataclass(frozen=True)
class LayeredRenderPacket:
    revision: int
    mesh_layers: tuple[MeshLayerPacket, ...] = ()
    point_layers: tuple[PointLayerPacket, ...] = ()
    line_layers: tuple[LineLayerPacket, ...] = ()
    visibility: LayerVisibility = LayerVisibility()
    style: RenderStyle = style_for_theme(RenderTheme.EDITOR_DARK)
    view_projection: tuple[float, ...] = IDENTITY_MATRIX
    camera_eye_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        if len(self.view_projection) != 16 or not all(math.isfinite(float(value)) for value in self.view_projection):
            raise ValueError("view_projection must be a finite 4x4 matrix")
        if len(self.camera_eye_world_m) != 3 or not all(
            math.isfinite(float(value)) for value in self.camera_eye_world_m
        ):
            raise ValueError("camera_eye_world_m must be a finite xyz tuple")
        layers = [item.layer for item in self.mesh_layers + self.point_layers + self.line_layers]
        if len(layers) != len(set(layers)):
            raise ValueError("a render packet can contain each layer at most once")
        object.__setattr__(self, "view_projection", tuple(float(value) for value in self.view_projection))
        object.__setattr__(self, "camera_eye_world_m", tuple(float(value) for value in self.camera_eye_world_m))

    def with_visibility(self, layer: RenderLayer | str, visible: bool) -> "LayeredRenderPacket":
        return replace(self, visibility=self.visibility.with_layer(layer, visible))

    def with_style(self, style: RenderStyle) -> "LayeredRenderPacket":
        return replace(self, style=style)


@dataclass(frozen=True)
class LayerUploadDelta:
    mesh_changed: tuple[MeshLayerPacket, ...] = ()
    mesh_removed: tuple[RenderLayer, ...] = ()
    point_changed: tuple[PointLayerPacket, ...] = ()
    point_removed: tuple[RenderLayer, ...] = ()
    line_changed: tuple[LineLayerPacket, ...] = ()
    line_removed: tuple[RenderLayer, ...] = ()


class LayerPayloadTracker:
    """Identify payload changes by immutable layer object identity."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mesh: dict[RenderLayer, MeshLayerPacket] = {}
        self._point: dict[RenderLayer, PointLayerPacket] = {}
        self._line: dict[RenderLayer, LineLayerPacket] = {}

    @staticmethod
    def _delta(previous, current):
        current_by_layer = {item.layer: item for item in current}
        changed = tuple(
            item for item in current
            if previous.get(item.layer) is not item
        )
        removed = tuple(layer for layer in previous if layer not in current_by_layer)
        return current_by_layer, changed, removed

    def update(self, packet: LayeredRenderPacket) -> LayerUploadDelta:
        self._mesh, mesh_changed, mesh_removed = self._delta(self._mesh, packet.mesh_layers)
        self._point, point_changed, point_removed = self._delta(self._point, packet.point_layers)
        self._line, line_changed, line_removed = self._delta(self._line, packet.line_layers)
        return LayerUploadDelta(
            mesh_changed, mesh_removed,
            point_changed, point_removed,
            line_changed, line_removed,
        )


class LatestLayeredPacket:
    """Bounded latest-only layer publication; submissions never form a queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._packet: LayeredRenderPacket | None = None

    def submit(self, packet: LayeredRenderPacket) -> int:
        if not isinstance(packet, LayeredRenderPacket):
            raise TypeError("packet must be a LayeredRenderPacket")
        with self._lock:
            self._generation += 1
            self._packet = packet
            return self._generation

    def snapshot(self) -> tuple[int, LayeredRenderPacket | None]:
        with self._lock:
            return self._generation, self._packet


def contrast_ratio(left: Color, right: Color) -> float:
    def luminance(color: Color) -> float:
        channels = []
        for component in color[:3]:
            channels.append(component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    high, low = sorted((luminance(left), luminance(right)), reverse=True)
    return (high + 0.05) / (low + 0.05)


__all__ = [
    "IDENTITY_MATRIX", "LINE_LAYERS", "MESH_LAYERS", "POINT_LAYERS", "TRANSLUCENT_MESH_LAYERS",
    "LayerPayloadTracker", "LayerUploadDelta", "LayerVisibility", "LayeredRenderPacket",
    "LatestLayeredPacket", "LineLayerPacket",
    "MeshDrawPacket", "MeshGeometryPacket", "MeshLayerPacket", "PointLayerPacket",
    "RenderLayer", "RenderStyle", "RenderTheme",
    "build_classified_point_layers", "build_outlier_layer",
    "build_inlier_layer", "build_line_layer", "build_mesh_geometry", "build_mesh_layer",
    "build_pending_inlier_layer", "build_point_layers", "build_seed_layer", "contrast_ratio",
    "emphasize_inlier_layer", "flatten_mesh_layer", "heatmap_color", "hover_context_color",
    "hover_emphasis_color",
    "recolor_inlier_layer",
    "merge_point_layers",
    "mesh_model_matrix", "style_for_theme",
]
