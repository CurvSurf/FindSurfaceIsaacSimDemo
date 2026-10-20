"""UI-independent dual-view layout, routing, and render-packet composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
from typing import Mapping, Sequence

from .camera import CameraRole, CameraSnapshot, look_orientation, view_projection_matrix
from .domain import Pose
from .geometry import MeshData
from .spatial_math import rotate_vector_unit_xyzw
from .render_layers import (
    LayeredRenderPacket,
    MeshLayerPacket,
    RenderLayer,
    build_line_layer,
    build_mesh_layer,
    mesh_model_matrix,
)


Vec3 = tuple[float, float, float]


class ViewSlot(str, Enum):
    PRIMARY = "primary"
    PIP = "pip"


class PointerTarget(str, Enum):
    PRIMARY = "primary"
    PIP_BLOCKED = "pip-blocked"
    OVERLAY = "overlay"
    OUTSIDE = "outside"


@dataclass(frozen=True)
class ViewRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("view rectangle must be finite")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("view rectangle size must be positive")

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def overlaps(self, other: "ViewRect") -> bool:
        return not (
            self.x + self.width <= other.x
            or other.x + other.width <= self.x
            or self.y + self.height <= other.y
            or other.y + other.height <= self.y
        )


@dataclass(frozen=True)
class DualViewLayout:
    primary: ViewRect
    pip: ViewRect
    sensor_overlay: ViewRect
    detection_overlay: ViewRect
    histogram_overlay: ViewRect

    @property
    def overlays(self) -> tuple[ViewRect, ...]:
        return self.sensor_overlay, self.detection_overlay, self.histogram_overlay


def compute_dual_view_layout(
    width: float,
    height: float,
    *,
    primary_aspect: float,
    pip_aspect: float,
    pip_width_fraction: float = 0.30,
    margin_px: float = 16.0,
) -> DualViewLayout:
    """Return a viewport-filling primary, aspect-preserving PIP, and overlays."""

    values = (width, height, primary_aspect, pip_aspect, margin_px)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("content size, aspects, and margin must be finite and positive")
    if not math.isfinite(pip_width_fraction) or not 0.0 < pip_width_fraction < 1.0:
        raise ValueError("pip_width_fraction must be in (0, 1)")

    # The primary projection follows this rectangle's live aspect. Keeping the
    # previous camera aspect here would letterbox content while the docked
    # viewport changes shape.
    primary = ViewRect(0.0, 0.0, width, height)
    available_width = max(1.0, width - 2.0 * margin_px)
    available_height = max(1.0, height - 2.0 * margin_px)
    pip_width = min(available_width, width * pip_width_fraction)
    pip_height = pip_width / pip_aspect
    if pip_height > available_height:
        pip_height = available_height
        pip_width = pip_height * pip_aspect
    pip = ViewRect(
        max(0.0, width - margin_px - pip_width),
        min(margin_px, max(0.0, height - pip_height)),
        pip_width,
        pip_height,
    )

    overlay_width = max(1.0, min(280.0, width * 0.27, available_width))
    overlay_height = max(1.0, min(74.0, height * 0.16, available_height))
    sensor = ViewRect(min(margin_px, max(0.0, width - overlay_width)), margin_px, overlay_width, overlay_height)
    histogram_height = max(1.0, min(148.0, height * 0.25, available_height))
    histogram = ViewRect(
        max(0.0, width - margin_px - overlay_width),
        max(0.0, height - margin_px - histogram_height),
        overlay_width,
        histogram_height,
    )
    detection_y = max(0.0, histogram.y - margin_px - overlay_height)
    detection = ViewRect(histogram.x, detection_y, overlay_width, overlay_height)
    return DualViewLayout(primary, pip, sensor, detection, histogram)


def route_pointer(
    x: float,
    y: float,
    layout: DualViewLayout,
    *,
    pip_visible: bool = True,
) -> PointerTarget:
    """UI overlays win; PIP is explicitly inert and never leaks into primary input."""

    if any(rect.contains(x, y) for rect in layout.overlays):
        return PointerTarget.OVERLAY
    if pip_visible and layout.pip.contains(x, y):
        return PointerTarget.PIP_BLOCKED
    if layout.primary.contains(x, y):
        return PointerTarget.PRIMARY
    return PointerTarget.OUTSIDE


@dataclass(frozen=True)
class DualViewSnapshot:
    generation: int = 0
    primary_role: CameraRole = CameraRole.SCANNER
    pip_visible: bool = True

    @property
    def pip_role(self) -> CameraRole:
        return CameraRole.SCENE if self.primary_role is CameraRole.SCANNER else CameraRole.SCANNER

    def role_for(self, slot: ViewSlot) -> CameraRole | None:
        if slot is ViewSlot.PIP and not self.pip_visible:
            return None
        return self.primary_role if slot is ViewSlot.PRIMARY else self.pip_role

    def slot_for(self, role: CameraRole) -> ViewSlot:
        return ViewSlot.PRIMARY if CameraRole(role) is self.primary_role else ViewSlot.PIP


class DualViewCompositor:
    """Own stable role resources while atomically changing only role-to-slot mapping."""

    def __init__(self, role_content: Mapping[CameraRole, object]) -> None:
        missing = set(CameraRole) - set(role_content)
        if missing:
            raise ValueError(f"missing role content: {', '.join(sorted(role.value for role in missing))}")
        self._role_content = {role: role_content[role] for role in CameraRole}
        self._snapshot = DualViewSnapshot()
        self._lock = threading.RLock()

    def snapshot(self) -> DualViewSnapshot:
        with self._lock:
            return self._snapshot

    def content_for(self, slot: ViewSlot) -> object | None:
        with self._lock:
            role = self._snapshot.role_for(ViewSlot(slot))
            return None if role is None else self._role_content[role]

    def content_for_role(self, role: CameraRole) -> object:
        return self._role_content[CameraRole(role)]

    def replace_role_content(self, role: CameraRole, content: object) -> object:
        if content is None:
            raise ValueError("role content cannot be None")
        with self._lock:
            role = CameraRole(role)
            previous = self._role_content[role]
            self._role_content[role] = content
            return previous

    def swap(self) -> DualViewSnapshot:
        with self._lock:
            primary = self._snapshot.pip_role
            self._snapshot = replace(
                self._snapshot,
                generation=self._snapshot.generation + 1,
                primary_role=primary,
            )
            return self._snapshot

    def set_primary(self, role: CameraRole) -> DualViewSnapshot:
        role = CameraRole(role)
        with self._lock:
            if self._snapshot.primary_role is role:
                return self._snapshot
            self._snapshot = replace(
                self._snapshot,
                generation=self._snapshot.generation + 1,
                primary_role=role,
            )
            return self._snapshot

    def set_pip_visible(self, visible: bool) -> DualViewSnapshot:
        with self._lock:
            visible = bool(visible)
            if self._snapshot.pip_visible == visible:
                return self._snapshot
            self._snapshot = replace(
                self._snapshot,
                generation=self._snapshot.generation + 1,
                pip_visible=visible,
            )
            return self._snapshot

    def reset(self) -> DualViewSnapshot:
        return self.set_primary(CameraRole.SCANNER)


@dataclass(frozen=True)
class ScannerFrustum:
    apex: Vec3
    far_corners: tuple[Vec3, Vec3, Vec3, Vec3]
    face_mesh: MeshData
    edge_vertices: tuple[Vec3, ...]


def build_scanner_frustum(camera: CameraSnapshot, *, depth_m: float | None = None) -> ScannerFrustum:
    """Build a pyramid from the exact scanner pose/FOV without a self-marker."""

    if camera.role is not CameraRole.SCANNER:
        raise ValueError("scanner frustum requires the scanner camera")
    depth = min(camera.intrinsics.far_m, 2.0) if depth_m is None else float(depth_m)
    if not math.isfinite(depth) or not camera.intrinsics.near_m < depth <= camera.intrinsics.far_m:
        raise ValueError("frustum depth must be finite and inside the scanner clip range")
    half_height = math.tan(camera.intrinsics.vertical_fov_radians * 0.5) * depth
    half_width = half_height * camera.intrinsics.aspect_ratio
    center = _add(camera.pose_world.position_m, _scale(camera.forward, depth))
    corners = (
        _add(_add(center, _scale(camera.right, -half_width)), _scale(camera.up, half_height)),
        _add(_add(center, _scale(camera.right, half_width)), _scale(camera.up, half_height)),
        _add(_add(center, _scale(camera.right, half_width)), _scale(camera.up, -half_height)),
        _add(_add(center, _scale(camera.right, -half_width)), _scale(camera.up, -half_height)),
    )
    apex = camera.pose_world.position_m
    vertices = (apex,) + corners
    # Normals are presentation-only; the current shader does not consume them.
    mesh = MeshData(
        vertices,
        tuple((0.0, 0.0, 1.0) for _ in vertices),
        (0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1),
        (0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1),
    )
    edges = tuple(vertices[index] for index in mesh.edge_indices)
    return ScannerFrustum(apex, corners, mesh, edges)


def compose_role_packets(
    base: LayeredRenderPacket,
    scanner_camera: CameraSnapshot,
    scene_camera: CameraSnapshot,
    *,
    frustum_depth_m: float | None = None,
) -> Mapping[CameraRole, LayeredRenderPacket]:
    """Share immutable scene payload while applying role camera/frustum policy."""

    if scanner_camera.role is not CameraRole.SCANNER or scene_camera.role is not CameraRole.SCENE:
        raise ValueError("camera roles do not match the requested dual-view packets")
    mesh_layers = tuple(layer for layer in base.mesh_layers if layer.layer is not RenderLayer.CAMERA_FRUSTUM_FACE)
    line_layers = tuple(layer for layer in base.line_layers if layer.layer is not RenderLayer.CAMERA_FRUSTUM)
    scanner = replace(
        base,
        mesh_layers=_screen_up_sphere_layers(mesh_layers, scanner_camera),
        line_layers=line_layers,
        view_projection=view_projection_matrix(scanner_camera),
        camera_eye_world_m=scanner_camera.pose_world.position_m,
    )
    frustum = build_scanner_frustum(scanner_camera, depth_m=frustum_depth_m)
    scene = replace(
        base,
        mesh_layers=_screen_up_sphere_layers(mesh_layers, scene_camera) + (build_mesh_layer(RenderLayer.CAMERA_FRUSTUM_FACE, (frustum.face_mesh,)),),
        line_layers=line_layers + (build_line_layer(RenderLayer.CAMERA_FRUSTUM, frustum.edge_vertices),),
        view_projection=view_projection_matrix(scene_camera),
        camera_eye_world_m=scene_camera.pose_world.position_m,
    )
    return {CameraRole.SCANNER: scanner, CameraRole.SCENE: scene}


def _screen_up_sphere_layers(
    layers: Sequence[MeshLayerPacket], camera: CameraSnapshot,
) -> tuple[MeshLayerPacket, ...]:
    """Rotate only world-baked result spheres about their centers, not their data."""

    orientation = None
    aligned = []
    for layer in layers:
        if layer.layer not in {RenderLayer.DETECTED_MESH, RenderLayer.CAPTURE}:
            aligned.append(layer)
            continue
        draws = []
        changed = False
        for draw in layer.draws:
            center = draw.sphere_pole_center_world_m
            if center is None:
                draws.append(draw)
                continue
            if orientation is None:
                orientation = look_orientation(tuple(-value for value in camera.up), camera.forward)
            rotated_center = rotate_vector_unit_xyzw(orientation, center)
            translation = tuple(value - rotated for value, rotated in zip(center, rotated_center))
            draws.append(replace(draw, model_matrix=mesh_model_matrix(Pose(translation, orientation))))
            changed = True
        aligned.append(replace(layer, draws=tuple(draws)) if changed else layer)
    return tuple(aligned)


def _fit_aspect(x: float, y: float, width: float, height: float, aspect: float) -> ViewRect:
    fitted_width = width
    fitted_height = fitted_width / aspect
    if fitted_height > height:
        fitted_height = height
        fitted_width = fitted_height * aspect
    return ViewRect(
        x + (width - fitted_width) * 0.5,
        y + (height - fitted_height) * 0.5,
        fitted_width,
        fitted_height,
    )


def _add(left: Sequence[float], right: Sequence[float]) -> Vec3:
    return tuple(float(a) + float(b) for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(value: Sequence[float], scale: float) -> Vec3:
    return tuple(float(component) * float(scale) for component in value)  # type: ignore[return-value]


__all__ = [
    "DualViewCompositor", "DualViewLayout", "DualViewSnapshot", "PointerTarget",
    "ScannerFrustum", "ViewRect", "ViewSlot", "build_scanner_frustum",
    "compose_role_packets", "compute_dual_view_layout", "route_pointer",
]
