"""Deterministic perspective point picking and selection state."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .spatial_math import cross_vec3
from .camera import CameraSnapshot, perspective_ray
from .domain import FrameIdentity, GenerationSnapshot, ScanFrame, SelectionSnapshot
from .native_picking import default_picker, native_picker_available
from .render_layers import (
    LineLayerPacket,
    PointLayerPacket,
    RenderLayer,
    build_line_layer,
    build_seed_layer,
)


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PickResult:
    point_index: int
    point_world_m: Vec3
    depth_m: float
    ray_distance_m: float
    cone_half_angle_radians: float
    used_fallback: bool = False


@dataclass(frozen=True)
class PickingState:
    frame_identity: FrameIdentity | None
    sensor_active: bool
    generations: GenerationSnapshot
    cursor_logical: tuple[float, float] | None = None
    hover: PickResult | None = None
    selection: SelectionSnapshot | None = None
    preview_revision: int = 0

    @property
    def point_selection_mode(self) -> bool:
        return not self.sensor_active and self.selection is not None

    @property
    def displayed_seed(self) -> PickResult | SelectionSnapshot | None:
        return self.selection if self.selection is not None else self.hover


@dataclass(frozen=True)
class PickingTransition:
    state: PickingState
    selection_invalidated: bool = False
    preview_invalidated: bool = False


@dataclass(frozen=True)
class PickingOverlay:
    seed: PointLayerPacket | None
    guide: LineLayerPacket
    seed_radius: LineLayerPacket | None = None


def cone_half_angle(
    intrinsics,
    logical_size: tuple[float, float],
    probe_radius_logical_px: float,
) -> float:
    width, height = _size(logical_size)
    radius = float(probe_radius_logical_px)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("probe radius must be finite and positive")
    tangent_y = math.tan(intrinsics.vertical_fov_radians * 0.5)
    tangent_x = tangent_y * intrinsics.aspect_ratio
    focal_x = width * 0.5 / tangent_x
    focal_y = height * 0.5 / tangent_y
    # A circular world cone uses the less restrictive pixel focal length so
    # the complete logical-pixel probe disk is covered with non-square pixels.
    return math.atan(radius / min(focal_x, focal_y))


def pick_perspective(
    frame: ScanFrame,
    camera: CameraSnapshot,
    cursor_logical: tuple[float, float],
    logical_size: tuple[float, float],
    *,
    probe_radius_logical_px: float = 6.0,
    fallback_to_nearest: bool = True,
    active_mask: Sequence[bool] | np.ndarray | None = None,
    prefer_native: bool = True,
) -> PickResult | None:
    """Pick by deterministic order, using the zero-copy native scan when packaged."""

    points = frame.points_world_m
    count = frame.compacted_count
    if count == 0:
        return None
    origin, direction = perspective_ray(camera, cursor_logical[0], cursor_logical[1], logical_size)
    parallel_radius_m = None
    if camera.intrinsics.orthographic_height_m is not None:
        width, height = _size(logical_size)
        radius = float(probe_radius_logical_px)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("probe radius must be finite and positive")
        parallel_radius_m = radius * camera.intrinsics.orthographic_height_m * max(
            camera.intrinsics.aspect_ratio / width, 1 / height)
    half_angle = 0.0 if parallel_radius_m is not None else cone_half_angle(camera.intrinsics, logical_size, probe_radius_logical_px)
    picker = default_picker() if prefer_native else None
    if picker is not None and parallel_radius_m is not None and not picker.supports_parallel:
        picker = None
    if picker is not None:
        native = picker.pick_perspective(
            points,
            origin,
            direction,
            half_angle,
            fallback_to_nearest=fallback_to_nearest,
            active_mask=active_mask,
            parallel_radius_m=parallel_radius_m,
        )
        if native is None:
            return None
        point = tuple(float(value) for value in points[native.point_index])
        return PickResult(
            native.point_index,
            point,  # type: ignore[arg-type]
            native.depth_m,
            native.ray_distance_m,
            half_angle,
            native.used_fallback,
        )
    return _pick_perspective_numpy(
        points,
        origin,
        direction,
        half_angle,
        fallback_to_nearest=fallback_to_nearest,
        active_mask=active_mask,
        parallel_radius_m=parallel_radius_m,
    )


def _pick_perspective_numpy(
    points: np.ndarray,
    origin: Vec3,
    direction: Vec3,
    half_angle: float,
    *,
    fallback_to_nearest: bool,
    active_mask: Sequence[bool] | np.ndarray | None,
    parallel_radius_m: float | None = None,
) -> PickResult | None:
    """Allocation-heavy compatibility path used only without a native library."""

    count = int(points.shape[0])
    relative = points.astype(np.float64, copy=False) - np.asarray(origin, dtype=np.float64)
    ray = np.asarray(direction, dtype=np.float64)
    depth = relative @ ray
    front = depth >= 0.0
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != (count,):
            raise ValueError("active_mask must contain one boolean per point")
        front &= mask
    if not np.any(front):
        return None
    perpendicular = relative - depth[:, None] * ray[None, :]
    distance = np.linalg.norm(perpendicular, axis=1)
    tolerance = np.maximum(1.0, np.abs(depth)) * 1.0e-12
    threshold = depth * math.tan(half_angle) if parallel_radius_m is None else parallel_radius_m
    inside = front & (distance <= threshold + tolerance)
    used_fallback = False
    candidates = np.flatnonzero(inside)
    if candidates.size:
        # np.lexsort's last key is primary.
        order = np.lexsort((candidates, distance[candidates], depth[candidates]))
    elif fallback_to_nearest:
        used_fallback = True
        candidates = np.flatnonzero(front)
        order = np.lexsort((candidates, depth[candidates], distance[candidates]))
    else:
        return None
    index = int(candidates[int(order[0])])
    point = tuple(float(value) for value in points[index])
    return PickResult(
        index,
        point,  # type: ignore[arg-type]
        float(depth[index]),
        float(distance[index]),
        half_angle,
        used_fallback,
    )


class PickingController:
    """Recompute hover inputs and own the disabled-sensor selection lock."""

    def __init__(
        self,
        frame: ScanFrame,
        camera: CameraSnapshot,
        logical_size: tuple[float, float],
        generations: GenerationSnapshot,
        *,
        sensor_active: bool,
        probe_radius_logical_px: float = 6.0,
        pick_nearest_on_probe_miss: bool = True,
    ) -> None:
        self._frame = frame
        self._camera = camera
        self._logical_size = _size(logical_size)
        self._probe_radius = float(probe_radius_logical_px)
        self._pick_nearest_on_probe_miss = bool(pick_nearest_on_probe_miss)
        self._selection_view = None
        cone_half_angle(camera.intrinsics, self._logical_size, self._probe_radius)
        _require_frame_generation(frame, generations)
        self.state = PickingState(frame.identity, bool(sensor_active), generations)

    def update_hover(self, cursor_logical: tuple[float, float]) -> PickingTransition:
        cursor = float(cursor_logical[0]), float(cursor_logical[1])
        if self.state.selection is not None:
            if self._selection_view is not None and self._selection_view[0].role == self._camera.role:
                self._selection_view = (self._camera, cursor, self._logical_size)
            self.state = replace(self.state, cursor_logical=cursor, hover=None)
            return PickingTransition(self.state, preview_invalidated=False)
        result = pick_perspective(
            self._frame,
            self._camera,
            cursor,
            self._logical_size,
            probe_radius_logical_px=self._probe_radius,
            fallback_to_nearest=self._pick_nearest_on_probe_miss,
        )
        # Screen-space distances/fallback provenance can change without changing
        # the native seed. Keep fresh visual picking data without a new request.
        changed = (None if result is None else result.point_index) != (
            None if self.state.hover is None else self.state.hover.point_index
        )
        generations = self.state.generations
        if changed and self.state.selection is None:
            generations = generations.advance("selection_revision")
        self.state = replace(
            self.state,
            generations=generations,
            cursor_logical=cursor,
            hover=result,
            preview_revision=self.state.preview_revision + int(changed and self.state.selection is None),
        )
        return PickingTransition(self.state, preview_invalidated=bool(changed and self.state.selection is None))

    def clear_hover(self) -> PickingTransition:
        changed = self.state.hover is not None or self.state.cursor_logical is not None
        generations = self.state.generations
        if changed and self.state.selection is None:
            generations = generations.advance("selection_revision")
        self.state = replace(
            self.state,
            generations=generations,
            cursor_logical=None,
            hover=None,
            preview_revision=self.state.preview_revision + int(changed and self.state.selection is None),
        )
        return PickingTransition(self.state, preview_invalidated=bool(changed and self.state.selection is None))

    def select_click(self, cursor_logical: tuple[float, float] | None = None) -> PickingTransition:
        if self.state.sensor_active:
            return PickingTransition(self.state)
        previous_selection = self.state.selection
        picked = self.state.hover
        cursor = self.state.cursor_logical
        if cursor_logical is not None:
            cursor = float(cursor_logical[0]), float(cursor_logical[1])
            picked = pick_perspective(
                self._frame,
                self._camera,
                cursor,
                self._logical_size,
                probe_radius_logical_px=self._probe_radius,
                # First-demo parity: a first click may use nearest fallback,
                # while a locked selection can be cleared by clicking empty space.
                fallback_to_nearest=(
                    self._pick_nearest_on_probe_miss and previous_selection is None
                ),
            )
        if (previous_selection is not None and self._selection_view is not None
                and self._selection_view[0].role != self._camera.role and picked is not None):
            # Observe the other view's locked cone until empty-click/ESC clears it.
            return PickingTransition(self.state)
        generations = self.state.generations.advance("selection_revision")
        selection = None if picked is None else SelectionSnapshot(
            generations.selection_revision,
            self._frame.identity,
            picked.point_index,
            picked.point_world_m,
            picked.used_fallback,
        )
        changed = selection != previous_selection
        self._selection_view = None if selection is None else (self._camera, cursor, self._logical_size)
        hover = None if selection is not None else picked
        self.state = replace(
            self.state,
            generations=generations,
            cursor_logical=cursor,
            hover=hover,
            selection=selection,
            preview_revision=self.state.preview_revision + int(changed),
        )
        return PickingTransition(
            self.state,
            selection_invalidated=previous_selection is not None and selection is None,
            preview_invalidated=changed,
        )

    def select_index(self, seed_index: int, *, used_fallback: bool = False) -> PickingTransition:
        """Lock a known current-frame seed, used by first-demo Sticky Alt+click."""

        if self.state.sensor_active:
            return PickingTransition(self.state)
        if (self.state.selection is not None and self._selection_view is not None
                and self._selection_view[0].role != self._camera.role):
            return PickingTransition(self.state)
        index = int(seed_index)
        if index < 0 or index >= self._frame.compacted_count:
            return PickingTransition(self.state)
        generations = self.state.generations.advance("selection_revision")
        point = tuple(float(value) for value in self._frame.points_world_m[index])
        selection = SelectionSnapshot(
            generations.selection_revision,
            self._frame.identity,
            index,
            point,  # type: ignore[arg-type]
            used_fallback,
        )
        changed = selection != self.state.selection
        self._selection_view = (self._camera, self.state.cursor_logical, self._logical_size)
        self.state = replace(
            self.state,
            generations=generations,
            hover=None,
            selection=selection,
            preview_revision=self.state.preview_revision + int(changed),
        )
        return PickingTransition(self.state, preview_invalidated=changed)

    def cancel_selection(self) -> PickingTransition:
        if self.state.selection is None:
            return PickingTransition(self.state)
        generations = self.state.generations.advance("selection_revision")
        self.state = replace(
            self.state,
            generations=generations,
            selection=None,
            preview_revision=self.state.preview_revision + 1,
        )
        return PickingTransition(self.state, selection_invalidated=True, preview_invalidated=True)

    def clear_interaction(self) -> PickingTransition:
        """Clear hover and locked selection together after capture or workflow reset."""

        changed = (
            self.state.selection is not None
            or self.state.hover is not None
            or self.state.cursor_logical is not None
        )
        if not changed:
            return PickingTransition(self.state)
        generations = self.state.generations.advance("selection_revision")
        self.state = replace(
            self.state,
            generations=generations,
            cursor_logical=None,
            hover=None,
            selection=None,
            preview_revision=self.state.preview_revision + 1,
        )
        return PickingTransition(
            self.state,
            selection_invalidated=True,
            preview_invalidated=True,
        )

    def update_context(
        self,
        frame: ScanFrame,
        camera: CameraSnapshot,
        logical_size: tuple[float, float],
        generations: GenerationSnapshot,
        *,
        sensor_active: bool,
    ) -> PickingTransition:
        _require_monotonic(self.state.generations, generations)
        _require_frame_generation(frame, generations)
        logical_size = _size(logical_size)
        frame_changed = frame.identity != self._frame.identity
        picking_context_changed = (
            frame_changed
            or camera != self._camera
            or logical_size != self._logical_size
        )
        preview_context_changed = (
            frame_changed
            or generations.session != self.state.generations.session
            or generations.scene != self.state.generations.scene
            or generations.options_revision != self.state.generations.options_revision
        )
        selection_invalidated = self.state.selection is not None and (
            frame_changed
            or generations.session != self.state.generations.session
            or generations.scene != self.state.generations.scene
            or bool(sensor_active)
        )
        if selection_invalidated:
            next_revision = max(
                self.state.generations.selection_revision,
                generations.selection_revision,
            ) + 1
            generations = replace(generations, selection_revision=next_revision)
        if self.state.selection is not None and self._selection_view is not None and self._camera.role == self._selection_view[0].role:
            self._selection_view = (self._camera, self.state.cursor_logical or self._selection_view[1], self._logical_size)
        self._frame = frame
        self._camera = camera
        self._logical_size = logical_size
        previous_hover = self.state.hover
        self.state = PickingState(
            frame.identity,
            bool(sensor_active),
            generations,
            self.state.cursor_logical,
            previous_hover if not picking_context_changed else None,
            None if selection_invalidated else self.state.selection,
            self.state.preview_revision + int(preview_context_changed or selection_invalidated),
        )
        hover_changed = False
        if self.state.cursor_logical is not None and picking_context_changed:
            result = pick_perspective(
                frame,
                camera,
                self.state.cursor_logical,
                logical_size,
                probe_radius_logical_px=self._probe_radius,
                fallback_to_nearest=self._pick_nearest_on_probe_miss,
            )
            hover_changed = result != previous_hover
            self.state = replace(self.state, hover=result)
        if hover_changed and self.state.selection is None and not selection_invalidated:
            self.state = replace(
                self.state,
                generations=self.state.generations.advance("selection_revision"),
            )
        preview_invalidated = preview_context_changed or selection_invalidated or (
            hover_changed and self.state.selection is None
        )
        if hover_changed and not (preview_context_changed or selection_invalidated) and self.state.selection is None:
            self.state = replace(self.state, preview_revision=self.state.preview_revision + 1)
        return PickingTransition(self.state, selection_invalidated, preview_invalidated)

    def update_settings(
        self,
        *,
        probe_radius_logical_px: float,
        pick_nearest_on_probe_miss: bool,
    ) -> PickingTransition:
        """Apply demo-1 picking settings and repick the current hover cursor."""

        radius = float(probe_radius_logical_px)
        cone_half_angle(self._camera.intrinsics, self._logical_size, radius)
        fallback = bool(pick_nearest_on_probe_miss)
        if radius == self._probe_radius and fallback == self._pick_nearest_on_probe_miss:
            return PickingTransition(self.state)
        self._probe_radius = radius
        self._pick_nearest_on_probe_miss = fallback
        if self.state.selection is not None or self.state.cursor_logical is None:
            return PickingTransition(self.state)
        previous = self.state.hover
        result = pick_perspective(
            self._frame,
            self._camera,
            self.state.cursor_logical,
            self._logical_size,
            probe_radius_logical_px=self._probe_radius,
            fallback_to_nearest=self._pick_nearest_on_probe_miss,
        )
        changed = (None if result is None else result.point_index) != (
            None if previous is None else previous.point_index
        )
        generations = self.state.generations.advance("selection_revision") if changed else self.state.generations
        self.state = replace(
            self.state,
            generations=generations,
            hover=result,
            preview_revision=self.state.preview_revision + int(changed),
        )
        return PickingTransition(self.state, preview_invalidated=changed)

    def overlay(
        self, *, guide_length_m: float | None = None, segments: int = 24,
        seed_radius_m: float = 0.0, show_hover_radius: bool = True,
    ) -> PickingOverlay:
        camera, cursor, size = self._camera, self.state.cursor_logical, self._logical_size
        if self.state.selection is not None and self._selection_view is not None and cursor is None:
            cursor = self._selection_view[1]
        if self.state.selection is not None and self._selection_view is not None and self._selection_view[0].role != camera.role:
            camera, cursor, size = self._selection_view
        cursor = cursor or (size[0] * 0.5, size[1] * 0.5)
        origin, direction = perspective_ray(camera, cursor[0], cursor[1], size)
        angle = cone_half_angle(camera.intrinsics, size, self._probe_radius)
        displayed = self.state.displayed_seed
        if guide_length_m is None:
            guide_length_m = (
                displayed.depth_m
                if isinstance(displayed, PickResult) and displayed.depth_m > 1.0e-6
                else min(10.0, self._camera.intrinsics.far_m)
            )
        height = camera.intrinsics.orthographic_height_m
        parallel_radius = None if height is None else self._probe_radius * height * max(camera.intrinsics.aspect_ratio / size[0], 1.0 / size[1])
        guide = build_cone_guide(origin, direction, angle, guide_length_m, segments=segments, parallel_radius_m=parallel_radius)
        if displayed is None:
            seed = None
        elif isinstance(displayed, SelectionSnapshot):
            seed = build_seed_layer(
                displayed.seed_world_m,
                used_fallback=displayed.used_fallback,
            )
        else:
            seed = build_seed_layer(
                displayed.point_world_m,
                used_fallback=displayed.used_fallback,
            )
        radius = None
        if displayed is not None:
            locked = isinstance(displayed, SelectionSnapshot)
            if locked or (show_hover_radius and self.state.cursor_logical is not None):
                radius = build_seed_radius_circle(
                    camera,
                    displayed.seed_world_m if locked else displayed.point_world_m,
                    seed_radius_m,
                    cursor_logical=None if locked else self.state.cursor_logical,
                    logical_size=size,
                )
        return PickingOverlay(seed, guide, radius)


def build_seed_radius_circle(
    camera: CameraSnapshot,
    seed_world_m: Vec3,
    radius_m: float,
    *,
    cursor_logical: tuple[float, float] | None = None,
    logical_size: tuple[float, float] = (1.0, 1.0),
    segments: int = 64,
) -> LineLayerPacket | None:
    """Demo-1 camera-plane circle, adapted to immutable perspective packets.

    Hover uses the cursor ray at the picked point's camera-forward depth;
    locked selection uses the actual point. The world-space radius projects
    to focal_length_px * radius_m / depth, not a fixed pixel probe radius.
    """

    radius_m = float(radius_m)
    if not math.isfinite(radius_m) or radius_m <= 1.0e-8:
        return None
    if segments < 8:
        raise ValueError("circle segments must be at least 8")
    origin = camera.pose_world.position_m
    forward = camera.forward
    depth = sum((point - eye) * axis for point, eye, axis in zip(seed_world_m, origin, forward))
    if not math.isfinite(depth) or depth < camera.intrinsics.near_m or depth > camera.intrinsics.far_m:
        return None
    center = seed_world_m
    if cursor_logical is not None:
        ray_origin, direction = perspective_ray(camera, *cursor_logical, logical_size)
        # PickResult.depth_m is measured along a ray. Using it here would
        # change the apparent radius/center toward the viewport corners.
        ray_forward = sum(component * axis for component, axis in zip(direction, forward))
        center = _add(ray_origin, _mul(direction, depth / ray_forward))
    right, up = camera.right, camera.up
    ring = [
        _add(center, _add(
            _mul(right, radius_m * math.cos(math.tau * index / segments)),
            _mul(up, radius_m * math.sin(math.tau * index / segments)),
        ))
        for index in range(segments)
    ]
    return build_line_layer(
        RenderLayer.SEED_RADIUS,
        [vertex for index in range(segments) for vertex in (ring[index], ring[(index + 1) % segments])],
    )


def build_cone_guide(
    origin: Vec3,
    direction: Vec3,
    half_angle_radians: float,
    length_m: float,
    *,
    segments: int = 24,
    parallel_radius_m: float | None = None,
) -> LineLayerPacket:
    length_m = float(length_m)
    if not math.isfinite(length_m) or length_m <= 0.0 or segments < 8:
        raise ValueError("guide length must be positive and segments at least 8")
    direction = _normalize(direction)
    helper = (0.0, 1.0, 0.0) if abs(direction[1]) < 0.9 else (1.0, 0.0, 0.0)
    right = _normalize(cross_vec3(direction, helper))
    up = _normalize(cross_vec3(right, direction))
    center = _add(origin, _mul(direction, length_m))
    radius = length_m * math.tan(float(half_angle_radians)) if parallel_radius_m is None else parallel_radius_m
    ring = [
        _add(center, _add(_mul(right, radius * math.cos(math.tau * index / segments)), _mul(up, radius * math.sin(math.tau * index / segments))))
        for index in range(segments)
    ]
    vertices: list[Vec3] = [origin, center]
    for index, point in enumerate(ring):
        vertices.extend((point, ring[(index + 1) % segments]))
    for index in range(0, segments, max(1, segments // 4)):
        start = origin if parallel_radius_m is None else _add(ring[index], _mul(direction, -length_m))
        vertices.extend((start, ring[index]))
    return build_line_layer(RenderLayer.PICKING_GUIDE, vertices)


def _require_monotonic(previous: GenerationSnapshot, current: GenerationSnapshot) -> None:
    for name in previous.__dataclass_fields__:
        if getattr(current, name) < getattr(previous, name):
            raise ValueError(f"generation axis {name} cannot move backward")


def _require_frame_generation(frame: ScanFrame, generations: GenerationSnapshot) -> None:
    if generations.sensor != frame.generation:
        raise ValueError("generation snapshot sensor must match frame generation")


def _size(value) -> tuple[float, float]:
    width, height = float(value[0]), float(value[1])
    if not all(math.isfinite(item) and item > 0.0 for item in (width, height)):
        raise ValueError("logical size must be finite and positive")
    return width, height


def _add(a, b): return tuple(left + right for left, right in zip(a, b))
def _mul(value, scale): return tuple(component * scale for component in value)
def _normalize(value):
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1.0e-12:
        raise ValueError("direction must be non-zero")
    return tuple(component / length for component in value)


__all__ = (
    "PickResult", "PickingController", "PickingOverlay", "PickingState",
    "PickingTransition", "build_cone_guide", "build_seed_radius_circle", "cone_half_angle", "pick_perspective",
    "native_picker_available",
)
