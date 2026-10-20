"""Orthographic scene-view math; UI integration is deliberately separate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from .camera import CameraRole


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("orthographic extent must be finite and positive")
    return value


def projection_matrix(height_m, aspect, near_m, far_m):
    """Column-major RH OpenGL matrix: camera -near/-far map to -1/+1."""
    height, aspect = _positive(height_m), _positive(aspect)
    near, far = _positive(near_m), _positive(far_m)
    if far <= near:
        raise ValueError("clip planes must be ordered")
    return (2 / (height * aspect), 0., 0., 0.,
            0., 2 / height, 0., 0.,
            0., 0., -2 / (far - near), 0.,
            0., 0., -(far + near) / (far - near), 1.)


def screen_ray(camera, x, y, logical_size, height_m):
    """Top-left logical cursor -> parallel ray on camera plane, not near plane."""
    if camera.role is not CameraRole.SCENE:
        raise ValueError("orthographic projection is scene-only")
    width, height = map(_positive, logical_size)
    extent = _positive(height_m)
    if not all(math.isfinite(float(v)) for v in (x, y)):
        raise ValueError("cursor must be finite")
    right = (float(x) / width - .5) * extent * camera.intrinsics.aspect_ratio
    up = (.5 - float(y) / height) * extent
    origin = tuple(camera.pose_world.position_m[i] + right * camera.right[i] + up * camera.up[i]
                   for i in range(3))
    return origin, camera.forward


def matching_height(camera):
    """Preserve perspective scale at the orbit target when switching modes."""
    if camera.role is not CameraRole.SCENE:
        raise ValueError("orthographic projection is scene-only")
    depth = sum((camera.target_world_m[i] - camera.pose_world.position_m[i]) * camera.forward[i]
                for i in range(3))
    return 2 * _positive(depth) * math.tan(camera.intrinsics.vertical_fov_radians / 2)


def ruler_length_m(height_m, viewport_height_px, ruler_width_px):
    """Demo 1 semantics: fixed screen length with a live world-length label."""
    return _positive(height_m) / _positive(viewport_height_px) * _positive(ruler_width_px)


@dataclass(frozen=True)
class RulerTick:
    offset_px: float
    level: int  # 0: major, 1: half, 2: tenth


@dataclass(frozen=True)
class RulerScale:
    significand: str
    exponent: int
    ticks: tuple[RulerTick, ...]


RULER_HALF_TICK_GAP_PX = 100.0
RULER_TENTH_TICK_GAP_PX = 200.0


def ruler_scale(length, span):
    """Scientific scale with screen-density-dependent, three-level ticks.

    Positions use the unrounded length so rounding the label never stretches
    the measured interval. The two endpoints are drawn separately by the UI.
    """
    length, span = _positive(length), _positive(span)
    significand, exponent_text = f"{length:.1e}".split("e")
    exponent = int(exponent_text)
    # Divide in log space to avoid overflow at extreme (but finite) extents.
    intervals = 10 ** (math.log10(length) - exponent)
    major_gap = span / intervals
    subdivisions = (10 if major_gap >= RULER_TENTH_TICK_GAP_PX else
                    2 if major_gap >= RULER_HALF_TICK_GAP_PX else 1)
    count = intervals * subdivisions
    ticks = []
    for index in range(1, math.ceil(count)):
        offset = span * index / count
        if offset >= span - 1e-9:
            continue
        level = (0 if index % subdivisions == 0 else
                 1 if subdivisions == 2 or index % 5 == 0 else 2)
        ticks.append(RulerTick(offset, level))
    return RulerScale(significand, exponent, tuple(ticks))


@dataclass
class MovableRuler:
    """Demo-1 anchor/grab-offset semantics; no UI objects or camera mutations."""
    PADDING: ClassVar[float] = 8.0
    PANEL_HEIGHT: ClassVar[float] = 60.0
    position: tuple[float, float] | None = None
    grab: tuple[float, float] | None = None
    size: tuple[float, float] = (1.0, 1.0)
    main_span: float = 180.0
    resize_grab: tuple | None = None
    EDGE_GRIP: ClassVar[float] = 8.0
    MIN_SPAN: ClassVar[float] = 48.0

    def layout(self, size, *, primary=True):
        self.size = tuple(map(float, size))
        width, height = self.size
        if primary:
            span = max(1.0, min(self.main_span, width - 2 * self.PADDING))
        else:
            span = max(1.0, 0.5 * min(180.0, width - 2 * self.PADDING - 16.0))
        x, y = self.position or (16.0, height - self.PANEL_HEIGHT - 16.0)
        x = max(0.0, min(x, width - span - 2 * self.PADDING))
        y = max(0.0, min(y, height - self.PANEL_HEIGHT))
        return x, y, span

    def cancel_drag(self):
        self.grab = None
        self.resize_grab = None

    def pointer(self, action, x, y, button):
        left, top, span = self.layout(self.size)
        hit = left <= x <= left + span + 2 * self.PADDING and top <= y <= top + self.PANEL_HEIGHT
        if action == "begin" and button == 0 and hit:
            self.cancel_drag()
            right = left + span + 2 * self.PADDING
            if x <= left + self.EDGE_GRIP or x >= right - self.EDGE_GRIP:
                self.resize_grab = (x <= left + self.EDGE_GRIP, x, left, right, top)
            else:
                self.grab = (x - left, y - top)
            return True
        if self.resize_grab is not None:
            from_left, start_x, start_left, start_right, top = self.resize_grab
            if action in ("move", "end"):
                minimum = min(self.MIN_SPAN, max(1.0, self.size[0] - 2 * self.PADDING)) + 2 * self.PADDING
                if from_left:
                    left = max(0.0, min(start_left + x - start_x, start_right - minimum))
                    right = start_right
                else:
                    left = start_left
                    right = min(self.size[0], max(start_right + x - start_x, left + minimum))
                self.main_span = right - left - 2 * self.PADDING
                self.position = (left, top)
            if action == "end":
                self.cancel_drag()
            return True
        if self.grab is not None:
            if action in ("move", "end"):
                self.position = (x - self.grab[0], y - self.grab[1])
                self.position = self.layout(self.size)[:2]
            if action == "end":
                self.grab = None
            return True
        return action == "move" and hit
