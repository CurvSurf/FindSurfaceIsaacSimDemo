"""Framework-light camera controls and a narrow Kit keyboard bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from .spatial_math import cross_vec3, quat_multiply_xyzw, rotate_vector_unit_xyzw
from .camera import (
    CameraRole, CameraSnapshot, CameraSnapshotCommand, OrbitMode,
    look_orientation, quat_axis_angle,
)
from .domain import Pose
from .kit_input import kit_enum_name, kit_key_name


class ShortcutIntent(str, Enum):
    RESET_PRIMARY_CAMERA = "reset-primary-camera"
    TOGGLE_PRIMARY_VIEW = "toggle-primary-view"


@dataclass(frozen=True)
class PointerResult:
    captured: bool = False
    pick_click: tuple[float, float] | None = None
    command: CameraSnapshotCommand | None = None


class CameraInputController:
    """Own key/pointer arbitration while publishing immutable camera values."""

    MOVEMENT_KEYS = frozenset(("W", "A", "S", "D", "UP", "LEFT", "DOWN", "RIGHT", "Q", "E", "Z", "C"))
    SCENE_DOLLY_LOG_STEP = 0.12

    def __init__(
        self,
        snapshot: CameraSnapshot,
        *,
        move_speed_m_s: float = 2.0,
        turn_radians_per_logical_pixel: float = 0.006,
        roll_radians_s: float = math.radians(60.0),
        drag_threshold_logical_px: float = 3.0,
    ) -> None:
        self.snapshot = snapshot
        self.move_speed_m_s = float(move_speed_m_s)
        self.turn_per_pixel = float(turn_radians_per_logical_pixel)
        self.roll_radians_s = float(roll_radians_s)
        self.drag_threshold = float(drag_threshold_logical_px)
        if min(self.move_speed_m_s, self.turn_per_pixel, self.roll_radians_s, self.drag_threshold) <= 0.0:
            raise ValueError("camera input rates and threshold must be positive")
        self._pressed: set[str] = set()
        self._modifiers: set[str] = set()
        self._focused = True
        self._pointer = None

    @property
    def pointer_captured(self) -> bool:
        return self._pointer is not None

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed)

    def restore(self, snapshot: CameraSnapshot) -> None:
        if snapshot.role is not self.snapshot.role:
            raise ValueError("camera role cannot change during restore")
        self.snapshot = snapshot

    def set_focused(self, focused: bool) -> None:
        self._focused = bool(focused)
        if not self._focused:
            self._pressed.clear()
            self._modifiers.clear()
            self._pointer = None

    def set_modifier(self, modifier: str, pressed: bool) -> None:
        if pressed:
            self._modifiers.add(modifier.lower())
        else:
            self._modifiers.discard(modifier.lower())

    @property
    def keyboard_speed_scale(self) -> float:
        return 0.05 if "ctrl" in self._modifiers else 10.0 if "shift" in self._modifiers else 1.0

    def key_event(self, key: str, pressed: bool, *, ui_consumed: bool = False, repeat: bool = False):
        key = str(key).upper()
        if ui_consumed or not self._focused:
            return None
        if key == "R" and pressed and not repeat:
            return ShortcutIntent.RESET_PRIMARY_CAMERA
        if key == "P" and pressed and not repeat:
            return ShortcutIntent.TOGGLE_PRIMARY_VIEW
        if key in self.MOVEMENT_KEYS:
            if pressed:
                self._pressed.add(key)
            else:
                self._pressed.discard(key)
        return None

    def update(self, dt_s: float) -> CameraSnapshot:
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and non-negative")
        if not self._focused or dt_s == 0.0:
            return self.snapshot
        before = self.snapshot
        forward = before.up if before.role is CameraRole.SCENE else before.forward
        right = before.right
        # Keep physical keys separate: releasing UP must not release a held W.
        up = bool(self._pressed & {"W", "UP"})
        down = bool(self._pressed & {"S", "DOWN"})
        left = bool(self._pressed & {"A", "LEFT"})
        right_pressed = bool(self._pressed & {"D", "RIGHT"})
        velocity = (0.0, 0.0, 0.0)
        velocity = _add(velocity, _mul(forward, float(up) - float(down)))
        velocity = _add(velocity, _mul(right, float(right_pressed) - float(left)))
        if before.role is CameraRole.SCANNER:
            velocity = _add(
                velocity,
                _mul(before.up, float("E" in self._pressed) - float("Q" in self._pressed)),
            )
        if _length(velocity) > 1.0e-12:
            delta = _mul(_normalize(velocity), self.move_speed_m_s * dt_s * self.keyboard_speed_scale)
            pose = replace(before.pose_world, position_m=_add(before.pose_world.position_m, delta))
            target = before.target_world_m if before.role is CameraRole.SCANNER else _add(before.target_world_m, delta)
            self.snapshot = replace(before, pose_world=pose, target_world_m=target)
        if before.role is CameraRole.SCANNER:
            roll = float("C" in self._pressed) - float("Z" in self._pressed)
            if roll:
                self._rotate_about(before.forward, roll * self.roll_radians_s * dt_s * self.keyboard_speed_scale)
        return self.snapshot

    def reset_scanner_up(self) -> CameraSnapshot:
        if self.snapshot.role is not CameraRole.SCANNER:
            return self.snapshot
        forward = self.snapshot.forward
        if abs(forward[1]) > 0.999:
            forward = _normalize((forward[0], math.copysign(0.999, forward[1]), forward[2]))
        self.snapshot = replace(
            self.snapshot,
            pose_world=replace(self.snapshot.pose_world, orientation_xyzw=look_orientation(forward, (0.0, 1.0, 0.0))),
        )
        return self.snapshot

    def pointer_begin(
        self, button: int, logical_x: float, logical_y: float,
        logical_size: tuple[float, float], *, ui_consumed: bool = False,
    ) -> PointerResult:
        if ui_consumed or not self._focused or self._pointer is not None:
            return PointerResult()
        button = int(button)
        allowed = button in (0, 1)
        if not allowed:
            return PointerResult()
        position = (float(logical_x), float(logical_y))
        self._pointer = {
            "button": button, "start": position, "last": position,
            "size": (float(logical_size[0]), float(logical_size[1])),
            "before": self.snapshot, "dragged": False,
        }
        return PointerResult(captured=True)

    def pointer_move(self, logical_x: float, logical_y: float, *, ui_consumed: bool = False) -> PointerResult:
        # A captured pointer continues through overlay boundaries. ui_consumed
        # applies only before capture, never halfway through a gesture.
        if self._pointer is None:
            return PointerResult()
        current = (float(logical_x), float(logical_y))
        start, last = self._pointer["start"], self._pointer["last"]
        total = _length2((current[0] - start[0], current[1] - start[1]))
        if total < self.drag_threshold and not self._pointer["dragged"]:
            return PointerResult(captured=True)
        self._pointer["dragged"] = True
        dx, dy = current[0] - last[0], current[1] - last[1]
        if self._pointer["button"] == 1 and self.snapshot.role is CameraRole.SCENE:
            width, height = self._pointer["size"]
            self._roll_about_screen_center(last, current, (width * 0.5, height * 0.5))
        elif self._pointer["button"] == 1:
            self._scanner_look(dx, dy)
        elif self.snapshot.role is CameraRole.SCENE:
            if self.snapshot.orbit_mode is OrbitMode.TRACKBALL:
                self._scene_orbit(
                    current[0] - start[0], current[1] - start[1],
                    start_camera=self._pointer["before"],
                )
            else:
                self._scene_orbit(dx, dy)
        self._pointer["last"] = current
        return PointerResult(captured=True)

    def pointer_end(self, button: int, logical_x: float, logical_y: float) -> PointerResult:
        if self._pointer is None or int(button) != self._pointer["button"]:
            return PointerResult()
        pointer = self._pointer
        self._pointer = None
        position = (float(logical_x), float(logical_y))
        if pointer["button"] == 0 and not pointer["dragged"]:
            return PointerResult(pick_click=position)
        if self.snapshot == pointer["before"]:
            return PointerResult()
        command = CameraSnapshotCommand(
            f"Move {self.snapshot.role.value} camera", self.restore, pointer["before"], self.snapshot
        )
        return PointerResult(command=command)

    def wheel_dolly(self, steps: float, *, ui_consumed: bool = False) -> PointerResult:
        """Move a perspective scene camera toward/away from its orbit target."""

        steps = float(steps)
        if not math.isfinite(steps):
            raise ValueError("wheel steps must be finite")
        if (
            ui_consumed
            or not self._focused
            or self._pointer is not None
            or self.snapshot.role is not CameraRole.SCENE
            or abs(steps) <= 1.0e-8
        ):
            return PointerResult()
        before = self.snapshot
        if before.intrinsics.orthographic_height_m is not None:
            log_height = math.log(before.intrinsics.orthographic_height_m) - steps * self.SCENE_DOLLY_LOG_STEP
            height = math.exp(max(math.log(0.0001), min(math.log(1000000.0), log_height)))
            self.snapshot = replace(before, intrinsics=replace(before.intrinsics, orthographic_height_m=height))
            return PointerResult(command=CameraSnapshotCommand(
                "Zoom orthographic scene camera", self.restore, before, self.snapshot))
        distance = _length(_sub(before.pose_world.position_m, before.target_world_m))
        minimum = max(0.001, before.intrinsics.near_m * 1.05)
        maximum = max(minimum, before.intrinsics.far_m * 0.95)
        distance = max(minimum, min(maximum, distance))
        log_distance = max(
            math.log(minimum),
            min(
                math.log(maximum),
                math.log(distance) - steps * self.SCENE_DOLLY_LOG_STEP,
            ),
        )
        next_distance = math.exp(log_distance)
        if abs(next_distance - distance) <= 1.0e-12:
            return PointerResult()
        position = _sub(before.target_world_m, _mul(before.forward, next_distance))
        self.snapshot = replace(
            before,
            pose_world=replace(before.pose_world, position_m=position),
        )
        return PointerResult(command=CameraSnapshotCommand(
            "Dolly scene camera",
            self.restore,
            before,
            self.snapshot,
        ))

    def command_since(self, before: CameraSnapshot, name: str | None = None) -> CameraSnapshotCommand | None:
        if before == self.snapshot:
            return None
        return CameraSnapshotCommand(name or f"Move {self.snapshot.role.value} camera", self.restore, before, self.snapshot)

    def _scanner_look(self, dx: float, dy: float) -> None:
        yaw = quat_axis_angle(self.snapshot.up, -dx * self.turn_per_pixel)
        orientation = quat_multiply_xyzw(yaw, self.snapshot.pose_world.orientation_xyzw)
        pitched = replace(self.snapshot, pose_world=replace(self.snapshot.pose_world, orientation_xyzw=orientation))
        # omni.ui Y grows downward: dragging upward (negative dy) must pitch
        # the scanner view upward. This is an input-space sign only; camera,
        # projection, picking, and renderer coordinates remain unchanged.
        pitch = quat_axis_angle(pitched.right, -dy * self.turn_per_pixel)
        orientation = quat_multiply_xyzw(pitch, orientation)
        self.snapshot = replace(pitched, pose_world=replace(pitched.pose_world, orientation_xyzw=orientation))

    def _scene_orbit(self, dx: float, dy: float, *, start_camera: CameraSnapshot | None = None) -> None:
        camera = start_camera or self.snapshot
        offset = _sub(camera.pose_world.position_m, camera.target_world_m)
        if camera.orbit_mode is OrbitMode.TURNTABLE:
            yaw = quat_axis_angle((0.0, 1.0, 0.0), -dx * self.turn_per_pixel)
            offset = rotate_vector_unit_xyzw(yaw, offset)
            right = _normalize(cross_vec3(_normalize(_mul(offset, -1.0)), (0.0, 1.0, 0.0)))
            pitch = quat_axis_angle(right, -dy * self.turn_per_pixel)
            candidate = rotate_vector_unit_xyzw(pitch, offset)
            if abs(_normalize(_mul(candidate, -1.0))[1]) < 0.999:
                offset = candidate
            up = (0.0, 1.0, 0.0)
        else:
            # Demo 1 object-like orbit: use the gesture-start camera basis and
            # total screen delta, so the result is independent of move events.
            axis = _normalize(_add(_mul(camera.right, -dy), _mul(camera.up, -dx)))
            rotation = quat_axis_angle(axis, math.hypot(dx, dy) * self.turn_per_pixel)
            offset = rotate_vector_unit_xyzw(rotation, offset)
            up = rotate_vector_unit_xyzw(rotation, camera.up)
        position = _add(camera.target_world_m, offset)
        self.snapshot = replace(camera, pose_world=Pose(position, look_orientation(_mul(offset, -1.0), up)))

    def _roll_about_screen_center(self, previous, current, center) -> None:
        a = (previous[0] - center[0], center[1] - previous[1])
        b = (current[0] - center[0], center[1] - current[1])
        if _length2(a) <= 1.0e-9 or _length2(b) <= 1.0e-9:
            return
        radians = math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
        self._rotate_about(self.snapshot.forward, radians)

    def _rotate_about(self, axis, radians) -> None:
        rotation = quat_axis_angle(axis, radians)
        orientation = quat_multiply_xyzw(rotation, self.snapshot.pose_world.orientation_xyzw)
        self.snapshot = replace(self.snapshot, pose_world=replace(self.snapshot.pose_world, orientation_xyzw=orientation))


class KitCameraKeyboardBridge:
    """Subscribe to Kit keyboard events without coupling camera math to Kit."""

    def __init__(self, controller: CameraInputController, *, input_interface=None, keyboard=None) -> None:
        self.controller = controller
        self._input = input_interface
        self._keyboard = keyboard
        self._subscription = None

    def start(self) -> None:
        if self._subscription is not None:
            return
        if self._input is None or self._keyboard is None:
            import carb.input
            import omni.appwindow
            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("the default Kit app window is unavailable")
            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self.on_keyboard_event)

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None
        self.controller.set_focused(False)

    def on_keyboard_event(self, event) -> bool:
        key = kit_key_name(getattr(event, "input", None))
        kind = kit_enum_name(getattr(event, "type", None))
        if kind not in ("KEY_PRESS", "KEY_REPEAT", "KEY_RELEASE"):
            return True
        repeat = kind == "KEY_REPEAT"
        pressed = kind in ("KEY_PRESS", "KEY_REPEAT")
        intent = self.controller.key_event(key, pressed, repeat=repeat)
        handled = key in self.controller.MOVEMENT_KEYS or intent is not None
        return not handled


def physical_to_logical(x: float, y: float, dpi_scale: float) -> tuple[float, float]:
    dpi_scale = float(dpi_scale)
    if not math.isfinite(dpi_scale) or dpi_scale <= 0.0:
        raise ValueError("dpi_scale must be finite and positive")
    return float(x) / dpi_scale, float(y) / dpi_scale


def update_event_dt_seconds(event) -> float:
    """Read Kit's mapping-style update event, with a payload fallback for tests."""

    try:
        value = event["dt"]
    except (KeyError, TypeError, AttributeError):
        payload = getattr(event, "payload", {}) or {}
        value = payload.get("dt", 0.0) if hasattr(payload, "get") else 0.0
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _mul(value, scale): return tuple(component * scale for component in value)
def _length(value): return math.sqrt(sum(component * component for component in value))
def _length2(value): return math.hypot(value[0], value[1])
def _normalize(value): return _mul(value, 1.0 / _length(value))


__all__ = (
    "CameraInputController", "KitCameraKeyboardBridge", "PointerResult",
    "ShortcutIntent", "physical_to_logical", "update_event_dt_seconds",
)
