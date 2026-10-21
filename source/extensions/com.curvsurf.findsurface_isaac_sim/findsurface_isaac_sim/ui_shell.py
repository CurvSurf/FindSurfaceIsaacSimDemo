"""Toolkit-neutral presentation model for the application chrome."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Mapping

from .command_facade import CommandFacade, CommandId, CommandPresentation, PrimaryView, SensorMode
from .preview_capture import CaptureAvailability, CaptureBlockReason
from .result_overlay import ResultOverlayPresentation
from .rtx_policy import PerformanceWarning, RtxSceneState


class PropertyTab(str, Enum):
    FINDSURFACE = "FindSurface"
    GEOMETRIES = "Geometries"
    OBJECTS = "Objects"
    CAMERA = "Camera"
    SCANNER = "Scanner"
    GRAPHICS = "Graphics"
    SETTINGS = "Settings"


class ModalKind(str, Enum):
    PROGRESS = "progress"
    ERROR = "error"
    PERFORMANCE = "performance"
    HELP = "help"


class ModalPhase(str, Enum):
    HIDDEN = "HIDDEN"
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ModalToken:
    generation: int


@dataclass(frozen=True)
class ModalSnapshot:
    generation: int = 0
    kind: ModalKind = ModalKind.PROGRESS
    phase: ModalPhase = ModalPhase.HIDDEN
    title: str = ""
    detail: str = ""
    progress: float = 0.0
    cancellable: bool = False
    error: str | None = None

    @property
    def visible(self) -> bool:
        return self.phase is not ModalPhase.HIDDEN

    @property
    def input_blocked(self) -> bool:
        return self.phase is ModalPhase.ACTIVE


class ModalController:
    """Generation-safe modal state; stale progress and completion are discarded."""

    def __init__(self) -> None:
        self._generation = 0
        self._snapshot = ModalSnapshot()

    @property
    def snapshot(self) -> ModalSnapshot:
        return self._snapshot

    def begin(
        self,
        kind: ModalKind,
        title: str,
        detail: str,
        *,
        cancellable: bool = False,
    ) -> ModalToken:
        self._generation += 1
        token = ModalToken(self._generation)
        self._snapshot = ModalSnapshot(
            token.generation,
            ModalKind(kind),
            ModalPhase.ACTIVE,
            str(title),
            str(detail),
            0.0,
            bool(cancellable),
        )
        return token

    def update(self, token: ModalToken, *, detail: str, progress: float) -> bool:
        if not self._active(token):
            return False
        self._snapshot = replace(
            self._snapshot,
            detail=str(detail),
            progress=max(0.0, min(1.0, float(progress))),
        )
        return True

    def succeed(self, token: ModalToken, detail: str = "Complete") -> bool:
        return self._finish(token, ModalPhase.SUCCEEDED, detail, None)

    def fail(self, token: ModalToken, error: BaseException | str) -> bool:
        message = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
        return self._finish(token, ModalPhase.FAILED, "The previous state was restored.", message)

    def cancel(self, token: ModalToken) -> bool:
        if not self._active(token) or not self._snapshot.cancellable:
            return False
        return self._finish(token, ModalPhase.CANCELLED, "Cancelled; the previous state was restored.", None)

    def cancel_current(self) -> bool:
        """Cancel the visible generation without exposing its token to the UI adapter."""
        return self.cancel(ModalToken(self._snapshot.generation))

    def dismiss(self, token: ModalToken | None = None) -> bool:
        if token is not None and token.generation != self._snapshot.generation:
            return False
        self._snapshot = replace(self._snapshot, phase=ModalPhase.HIDDEN)
        return True

    def _active(self, token: ModalToken) -> bool:
        return token.generation == self._snapshot.generation and self._snapshot.phase is ModalPhase.ACTIVE

    def _finish(self, token: ModalToken, phase: ModalPhase, detail: str, error: str | None) -> bool:
        if not self._active(token):
            return False
        self._snapshot = replace(
            self._snapshot,
            phase=phase,
            detail=str(detail),
            progress=1.0,
            error=error,
        )
        return True


@dataclass(frozen=True)
class SensorUiStatus:
    phase: str = "DISABLED"
    profile_id: str = "default"
    frame_id: int | None = None
    point_count: int = 0
    update_hz: float = 0.0
    complete_scan_hz: float = 0.0
    follows_camera: bool = False
    mode: SensorMode = SensorMode.FROZEN
    no_returns: bool = False
    range_summary: str = ""
    findsurface_ms: float | None = None


@dataclass(frozen=True)
class PropertySection:
    tab: PropertyTab
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OverlayPresentation:
    sensor: str
    detection: str
    analysis: str
    rtx: str
    result: ResultOverlayPresentation = ResultOverlayPresentation()


@dataclass(frozen=True)
class HelpPresentation:
    shortcuts: tuple[tuple[str, str], ...]
    capture_contract: str
    camera_contract: str
    exit_contract: str


@dataclass(frozen=True)
class PerformanceWarningPresentation:
    title: str
    rates: str
    point_status: str
    sensor_cost_notice: str
    transition_notice: str
    keep_label: str = "Keep RTX"
    switch_label: str = "Switch to OpenGL"


@dataclass(frozen=True)
class ShellSnapshot:
    commands: tuple[CommandPresentation, ...]
    properties: tuple[PropertySection, ...]
    overlays: OverlayPresentation
    modal: ModalSnapshot
    help: HelpPresentation

    def command(self, command_id: CommandId | str) -> CommandPresentation:
        command_id = CommandId(command_id)
        return next(item for item in self.commands if item.command_id is command_id)


HELP_PRESENTATION = HelpPresentation(
    shortcuts=(
        ("Ctrl+C", "Capture (takes priority over Copy)"),
        ("Ctrl+Z / Ctrl+Shift+Z", "Undo / Redo"),
        ("Ctrl+R", "Reset captured geometries"),
        ("P / R", "Swap primary view / reset primary camera"),
        ("` / 1-6", "Inspect/default / Auto-Torus"),
        ("I / M / B / O", "Inlier / Mesh.D / Mesh.B / Outlier"),
        ("Alt/Option+Hover / Alt/Option+LMB", "Show / select the last successful seed; hover detection continues"),
        ("Shift+/", "Help"),
        ("Alt+F4 or Ctrl+Shift+Q", "Safe Exit"),
    ),
    capture_contract=(
        "Capture requires a disabled sensor, point-selection mode, and the latest successful result "
        "for the current frame, selection, and options. Capture never disables the sensor automatically."
    ),
    camera_contract=(
        "Scanner uses FPS controls; Scene uses orbit/trackball controls and mouse-wheel perspective dolly. "
        "PIP is intentionally inert."
    ),
    exit_contract="Toolbar, Alt+F4, Ctrl+Shift+Q, and window close use one ordered Safe Exit path.",
)


class ClassicShellModel:
    """Project domain owners into one immutable toolbar/property/overlay snapshot."""

    def __init__(self, facade: CommandFacade, modal: ModalController | None = None) -> None:
        self.facade = facade
        self.modal = modal or ModalController()

    def snapshot(
        self,
        *,
        capture: CaptureAvailability | None = None,
        sensor: SensorUiStatus | None = None,
        rtx: RtxSceneState | None = None,
        detection_status: str = "No current detection",
        analysis_status: str = "No current analysis",
        result_overlay: ResultOverlayPresentation | None = None,
    ) -> ShellSnapshot:
        capture = capture or CaptureAvailability(
            False,
            (CaptureBlockReason.NO_SUCCESSFUL_RESULT,),
            "No successful selected-point detection is available.",
        )
        sensor = sensor or SensorUiStatus(
            profile_id=self.facade.snapshot.sensor.profile_id,
            phase="RUNNING" if self.facade.snapshot.sensor.enabled else "DISABLED",
            follows_camera=self.facade.snapshot.sensor.follows_camera,
            mode=self.facade.snapshot.sensor.mode,
        )
        commands = list(self.facade.command_matrix(capture_ready=capture.enabled))
        for index, item in enumerate(commands):
            if self.modal.snapshot.input_blocked and item.command_id is not CommandId.SAFE_EXIT:
                commands[index] = replace(item, enabled=False, tooltip="Unavailable while a modal operation is active.")
            elif item.command_id is CommandId.CAPTURE:
                commands[index] = replace(item, tooltip=capture.tooltip)
            elif item.command_id in {
                CommandId.SENSOR_FROZEN,
                CommandId.SENSOR_LIVE_FIXED,
                CommandId.SENSOR_LIVE_FOLLOW,
            }:
                ready = sensor.phase in {"RUNNING", "DISABLED"}
                mode = {
                    CommandId.SENSOR_FROZEN: SensorMode.FROZEN,
                    CommandId.SENSOR_LIVE_FIXED: SensorMode.LIVE_FIXED,
                    CommandId.SENSOR_LIVE_FOLLOW: SensorMode.LIVE_FOLLOW,
                }[item.command_id]
                checked = sensor.mode is mode
                tooltip = {
                    SensorMode.FROZEN: "Freeze sensor pose and stop point-cloud updates.",
                    SensorMode.LIVE_FIXED: "Keep sensor pose fixed while updating the point cloud.",
                    SensorMode.LIVE_FOLLOW: "Match scanner camera pose and continuously update the point cloud.",
                }[mode]
                if not ready:
                    tooltip = f"Sensor control is unavailable while status is {sensor.phase}."
                commands[index] = replace(item, enabled=ready, checked=checked, tooltip=tooltip)
        properties = self._properties(sensor, rtx, detection_status)
        rtx_text = "Scene renderer: OpenGL"
        if rtx is not None:
            rtx_text = f"Scene renderer: requested {rtx.requested.value}, effective {rtx.effective.value}"
            if rtx.fallback_reason:
                rtx_text += f" - {rtx.fallback_reason}"
            rtx_text += f"; RTX points {rtx.point_overlay.status.value}"
            if rtx.point_overlay.reason:
                rtx_text += f" - {rtx.point_overlay.reason}"
        sensor_overlay = (
            f"{sensor.phase} · {sensor.profile_id} · frame {sensor.frame_id if sensor.frame_id is not None else 'N/A'} · "
            f"{sensor.point_count:,} points · App Scan {sensor.complete_scan_hz:.1f} Hz · "
            f"FindSurface {f'{sensor.findsurface_ms:.2f} ms' if sensor.findsurface_ms is not None else '-- ms'}"
        )
        if sensor.no_returns:
            sensor_overlay += " · no returns in valid range/FOV"
        overlays = OverlayPresentation(
            sensor=sensor_overlay,
            detection=str(detection_status),
            analysis=str(analysis_status),
            rtx=rtx_text,
            result=result_overlay or ResultOverlayPresentation(),
        )
        return ShellSnapshot(tuple(commands), properties, overlays, self.modal.snapshot, HELP_PRESENTATION)

    def _properties(
        self,
        sensor: SensorUiStatus,
        rtx: RtxSceneState | None,
        detection_status: str,
    ) -> tuple[PropertySection, ...]:
        state = self.facade.snapshot
        detection = state.detection
        objects = ", ".join(f"{item.geometry_type.value}:{'on' if item.enabled else 'off'}" for item in state.objects)
        captures = ", ".join(item.capture_id for item in state.captures) or "None"
        renderer = "OpenGL" if rtx is None else f"{rtx.requested.value} -> {rtx.effective.value}"
        return (
            PropertySection(PropertyTab.FINDSURFACE, (
                ("Mode", "Inspect/default" if state.inspect_mode else detection.geometry_mode.value),
                ("Measurement accuracy", f"{detection.measurement_accuracy_m:.4f} m"),
                ("Mean distance", f"{detection.mean_distance_m:.4f} m"),
                ("Seed radius", f"{detection.seed_radius_m:.4f} m"),
                ("Result", str(detection_status)),
            )),
            PropertySection(PropertyTab.GEOMETRIES, (
                ("Captured", captures),
                ("Selected", state.selected_capture_id or "None"),
            )),
            PropertySection(PropertyTab.OBJECTS, (("As-designed", objects),)),
            PropertySection(PropertyTab.CAMERA, (
                ("Scanner position", _vector(state.scanner_pose.position_m)),
                ("Scene position", _vector(state.scene_pose.position_m)),
                ("Primary", state.primary_view.value),
            )),
            PropertySection(PropertyTab.SCANNER, (
                ("Source", state.sensor.sensor_kind.value),
                ("Profile", sensor.profile_id),
                ("Status", sensor.phase),
                ("Mode", {
                    SensorMode.FROZEN: "Frozen pose / updates off",
                    SensorMode.LIVE_FIXED: "Fixed pose / updates on",
                    SensorMode.LIVE_FOLLOW: "Follow camera / updates on",
                }[sensor.mode]),
                ("Pose", "Follow camera" if sensor.follows_camera else "Fixed"),
                ("Frame / points", f"{sensor.frame_id if sensor.frame_id is not None else 'N/A'} / {sensor.point_count:,}"),
                ("Returns", (
                    f"No object in valid range/FOV ({sensor.range_summary})"
                    if sensor.no_returns and sensor.range_summary
                    else "No object in valid range/FOV"
                    if sensor.no_returns
                    else "Receiving points"
                )),
            )),
            PropertySection(PropertyTab.GRAPHICS, (
                ("Mesh categories", "Plane / Sphere / Cylinder / Cone / Torus"),
                ("Mesh states", "Original / Detected / Captured"),
                ("Point clouds", "Sizes and colors"),
            )),
            PropertySection(PropertyTab.SETTINGS, (
                ("Picking probe", f"{state.interaction.probe_radius_px:.1f} px"),
                ("Probe miss", "Nearest" if state.interaction.pick_nearest_on_probe_miss else "No pick"),
                ("Sticky preview", (
                    f"{state.interaction.sticky_preview_timeout_ms} ms"
                    if state.interaction.sticky_auto_preview else "Off"
                )),
                ("Minimum geometry length", f"{state.interaction.minimum_geometry_length_m:.4f} m"),
                ("Scene renderer", renderer),
                ("Scanner renderer", "OpenGL"),
                ("PIP", "Scene" if state.primary_view is PrimaryView.SCANNER else "Scanner"),
            )),
        )


class ShellCommandController:
    """Dispatch enabled shell commands and shortcuts through one registry."""

    def __init__(
        self,
        model: ClassicShellModel,
        snapshot_provider: Callable[[], ShellSnapshot],
        callbacks: Mapping[CommandId | str, Callable[[], object]],
    ) -> None:
        self.model = model
        self._snapshot_provider = snapshot_provider
        self._callbacks = {CommandId(key): value for key, value in callbacks.items()}
        if any(not callable(value) for value in self._callbacks.values()):
            raise TypeError("command callbacks must be callable")

    def dispatch(self, command_id: CommandId | str):
        command_id = CommandId(command_id)
        presentation = self._snapshot_provider().command(command_id)
        callback = self._callbacks.get(command_id)
        if callback is None or not presentation.enabled:
            return False
        return callback()

    def dispatch_shortcut(self, key: str, *, ctrl: bool = False, shift: bool = False, alt: bool = False):
        command = shortcut_command(
            key,
            ctrl=ctrl,
            shift=shift,
            alt=alt,
            primary=self.model.facade.snapshot.primary_view,
        )
        return False if command is None else self.dispatch(command)


def shortcut_command(
    key: str,
    *,
    ctrl: bool = False,
    shift: bool = False,
    alt: bool = False,
    primary: PrimaryView = PrimaryView.SCANNER,
) -> CommandId | None:
    key = str(key).upper()
    if alt and key == "F4" or ctrl and shift and key == "Q":
        return CommandId.SAFE_EXIT
    if ctrl and shift and key == "Z":
        return CommandId.REDO
    if ctrl and key == "C":
        return CommandId.CAPTURE
    if ctrl and key == "Z":
        return CommandId.UNDO
    if ctrl and key == "R":
        return CommandId.RESET_CAPTURES
    if shift and key in ("/", "SLASH"):
        return CommandId.HELP
    if key == "P":
        return CommandId.PRIMARY_SCENE if primary is PrimaryView.SCANNER else CommandId.PRIMARY_SCANNER
    if key == "R":
        return CommandId.CAMERA_RESET_SCANNER if primary is PrimaryView.SCANNER else CommandId.CAMERA_RESET_SCENE
    direct = {
        "`": CommandId.MODE_INSPECT,
        "GRAVE_ACCENT": CommandId.MODE_INSPECT,
        "1": CommandId.MODE_AUTO,
        "2": CommandId.MODE_PLANE,
        "3": CommandId.MODE_SPHERE,
        "4": CommandId.MODE_CYLINDER,
        "5": CommandId.MODE_CONE,
        "6": CommandId.MODE_TORUS,
        "I": CommandId.VISIBILITY_INLIER,
        "M": CommandId.VISIBILITY_DESIGNED,
        "B": CommandId.VISIBILITY_DETECTED,
        "O": CommandId.VISIBILITY_OUTLIER,
    }
    return direct.get(key)


def build_performance_warning(warning: PerformanceWarning) -> PerformanceWarningPresentation:
    window = warning.window
    return PerformanceWarningPresentation(
        "Performance Warning",
        (
            f"{warning.profile_id}: updates {window.update_hz:.1f}/s, complete scans "
            f"{window.complete_scan_hz:.1f}/s; target {window.target_hz:.1f}/s (90% = {window.minimum_healthy_hz:.1f}/s)."
        ),
        f"RTX point-cloud overlay: {warning.point_overlay_status.value}.",
        "Changing only the scene view does not remove the RTX cost of DepthSensor.",
        "Switching can pause rendering briefly. This session-only choice does not modify app_defaults.json.",
    )


def _vector(values) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values) + ") m"


__all__ = [
    "ClassicShellModel", "HELP_PRESENTATION", "HelpPresentation", "ModalController", "ModalKind",
    "ModalPhase", "ModalSnapshot", "ModalToken", "OverlayPresentation", "PerformanceWarningPresentation",
    "PropertySection", "PropertyTab", "SensorUiStatus", "ShellCommandController", "ShellSnapshot",
    "build_performance_warning", "shortcut_command",
]
