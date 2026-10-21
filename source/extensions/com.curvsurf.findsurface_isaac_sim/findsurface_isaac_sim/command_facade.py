"""Toolkit-neutral commands, presentation state, and unified history."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import inspect
import math
from typing import Awaitable, Callable, Mapping

from .analysis_pipeline import (
    AnalysisPresentation,
    ResidualBasis,
    SurfaceSideScale,
    SurfaceSideScaleMode,
)
from .domain import CapturedSurface, DetectionOptions, GeometryMode, Pose, SensorKind
from .scene import SceneCatalog, SceneObject
from .sensor_material import SensorMaterialSetting


class CommandId(str, Enum):
    CAPTURE = "action.capture"
    UNDO = "action.undo"
    REDO = "action.redo"
    # Preserve the legacy canonical name/value; use the precise alias internally.
    RESET_ALL = "action.reset_all"
    RESET_CAPTURES = RESET_ALL
    SENSOR_FROZEN = "sensor.mode.frozen"
    SENSOR_LIVE_FIXED = "sensor.mode.live_fixed"
    SENSOR_LIVE_FOLLOW = "sensor.mode.live_follow"
    PRIMARY_SCENE = "view.primary.scene"
    PRIMARY_SCANNER = "view.primary.scanner"
    CAMERA_RESET_SCENE = "camera.reset.scene"
    CAMERA_SCENE_POS_X = "camera.scene.pos_x"
    CAMERA_SCENE_POS_Y = "camera.scene.pos_y"
    CAMERA_SCENE_POS_Z = "camera.scene.pos_z"
    CAMERA_RESET_SCANNER = "camera.reset.scanner"
    CAMERA_RESET_SCANNER_UP = "camera.reset.scanner_up"
    MODE_INSPECT = "findsurface.mode.inspect"
    MODE_AUTO = "findsurface.mode.auto"
    MODE_PLANE = "findsurface.mode.plane"
    MODE_SPHERE = "findsurface.mode.sphere"
    MODE_CYLINDER = "findsurface.mode.cylinder"
    MODE_CONE = "findsurface.mode.cone"
    MODE_TORUS = "findsurface.mode.torus"
    VISIBILITY_INLIER = "visibility.inlier"
    VISIBILITY_DESIGNED = "visibility.designed"
    VISIBILITY_DETECTED = "visibility.detected"
    VISIBILITY_CAPTURED = "visibility.captured"
    VISIBILITY_OUTLIER = "visibility.outlier"
    HELP = "misc.help"
    SAFE_EXIT = "misc.safe_exit"


class PrimaryView(str, Enum):
    SCANNER = "scanner"
    SCENE = "scene"


class SensorMode(str, Enum):
    FROZEN = "frozen"
    LIVE_FIXED = "live-fixed"
    LIVE_FOLLOW = "live-follow"


class FacadePhase(str, Enum):
    READY = "READY"
    BUSY = "BUSY"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class VisibilityState:
    inlier: bool = True
    designed: bool = True
    detected: bool = True
    captured: bool = True
    outlier: bool = True


@dataclass(frozen=True)
class MeshAppearance:
    subdivision: int = 96
    face_rgba: tuple[float, float, float, float] = (0.72, 0.96, 0.92, 0.25)
    wire_rgba: tuple[float, float, float, float] = (0.15, 1.0, 0.65, 1.0)

    def __post_init__(self) -> None:
        subdivision = int(self.subdivision)
        if subdivision != self.subdivision or not 3 <= subdivision <= 256:
            raise ValueError("mesh subdivision must be an integer between 3 and 256")
        object.__setattr__(self, "subdivision", subdivision)
        for name in ("face_rgba", "wire_rgba"):
            color = tuple(float(component) for component in getattr(self, name))
            if len(color) != 4 or not all(
                math.isfinite(component) and 0.0 <= component <= 1.0 for component in color
            ):
                raise ValueError(f"{name} must contain four finite RGBA values in [0, 1]")
            object.__setattr__(self, name, color)


def _mesh_appearances(default: MeshAppearance) -> tuple:
    return tuple(
        (mode, default) for mode in GeometryMode if mode is not GeometryMode.AUTO
    )


@dataclass(frozen=True)
class InteractionSettings:
    point_rendering_size: float = 2.5
    ruler_main_opacity: float = 0.8
    ruler_pip_opacity: float = 0.35
    inlier_rendering_size: float = 2.5
    outlier_color_rgba: tuple[float, float, float, float] = (0.70, 0.78, 0.88, 1.0)
    inlier_negative_rgba: tuple[float, float, float, float] = (0.0, 0.22, 1.0, 1.0)
    inlier_zero_rgba: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
    inlier_positive_rgba: tuple[float, float, float, float] = (1.0, 0.02, 0.0, 1.0)
    detected_preview_subdivision: int = 96
    detected_preview_face_rgba: tuple[float, float, float, float] = (0.72, 0.96, 0.92, 0.25)
    detected_preview_wire_rgba: tuple[float, float, float, float] = (0.15, 1.0, 0.65, 1.0)
    detected_mesh_appearances: tuple = ()
    captured_mesh_appearances: tuple = ()
    fs_parameter_ranges: tuple = ((0.001, 0.1), (0.01, 0.2), (0.05, 2.0))
    probe_radius_px: float = 6.0
    pick_nearest_on_probe_miss: bool = True
    sticky_auto_preview: bool = True
    sticky_preview_timeout_ms: int = 500
    minimum_geometry_length_m: float = 0.01
    surface_side_scale_mode: SurfaceSideScaleMode = SurfaceSideScaleMode.SIGMA
    surface_side_fixed_scale_m: float = 0.01
    surface_side_sigma_multiplier: float = 3.0
    detection_average_window: int = 10

    def __post_init__(self) -> None:
        for opacity in (self.ruler_main_opacity, self.ruler_pip_opacity):
            if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
                raise ValueError("ruler background opacity must be between 0 and 1")
        if (isinstance(self.detection_average_window, bool)
                or not isinstance(self.detection_average_window, int)
                or not 1 <= self.detection_average_window <= 120):
            raise ValueError("detection average window must be an integer between 1 and 120")
        object.__setattr__(
            self,
            "surface_side_scale_mode",
            SurfaceSideScaleMode(self.surface_side_scale_mode),
        )
        if not math.isfinite(self.point_rendering_size) or not 1.0 <= self.point_rendering_size <= 16.0:
            raise ValueError("point rendering size must be between 1 and 16")
        ranges = tuple(tuple(float(v) for v in pair) for pair in self.fs_parameter_ranges)
        if len(ranges) != 3 or any(len(pair) != 2 or not all(math.isfinite(v) for v in pair) or not 0 < pair[0] <= pair[1] for pair in ranges):
            raise ValueError("FindSurface ranges must contain three finite positive min/max pairs")
        object.__setattr__(self, "fs_parameter_ranges", ranges)
        if not math.isfinite(self.inlier_rendering_size) or not 1.0 <= self.inlier_rendering_size <= 16.0:
            raise ValueError("inlier rendering size must be between 1 and 16")
        for name in (
            "outlier_color_rgba", "inlier_negative_rgba", "inlier_zero_rgba", "inlier_positive_rgba",
            "detected_preview_face_rgba", "detected_preview_wire_rgba",
        ):
            color = tuple(float(component) for component in getattr(self, name))
            if len(color) != 4 or not all(
                math.isfinite(component) and 0.0 <= component <= 1.0 for component in color
            ):
                raise ValueError(f"{name} must contain four finite RGBA values in [0, 1]")
            object.__setattr__(self, name, color)
        subdivision = int(self.detected_preview_subdivision)
        if subdivision != self.detected_preview_subdivision or not 3 <= subdivision <= 256:
            raise ValueError("detected preview subdivision must be an integer between 3 and 256")
        object.__setattr__(self, "detected_preview_subdivision", subdivision)
        detected_fallback = MeshAppearance(
            subdivision,
            self.detected_preview_face_rgba,
            self.detected_preview_wire_rgba,
        )
        captured_fallback = MeshAppearance(
            96,
            (1.0, 0.72, 0.18, 0.22),
            (1.0, 0.78, 0.16, 1.0),
        )
        for name, fallback in (
            ("detected_mesh_appearances", detected_fallback),
            ("captured_mesh_appearances", captured_fallback),
        ):
            supplied = dict(getattr(self, name) or _mesh_appearances(fallback))
            normalized = tuple(
                (mode, supplied.get(mode, supplied.get(mode.value, fallback)))
                for mode in GeometryMode if mode is not GeometryMode.AUTO
            )
            normalized = tuple(
                (mode, value if isinstance(value, MeshAppearance) else MeshAppearance(*value))
                for mode, value in normalized
            )
            object.__setattr__(self, name, normalized)
        if not math.isfinite(self.probe_radius_px) or not 1.0 <= self.probe_radius_px <= 80.0:
            raise ValueError("probe radius must be between 1 and 80 pixels")
        if not 100 <= int(self.sticky_preview_timeout_ms) <= 60000:
            raise ValueError("sticky preview timeout must be between 100 and 60000 ms")
        if (
            not math.isfinite(self.minimum_geometry_length_m)
            or not 0.0001 <= self.minimum_geometry_length_m <= 100.0
        ):
            raise ValueError("minimum geometry length must be between 0.0001 and 100 metres")
        # Share the exact range validation with analysis/render consumers.
        SurfaceSideScale(
            self.surface_side_scale_mode,
            self.surface_side_fixed_scale_m,
            self.surface_side_sigma_multiplier,
        )

    @property
    def surface_side_scale(self) -> SurfaceSideScale:
        return SurfaceSideScale(
            self.surface_side_scale_mode,
            self.surface_side_fixed_scale_m,
            self.surface_side_sigma_multiplier,
        )

    def mesh_appearance(self, kind: str, mode: GeometryMode | str) -> MeshAppearance:
        mode = GeometryMode(mode)
        entries = (
            self.detected_mesh_appearances
            if kind == "detected"
            else self.captured_mesh_appearances
            if kind == "captured"
            else ()
        )
        if not entries:
            raise ValueError(f"unknown mesh appearance kind: {kind}")
        return dict(entries)[mode]


@dataclass(frozen=True)
class SensorSetting:
    sensor_kind: SensorKind = SensorKind.DEPTH
    profile_id: str = "default"
    enabled: bool = False
    follows_camera: bool = False
    target_material: SensorMaterialSetting = SensorMaterialSetting()

    def __post_init__(self) -> None:
        if self.follows_camera and not self.enabled:
            raise ValueError("a disabled sensor cannot follow the scanner camera")

    @property
    def mode(self) -> SensorMode:
        if not self.enabled:
            return SensorMode.FROZEN
        return SensorMode.LIVE_FOLLOW if self.follows_camera else SensorMode.LIVE_FIXED

    def with_mode(self, mode: SensorMode | str) -> "SensorSetting":
        mode = SensorMode(mode)
        return replace(
            self,
            enabled=mode is not SensorMode.FROZEN,
            follows_camera=mode is SensorMode.LIVE_FOLLOW,
        )


@dataclass(frozen=True)
class CapturePresentation:
    surface: CapturedSurface
    analysis: AnalysisPresentation
    mesh_visible: bool = True
    inliers_visible: bool = True
    heatmap_basis: ResidualBasis = ResidualBasis.DETECTED
    mesh_subdivision: int = 96
    mesh_face_rgba: tuple[float, float, float, float] = (1.0, 0.72, 0.18, 0.22)
    mesh_wire_rgba: tuple[float, float, float, float] = (1.0, 0.78, 0.16, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "heatmap_basis", ResidualBasis(self.heatmap_basis))
        subdivision = int(self.mesh_subdivision)
        if subdivision != self.mesh_subdivision or not 3 <= subdivision <= 256:
            raise ValueError("capture mesh subdivision must be an integer between 3 and 256")
        object.__setattr__(self, "mesh_subdivision", subdivision)
        for name in ("mesh_face_rgba", "mesh_wire_rgba"):
            color = tuple(float(component) for component in getattr(self, name))
            if len(color) != 4 or not all(
                math.isfinite(component) and 0.0 <= component <= 1.0 for component in color
            ):
                raise ValueError(f"{name} must contain four finite RGBA values in [0, 1]")
            object.__setattr__(self, name, color)

    @property
    def capture_id(self) -> str:
        return self.surface.capture_id

    @property
    def visible(self) -> bool:
        """Compatibility summary for callers that toggle the complete capture."""

        return self.mesh_visible and self.inliers_visible


@dataclass(frozen=True)
class SessionSnapshot:
    objects: tuple[SceneObject, ...]
    scanner_pose: Pose
    scene_pose: Pose
    detection: DetectionOptions
    inspect_mode: bool
    visibility: VisibilityState
    primary_view: PrimaryView
    sensor: SensorSetting
    point_depth_offset_m: float
    interaction: InteractionSettings
    captures: tuple[CapturePresentation, ...] = ()
    selected_capture_id: str | None = None
    camera_views: tuple = ()
    scanner_manual_fov: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.point_depth_offset_m) or self.point_depth_offset_m < 0.0:
            raise ValueError("point_depth_offset_m must be finite and nonnegative")
        capture_ids = tuple(item.capture_id for item in self.captures)
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("capture ids must be unique")
        geometry_types = tuple(item.surface.geometry_type for item in self.captures)
        if len(geometry_types) != len(set(geometry_types)):
            raise ValueError("only one captured geometry per type is allowed")
        if self.selected_capture_id is not None and self.selected_capture_id not in capture_ids:
            raise ValueError("selected capture must exist in captures")


@dataclass(frozen=True)
class CommandPresentation:
    command_id: CommandId
    label: str
    shortcut: str | None
    enabled: bool
    checked: bool | None
    tooltip: str


@dataclass(frozen=True)
class FacadeStatus:
    phase: FacadePhase
    message: str
    error: str | None = None


Transition = Callable[[SensorSetting, SensorSetting], Awaitable[None] | None]
SettingsTransition = Callable[[SessionSnapshot, SessionSnapshot], Awaitable[None] | None]


@dataclass(frozen=True)
class _HistoryEntry:
    name: str
    before: SessionSnapshot
    after: SessionSnapshot
    transition: Transition | None = None
    settings_transition: SettingsTransition | None = None


_MODE_COMMANDS = {
    CommandId.MODE_AUTO: GeometryMode.AUTO,
    CommandId.MODE_PLANE: GeometryMode.PLANE,
    CommandId.MODE_SPHERE: GeometryMode.SPHERE,
    CommandId.MODE_CYLINDER: GeometryMode.CYLINDER,
    CommandId.MODE_CONE: GeometryMode.CONE,
    CommandId.MODE_TORUS: GeometryMode.TORUS,
}

_STATIC_COMMANDS = (
    (CommandId.CAPTURE, "Capture", "Ctrl+C"),
    (CommandId.UNDO, "Undo", "Ctrl+Z"),
    (CommandId.REDO, "Redo", "Ctrl+Shift+Z"),
    (CommandId.RESET_CAPTURES, "Reset captures", "Ctrl+R"),
    (CommandId.SENSOR_FROZEN, "Freeze sensor and point cloud", None),
    (CommandId.SENSOR_LIVE_FIXED, "Update cloud at fixed sensor pose", None),
    (CommandId.SENSOR_LIVE_FOLLOW, "Follow scanner camera and update cloud", None),
    (CommandId.PRIMARY_SCENE, "Scene primary", "P"),
    (CommandId.PRIMARY_SCANNER, "Scanner primary", "P"),
    (CommandId.CAMERA_RESET_SCENE, "Reset scene camera", "R"),
    (CommandId.CAMERA_SCENE_POS_X, "Scene +X", None),
    (CommandId.CAMERA_SCENE_POS_Y, "Scene +Y", None),
    (CommandId.CAMERA_SCENE_POS_Z, "Scene +Z", None),
    (CommandId.CAMERA_RESET_SCANNER, "Reset scanner camera", "R"),
    (CommandId.CAMERA_RESET_SCANNER_UP, "Reset scanner up", None),
    (CommandId.MODE_INSPECT, "Inspect (pause detection)", "`"),
    (CommandId.MODE_AUTO, "Auto", "1"),
    (CommandId.MODE_PLANE, "Plane", "2"),
    (CommandId.MODE_SPHERE, "Sphere", "3"),
    (CommandId.MODE_CYLINDER, "Cylinder", "4"),
    (CommandId.MODE_CONE, "Cone", "5"),
    (CommandId.MODE_TORUS, "Torus", "6"),
    (CommandId.VISIBILITY_OUTLIER, "Outlier", "O"),
    (CommandId.VISIBILITY_INLIER, "Inlier", "I"),
    (CommandId.VISIBILITY_DESIGNED, "Mesh (original)", "M"),
    (CommandId.VISIBILITY_DETECTED, "Mesh (hover preview)", "B"),
    (CommandId.VISIBILITY_CAPTURED, "Mesh (captured)", None),
    (CommandId.HELP, "Help", "Shift+/"),
    (CommandId.SAFE_EXIT, "Safe Exit", "Alt+F4 / Ctrl+Shift+Q"),
)


class CommandFacade:
    """One immutable session state with ordered sync/async undo and redo."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._defaults = session_from_config(config)
        self._snapshot = self._defaults
        self._undo: list[_HistoryEntry] = []
        self._redo: list[_HistoryEntry] = []
        self._status = FacadeStatus(FacadePhase.READY, "Ready")

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot

    @property
    def status(self) -> FacadeStatus:
        return self._status

    @property
    def can_undo(self) -> bool:
        return bool(self._undo) and self._status.phase is FacadePhase.READY

    @property
    def can_redo(self) -> bool:
        return bool(self._redo) and self._status.phase is FacadePhase.READY

    @property
    def history_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._undo)

    def command_matrix(self, *, capture_ready: bool = False) -> tuple[CommandPresentation, ...]:
        busy = self._status.phase in {FacadePhase.BUSY, FacadePhase.CLOSED}
        result = []
        for command_id, label, shortcut in _STATIC_COMMANDS:
            checked = self._checked(command_id)
            enabled = not busy
            tooltip = label
            if command_id is CommandId.CAPTURE:
                enabled = enabled and capture_ready
                tooltip = "Capture the selected current result." if enabled else "Preview and select a current fixed-cloud result first."
            elif command_id is CommandId.UNDO:
                enabled = self.can_undo
            elif command_id is CommandId.REDO:
                enabled = self.can_redo
            elif command_id is CommandId.RESET_CAPTURES:
                enabled = enabled and bool(self._snapshot.captures)
                tooltip = "Clear all captured geometries." if enabled else "There are no captured geometries to reset."
            elif command_id is CommandId.CAMERA_RESET_SCENE:
                enabled = enabled and self._snapshot.primary_view is PrimaryView.SCENE
            elif command_id is CommandId.CAMERA_RESET_SCANNER:
                enabled = enabled and self._snapshot.primary_view is PrimaryView.SCANNER
            result.append(CommandPresentation(command_id, label, shortcut, enabled, checked, tooltip))
        return tuple(result)

    def execute(self, command_id: CommandId | str, value=None) -> SessionSnapshot:
        self._require_ready()
        command_id = CommandId(command_id)
        before = self._snapshot
        after, name = self._reduce(command_id, value)
        if after == before:
            return before
        self._record(_HistoryEntry(name, before, after))
        return self._snapshot

    def replace_object(self, replacement: SceneObject) -> SessionSnapshot:
        self._require_ready()
        before = self._snapshot
        objects = tuple(replacement if item.geometry_type is replacement.geometry_type else item for item in before.objects)
        self._record(_HistoryEntry(f"Change {replacement.geometry_type.value} object", before, replace(before, objects=objects)))
        return self._snapshot

    def preview_object(self, replacement: SceneObject) -> SessionSnapshot:
        """Replace one object without history while a continuous UI gesture is active."""

        self._require_ready()
        objects = tuple(
            replacement if item.geometry_type is replacement.geometry_type else item
            for item in self._snapshot.objects
        )
        self._snapshot = replace(self._snapshot, objects=objects)
        return self._snapshot

    def commit_object_preview(self, before: SessionSnapshot, geometry_type: GeometryMode) -> SessionSnapshot:
        """Commit the current continuous object preview as one undo entry."""

        self._require_ready()
        after = self._snapshot
        if after != before:
            self._record(_HistoryEntry(f"Rotate {geometry_type.value} object", before, after))
        return self._snapshot

    def replace_camera(self, role: str, pose: Pose, *, view=None, manual_fov=None) -> SessionSnapshot:
        self._require_ready()
        before = self._snapshot
        if role == "scanner":
            after = replace(before, scanner_pose=pose)
        elif role == "scene":
            after = replace(before, scene_pose=pose)
        else:
            raise ValueError("camera role must be scanner or scene")
        if view is not None:
            if view.role.value != role or view.pose_world != pose:
                raise ValueError("camera view and pose must match")
            after = replace(after, camera_views=tuple(v for v in before.camera_views if v.role.value != role) + (view,))
        else:
            after = replace(after, camera_views=tuple(replace(v, pose_world=pose) if v.role.value == role else v for v in before.camera_views))
        if manual_fov is not None:
            after = replace(after, scanner_manual_fov=bool(manual_fov))
        self._record(_HistoryEntry(f"Change {role} camera", before, after))
        return self._snapshot

    def replace_detection(self, options: DetectionOptions) -> SessionSnapshot:
        self._require_ready()
        before = self._snapshot
        if options == before.detection:
            return before
        if options.revision <= before.detection.revision:
            options = replace(options, revision=before.detection.revision + 1)
        self._record(_HistoryEntry("Change FindSurface parameters", before, replace(before, detection=options)))
        return self._snapshot


    def set_point_depth_offset(self, offset_m: float) -> SessionSnapshot:
        self._require_ready()
        offset_m = float(offset_m)
        if not math.isfinite(offset_m) or not 0.0 <= offset_m <= 1.0:
            raise ValueError("point depth offset must be between 0 and 1 metre")
        before = self._snapshot
        if offset_m == before.point_depth_offset_m:
            return before
        self._record(_HistoryEntry(
            "Change point-cloud depth offset",
            before,
            replace(before, point_depth_offset_m=offset_m),
        ))
        return self._snapshot

    def set_interaction(self, **changes) -> SessionSnapshot:
        """Apply an undoable Settings-window display/picking policy change."""

        self._require_ready()
        before = self._snapshot
        legacy_detected_fields = {
            "detected_preview_subdivision",
            "detected_preview_face_rgba",
            "detected_preview_wire_rgba",
        }
        if legacy_detected_fields.intersection(changes) and "detected_mesh_appearances" not in changes:
            detected = {}
            for mode, appearance in before.interaction.detected_mesh_appearances:
                detected[mode] = replace(
                    appearance,
                    subdivision=int(changes.get(
                        "detected_preview_subdivision", appearance.subdivision,
                    )),
                    face_rgba=tuple(changes.get(
                        "detected_preview_face_rgba", appearance.face_rgba,
                    )),
                    wire_rgba=tuple(changes.get(
                        "detected_preview_wire_rgba", appearance.wire_rgba,
                    )),
                )
            changes["detected_mesh_appearances"] = tuple(detected.items())
        interaction = replace(before.interaction, **changes)
        if interaction == before.interaction:
            return before
        labels = {
            "point_rendering_size": "point rendering size",
            "ruler_main_opacity": "main ruler background opacity",
            "ruler_pip_opacity": "PIP ruler background opacity",
            "inlier_rendering_size": "inlier rendering size",
            "outlier_color_rgba": "outlier point color",
            "inlier_negative_rgba": "negative inlier color",
            "inlier_zero_rgba": "zero inlier color",
            "inlier_positive_rgba": "positive inlier color",
            "detected_preview_subdivision": "detected preview subdivision",
            "detected_preview_face_rgba": "detected preview face color",
            "detected_preview_wire_rgba": "detected preview outline color",
            "probe_radius_px": "picking probe radius",
            "pick_nearest_on_probe_miss": "probe miss fallback",
            "sticky_auto_preview": "Sticky Auto Preview",
            "sticky_preview_timeout_ms": "Sticky Preview timeout",
            "detection_average_window": "FindSurface timing average window",
            "minimum_geometry_length_m": "minimum geometry length",
            "surface_side_scale_mode": "Surface side scale mode",
            "surface_side_fixed_scale_m": "Surface side fixed scale",
            "surface_side_sigma_multiplier": "Surface side sigma range",
        }
        name = labels.get(next(iter(changes), ""), "interaction settings")
        self._record(_HistoryEntry(f"Change {name}", before, replace(before, interaction=interaction)))
        return self._snapshot

    def add_capture(self, surface: CapturedSurface, analysis: AnalysisPresentation) -> CapturePresentation:
        self._require_ready()
        if any(item.capture_id == surface.capture_id for item in self._snapshot.captures):
            raise ValueError(f"duplicate capture id: {surface.capture_id}")
        appearance = self._snapshot.interaction.mesh_appearance(
            "captured", surface.geometry_type,
        )
        item = CapturePresentation(
            surface,
            analysis,
            mesh_subdivision=appearance.subdivision,
            mesh_face_rgba=appearance.face_rgba,
            mesh_wire_rgba=appearance.wire_rgba,
        )
        before = self._snapshot
        replaced = next(
            (
                existing for existing in before.captures
                if existing.surface.geometry_type is surface.geometry_type
            ),
            None,
        )
        captures = tuple(
            item if existing.surface.geometry_type is surface.geometry_type else existing
            for existing in before.captures
        )
        if replaced is None:
            captures += (item,)
        after = replace(before, captures=captures, selected_capture_id=item.capture_id)
        name = (
            f"Replace {surface.geometry_type.value} capture"
            if replaced is not None
            else f"Capture {item.capture_id}"
        )
        self._record(_HistoryEntry(name, before, after))
        return item

    def select_capture(self, capture_id: str | None) -> None:
        """Select a captured geometry without adding a history transaction."""

        self._require_ready()
        if capture_id is not None and not any(
            item.capture_id == capture_id for item in self._snapshot.captures
        ):
            raise KeyError(capture_id)
        self._snapshot = replace(self._snapshot, selected_capture_id=capture_id)
        self._status = FacadeStatus(
            FacadePhase.READY,
            "No captured geometry selected" if capture_id is None else f"Selected {capture_id}",
        )

    def set_capture_visibility(self, capture_id: str, visible: bool) -> None:
        self._mutate_capture(
            capture_id,
            lambda item: replace(
                item,
                mesh_visible=bool(visible),
                inliers_visible=bool(visible),
            ),
            f"Set {capture_id} visibility",
        )

    def set_capture_mesh_visibility(self, capture_id: str, visible: bool) -> None:
        self._mutate_capture(
            capture_id,
            lambda item: replace(item, mesh_visible=bool(visible)),
            f"Set {capture_id} mesh visibility",
        )

    def set_capture_inliers_visibility(self, capture_id: str, visible: bool) -> None:
        self._mutate_capture(
            capture_id,
            lambda item: replace(item, inliers_visible=bool(visible)),
            f"Set {capture_id} inlier visibility",
        )

    def set_capture_heatmap_basis(
        self,
        capture_id: str,
        basis: ResidualBasis | str,
    ) -> None:
        basis = ResidualBasis(basis)
        current = next(
            (item for item in self._snapshot.captures if item.capture_id == capture_id),
            None,
        )
        if current is None:
            raise KeyError(capture_id)
        if current.heatmap_basis is basis:
            return
        self._mutate_capture(
            capture_id,
            lambda item: replace(item, heatmap_basis=basis),
            f"Set {capture_id} heatmap basis to {basis.value}",
        )

    def set_capture_mesh_appearance(
        self,
        capture_id: str,
        *,
        subdivision: int,
        face_rgba,
        wire_rgba,
    ) -> None:
        self._mutate_capture(
            capture_id,
            lambda item: replace(
                item,
                mesh_subdivision=int(subdivision),
                mesh_face_rgba=tuple(face_rgba),
                mesh_wire_rgba=tuple(wire_rgba),
            ),
            f"Change {capture_id} mesh appearance",
        )

    def set_geometry_mesh_appearance(
        self,
        kind: str,
        mode: GeometryMode | str,
        *,
        subdivision: int,
        face_rgba,
        wire_rgba,
    ) -> None:
        """Set a per-type detected/captured style in one undoable transaction."""

        self._require_ready()
        if kind not in {"detected", "captured"}:
            raise ValueError(f"unknown mesh appearance kind: {kind}")
        mode = GeometryMode(mode)
        appearance = MeshAppearance(int(subdivision), tuple(face_rgba), tuple(wire_rgba))
        before = self._snapshot
        field = f"{kind}_mesh_appearances"
        updated = dict(getattr(before.interaction, field))
        updated[mode] = appearance
        interaction = replace(before.interaction, **{field: tuple(updated.items())})
        captures = before.captures
        if kind == "captured":
            captures = tuple(
                replace(
                    item,
                    mesh_subdivision=appearance.subdivision,
                    mesh_face_rgba=appearance.face_rgba,
                    mesh_wire_rgba=appearance.wire_rgba,
                )
                if item.surface.geometry_type is mode else item
                for item in captures
            )
        after = replace(before, interaction=interaction, captures=captures)
        if after == before:
            return
        self._record(_HistoryEntry(
            f"Change {mode.value} {kind} mesh appearance", before, after,
        ))

    def set_all_capture_visibility(self, visible: bool) -> None:
        self._require_ready()
        before = self._snapshot
        captures = tuple(
            replace(item, mesh_visible=bool(visible), inliers_visible=bool(visible))
            for item in before.captures
        )
        if captures == before.captures:
            return
        label = "Show" if visible else "Hide"
        self._record(_HistoryEntry(
            f"{label} all captured geometries",
            before,
            replace(before, captures=captures),
        ))

    def set_all_capture_mesh_visibility(self, visible: bool) -> None:
        self._set_all_capture_component_visibility("mesh_visible", visible, "meshes")

    def set_all_capture_inliers_visibility(self, visible: bool) -> None:
        self._set_all_capture_component_visibility("inliers_visible", visible, "inliers")

    def _set_all_capture_component_visibility(
        self,
        field_name: str,
        visible: bool,
        label: str,
    ) -> None:
        self._require_ready()
        before = self._snapshot
        captures = tuple(
            replace(item, **{field_name: bool(visible)}) for item in before.captures
        )
        if captures == before.captures:
            return
        verb = "Show" if visible else "Hide"
        self._record(_HistoryEntry(
            f"{verb} all captured {label}",
            before,
            replace(before, captures=captures),
        ))

    def delete_capture(self, capture_id: str) -> None:
        self._require_ready()
        before = self._snapshot
        if not any(item.capture_id == capture_id for item in before.captures):
            raise KeyError(capture_id)
        after = replace(
            before,
            captures=tuple(item for item in before.captures if item.capture_id != capture_id),
            selected_capture_id=None if before.selected_capture_id == capture_id else before.selected_capture_id,
        )
        self._record(_HistoryEntry(f"Delete {capture_id}", before, after))

    async def select_sensor(self, setting: SensorSetting, transition: Transition) -> bool:
        self._require_ready()
        before = self._snapshot
        after = replace(before, sensor=setting)
        if after == before:
            return True
        entry = _HistoryEntry(f"Select sensor {setting.sensor_kind.value}/{setting.profile_id}", before, after, transition)
        return await self._apply_async_entry(entry)

    async def set_sensor_material(
        self, material: SensorMaterialSetting, transition: Transition,
    ) -> bool:
        """Apply one undoable non-visual target-material transaction."""

        self._require_ready()
        if not isinstance(material, SensorMaterialSetting):
            raise TypeError("material must be SensorMaterialSetting")
        before = self._snapshot
        after = replace(before, sensor=replace(before.sensor, target_material=material))
        if after == before:
            return True
        entry = _HistoryEntry(
            f"Change target sensor material to {material.summary}",
            before,
            after,
            transition,
        )
        return await self._apply_async_entry(entry)

    async def reset_captures(self, transition: Transition | None = None) -> bool:
        self._require_ready()
        before = self._snapshot
        after = replace(before, captures=(), selected_capture_id=None)
        if after == before:
            self._status = FacadeStatus(FacadePhase.READY, "No captured geometries to reset")
            return False
        entry = _HistoryEntry("Reset captures", before, after)
        return await self._apply_async_entry(entry)

    async def reset(self, transition: Transition | None = None) -> bool:
        """Compatibility entry point; only captures are reset, not settings."""
        return await self.reset_captures(transition)

    async def restore_settings(self, config, transition=None) -> bool:
        """Restore packaged defaults without discarding captures, view, or run mode."""
        self._require_ready()
        before = self._snapshot
        defaults = session_from_config(config)
        after = replace(
            before,
            objects=defaults.objects,
            scanner_pose=defaults.scanner_pose,
            scene_pose=defaults.scene_pose,
            camera_views=defaults.camera_views,
            scanner_manual_fov=defaults.scanner_manual_fov,
            detection=defaults.detection,
            inspect_mode=defaults.inspect_mode,
            visibility=defaults.visibility,
            sensor=replace(
                defaults.sensor,
                enabled=before.sensor.enabled,
                follows_camera=before.sensor.follows_camera,
            ),
            point_depth_offset_m=defaults.point_depth_offset_m,
            interaction=defaults.interaction,
        )
        return await self._apply_async_entry(_HistoryEntry(
            "Restore settings defaults", before, after, settings_transition=transition,
        ))

    def bind_startup_camera_views(self, views) -> None:
        """Complete the immutable camera baseline once runtime views exist."""
        if getattr(self, "_startup_views_bound", False):
            return
        self._defaults = replace(self._defaults, camera_views=tuple(views))
        self._snapshot = replace(self._snapshot, camera_views=tuple(views))
        self._startup_views_bound = True

    async def restore_startup_scope(self, scope: str, transition=None) -> bool:
        """Restore a startup domain scope through a reversible full-state adapter.

        _defaults is created from the preference-overlaid startup config, not
        from the mutable current snapshot. Runtime/UI adapters supply transition.
        """
        self._require_ready()
        before, startup = self._snapshot, self._defaults
        fields = {
            "camera": ("scanner_pose", "scene_pose", "camera_views", "scanner_manual_fov", "point_depth_offset_m"),
            "objects": ("objects",),
            "findsurface": ("detection",),
            "scanner": ("sensor",),
            "geometries": ("captures", "selected_capture_id"),
        }
        if scope == "all":
            after = startup
        elif scope in fields:
            after = replace(before, **{name: getattr(startup, name) for name in fields[scope]})
            if scope == "findsurface":
                after = replace(after, interaction=replace(before.interaction,
                    fs_parameter_ranges=startup.interaction.fs_parameter_ranges))
        else:
            raise ValueError(f"Unknown startup reset scope: {scope}")
        return await self._apply_async_entry(_HistoryEntry(
            f"Restore startup {scope}", before, after, settings_transition=transition,
        ))

    async def undo(self) -> bool:
        self._require_ready()
        if not self._undo:
            return False
        entry = self._undo[-1]
        if not await self._transition(entry, entry.after.sensor, entry.before.sensor, "Undo"):
            return False
        self._undo.pop()
        self._snapshot = entry.before
        self._redo.append(entry)
        self._status = FacadeStatus(FacadePhase.READY, f"Undid {entry.name}")
        return True

    async def redo(self) -> bool:
        self._require_ready()
        if not self._redo:
            return False
        entry = self._redo[-1]
        if not await self._transition(entry, entry.before.sensor, entry.after.sensor, "Redo"):
            return False
        self._redo.pop()
        self._snapshot = entry.after
        self._undo.append(entry)
        self._status = FacadeStatus(FacadePhase.READY, f"Redid {entry.name}")
        return True

    def close(self) -> None:
        self._status = FacadeStatus(FacadePhase.CLOSED, "Closed")

    def _record(self, entry: _HistoryEntry) -> None:
        self._snapshot = entry.after
        self._undo.append(entry)
        self._redo.clear()
        self._status = FacadeStatus(FacadePhase.READY, entry.name)

    async def _apply_async_entry(self, entry: _HistoryEntry) -> bool:
        if not await self._transition(entry, entry.before.sensor, entry.after.sensor, "Apply"):
            return False
        self._record(entry)
        return True

    async def _transition(self, entry: _HistoryEntry, before: SensorSetting, after: SensorSetting, verb: str) -> bool:
        if entry.settings_transition is None and (entry.transition is None or before == after):
            return True
        self._status = FacadeStatus(FacadePhase.BUSY, f"{verb}: {entry.name}")
        try:
            if entry.settings_transition is not None:
                source, target = (entry.after, entry.before) if verb == "Undo" else (entry.before, entry.after)
                result = entry.settings_transition(source, target)
            else:
                result = entry.transition(before, after)
            if inspect.isawaitable(result):
                await result
        except Exception as error:
            self._status = FacadeStatus(FacadePhase.ERROR, f"{verb} failed; state rolled back", f"{type(error).__name__}: {error}")
            return False
        self._status = FacadeStatus(FacadePhase.READY, f"{verb}: {entry.name}")
        return True

    def _reduce(self, command_id: CommandId, value) -> tuple[SessionSnapshot, str]:
        state = self._snapshot
        if command_id in _MODE_COMMANDS:
            mode = _MODE_COMMANDS[command_id]
            options = replace(state.detection, geometry_mode=mode, revision=state.detection.revision + 1)
            return replace(state, detection=options, inspect_mode=False), f"Set FindSurface mode {mode.value}"
        if command_id is CommandId.MODE_INSPECT:
            return replace(state, inspect_mode=True), "Inspect/default"
        if command_id in {CommandId.PRIMARY_SCENE, CommandId.PRIMARY_SCANNER}:
            primary = PrimaryView.SCENE if command_id is CommandId.PRIMARY_SCENE else PrimaryView.SCANNER
            return replace(state, primary_view=primary), f"Set {primary.value} primary"
        visibility_names = {
            CommandId.VISIBILITY_INLIER: "inlier",
            CommandId.VISIBILITY_DESIGNED: "designed",
            CommandId.VISIBILITY_DETECTED: "detected",
            CommandId.VISIBILITY_CAPTURED: "captured",
            CommandId.VISIBILITY_OUTLIER: "outlier",
        }
        if command_id in visibility_names:
            name = visibility_names[command_id]
            visibility = replace(state.visibility, **{name: not getattr(state.visibility, name)})
            return replace(state, visibility=visibility), f"Toggle {name} visibility"
        raise ValueError(f"command requires a dedicated handler or is presentation-only: {command_id.value}")

    def _mutate_capture(self, capture_id, mutation, name) -> None:
        self._require_ready()
        before = self._snapshot
        found = False
        captures = []
        for item in before.captures:
            if item.capture_id == capture_id:
                item = mutation(item)
                found = True
            captures.append(item)
        if not found:
            raise KeyError(capture_id)
        self._record(_HistoryEntry(name, before, replace(before, captures=tuple(captures))))

    def _checked(self, command_id: CommandId) -> bool | None:
        mode = _MODE_COMMANDS.get(command_id)
        if mode is not None:
            return not self._snapshot.inspect_mode and self._snapshot.detection.geometry_mode is mode
        if command_id is CommandId.MODE_INSPECT:
            return self._snapshot.inspect_mode
        if command_id is CommandId.PRIMARY_SCENE:
            return self._snapshot.primary_view is PrimaryView.SCENE
        if command_id is CommandId.PRIMARY_SCANNER:
            return self._snapshot.primary_view is PrimaryView.SCANNER
        sensor_modes = {
            CommandId.SENSOR_FROZEN: SensorMode.FROZEN,
            CommandId.SENSOR_LIVE_FIXED: SensorMode.LIVE_FIXED,
            CommandId.SENSOR_LIVE_FOLLOW: SensorMode.LIVE_FOLLOW,
        }
        if command_id in sensor_modes:
            return self._snapshot.sensor.mode is sensor_modes[command_id]
        visibility = {
            CommandId.VISIBILITY_INLIER: self._snapshot.visibility.inlier,
            CommandId.VISIBILITY_DESIGNED: self._snapshot.visibility.designed,
            CommandId.VISIBILITY_DETECTED: self._snapshot.visibility.detected,
            CommandId.VISIBILITY_CAPTURED: self._snapshot.visibility.captured,
            CommandId.VISIBILITY_OUTLIER: self._snapshot.visibility.outlier,
        }
        return visibility.get(command_id)

    def _require_ready(self) -> None:
        if self._status.phase is FacadePhase.CLOSED:
            raise RuntimeError("command facade is closed")
        if self._status.phase is FacadePhase.BUSY:
            raise RuntimeError("an asynchronous command is already active")
        if self._status.phase is FacadePhase.ERROR:
            self._status = FacadeStatus(FacadePhase.READY, "Recovered from previous command error")


def session_from_config(config: Mapping[str, object]) -> SessionSnapshot:
    from .camera import CameraRole, CameraSnapshot, CameraIntrinsics, aligned_orbit_target
    objects = SceneCatalog.from_config(config["objects"]).objects
    cameras = config["cameras"]
    detection = config["detection"]
    sensor = config["sensor"]
    visibility = config["visibility"]
    display = config.get("display", {})
    picking = config.get("picking", {})
    preview = config.get("preview", {})
    heatmap = config.get("heatmap", {})
    def mesh_appearance_entries(key: str, fallback: MeshAppearance) -> tuple:
        configured = display.get(key, {})
        return tuple(
            (
                mode,
                MeshAppearance(
                    configured.get(mode.value.lower(), {}).get("subdivision", fallback.subdivision),
                    configured.get(mode.value.lower(), {}).get("face_rgba", fallback.face_rgba),
                    configured.get(mode.value.lower(), {}).get("wire_rgba", fallback.wire_rgba),
                ),
            )
            for mode in GeometryMode if mode is not GeometryMode.AUTO
        )
    source = sensor["source"]
    if source not in ("DepthSensor", "RTX"):
        raise ValueError(f"unsupported sensor source: {source}")
    kind = SensorKind.DEPTH if source == "DepthSensor" else SensorKind.RTX_LIDAR
    material = sensor.get("target_material", {})
    views = []
    object_center = SceneCatalog(objects).bounds.center_m
    for role, values in cameras.items():
        projection = values.get("projection", "perspective")
        if projection not in ("perspective", "orthographic"):
            raise ValueError("invalid camera projection")
        pose = Pose(values["position_m"], values["orientation_xyzw"])
        views.append(CameraSnapshot(CameraRole(role), pose,
            CameraIntrinsics(float(values.get("vertical_fov_degrees", 52.0)),
                near_m=float(values.get("near_m", 0.05)), far_m=float(values.get("far_m", 100.0)),
                orthographic_height_m=float(values.get("orthographic_height_m", 10.0)) if projection == "orthographic" else None),
            tuple(values["target_world_m"]) if "target_world_m" in values else
            aligned_orbit_target(pose, object_center)))
    return SessionSnapshot(
        objects,
        Pose(cameras["scanner"]["position_m"], cameras["scanner"]["orientation_xyzw"]),
        Pose(cameras["scene"]["position_m"], cameras["scene"]["orientation_xyzw"]),
        DetectionOptions(
            geometry_mode=GeometryMode(detection["type"]),
            seed_radius_m=detection["seed_radius_m"],
            measurement_accuracy_m=detection["measurement_accuracy_m"],
            mean_distance_m=detection["mean_distance_m"],
            radial_expansion=detection["radial_expansion"],
            lateral_extension=detection["lateral_extension"],
            smart_convert_cone_to_cylinder=bool(
                detection.get("smart_convert_cone_to_cylinder", True)
            ),
            smart_convert_torus_to_sphere=bool(
                detection.get("smart_convert_torus_to_sphere", True)
            ),
            smart_convert_torus_to_cylinder=bool(
                detection.get("smart_convert_torus_to_cylinder", True)
            ),
        ),
        False,
        VisibilityState(
            inlier=bool(visibility.get("points", True)),
            designed=bool(visibility.get("source", True)),
            detected=bool(visibility.get("detected", True)),
            captured=bool(visibility.get("captured", visibility.get("detected", True))),
            outlier=bool(visibility.get("points", True)),
        ),
        PrimaryView.SCANNER,
        SensorSetting(
            kind,
            str(sensor["profile"]),
            bool(sensor["enabled"]),
            bool(sensor.get("follows_camera", False)) if bool(sensor["enabled"]) else False,
            SensorMaterialSetting(
                str(material.get("base", "concrete")),
                str(material.get("coating", "paint")),
                str(material.get("attribute", "emissive")),
            ),
        ),
        float(config["renderer"].get("point_depth_offset_m", 0.006)),
        InteractionSettings(
            point_rendering_size=float(display.get("point_size", 2.5)),
            ruler_main_opacity=float(display.get("ruler_main_opacity", 0.8)),
            ruler_pip_opacity=float(display.get("ruler_pip_opacity", 0.35)),
            inlier_rendering_size=float(display.get("inlier_size", 2.5)),
            outlier_color_rgba=display.get("outlier_color_rgba", (0.70, 0.78, 0.88, 1.0)),
            inlier_negative_rgba=display.get("inlier_negative_rgba", (0.0, 0.22, 1.0, 1.0)),
            inlier_zero_rgba=display.get("inlier_zero_rgba", (1.0, 1.0, 0.0, 1.0)),
            inlier_positive_rgba=display.get("inlier_positive_rgba", (1.0, 0.02, 0.0, 1.0)),
            detected_preview_subdivision=display.get("detected_preview_subdivision", 96),
            detected_preview_face_rgba=display.get("detected_preview_face_rgba", (0.72, 0.96, 0.92, 0.25)),
            detected_preview_wire_rgba=display.get("detected_preview_wire_rgba", (0.15, 1.0, 0.65, 1.0)),
            detected_mesh_appearances=mesh_appearance_entries(
                "detected_mesh_appearances",
                MeshAppearance(
                    display.get("detected_preview_subdivision", 96),
                    display.get("detected_preview_face_rgba", (0.72, 0.96, 0.92, 0.25)),
                    display.get("detected_preview_wire_rgba", (0.15, 1.0, 0.65, 1.0)),
                ),
            ),
            captured_mesh_appearances=mesh_appearance_entries(
                "captured_mesh_appearances",
                MeshAppearance(96, (1.0, 0.72, 0.18, 0.22), (1.0, 0.78, 0.16, 1.0)),
            ),
            fs_parameter_ranges=config.get("findsurface_ui", {}).get("ranges", ((0.001, 0.1), (0.01, 0.2), (0.05, 2.0))),
            probe_radius_px=float(picking.get("probe_radius_px", 6.0)),
            pick_nearest_on_probe_miss=bool(picking.get("pick_nearest_on_probe_miss", True)),
            sticky_auto_preview=bool(preview.get("sticky_auto_preview", True)),
            sticky_preview_timeout_ms=int(preview.get("sticky_auto_preview_timeout_ms", 500)),
            detection_average_window=preview.get("detection_average_window", 10),
            minimum_geometry_length_m=float(display.get("minimum_geometry_length_m", 0.01)),
            surface_side_scale_mode=SurfaceSideScaleMode(
                heatmap.get("surface_side_scale_mode", SurfaceSideScaleMode.SIGMA.value)
            ),
            surface_side_fixed_scale_m=float(
                heatmap.get("surface_side_fixed_scale_m", 0.01)
            ),
            surface_side_sigma_multiplier=float(
                heatmap.get("surface_side_sigma_multiplier", 3.0)
            ),
        ),
        camera_views=tuple(views),
        scanner_manual_fov=bool(cameras["scanner"].get("manual_fov", False)),
    )


__all__ = [
    "CapturePresentation", "CommandFacade", "CommandId", "CommandPresentation", "FacadePhase",
    "FacadeStatus", "InteractionSettings", "PrimaryView", "SensorMode", "SensorSetting", "SessionSnapshot", "VisibilityState",
    "MeshAppearance", "session_from_config",
]
