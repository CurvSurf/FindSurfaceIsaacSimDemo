"""Canonical scene object catalog, bounds, and object-change intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Mapping, Sequence

from .spatial_math import quat_conjugate_xyzw, quat_multiply_xyzw, rotate_vector_xyzw
from .domain import GeometryMode, Pose, StageMetadata


ROOT_PATH = "/World/FindSurfaceValidation"
OBJECTS_PATH = f"{ROOT_PATH}/Objects"
SENSOR_RIG_PATH = f"{ROOT_PATH}/SensorRig"
DETECTED_PATH = f"{ROOT_PATH}/Detected"

OBJECT_ORDER = (
    GeometryMode.PLANE,
    GeometryMode.SPHERE,
    GeometryMode.CYLINDER,
    GeometryMode.CONE,
    GeometryMode.TORUS,
)

PARAMETER_NAMES: dict[GeometryMode, tuple[str, ...]] = {
    GeometryMode.PLANE: ("width_m", "height_m"),
    GeometryMode.SPHERE: ("radius_m",),
    GeometryMode.CYLINDER: ("radius_m", "length_m"),
    GeometryMode.CONE: ("top_radius_m", "bottom_radius_m", "length_m"),
    GeometryMode.TORUS: (
        "major_radius_m",
        "minor_radius_m",
        "sweep_start_deg",
        "sweep_angle_deg",
    ),
}

MINIMUM_TORUS_SWEEP_DEG = 0.1
MINIMUM_OBJECT_SUBDIVISION = 3
MAXIMUM_OBJECT_SUBDIVISION = 256
DEFAULT_OBJECT_SUBDIVISION = 96

GEOMETRY_LENGTH_PARAMETERS = frozenset(
    name
    for names in PARAMETER_NAMES.values()
    for name in names
    if name.endswith("_m")
)


def constrain_geometry_parameter_edit(
    geometry_type: GeometryMode | str,
    parameters: Mapping[str, Any],
    *,
    edited_parameter: str | None,
    minimum_length_m: float,
) -> dict[str, float]:
    """Clamp one edited dimension without retroactively rewriting untouched values."""

    mode = GeometryMode(geometry_type)
    minimum = _finite(minimum_length_m, "minimum_length_m")
    if minimum <= 0.0:
        raise ValueError("minimum_length_m must be greater than zero")
    values = {name: _finite(value, name) for name, value in parameters.items()}
    name = None if edited_parameter is None else str(edited_parameter)
    if name not in PARAMETER_NAMES[mode]:
        return values

    if mode is GeometryMode.TORUS and name == "sweep_start_deg":
        values[name] = min(360.0, max(0.0, values[name]))
        return values
    if mode is GeometryMode.TORUS and name == "sweep_angle_deg":
        values[name] = min(360.0, max(MINIMUM_TORUS_SWEEP_DEG, values[name]))
        return values
    if name not in GEOMETRY_LENGTH_PARAMETERS:
        return values

    values[name] = max(minimum, values[name])
    if mode is not GeometryMode.TORUS:
        return values

    major_name, minor_name = "major_radius_m", "minor_radius_m"
    major, minor = values[major_name], values[minor_name]
    if name == minor_name:
        # Keep the non-edited major radius stable when possible. If a newly
        # raised minimum makes that impossible, expand the major radius just
        # enough to retain the analytic torus invariant.
        if major < 2.0 * minimum:
            major = 2.0 * minimum
        minor = min(max(minor, minimum), major - minimum)
    else:
        major = max(major, minimum, minor + minimum)
    values[major_name], values[minor_name] = major, minor
    return values


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _validate_parameters(
    geometry_type: GeometryMode, parameters: Mapping[str, Any]
) -> tuple[tuple[str, float], ...]:
    expected = PARAMETER_NAMES[geometry_type]
    missing = set(expected) - set(parameters)
    extra = set(parameters) - set(expected)
    if missing or extra:
        raise ValueError(
            f"{geometry_type.value} parameters differ from the contract; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    values = {name: _finite(parameters[name], name) for name in expected}
    if geometry_type is GeometryMode.CONE:
        for name in ("top_radius_m", "bottom_radius_m"):
            if values[name] < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if values["top_radius_m"] == 0.0 and values["bottom_radius_m"] == 0.0:
            raise ValueError("a cone must have at least one non-zero radius")
        _positive(values["length_m"], "length_m")
    elif geometry_type is GeometryMode.TORUS:
        _positive(values["major_radius_m"], "major_radius_m")
        _positive(values["minor_radius_m"], "minor_radius_m")
        if values["major_radius_m"] <= values["minor_radius_m"]:
            raise ValueError("major_radius_m must be greater than minor_radius_m")
        if not 0.0 <= values["sweep_start_deg"] <= 360.0:
            raise ValueError("torus sweep start must be between 0 and 360 degrees")
        if not 0.0 < values["sweep_angle_deg"] <= 360.0:
            raise ValueError("torus sweep angle must be greater than 0 and at most 360 degrees")
    else:
        for name in expected:
            _positive(values[name], name)
    return tuple((name, values[name]) for name in expected)


@dataclass(frozen=True)
class SceneObject:
    geometry_type: GeometryMode
    enabled: bool
    pose_world_m: Pose
    parameters: tuple[tuple[str, float], ...]
    subdivision: int
    display_subdivision: int
    face_color_rgba: tuple[float, float, float, float]
    wire_color_rgba: tuple[float, float, float, float]

    def __init__(
        self,
        geometry_type: GeometryMode | str,
        enabled: bool,
        pose_world_m: Pose,
        parameters: Mapping[str, Any] | Sequence[tuple[str, Any]],
        subdivision: Any = DEFAULT_OBJECT_SUBDIVISION,
        display_subdivision: Any | None = None,
        face_color_rgba: Sequence[Any] = (0.36, 0.72, 0.94, 0.25),
        wire_color_rgba: Sequence[Any] = (0.10, 0.43, 0.62, 1.0),
    ) -> None:
        mode = GeometryMode(geometry_type)
        if mode is GeometryMode.AUTO:
            raise ValueError("AUTO is not an as-designed object type")
        payload = dict(parameters)
        subdivision_value = int(subdivision)
        if subdivision_value != float(subdivision):
            raise ValueError("subdivision must be an integer")
        if not MINIMUM_OBJECT_SUBDIVISION <= subdivision_value <= MAXIMUM_OBJECT_SUBDIVISION:
            raise ValueError(
                f"subdivision must be between {MINIMUM_OBJECT_SUBDIVISION} and "
                f"{MAXIMUM_OBJECT_SUBDIVISION}"
            )
        display_subdivision_value = subdivision_value if display_subdivision is None else int(display_subdivision)
        if display_subdivision_value != float(
            subdivision_value if display_subdivision is None else display_subdivision
        ):
            raise ValueError("display_subdivision must be an integer")
        if not MINIMUM_OBJECT_SUBDIVISION <= display_subdivision_value <= MAXIMUM_OBJECT_SUBDIVISION:
            raise ValueError(
                f"display_subdivision must be between {MINIMUM_OBJECT_SUBDIVISION} and "
                f"{MAXIMUM_OBJECT_SUBDIVISION}"
            )

        def color(value: Sequence[Any], name: str) -> tuple[float, float, float, float]:
            result = tuple(float(component) for component in value)
            if len(result) != 4 or not all(math.isfinite(component) and 0.0 <= component <= 1.0 for component in result):
                raise ValueError(f"{name} must contain four finite RGBA values in [0, 1]")
            return result  # type: ignore[return-value]
        object.__setattr__(self, "geometry_type", mode)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "pose_world_m", pose_world_m)
        object.__setattr__(self, "parameters", _validate_parameters(mode, payload))
        object.__setattr__(self, "subdivision", subdivision_value)
        object.__setattr__(self, "display_subdivision", display_subdivision_value)
        object.__setattr__(self, "face_color_rgba", color(face_color_rgba, "face_color_rgba"))
        object.__setattr__(self, "wire_color_rgba", color(wire_color_rgba, "wire_color_rgba"))

    @property
    def object_id(self) -> str:
        return self.geometry_type.value

    @property
    def prim_path(self) -> str:
        return f"{OBJECTS_PATH}/{self.object_id}"

    @property
    def parameter_dict(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def sensor_signature(self) -> tuple[object, ...]:
        """State that can alter the sensor-authoritative USD surface."""

        return (
            self.geometry_type,
            self.enabled,
            self.pose_world_m,
            self.parameters,
            self.subdivision,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "subdivision": self.subdivision,
            "display_subdivision": self.display_subdivision,
            "face_color_rgba": list(self.face_color_rgba),
            "wire_color_rgba": list(self.wire_color_rgba),
            **self.pose_world_m.to_dict(),
            **self.parameter_dict,
        }

    @classmethod
    def from_config(
        cls, geometry_type: GeometryMode | str, payload: Mapping[str, Any]
    ) -> "SceneObject":
        mode = GeometryMode(geometry_type)
        pose = Pose(
            payload.get("position_m", (0.0, 0.0, 0.0)),
            payload.get("orientation_xyzw", (0.0, 0.0, 0.0, 1.0)),
        )
        return cls(
            mode,
            payload.get("enabled", True),
            pose,
            {name: payload[name] for name in PARAMETER_NAMES[mode]},
            payload.get("subdivision", DEFAULT_OBJECT_SUBDIVISION),
            payload.get("display_subdivision", payload.get("subdivision", DEFAULT_OBJECT_SUBDIVISION)),
            payload.get("face_color_rgba", (0.36, 0.72, 0.94, 0.25)),
            payload.get("wire_color_rgba", (0.10, 0.43, 0.62, 1.0)),
        )


@dataclass(frozen=True)
class SceneBounds:
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]
    center_m: tuple[float, float, float]
    bounding_sphere_radius_m: float
    enabled_object_count: int

    @property
    def detection_available(self) -> bool:
        return self.enabled_object_count > 0


@dataclass(frozen=True)
class SceneCatalog:
    objects: tuple[SceneObject, ...]

    def __post_init__(self) -> None:
        observed = tuple(item.geometry_type for item in self.objects)
        if observed != OBJECT_ORDER:
            raise ValueError("scene catalog must contain exactly one object of each type")

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, Any]]) -> "SceneCatalog":
        return cls(
            tuple(
                SceneObject.from_config(mode, config[mode.value.lower()])
                for mode in OBJECT_ORDER
            )
        )

    def object(self, geometry_type: GeometryMode | str) -> SceneObject:
        mode = GeometryMode(geometry_type)
        return self.objects[OBJECT_ORDER.index(mode)]

    def replacing(self, replacement: SceneObject) -> "SceneCatalog":
        return SceneCatalog(
            tuple(
                replacement if item.geometry_type is replacement.geometry_type else item
                for item in self.objects
            )
        )

    @property
    def bounds(self) -> SceneBounds:
        object_bounds = [_world_bounds(item) for item in self.objects if item.enabled]
        if not object_bounds:
            return SceneBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), math.sqrt(3.0), 0)
        minimum = tuple(min(bounds[0][axis] for bounds in object_bounds) for axis in range(3))
        maximum = tuple(max(bounds[1][axis] for bounds in object_bounds) for axis in range(3))
        center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
        radius = math.sqrt(
            sum(((maximum[axis] - minimum[axis]) * 0.5) ** 2 for axis in range(3))
        )
        return SceneBounds(minimum, maximum, center, radius, len(object_bounds))


class ScanIntentKind(str, Enum):
    ONE_SHOT_AFTER_OBJECT_CHANGE = "one-shot-after-object-change"


@dataclass(frozen=True)
class ScanIntent:
    sequence: int
    kind: ScanIntentKind = ScanIntentKind.ONE_SHOT_AFTER_OBJECT_CHANGE


@dataclass(frozen=True)
class ObjectChangeEvent:
    before: SceneObject
    after: SceneObject
    scene_generation: int
    selection_revision: int
    selection_invalidated: bool
    scan_intents: tuple[ScanIntent, ...]

    def __post_init__(self) -> None:
        if len(self.scan_intents) != 1:
            raise ValueError("an object change must emit exactly one scan intent")


class SceneState:
    """Mutable transaction target with sensor-execution subscriptions."""

    def __init__(
        self,
        catalog: SceneCatalog,
        *,
        object_writer: Callable[[SceneObject], None] | None = None,
        event_sink: Callable[[ObjectChangeEvent], None] | None = None,
    ) -> None:
        self.catalog = catalog
        self.scene_generation = 0
        self.selection_revision = 0
        self._scan_sequence = 0
        self._object_writer = object_writer
        self._event_sink = event_sink

    def replace_object(self, replacement: SceneObject) -> ObjectChangeEvent:
        before = self.catalog.object(replacement.geometry_type)
        if replacement == before:
            raise ValueError("object replacement must change enable, pose, or parameters")
        if self._object_writer is not None:
            self._object_writer(replacement)
        self.catalog = self.catalog.replacing(replacement)
        self.scene_generation += 1
        self.selection_revision += 1
        self._scan_sequence += 1
        event = ObjectChangeEvent(
            before,
            replacement,
            self.scene_generation,
            self.selection_revision,
            True,
            (ScanIntent(self._scan_sequence),),
        )
        if self._event_sink is not None:
            self._event_sink(event)
        return event

    def replace_presentation(self, replacement: SceneObject) -> bool:
        """Update display-only object state without touching USD or scan generations."""

        before = self.catalog.object(replacement.geometry_type)
        if before.sensor_signature != replacement.sensor_signature:
            raise ValueError("presentation replacement cannot change sensor-authoritative state")
        if replacement == before:
            return False
        self.catalog = self.catalog.replacing(replacement)
        return True

    def command(self, replacement: SceneObject) -> "SceneObjectCommand":
        return SceneObjectCommand(self, self.catalog.object(replacement.geometry_type), replacement)


@dataclass(frozen=True)
class SceneObjectCommand:
    state: SceneState
    before: SceneObject
    after: SceneObject
    name: str = field(default="Change as-designed object", init=False)

    @property
    def scope(self):
        from .transactions import CommandScope

        return CommandScope.DURABLE

    def apply(self) -> None:
        self.state.replace_object(self.after)

    def undo(self) -> None:
        self.state.replace_object(self.before)


@dataclass(frozen=True)
class StagePose:
    position_stage_units: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


class StageSpace:
    """Convert canonical right-handed Z-up metres at the USD boundary."""

    def __init__(self, metadata: StageMetadata) -> None:
        self.metadata = metadata
        half = math.sqrt(0.5)
        self._stage_to_world = {
            "X": (0.0, -half, 0.0, half),
            "Y": (half, 0.0, 0.0, half),
            "Z": (0.0, 0.0, 0.0, 1.0),
        }[metadata.up_axis]

    def to_world_point(self, point_stage: Sequence[float]) -> tuple[float, float, float]:
        scaled = tuple(float(value) * self.metadata.meters_per_unit for value in point_stage)
        return _rotate(self._stage_to_world, scaled)

    def to_stage_point(self, point_world_m: Sequence[float]) -> tuple[float, float, float]:
        rotated = _rotate(quat_conjugate_xyzw(self._stage_to_world), tuple(point_world_m))
        return tuple(value / self.metadata.meters_per_unit for value in rotated)

    def to_stage_pose(self, pose_world_m: Pose) -> StagePose:
        inverse = quat_conjugate_xyzw(self._stage_to_world)
        orientation = quat_multiply_xyzw(
            quat_multiply_xyzw(inverse, pose_world_m.orientation_xyzw), self._stage_to_world,
        )
        return StagePose(self.to_stage_point(pose_world_m.position_m), orientation)

    def to_world_pose(self, pose_stage: StagePose) -> Pose:
        basis = self._stage_to_world
        orientation = quat_multiply_xyzw(
            quat_multiply_xyzw(basis, pose_stage.orientation_xyzw), quat_conjugate_xyzw(basis),
        )
        return Pose(self.to_world_point(pose_stage.position_stage_units), orientation)


def _local_half_extents(item: SceneObject) -> tuple[float, float, float]:
    values = item.parameter_dict
    if item.geometry_type is GeometryMode.PLANE:
        return values["width_m"] * 0.5, values["height_m"] * 0.5, 0.0
    if item.geometry_type is GeometryMode.SPHERE:
        return (values["radius_m"],) * 3
    if item.geometry_type in (GeometryMode.CYLINDER, GeometryMode.CONE):
        radius = values.get("radius_m", max(values.get("top_radius_m", 0.0), values.get("bottom_radius_m", 0.0)))
        return radius, radius, values["length_m"] * 0.5
    extent = values["major_radius_m"] + values["minor_radius_m"]
    return extent, extent, values["minor_radius_m"]


def _world_bounds(item: SceneObject):
    half = _local_half_extents(item)
    corners = []
    for x in (-half[0], half[0]):
        for y in (-half[1], half[1]):
            for z in (-half[2], half[2]):
                rotated = _rotate(item.pose_world_m.orientation_xyzw, (x, y, z))
                corners.append(tuple(rotated[axis] + item.pose_world_m.position_m[axis] for axis in range(3)))
    return (
        tuple(min(point[axis] for point in corners) for axis in range(3)),
        tuple(max(point[axis] for point in corners) for axis in range(3)),
    )


def _rotate(rotation, value):
    vector = (float(value[0]), float(value[1]), float(value[2]))
    return rotate_vector_xyzw(rotation, vector)


__all__ = [
    "DETECTED_PATH",
    "DEFAULT_OBJECT_SUBDIVISION",
    "GEOMETRY_LENGTH_PARAMETERS",
    "MINIMUM_TORUS_SWEEP_DEG",
    "MAXIMUM_OBJECT_SUBDIVISION",
    "MINIMUM_OBJECT_SUBDIVISION",
    "OBJECTS_PATH",
    "OBJECT_ORDER",
    "ObjectChangeEvent",
    "PARAMETER_NAMES",
    "ROOT_PATH",
    "SENSOR_RIG_PATH",
    "ScanIntent",
    "ScanIntentKind",
    "SceneBounds",
    "SceneCatalog",
    "SceneObject",
    "SceneObjectCommand",
    "SceneState",
    "StagePose",
    "StageSpace",
    "constrain_geometry_parameter_edit",
]
