"""Framework-free immutable value objects shared across every app boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np


UINT64_MAX = (1 << 64) - 1
Vec3 = tuple[float, float, float]
QuaternionXyzw = tuple[float, float, float, float]
FrozenValue = bool | int | float | str | None | tuple["FrozenValue", ...] | tuple[tuple[str, "FrozenValue"], ...]
FrozenMapping = tuple[tuple[str, FrozenValue], ...]


def _uint64(value: int, name: str) -> int:
    result = int(value)
    if result < 0 or result > UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return result


def _finite_tuple(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vec3(value: Sequence[float], name: str = "vector") -> Vec3:
    return _finite_tuple(value, 3, name)  # type: ignore[return-value]


def freeze(value: Any) -> FrozenValue:
    """Recursively convert JSON-like values into deterministic immutable values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("frozen numeric values must be finite")
        return value
    if isinstance(value, Mapping):
        return tuple(
            (str(key), freeze(item))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze(item) for item in value)
    raise TypeError(f"unsupported mutable or opaque value: {type(value).__name__}")


def freeze_mapping(value: Mapping[str, Any] | Sequence[tuple[str, Any]] | None) -> FrozenMapping:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted(((str(key), freeze(item)) for key, item in items), key=lambda entry: entry[0]))


def thaw(value: FrozenValue) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {item[0]: thaw(item[1]) for item in value}
        return [thaw(item) for item in value]
    return value


def immutable_float32_points(value: Any) -> np.ndarray:
    """Snapshot mutable input or reuse validated, bytes-backed float32[N,3] storage."""

    array = np.asarray(value, dtype=np.float32, order="C")
    if array.size == 0 and array.shape != (0, 3):
        array = np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points_world_m must have shape [N, 3]")
    if not np.isfinite(array).all():
        raise ValueError("points_world_m must contain only finite values")
    owner = array
    while type(owner) is np.ndarray:
        owner = owner.base
    if type(owner) is bytes:
        return array.view()
    raw = array.tobytes(order="C")
    result = np.frombuffer(raw, dtype=np.float32).reshape((-1, 3))
    if result.flags.writeable or not result.flags.c_contiguous:
        raise AssertionError("immutable point storage construction failed")
    return result


@dataclass(frozen=True)
class Pose:
    position_m: Vec3 = (0.0, 0.0, 0.0)
    orientation_xyzw: QuaternionXyzw = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))
        orientation = _finite_tuple(self.orientation_xyzw, 4, "orientation_xyzw")
        magnitude = math.sqrt(sum(component * component for component in orientation))
        if magnitude <= 1e-12:
            raise ValueError("orientation_xyzw must have non-zero length")
        object.__setattr__(self, "orientation_xyzw", tuple(component / magnitude for component in orientation))

    def to_dict(self) -> dict[str, object]:
        return {"position_m": list(self.position_m), "orientation_xyzw": list(self.orientation_xyzw)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Pose":
        return cls(payload["position_m"], payload["orientation_xyzw"])


@dataclass(frozen=True)
class StageMetadata:
    up_axis: str = "Z"
    meters_per_unit: float = 1.0
    right_handed: bool = True

    def __post_init__(self) -> None:
        axis = str(self.up_axis).upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("up_axis must be X, Y, or Z")
        if not math.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0.0:
            raise ValueError("meters_per_unit must be finite and greater than zero")
        if not self.right_handed:
            raise ValueError("internal stage metadata must resolve to right-handed coordinates")
        object.__setattr__(self, "up_axis", axis)


class SensorKind(str, Enum):
    DEPTH = "DEPTH"
    RTX_LIDAR = "RTX_LIDAR"


@dataclass(frozen=True)
class SensorProfileDescriptor:
    profile_id: str
    sensor_kind: SensorKind
    model: FrozenMapping = ()
    calibration_provenance: FrozenMapping = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        object.__setattr__(self, "sensor_kind", SensorKind(self.sensor_kind))
        object.__setattr__(self, "model", freeze_mapping(self.model))
        object.__setattr__(self, "calibration_provenance", freeze_mapping(self.calibration_provenance))


@dataclass(frozen=True)
class NativePayloadProvenance:
    adapter: str
    native_frame_id: str
    native_units: str
    native_axes: str
    invalid_point_count: int = 0
    out_of_range_point_count: int = 0
    details: FrozenMapping = ()

    def __post_init__(self) -> None:
        if not self.adapter or not self.native_frame_id or not self.native_units or not self.native_axes:
            raise ValueError("native provenance identifiers cannot be empty")
        if self.invalid_point_count < 0 or self.out_of_range_point_count < 0:
            raise ValueError("native rejected-point counts cannot be negative")
        object.__setattr__(self, "details", freeze_mapping(self.details))


@dataclass(frozen=True, eq=False)
class ScanFrame:
    generation: int
    sequence: int
    timestamp_s: float
    sensor_kind: SensorKind
    profile_id: str
    points_world_m: np.ndarray = field(repr=False)
    native_payload_provenance: NativePayloadProvenance
    sensor_pose_world: Pose = field(default_factory=Pose)
    sensor_model: FrozenMapping = ()
    stage_metadata: StageMetadata = field(default_factory=StageMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", _uint64(self.generation, "generation"))
        object.__setattr__(self, "sequence", _uint64(self.sequence, "sequence"))
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        object.__setattr__(self, "sensor_kind", SensorKind(self.sensor_kind))
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        object.__setattr__(self, "points_world_m", immutable_float32_points(self.points_world_m))
        object.__setattr__(self, "sensor_model", freeze_mapping(self.sensor_model))

    @property
    def compacted_count(self) -> int:
        return int(self.points_world_m.shape[0])

    @property
    def identity(self) -> "FrameIdentity":
        return FrameIdentity(self.generation, self.sequence)

    def to_dict(self, *, include_points: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "generation": self.generation,
            "sequence": self.sequence,
            "timestamp_s": self.timestamp_s,
            "sensor_kind": self.sensor_kind.value,
            "profile_id": self.profile_id,
            "compacted_count": self.compacted_count,
            "sensor_pose_world": self.sensor_pose_world.to_dict(),
            "sensor_model": thaw(self.sensor_model),
            "stage_metadata": asdict(self.stage_metadata),
            "native_payload_provenance": {
                **asdict(self.native_payload_provenance),
                "details": thaw(self.native_payload_provenance.details),
            },
        }
        if include_points:
            payload["points_world_m"] = self.points_world_m.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScanFrame":
        provenance = dict(payload["native_payload_provenance"])
        return cls(
            generation=payload["generation"],
            sequence=payload["sequence"],
            timestamp_s=payload["timestamp_s"],
            sensor_kind=payload["sensor_kind"],
            profile_id=payload["profile_id"],
            points_world_m=payload["points_world_m"],
            sensor_pose_world=Pose.from_dict(payload["sensor_pose_world"]),
            sensor_model=payload.get("sensor_model", {}),
            stage_metadata=StageMetadata(**payload["stage_metadata"]),
            native_payload_provenance=NativePayloadProvenance(**provenance),
        )


@dataclass(frozen=True, order=True)
class FrameIdentity:
    generation: int
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", _uint64(self.generation, "generation"))
        object.__setattr__(self, "sequence", _uint64(self.sequence, "sequence"))


@dataclass(frozen=True)
class GenerationSnapshot:
    session: int = 0
    scene: int = 0
    sensor: int = 0
    frame: int = 0
    options_revision: int = 0
    selection_revision: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _uint64(getattr(self, name), name))

    def advance(self, *axes: str) -> "GenerationSnapshot":
        if not axes:
            raise ValueError("at least one generation axis must advance")
        if len(set(axes)) != len(axes):
            raise ValueError("generation axes cannot be advanced more than once per transition")
        unknown = set(axes) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown generation axes: {sorted(unknown)}")
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for axis in axes:
            values[axis] = _uint64(values[axis] + 1, axis)
        return replace(self, **values)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class GeometryMode(str, Enum):
    AUTO = "Auto"
    PLANE = "Plane"
    SPHERE = "Sphere"
    CYLINDER = "Cylinder"
    CONE = "Cone"
    TORUS = "Torus"


@dataclass(frozen=True)
class DetectionOptions:
    geometry_mode: GeometryMode = GeometryMode.AUTO
    seed_radius_m: float = 0.15
    measurement_accuracy_m: float = 0.003
    mean_distance_m: float = 0.05
    radial_expansion: int = 7
    lateral_extension: int = 5
    smart_convert_cone_to_cylinder: bool = True
    smart_convert_torus_to_sphere: bool = True
    smart_convert_torus_to_cylinder: bool = True
    revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_mode", GeometryMode(self.geometry_mode))
        for name in ("seed_radius_m", "measurement_accuracy_m", "mean_distance_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if not 0 <= self.radial_expansion <= 10 or not 0 <= self.lateral_extension <= 10:
            raise ValueError("expansion values must be between 0 and 10")
        for name in (
            "smart_convert_cone_to_cylinder",
            "smart_convert_torus_to_sphere",
            "smart_convert_torus_to_cylinder",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "revision", _uint64(self.revision, "revision"))


@dataclass(frozen=True, order=True)
class DetectionRequestKey:
    scan_generation: int
    scan_sequence: int
    seed_index: int
    geometry_mode: GeometryMode
    option_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scan_generation", _uint64(self.scan_generation, "scan_generation"))
        object.__setattr__(self, "scan_sequence", _uint64(self.scan_sequence, "scan_sequence"))
        if self.seed_index < 0:
            raise ValueError("seed_index cannot be negative")
        object.__setattr__(self, "geometry_mode", GeometryMode(self.geometry_mode))
        object.__setattr__(self, "option_revision", _uint64(self.option_revision, "option_revision"))


@dataclass(frozen=True)
class SelectionSnapshot:
    revision: int
    frame: FrameIdentity
    seed_index: int
    seed_world_m: Vec3
    used_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _uint64(self.revision, "revision"))
        if self.seed_index < 0:
            raise ValueError("seed_index cannot be negative")
        object.__setattr__(self, "seed_world_m", _vec3(self.seed_world_m, "seed_world_m"))
        object.__setattr__(self, "used_fallback", bool(self.used_fallback))


@dataclass(frozen=True)
class DetectionRequest:
    frame: ScanFrame
    selection: SelectionSnapshot
    options: DetectionOptions
    generations: GenerationSnapshot

    def __post_init__(self) -> None:
        if self.selection.frame != self.frame.identity:
            raise ValueError("selection must reference the request frame")
        if self.selection.seed_index >= self.frame.compacted_count:
            raise ValueError("selection seed_index is outside the request frame")
        expected_seed = self.frame.points_world_m[self.selection.seed_index]
        observed_seed = np.asarray(self.selection.seed_world_m, dtype=np.float32)
        if not np.array_equal(expected_seed, observed_seed):
            raise ValueError("selection seed_world_m must match the request frame point")
        if self.generations.sensor != self.frame.generation:
            raise ValueError("generation snapshot sensor must match frame generation")
        if self.generations.options_revision != self.options.revision:
            raise ValueError("generation snapshot options_revision must match options")
        if self.generations.selection_revision != self.selection.revision:
            raise ValueError("generation snapshot selection_revision must match selection")

    @property
    def key(self) -> DetectionRequestKey:
        return DetectionRequestKey(
            self.frame.generation,
            self.frame.sequence,
            self.selection.seed_index,
            self.options.geometry_mode,
            self.options.revision,
        )


@dataclass(frozen=True)
class DetectionResult:
    request_key: DetectionRequestKey
    generations: GenerationSnapshot
    geometry_type: GeometryMode
    parameters: FrozenMapping
    inlier_indices: tuple[int, ...]
    rms_error_m: float

    def __post_init__(self) -> None:
        if self.request_key.scan_generation != self.generations.sensor:
            raise ValueError("request scan_generation must match result sensor generation")
        if self.request_key.option_revision != self.generations.options_revision:
            raise ValueError("request option_revision must match result generations")
        object.__setattr__(self, "geometry_type", GeometryMode(self.geometry_type))
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))
        indices = tuple(int(index) for index in self.inlier_indices)
        if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("inlier_indices must be unique and non-negative")
        object.__setattr__(self, "inlier_indices", indices)
        if not math.isfinite(self.rms_error_m) or self.rms_error_m < 0.0:
            raise ValueError("rms_error_m must be finite and non-negative")


@dataclass(frozen=True, eq=False)
class CapturedSurface:
    capture_id: str
    request_key: DetectionRequestKey
    generations: GenerationSnapshot
    geometry_type: GeometryMode
    parameters: FrozenMapping
    inlier_points_world_m: np.ndarray = field(repr=False)
    seed_world_m: Vec3
    detection_options: DetectionOptions
    frame_provenance: NativePayloadProvenance
    fit_rms_error_m: float = 0.0
    source_inlier_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.capture_id:
            raise ValueError("capture_id cannot be empty")
        object.__setattr__(self, "geometry_type", GeometryMode(self.geometry_type))
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))
        object.__setattr__(self, "inlier_points_world_m", immutable_float32_points(self.inlier_points_world_m))
        object.__setattr__(self, "seed_world_m", _vec3(self.seed_world_m, "seed_world_m"))
        if not isinstance(self.detection_options, DetectionOptions):
            raise TypeError("detection_options must be DetectionOptions")
        if self.detection_options.revision != self.request_key.option_revision:
            raise ValueError("capture options must match the detection request revision")
        if not math.isfinite(self.fit_rms_error_m) or self.fit_rms_error_m < 0.0:
            raise ValueError("capture fit_rms_error_m must be finite and non-negative")
        indices = tuple(int(index) for index in self.source_inlier_indices)
        if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
            raise ValueError("capture source_inlier_indices must be unique and non-negative")
        object.__setattr__(self, "source_inlier_indices", indices)

    @classmethod
    def from_detection(
        cls,
        capture_id: str,
        result: DetectionResult,
        frame: ScanFrame,
        request: DetectionRequest | None = None,
    ) -> "CapturedSurface":
        if result.request_key.scan_generation != frame.generation or result.request_key.scan_sequence != frame.sequence:
            raise ValueError("detection result does not belong to the capture frame")
        if result.request_key.seed_index >= frame.compacted_count:
            raise ValueError("detection result contains an out-of-range seed index")
        if any(index >= frame.compacted_count for index in result.inlier_indices):
            raise ValueError("detection result contains an out-of-range inlier index")
        if request is not None:
            if request.key != result.request_key or request.frame is not frame:
                raise ValueError("capture request must be the exact detection request and frame")
            seed_world_m = request.selection.seed_world_m
            options = request.options
        else:
            seed_world_m = tuple(float(value) for value in frame.points_world_m[result.request_key.seed_index])
            options = DetectionOptions(
                geometry_mode=result.request_key.geometry_mode,
                revision=result.request_key.option_revision,
            )
        coordinates = frame.points_world_m[list(result.inlier_indices)]
        return cls(
            capture_id=capture_id,
            request_key=result.request_key,
            generations=result.generations,
            geometry_type=result.geometry_type,
            parameters=result.parameters,
            inlier_points_world_m=coordinates,
            seed_world_m=seed_world_m,
            detection_options=options,
            frame_provenance=frame.native_payload_provenance,
            fit_rms_error_m=result.rms_error_m,
            source_inlier_indices=result.inlier_indices,
        )


@dataclass(frozen=True)
class PreviewRecord:
    result: DetectionResult
    accepted: bool
    stale_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RendererPacket:
    frame: ScanFrame
    generations: GenerationSnapshot
    visibility_revision: int = 0

    def __post_init__(self) -> None:
        if self.generations.sensor != self.frame.generation:
            raise ValueError("renderer packet sensor generation must match frame")
        object.__setattr__(self, "visibility_revision", _uint64(self.visibility_revision, "visibility_revision"))


@dataclass(frozen=True)
class PickQuery:
    origin_world_m: Vec3
    direction_world: Vec3
    probe_radius_px: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_world_m", _vec3(self.origin_world_m, "origin_world_m"))
        direction = _vec3(self.direction_world, "direction_world")
        magnitude = math.sqrt(sum(component * component for component in direction))
        if magnitude <= 1e-12:
            raise ValueError("direction_world must have non-zero length")
        object.__setattr__(self, "direction_world", tuple(component / magnitude for component in direction))
        if not math.isfinite(self.probe_radius_px) or self.probe_radius_px <= 0.0:
            raise ValueError("probe_radius_px must be finite and greater than zero")


@dataclass(frozen=True)
class GeometrySnapshot:
    geometry_id: str
    geometry_type: GeometryMode
    parameters: FrozenMapping

    def __post_init__(self) -> None:
        if not self.geometry_id:
            raise ValueError("geometry_id cannot be empty")
        object.__setattr__(self, "geometry_type", GeometryMode(self.geometry_type))
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class MetricResult:
    available: bool
    rms_m: float | None
    sample_count: int
    details: FrozenMapping = ()

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        if self.available:
            if self.rms_m is None or not math.isfinite(self.rms_m) or self.rms_m < 0.0:
                raise ValueError("available metric requires a finite non-negative rms_m")
        elif self.rms_m is not None:
            raise ValueError("unavailable metric cannot carry rms_m")
        object.__setattr__(self, "details", freeze_mapping(self.details))


class SensorLifecycle(str, Enum):
    STOPPED = "STOPPED"
    STOPPING = "STOPPING"
    CREATING = "CREATING"
    WARMING = "WARMING"
    RUNNING = "RUNNING"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class SensorStatus:
    lifecycle: SensorLifecycle = SensorLifecycle.STOPPED
    generation: int = 0
    profile_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifecycle", SensorLifecycle(self.lifecycle))
        object.__setattr__(self, "generation", _uint64(self.generation, "generation"))
        if self.lifecycle is SensorLifecycle.ERROR and not self.error:
            raise ValueError("ERROR sensor status requires an error")
        if self.lifecycle is not SensorLifecycle.ERROR and self.error is not None:
            raise ValueError("error is only valid in ERROR lifecycle")
