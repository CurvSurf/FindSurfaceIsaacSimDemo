"""Sensor-native normalization and bounded immutable ScanFrame publication."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import threading
from typing import Any, Mapping, Sequence

import numpy as np
from .axial_error import distance_error_sigmas_m, validate_distance_error

from .domain import (
    NativePayloadProvenance,
    Pose,
    ScanFrame,
    SensorProfileDescriptor,
    StageMetadata,
    freeze_mapping,
)


@dataclass(frozen=True)
class NativePointPayload:
    """One complete native point payload plus its explicit conversion contract."""

    points_native: Any = field(repr=False, compare=False)
    timestamp_s: float
    native_frame_id: str
    native_units: str
    native_axes: str
    metres_per_native_unit: float = 1.0
    metres_per_native_axis: tuple[float, float, float] | None = None
    axis_transform_to_sensor: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    minimum_range_m: float = 0.0
    maximum_range_m: float = math.inf
    reject_origin: bool = True
    axial_error_sigma_m: float = 0.0
    axial_error_seed: int = 0
    details: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None
    axial_error_model: str = "constant"
    axial_error_slope_m_per_m: float = 0.0
    axial_error_intercept_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if not self.native_frame_id or not self.native_units or not self.native_axes:
            raise ValueError("native payload identifiers cannot be empty")
        if not math.isfinite(self.metres_per_native_unit) or self.metres_per_native_unit <= 0.0:
            raise ValueError("metres_per_native_unit must be finite and greater than zero")
        if self.metres_per_native_axis is not None:
            scales = tuple(float(value) for value in self.metres_per_native_axis)
            if len(scales) != 3 or not all(math.isfinite(value) and value > 0.0 for value in scales):
                raise ValueError("metres_per_native_axis must contain three finite positive values")
            object.__setattr__(self, "metres_per_native_axis", scales)
        if not math.isfinite(self.minimum_range_m) or self.minimum_range_m < 0.0:
            raise ValueError("minimum_range_m must be finite and non-negative")
        if math.isnan(self.maximum_range_m) or self.maximum_range_m <= self.minimum_range_m:
            raise ValueError("maximum_range_m must be greater than minimum_range_m")
        sigma = float(self.axial_error_sigma_m)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("axial_error_sigma_m must be finite and non-negative")
        seed = int(self.axial_error_seed)
        if seed != self.axial_error_seed or seed < 0:
            raise ValueError("axial_error_seed must be a non-negative integer")
        matrix = np.asarray(self.axis_transform_to_sensor, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("axis_transform_to_sensor must be a finite 3x3 matrix")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-9, rtol=0.0):
            raise ValueError("axis_transform_to_sensor must be orthonormal")
        object.__setattr__(self, "axis_transform_to_sensor", tuple(tuple(float(v) for v in row) for row in matrix))
        object.__setattr__(self, "axial_error_sigma_m", sigma)
        object.__setattr__(self, "axial_error_seed", seed)
        model, slope, intercept = validate_distance_error(
            self.axial_error_model, self.axial_error_slope_m_per_m, self.axial_error_intercept_m)
        object.__setattr__(self, "axial_error_model", model)
        object.__setattr__(self, "axial_error_slope_m_per_m", slope)
        object.__setattr__(self, "axial_error_intercept_m", intercept)
        object.__setattr__(self, "details", freeze_mapping(self.details))


@dataclass(frozen=True)
class NormalizationResult:
    frame: ScanFrame
    returned_point_count: int
    invalid_point_count: int
    out_of_range_point_count: int


def _rotation_matrix(pose: Pose) -> np.ndarray:
    x, y, z, w = pose.orientation_xyzw
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def normalize_native_payload(
    *,
    adapter: str,
    profile: SensorProfileDescriptor,
    generation: int,
    sequence: int,
    payload: NativePointPayload,
    sensor_pose_world: Pose,
    stage_metadata: StageMetadata = StageMetadata(),
) -> NormalizationResult:
    """Validate, compact, and transform a complete payload exactly once."""

    if not adapter:
        raise ValueError("adapter cannot be empty")
    native = np.asarray(payload.points_native, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 3:
        raise ValueError("native points must have shape [N, 3]")
    returned_count = int(native.shape[0])
    finite = np.isfinite(native).all(axis=1)
    invalid_count = int(returned_count - np.count_nonzero(finite))

    unit_scale = payload.metres_per_native_axis or (payload.metres_per_native_unit,) * 3
    sensor_points = native[finite] * np.asarray(unit_scale, dtype=np.float64)
    sensor_points = sensor_points @ np.asarray(payload.axis_transform_to_sensor, dtype=np.float64).T
    distances = np.linalg.norm(sensor_points, axis=1)
    in_range = (distances >= payload.minimum_range_m) & (distances <= payload.maximum_range_m)
    if payload.reject_origin:
        origin = distances <= 1.0e-12
        invalid_count += int(np.count_nonzero(origin))
        in_range &= ~origin
    out_of_range_count = int(np.count_nonzero(~in_range & (distances > 1.0e-12)))
    sensor_points = sensor_points[in_range]
    error_active = (payload.axial_error_sigma_m > 0.0 if payload.axial_error_model == "constant"
                    else payload.axial_error_slope_m_per_m > 0.0 or payload.axial_error_intercept_m > 0.0)
    if error_active and sensor_points.size:
        ray_lengths = distances[in_range]
        ray_directions = sensor_points / ray_lengths[:, None]
        generator = np.random.Generator(np.random.PCG64(payload.axial_error_seed))
        sigmas = (payload.axial_error_sigma_m if payload.axial_error_model == "constant" else
                  distance_error_sigmas_m(ray_lengths, payload.axial_error_slope_m_per_m,
                                         payload.axial_error_intercept_m))
        axial_error = generator.normal(0.0, sigmas, sensor_points.shape[0])
        sensor_points = sensor_points + ray_directions * axial_error[:, None]

    rotation = _rotation_matrix(sensor_pose_world)
    world = sensor_points @ rotation.T + np.asarray(sensor_pose_world.position_m, dtype=np.float64)
    provenance = NativePayloadProvenance(
        adapter=adapter,
        native_frame_id=payload.native_frame_id,
        native_units=payload.native_units,
        native_axes=payload.native_axes,
        invalid_point_count=invalid_count,
        out_of_range_point_count=out_of_range_count,
        details={
            "returned_point_count": returned_count,
            "metres_per_native_unit": payload.metres_per_native_unit,
            "metres_per_native_axis": unit_scale,
            "axis_transform_to_sensor": payload.axis_transform_to_sensor,
            "application_axial_error_sigma_m": payload.axial_error_sigma_m,
            "application_axial_error_seed": payload.axial_error_seed,
            "application_axial_error_model": payload.axial_error_model,
            "application_axial_error_slope_m_per_m": payload.axial_error_slope_m_per_m,
            "application_axial_error_intercept_m": payload.axial_error_intercept_m,
            "application_axial_error_distance_definition": "pre-noise sensor-to-point Euclidean range in metres",
            "application_axial_error_distance_formula": "max(0, a * range_m + b)",
            "profile_calibration": profile.calibration_provenance,
            "payload": payload.details,
        },
    )
    frame = ScanFrame(
        generation=generation,
        sequence=sequence,
        timestamp_s=payload.timestamp_s,
        sensor_kind=profile.sensor_kind,
        profile_id=profile.profile_id,
        points_world_m=np.ascontiguousarray(world, dtype=np.float32),
        native_payload_provenance=provenance,
        sensor_pose_world=sensor_pose_world,
        sensor_model=profile.model,
        stage_metadata=stage_metadata,
    )
    return NormalizationResult(frame, returned_count, invalid_count, out_of_range_count)


@dataclass(frozen=True)
class FramePublication:
    revision: int
    frame: ScanFrame


@dataclass(frozen=True)
class DispatcherTelemetry:
    publication_attempts: int = 0
    published_frames: int = 0
    stale_frames: int = 0
    overwritten_deliveries: int = 0
    delivered_frames: int = 0
    consumer_count: int = 0
    pending_delivery_count: int = 0


class _ConsumerMailbox:
    def __init__(self) -> None:
        self.pending: FramePublication | None = None


class ScanSubscription:
    def __init__(self, dispatcher: "ScanDispatcher", name: str) -> None:
        self._dispatcher = dispatcher
        self.name = name

    def take_latest(self) -> FramePublication | None:
        dispatcher = self._dispatcher
        if dispatcher is None:
            return None
        return dispatcher._take_latest(self.name)

    def close(self) -> None:
        dispatcher, self._dispatcher = self._dispatcher, None
        if dispatcher is not None:
            dispatcher._unsubscribe(self.name)


class ScanDispatcher:
    """One current publication plus one overwrite-only mailbox per consumer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_generation: int | None = None
        self._latest: FramePublication | None = None
        self._mailboxes: dict[str, _ConsumerMailbox] = {}
        self._telemetry = DispatcherTelemetry()

    def activate_generation(self, generation: int) -> None:
        generation = int(generation)
        if generation < 0:
            raise ValueError("generation cannot be negative")
        with self._lock:
            if self._active_generation is not None and generation <= self._active_generation:
                raise ValueError("active generation must increase monotonically")
            self._active_generation = generation
            for mailbox in self._mailboxes.values():
                mailbox.pending = None

    def activate_and_publish(self, frame: ScanFrame) -> FramePublication:
        """Atomically replace the active generation with its first valid frame."""

        with self._lock:
            if self._active_generation is not None and frame.generation <= self._active_generation:
                raise ValueError("active generation must increase monotonically")
            self._active_generation = frame.generation
            revision = 1 if self._latest is None else self._latest.revision + 1
            publication = FramePublication(revision, frame)
            overwritten = sum(mailbox.pending is not None for mailbox in self._mailboxes.values())
            for mailbox in self._mailboxes.values():
                mailbox.pending = publication
            self._latest = publication
            self._telemetry = replace(
                self._telemetry,
                publication_attempts=self._telemetry.publication_attempts + 1,
                published_frames=self._telemetry.published_frames + 1,
                overwritten_deliveries=self._telemetry.overwritten_deliveries + overwritten,
            )
            return publication

    def activate_empty_generation(self, generation: int) -> FramePublication | None:
        """Atomically activate a ready sensor generation that has no returns."""

        generation = int(generation)
        if generation < 0:
            raise ValueError("generation cannot be negative")
        with self._lock:
            if self._active_generation is not None and generation <= self._active_generation:
                raise ValueError("active generation must increase monotonically")
            self._active_generation = generation
            latest, self._latest = self._latest, None
            for mailbox in self._mailboxes.values():
                mailbox.pending = None
            return latest

    def subscribe(self, name: str) -> ScanSubscription:
        if not name:
            raise ValueError("consumer name cannot be empty")
        with self._lock:
            if name in self._mailboxes:
                raise ValueError(f"consumer already subscribed: {name}")
            self._mailboxes[name] = _ConsumerMailbox()
        return ScanSubscription(self, name)

    def publish(self, frame: ScanFrame) -> FramePublication | None:
        with self._lock:
            telemetry = replace(self._telemetry, publication_attempts=self._telemetry.publication_attempts + 1)
            if frame.generation != self._active_generation or (
                self._latest is not None and frame.identity <= self._latest.frame.identity
            ):
                self._telemetry = replace(telemetry, stale_frames=telemetry.stale_frames + 1)
                return None
            revision = 1 if self._latest is None else self._latest.revision + 1
            publication = FramePublication(revision, frame)
            overwritten = sum(mailbox.pending is not None for mailbox in self._mailboxes.values())
            for mailbox in self._mailboxes.values():
                mailbox.pending = publication
            self._latest = publication
            self._telemetry = replace(
                telemetry,
                published_frames=telemetry.published_frames + 1,
                overwritten_deliveries=telemetry.overwritten_deliveries + overwritten,
            )
            return publication

    def snapshot(self) -> FramePublication | None:
        with self._lock:
            return self._latest

    def telemetry(self) -> DispatcherTelemetry:
        with self._lock:
            return replace(
                self._telemetry,
                consumer_count=len(self._mailboxes),
                pending_delivery_count=sum(mailbox.pending is not None for mailbox in self._mailboxes.values()),
            )

    def clear(self) -> FramePublication | None:
        with self._lock:
            latest, self._latest = self._latest, None
            for mailbox in self._mailboxes.values():
                mailbox.pending = None
            return latest

    def clear_generation(self, generation: int) -> FramePublication | None:
        """Clear only if ``generation`` still owns the dispatcher."""

        with self._lock:
            if self._active_generation != int(generation):
                return None
            latest, self._latest = self._latest, None
            for mailbox in self._mailboxes.values():
                mailbox.pending = None
            return latest

    def _take_latest(self, name: str) -> FramePublication | None:
        with self._lock:
            mailbox = self._mailboxes.get(name)
            if mailbox is None:
                return None
            publication, mailbox.pending = mailbox.pending, None
            if publication is not None:
                self._telemetry = replace(
                    self._telemetry,
                    delivered_frames=self._telemetry.delivered_frames + 1,
                )
            return publication

    def _unsubscribe(self, name: str) -> None:
        with self._lock:
            self._mailboxes.pop(name, None)


__all__ = [
    "DispatcherTelemetry",
    "FramePublication",
    "NativePointPayload",
    "NormalizationResult",
    "ScanDispatcher",
    "ScanSubscription",
    "normalize_native_payload",
]
