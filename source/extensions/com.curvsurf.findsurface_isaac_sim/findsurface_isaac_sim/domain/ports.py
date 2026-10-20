"""Framework-free Protocol boundaries for later runtime adapters."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .models import (
    CapturedSurface,
    DetectionRequest,
    DetectionResult,
    GeometrySnapshot,
    MetricResult,
    PickQuery,
    RendererPacket,
    ScanFrame,
    SensorProfileDescriptor,
    SensorStatus,
    Pose,
)


@runtime_checkable
class PointCloudProvider(Protocol):
    def list_profiles(self) -> Sequence[SensorProfileDescriptor]: ...

    def start(self, profile: SensorProfileDescriptor, pose_world: Pose) -> int: ...

    def disable(self) -> None: ...

    def enable(self) -> None: ...

    def scan_once(self) -> None: ...

    def replace(self, profile: SensorProfileDescriptor) -> int: ...

    def latest_frame(self) -> ScanFrame | None: ...

    def latest_status(self) -> SensorStatus: ...

    def close(self) -> None: ...


@runtime_checkable
class Renderer(Protocol):
    def present(self, packet: RendererPacket | None) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class PointPicker(Protocol):
    def pick(self, frame: ScanFrame, query: PickQuery) -> int | None: ...


@runtime_checkable
class Detector(Protocol):
    def detect(self, request: DetectionRequest) -> DetectionResult: ...

    def close(self) -> None: ...


@runtime_checkable
class MetricEvaluator(Protocol):
    def evaluate(self, reference: GeometrySnapshot, capture: CapturedSurface) -> MetricResult: ...
