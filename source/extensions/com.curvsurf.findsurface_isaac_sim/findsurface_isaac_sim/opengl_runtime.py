"""Framework-free OpenGL packet and resource lifetime contracts."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
from typing import Callable, Iterable, Sequence


class GpuResourceKind(str, Enum):
    SHADER = "shader"
    PROGRAM = "program"
    VERTEX_ARRAY = "vertex_array"
    BUFFER = "buffer"
    TEXTURE = "texture"
    FRAMEBUFFER = "framebuffer"


@dataclass(frozen=True)
class GpuResource:
    kind: GpuResourceKind
    handle: int
    context_generation: int


class GpuResourceOwner:
    """Track handles by their creating context and retire them in reverse order."""

    def __init__(self) -> None:
        self._context_generation = 0
        self._resources: list[GpuResource] = []
        self._stale_discard_count = 0

    @property
    def context_generation(self) -> int:
        return self._context_generation

    @property
    def resource_count(self) -> int:
        return len(self._resources)

    @property
    def stale_discard_count(self) -> int:
        return self._stale_discard_count

    def begin_context(self, *, previous_context_lost: bool = False) -> int:
        if self._resources:
            if not previous_context_lost:
                raise RuntimeError("live GPU resources must be released before context recreation")
            self._stale_discard_count += len(self._resources)
            self._resources.clear()
        self._context_generation += 1
        return self._context_generation

    def own(self, kind: GpuResourceKind, handle: int) -> int:
        value = int(handle)
        if self._context_generation <= 0:
            raise RuntimeError("a GL context generation must begin before owning resources")
        if value <= 0:
            raise ValueError("GPU handles must be positive")
        self._resources.append(GpuResource(GpuResourceKind(kind), value, self._context_generation))
        return value

    def forget(self, kind: GpuResourceKind, handle: int) -> None:
        target = (GpuResourceKind(kind), int(handle))
        for index in range(len(self._resources) - 1, -1, -1):
            resource = self._resources[index]
            if (resource.kind, resource.handle) == target:
                del self._resources[index]
                return
        raise KeyError(f"GPU resource is not owned: {target[0].value} {target[1]}")

    def is_current(self, kind: GpuResourceKind, handle: int) -> bool:
        target = (GpuResourceKind(kind), int(handle), self._context_generation)
        return any(
            (item.kind, item.handle, item.context_generation) == target
            for item in self._resources
        )

    def release(self, delete: Callable[[GpuResource], None]) -> int:
        released = 0
        while self._resources:
            resource = self._resources.pop()
            if resource.context_generation != self._context_generation:
                self._stale_discard_count += 1
                continue
            delete(resource)
            released += 1
        return released


def _float32_bytes(values: Iterable[Sequence[float]], name: str) -> tuple[bytes, int]:
    packed = array("f")
    count = 0
    for value in values:
        if len(value) != 3:
            raise ValueError(f"{name} entries must contain exactly three coordinates")
        coordinates = tuple(float(component) for component in value)
        if not all(math.isfinite(component) for component in coordinates):
            raise ValueError(f"{name} coordinates must be finite")
        packed.extend(coordinates)
        count += 1
    return packed.tobytes(), count


@dataclass(frozen=True)
class OpenGLRenderPacket:
    """Immutable CPU packet published without exposing mutable domain state."""

    revision: int
    triangle_vertices_f32: bytes
    triangle_vertex_count: int
    point_vertices_f32: bytes
    point_count: int

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        if self.triangle_vertex_count < 0 or self.point_count < 0:
            raise ValueError("vertex counts cannot be negative")
        if self.triangle_vertex_count % 3:
            raise ValueError("triangle vertices must be provided in groups of three")
        if len(self.triangle_vertices_f32) != self.triangle_vertex_count * 12:
            raise ValueError("triangle byte size does not match triangle_vertex_count")
        if len(self.point_vertices_f32) != self.point_count * 12:
            raise ValueError("point byte size does not match point_count")

    @classmethod
    def build(
        cls,
        revision: int,
        *,
        triangles: Iterable[Sequence[float]] = (),
        points: Iterable[Sequence[float]] = (),
    ) -> "OpenGLRenderPacket":
        triangle_bytes, triangle_count = _float32_bytes(triangles, "triangles")
        point_bytes, point_count = _float32_bytes(points, "points")
        return cls(int(revision), triangle_bytes, triangle_count, point_bytes, point_count)

    @classmethod
    def empty(cls) -> "OpenGLRenderPacket":
        return cls(0, b"", 0, b"", 0)


class LatestRenderPacket:
    """One-slot atomic publication boundary used by UI/update and render callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet = OpenGLRenderPacket.empty()

    def submit(self, packet: OpenGLRenderPacket) -> None:
        if not isinstance(packet, OpenGLRenderPacket):
            raise TypeError("packet must be an OpenGLRenderPacket")
        with self._lock:
            self._packet = packet

    def snapshot(self) -> OpenGLRenderPacket:
        with self._lock:
            return self._packet


@dataclass(frozen=True)
class OpenGLTelemetry:
    initialize_count: int = 0
    context_generation: int = 0
    draw_count: int = 0
    upload_count: int = 0
    last_upload_ms: float = 0.0
    last_draw_ms: float = 0.0
    packet_revision: int = -1
    triangle_vertex_count: int = 0
    point_count: int = 0
    logical_width: float = 0.0
    logical_height: float = 0.0
    framebuffer_width: int = 0
    framebuffer_height: int = 0
    dpi_scale_x: float = 1.0
    dpi_scale_y: float = 1.0
    recreate_count: int = 0
    stale_handle_discard_count: int = 0
    release_count: int = 0
    resources_released: bool = False
    gl_error_count: int = 0
    last_gl_error: int | None = None
    error: str | None = None


class OpenGLRuntimeState:
    """Shared state machine for a real Kit scene or deterministic headless fake."""

    def __init__(self, error_hook: Callable[[int], None] | None = None) -> None:
        self.packets = LatestRenderPacket()
        self.resources = GpuResourceOwner()
        self._lock = threading.Lock()
        self._telemetry = OpenGLTelemetry()
        self._close_requested = False
        self._recreate_requested = False
        self.error_hook = error_hook

    def telemetry(self) -> OpenGLTelemetry:
        with self._lock:
            return self._telemetry

    def update_telemetry(self, **changes) -> None:
        with self._lock:
            self._telemetry = replace(self._telemetry, **changes)

    def set_logical_size(self, width: float, height: float) -> None:
        width = max(0.0, float(width))
        height = max(0.0, float(height))
        current = self.telemetry()
        self.update_telemetry(
            logical_width=width,
            logical_height=height,
            dpi_scale_x=(current.framebuffer_width / width if width else 1.0),
            dpi_scale_y=(current.framebuffer_height / height if height else 1.0),
        )

    def set_framebuffer_size(self, width: int, height: int) -> None:
        width = max(0, int(width))
        height = max(0, int(height))
        current = self.telemetry()
        self.update_telemetry(
            framebuffer_width=width,
            framebuffer_height=height,
            dpi_scale_x=(width / current.logical_width if current.logical_width else 1.0),
            dpi_scale_y=(height / current.logical_height if current.logical_height else 1.0),
        )

    def request_close(self) -> None:
        self._close_requested = True

    def request_recreate(self) -> None:
        if not self._close_requested:
            self._recreate_requested = True

    @property
    def close_requested(self) -> bool:
        return self._close_requested

    def consume_recreate_request(self) -> bool:
        requested, self._recreate_requested = self._recreate_requested, False
        return requested

    def release_complete(self, released: int) -> None:
        current = self.telemetry()
        self._close_requested = False
        self.update_telemetry(
            release_count=current.release_count + int(released),
            resources_released=True,
            stale_handle_discard_count=self.resources.stale_discard_count,
        )


class HeadlessOpenGLScene:
    """No-op renderer with the same packet/context lifecycle for host and headless use."""

    def __init__(self) -> None:
        self._state = OpenGLRuntimeState()
        self._initialized = False

    def submit(self, packet: OpenGLRenderPacket) -> None:
        self._state.packets.submit(packet)

    def set_logical_size(self, width: float, height: float) -> None:
        self._state.set_logical_size(width, height)

    def initialize(self, width: int, height: int, *, context_lost: bool = False) -> None:
        if self._initialized and not context_lost:
            return
        generation = self._state.resources.begin_context(previous_context_lost=context_lost)
        self._initialized = True
        self._state.set_framebuffer_size(width, height)
        telemetry = self._state.telemetry()
        self._state.update_telemetry(
            initialize_count=telemetry.initialize_count + 1,
            context_generation=generation,
            stale_handle_discard_count=self._state.resources.stale_discard_count,
            resources_released=False,
        )

    def own_for_test(self, kind: GpuResourceKind, handle: int) -> int:
        return self._state.resources.own(kind, handle)

    def draw(self, width: int, height: int) -> None:
        if not self._initialized or self._state.close_requested:
            return
        self._state.set_framebuffer_size(width, height)
        packet = self._state.packets.snapshot()
        telemetry = self._state.telemetry()
        uploaded = packet.revision != telemetry.packet_revision
        self._state.update_telemetry(
            draw_count=telemetry.draw_count + 1,
            upload_count=telemetry.upload_count + int(uploaded),
            packet_revision=packet.revision,
            triangle_vertex_count=packet.triangle_vertex_count,
            point_count=packet.point_count,
        )

    def close(self) -> None:
        if not self._initialized:
            return
        self._state.request_close()
        released = self._state.resources.release(lambda _resource: None)
        self._initialized = False
        self._state.release_complete(released)

    def telemetry(self) -> OpenGLTelemetry:
        return self._state.telemetry()


__all__ = [
    "GpuResource",
    "GpuResourceKind",
    "GpuResourceOwner",
    "HeadlessOpenGLScene",
    "LatestRenderPacket",
    "OpenGLRenderPacket",
    "OpenGLRuntimeState",
    "OpenGLTelemetry",
]
