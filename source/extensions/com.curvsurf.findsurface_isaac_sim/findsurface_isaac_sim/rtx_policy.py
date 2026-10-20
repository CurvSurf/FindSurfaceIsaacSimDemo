"""Host-safe RTX scene backend, point capability, and warning policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Hashable


class SceneRendererMode(str, Enum):
    AUTO = "Auto"
    RTX = "RTX (Experimental)"
    OPENGL = "OpenGL"


class EffectiveSceneRenderer(str, Enum):
    RTX = "RTX"
    OPENGL = "OpenGL"


class RtxPointOverlayStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PERFORMANCE_DISABLED = "performance-disabled"


@dataclass(frozen=True)
class RtxPointOverlayCapability:
    status: RtxPointOverlayStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RtxPointOverlayStatus(self.status))
        if self.status is RtxPointOverlayStatus.AVAILABLE and self.reason is not None:
            raise ValueError("available point overlay cannot have a failure reason")
        if self.status is not RtxPointOverlayStatus.AVAILABLE and not str(self.reason or "").strip():
            raise ValueError("unavailable point overlay requires a reason")

    @property
    def available(self) -> bool:
        return self.status is RtxPointOverlayStatus.AVAILABLE


def probe_rtx_point_overlay(viewport_api: object) -> RtxPointOverlayCapability:
    """Accept only an existing GPU-buffer consumer; per-frame USD Points is forbidden."""

    consumer_names = ("submit_scan_frame_gpu", "set_external_point_buffer")
    if any(callable(getattr(viewport_api, name, None)) for name in consumer_names):
        return RtxPointOverlayCapability(RtxPointOverlayStatus.AVAILABLE)
    return RtxPointOverlayCapability(
        RtxPointOverlayStatus.UNAVAILABLE,
        "ViewportWidget has no ScanFrame GPU overlay consumer; per-frame USD point authoring is disabled",
    )


@dataclass(frozen=True)
class RtxSceneState:
    generation: int = 0
    requested: SceneRendererMode = SceneRendererMode.AUTO
    effective: EffectiveSceneRenderer = EffectiveSceneRenderer.OPENGL
    fallback_reason: str | None = "RTX capability has not been evaluated"
    point_overlay: RtxPointOverlayCapability = RtxPointOverlayCapability(
        RtxPointOverlayStatus.UNAVAILABLE,
        "RTX point-overlay capability has not been evaluated",
    )
    performance_override: bool = False


class RtxScenePolicy:
    """Separate persistent request from effective/session-only renderer decisions."""

    PERFORMANCE_REASON = "session performance choice: OpenGL scene view"

    def __init__(self) -> None:
        self._state = RtxSceneState()

    def snapshot(self) -> RtxSceneState:
        return self._state

    def apply_request(
        self,
        requested: SceneRendererMode,
        *,
        rtx_available: bool,
        point_overlay: RtxPointOverlayCapability | None = None,
        failure_reason: str | None = None,
    ) -> RtxSceneState:
        requested = SceneRendererMode(requested)
        if requested is SceneRendererMode.OPENGL:
            effective = EffectiveSceneRenderer.OPENGL
            reason = None
        elif rtx_available:
            effective = EffectiveSceneRenderer.RTX
            reason = None
        else:
            effective = EffectiveSceneRenderer.OPENGL
            reason = failure_reason or "RTX scene renderer is unavailable"
        return self._replace(
            requested=requested,
            effective=effective,
            fallback_reason=reason,
            point_overlay=point_overlay or self._state.point_overlay,
            performance_override=False,
        )

    def choose_keep_rtx(self) -> RtxSceneState:
        return self._state

    def choose_opengl_for_performance(self) -> RtxSceneState:
        if self._state.requested is not SceneRendererMode.AUTO:
            return self._state
        if self._state.effective is not EffectiveSceneRenderer.RTX:
            return self._state
        return self._replace(
            effective=EffectiveSceneRenderer.OPENGL,
            fallback_reason=self.PERFORMANCE_REASON,
            performance_override=True,
        )

    def _replace(self, **changes) -> RtxSceneState:
        candidate = replace(self._state, **changes)
        if candidate == self._state:
            return self._state
        self._state = replace(candidate, generation=self._state.generation + 1)
        return self._state


class PerformancePhase(str, Enum):
    INACTIVE = "INACTIVE"
    WARMING = "WARMING"
    OBSERVING = "OBSERVING"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class PerformancePolicy:
    warmup_s: float = 5.0
    window_s: float = 5.0
    minimum_rate_ratio: float = 0.90
    required_consecutive_windows: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.warmup_s) or self.warmup_s < 0.0:
            raise ValueError("warmup_s must be finite and nonnegative")
        if not math.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("window_s must be finite and positive")
        if not math.isfinite(self.minimum_rate_ratio) or not 0.0 < self.minimum_rate_ratio <= 1.0:
            raise ValueError("minimum_rate_ratio must be in (0, 1]")
        if self.required_consecutive_windows <= 0:
            raise ValueError("required_consecutive_windows must be positive")


@dataclass(frozen=True)
class PerformanceWindow:
    duration_s: float
    update_hz: float
    complete_scan_hz: float
    target_hz: float
    minimum_healthy_hz: float
    update_count: int
    complete_scan_count: int


@dataclass(frozen=True)
class PerformanceWarning:
    profile_id: str
    point_overlay_status: RtxPointOverlayStatus
    window: PerformanceWindow
    consecutive_deficient_windows: int
    deficient_metrics: tuple[str, ...]


@dataclass(frozen=True)
class PerformanceSnapshot:
    phase: PerformancePhase
    consecutive_deficient_windows: int
    latest_window: PerformanceWindow | None
    pending_warning: PerformanceWarning | None
    warning_issued: bool


class RtxPerformanceMonitor:
    """Emit once after warmup and two deficient windows; never changes renderer state."""

    def __init__(self, policy: PerformancePolicy | None = None) -> None:
        self.policy = policy or PerformancePolicy()
        self._phase = PerformancePhase.INACTIVE
        self._context_key: Hashable | None = None
        self._warmup_started_s: float | None = None
        self._window_started_s: float | None = None
        self._update_count = 0
        self._scan_count = 0
        self._consecutive = 0
        self._latest: PerformanceWindow | None = None
        self._pending: PerformanceWarning | None = None
        self._issued = False
        self._last_now: float | None = None

    def snapshot(self) -> PerformanceSnapshot:
        return PerformanceSnapshot(self._phase, self._consecutive, self._latest, self._pending, self._issued)

    def observe(
        self,
        *,
        now_s: float,
        eligible: bool,
        context_key: Hashable | None,
        profile_id: str,
        target_hz: float,
        complete_scan: bool,
        point_overlay_status: RtxPointOverlayStatus,
    ) -> PerformanceWarning | None:
        now_s, target_hz = float(now_s), float(target_hz)
        if not math.isfinite(now_s) or now_s < 0.0:
            raise ValueError("now_s must be finite and nonnegative")
        if not math.isfinite(target_hz) or target_hz <= 0.0:
            raise ValueError("target_hz must be finite and positive")
        rewound = self._last_now is not None and now_s < self._last_now
        self._last_now = now_s
        if self._issued:
            return None
        if not eligible or context_key is None:
            self._reset(PerformancePhase.INACTIVE)
            return None
        if rewound or context_key != self._context_key or self._warmup_started_s is None:
            self._start(now_s, context_key)
            return None
        if now_s - self._warmup_started_s < self.policy.warmup_s:
            self._phase = PerformancePhase.WARMING
            return None
        if self._window_started_s is None:
            self._window_started_s = now_s
        self._phase = PerformancePhase.OBSERVING
        self._update_count += 1
        self._scan_count += int(bool(complete_scan))
        elapsed = now_s - self._window_started_s
        if elapsed < self.policy.window_s:
            return None
        minimum = target_hz * self.policy.minimum_rate_ratio
        window = PerformanceWindow(
            elapsed,
            self._update_count / elapsed,
            self._scan_count / elapsed,
            target_hz,
            minimum,
            self._update_count,
            self._scan_count,
        )
        self._latest = window
        deficient = tuple(
            name for name, rate in (("update_hz", window.update_hz), ("complete_scan_hz", window.complete_scan_hz))
            if rate < minimum
        )
        self._consecutive = self._consecutive + 1 if deficient else 0
        self._window_started_s = now_s
        self._update_count = self._scan_count = 0
        if self._consecutive < self.policy.required_consecutive_windows:
            return None
        warning = PerformanceWarning(
            str(profile_id), RtxPointOverlayStatus(point_overlay_status), window, self._consecutive, deficient
        )
        self._pending = warning
        self._issued = True
        self._phase = PerformancePhase.PENDING
        return warning

    def resolve(self) -> None:
        if self._issued:
            self._pending = None
            self._phase = PerformancePhase.RESOLVED

    def _start(self, now_s: float, key: Hashable) -> None:
        self._context_key = key
        self._warmup_started_s = now_s
        self._window_started_s = None
        self._update_count = self._scan_count = self._consecutive = 0
        self._latest = self._pending = None
        self._phase = PerformancePhase.WARMING

    def _reset(self, phase: PerformancePhase) -> None:
        self._context_key = self._warmup_started_s = self._window_started_s = None
        self._update_count = self._scan_count = self._consecutive = 0
        self._latest = self._pending = None
        self._phase = phase


__all__ = [
    "EffectiveSceneRenderer", "PerformancePhase", "PerformancePolicy", "PerformanceSnapshot",
    "PerformanceWarning", "PerformanceWindow", "RtxPerformanceMonitor", "RtxPointOverlayCapability",
    "RtxPointOverlayStatus", "RtxScenePolicy", "RtxSceneState", "SceneRendererMode",
    "probe_rtx_point_overlay",
]
