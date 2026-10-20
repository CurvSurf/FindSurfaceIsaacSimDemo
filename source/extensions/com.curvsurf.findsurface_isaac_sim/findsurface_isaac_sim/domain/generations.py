"""Pure stale-result and bounded latest-frame rules."""

from __future__ import annotations

from enum import Enum

from .models import DetectionResult, FrameIdentity, GenerationSnapshot, PreviewRecord, ScanFrame


class StaleReason(str, Enum):
    SESSION = "session_generation"
    SCENE = "scene_generation"
    SENSOR = "sensor_generation"
    FRAME = "frame_generation"
    OPTIONS = "options_revision"
    SELECTION = "selection_revision"
    SCAN = "scan_identity"


def stale_reasons(
    result: DetectionResult,
    current: GenerationSnapshot,
    current_frame: FrameIdentity | None,
) -> tuple[StaleReason, ...]:
    reasons: list[StaleReason] = []
    pairs = (
        (StaleReason.SESSION, result.generations.session, current.session),
        (StaleReason.SCENE, result.generations.scene, current.scene),
        (StaleReason.SENSOR, result.generations.sensor, current.sensor),
        (StaleReason.FRAME, result.generations.frame, current.frame),
        (StaleReason.OPTIONS, result.generations.options_revision, current.options_revision),
        (StaleReason.SELECTION, result.generations.selection_revision, current.selection_revision),
    )
    reasons.extend(reason for reason, observed, expected in pairs if observed != expected)
    if current_frame is None or (
        result.request_key.scan_generation,
        result.request_key.scan_sequence,
    ) != (current_frame.generation, current_frame.sequence):
        reasons.append(StaleReason.SCAN)
    return tuple(reasons)


def preview_record(
    result: DetectionResult,
    current: GenerationSnapshot,
    current_frame: FrameIdentity | None,
) -> PreviewRecord:
    reasons = stale_reasons(result, current, current_frame)
    return PreviewRecord(result=result, accepted=not reasons, stale_reasons=tuple(reason.value for reason in reasons))


class GenerationGate:
    """Reject completions that do not exactly match current durable/transient state."""

    def __init__(self, current: GenerationSnapshot | None = None) -> None:
        self._current = current or GenerationSnapshot()
        self._frame: FrameIdentity | None = None

    @property
    def current(self) -> GenerationSnapshot:
        return self._current

    @property
    def frame(self) -> FrameIdentity | None:
        return self._frame

    def update(self, current: GenerationSnapshot, frame: FrameIdentity | None) -> None:
        previous = self._current
        for name in current.__dataclass_fields__:
            if getattr(current, name) < getattr(previous, name):
                raise ValueError(f"generation axis cannot move backwards: {name}")
        self._current = current
        self._frame = frame

    def evaluate(self, result: DetectionResult) -> PreviewRecord:
        return preview_record(result, self._current, self._frame)


class LatestFrameSlot:
    """Own at most one frame and reject duplicate or older publications."""

    def __init__(self) -> None:
        self._latest: ScanFrame | None = None

    @property
    def latest(self) -> ScanFrame | None:
        return self._latest

    def publish(self, frame: ScanFrame) -> bool:
        latest = self._latest
        if latest is not None and frame.identity <= latest.identity:
            return False
        self._latest = frame
        return True

    def clear(self) -> ScanFrame | None:
        latest, self._latest = self._latest, None
        return latest
