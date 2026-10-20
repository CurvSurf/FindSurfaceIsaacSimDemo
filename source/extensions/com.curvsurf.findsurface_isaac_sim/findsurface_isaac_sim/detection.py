"""Latest-only serialized detection scheduling and native-context ownership."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import Enum
import math
from threading import Condition, Thread, current_thread
from time import monotonic, perf_counter
from typing import Callable

from .domain import (
    DetectionRequest,
    DetectionRequestKey,
    DetectionResult,
    Detector,
    FrameIdentity,
    GenerationGate,
    GenerationSnapshot,
)


class DetectionSchedulerClosedError(RuntimeError):
    """A request was submitted after scheduler shutdown began."""


class DetectionSubmit(str, Enum):
    STARTED = "STARTED"
    BUSY_LATEST = "BUSY_LATEST"


class DetectionCompletionStatus(str, Enum):
    PUBLISHED = "PUBLISHED"
    STALE = "STALE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DetectionCompletion:
    request_key: DetectionRequestKey
    generations: GenerationSnapshot
    status: DetectionCompletionStatus
    result: DetectionResult | None
    stale_reasons: tuple[str, ...]
    error: str | None
    latency_s: float
    latest_retry: bool
    request: DetectionRequest | None = None


@dataclass(frozen=True)
class DetectionTelemetry:
    attempts: int
    completed: int
    failed: int
    stale: int
    published: int
    busy_skips: int
    latest_retries: int
    callback_failures: int
    max_concurrency: int
    latency_p50_s: float | None
    latency_p95_s: float | None


def _percentile(samples: list[float], fraction: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class SerializedDetectionScheduler:
    """Own one detector on one worker and collapse busy work to one latest request."""

    def __init__(
        self,
        detector_factory: Callable[[], Detector],
        *,
        current_generations: GenerationSnapshot | None = None,
        current_frame: FrameIdentity | None = None,
        on_result: Callable[[DetectionResult], None] | None = None,
        on_completion: Callable[[DetectionCompletion], None] | None = None,
        completion_ack_required: bool = False,
        thread_name: str = "findsurface-detection",
    ) -> None:
        self._factory = detector_factory
        self._on_result = on_result
        self._on_completion = on_completion
        self._completion_ack_required = bool(completion_ack_required and on_completion is not None)
        self._condition = Condition()
        self._gate = GenerationGate(current_generations)
        if current_generations is not None or current_frame is not None:
            self._gate.update(current_generations or GenerationSnapshot(), current_frame)
        self._next: tuple[DetectionRequest, bool] | None = None
        self._dirty: DetectionRequest | None = None
        self._active_key: DetectionRequestKey | None = None
        self._awaiting_ack: tuple[DetectionRequestKey, GenerationSnapshot] | None = None
        self._dispatching_completion = False
        self._closing = False
        self._closed = False
        self._startup_error: str | None = None
        self._attempts = 0
        self._completed = 0
        self._failed = 0
        self._stale = 0
        self._published = 0
        self._busy_skips = 0
        self._latest_retries = 0
        self._callback_failures = 0
        self._active_calls = 0
        self._max_concurrency = 0
        self._latencies: list[float] = []
        self._native_call_samples = deque(maxlen=120)
        self._thread = Thread(target=self._run, name=thread_name, daemon=True)
        self._thread.start()

    @property
    def startup_error(self) -> str | None:
        with self._condition:
            return self._startup_error

    @property
    def busy(self) -> bool:
        with self._condition:
            return (
                self._active_key is not None
                or self._next is not None
                or self._awaiting_ack is not None
                or self._dispatching_completion
            )

    def update_current(self, generations: GenerationSnapshot, frame: FrameIdentity | None) -> None:
        with self._condition:
            if self._closed:
                raise DetectionSchedulerClosedError("detection scheduler is closed")
            self._gate.update(generations, frame)

    def submit(self, request: DetectionRequest) -> DetectionSubmit:
        with self._condition:
            if self._closing or self._closed:
                raise DetectionSchedulerClosedError("detection scheduler is closing")
            if self._startup_error is not None:
                raise RuntimeError(self._startup_error)
            if self._active_key is None and self._next is None and self._awaiting_ack is None:
                self._next = (request, False)
                self._condition.notify_all()
                return DetectionSubmit.STARTED
            if self._active_key is None and self._next is not None and self._next[1]:
                self._next = (request, True)
                self._busy_skips += 1
                self._condition.notify_all()
                return DetectionSubmit.BUSY_LATEST
            self._dirty = request
            self._busy_skips += 1
            return DetectionSubmit.BUSY_LATEST

    def acknowledge_completion(
        self,
        request_key: DetectionRequestKey,
        generations: GenerationSnapshot,
    ) -> bool:
        """Release a presented completion and schedule at most the latest demand."""

        with self._condition:
            token = request_key, generations
            if self._awaiting_ack != token:
                return False
            self._awaiting_ack = None
            if not self._closing:
                dirty, self._dirty = self._dirty, None
                if dirty is not None and (
                    dirty.key != request_key or dirty.generations != generations
                ):
                    self._next = (dirty, True)
                    self._latest_retries += 1
            else:
                self._dirty = None
            self._condition.notify_all()
            return True

    def discard_pending_retry(self) -> None:
        """Forget demand that has not entered the native call.

        The active call is deliberately not cancelled.  This is used when the
        current hover/selection disappears so a previously recorded busy skip
        cannot become a pointless native retry after the active call returns.
        """

        with self._condition:
            if self._closing or self._closed:
                return
            self._dirty = None
            if self._next is not None and self._next[1]:
                self._next = None
            self._condition.notify_all()

    def successful_call_mean_ms(self, window: int) -> float | None:
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 120:
            raise ValueError("successful call window must be an integer between 1 and 120")
        with self._condition:
            samples = [duration for duration, successful in tuple(self._native_call_samples)[-window:] if successful]
            return 1000.0 * math.fsum(samples) / len(samples) if samples else None

    def telemetry(self) -> DetectionTelemetry:
        with self._condition:
            return DetectionTelemetry(
                attempts=self._attempts,
                completed=self._completed,
                failed=self._failed,
                stale=self._stale,
                published=self._published,
                busy_skips=self._busy_skips,
                latest_retries=self._latest_retries,
                callback_failures=self._callback_failures,
                max_concurrency=self._max_concurrency,
                latency_p50_s=_percentile(self._latencies, 0.50),
                latency_p95_s=_percentile(self._latencies, 0.95),
            )

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else monotonic() + timeout_s
        with self._condition:
            while (
                self._active_key is not None
                or self._next is not None
                or self._dirty is not None
                or self._awaiting_ack is not None
                or self._dispatching_completion
            ):
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout_s: float | None = None) -> None:
        if current_thread() is self._thread:
            raise RuntimeError("detection scheduler cannot join itself")
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._next = None
            self._dirty = None
            self._awaiting_ack = None
            self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("native detection is still running; context was not released")
        with self._condition:
            self._closed = True

    def _run(self) -> None:
        detector: Detector | None = None
        try:
            try:
                detector = self._factory()
            except Exception as exception:
                with self._condition:
                    self._startup_error = (
                        f"detector initialization failed: {type(exception).__name__}: {exception}"
                    )
                    self._next = None
                    self._dirty = None
                    self._closing = True
                    self._condition.notify_all()
                return
            while True:
                with self._condition:
                    while self._next is None and not self._closing:
                        self._condition.wait()
                    if self._closing:
                        return
                    request, is_retry = self._next
                    self._next = None
                    self._active_key = request.key
                    self._attempts += 1
                    self._active_calls += 1
                    self._max_concurrency = max(self._max_concurrency, self._active_calls)
                started = perf_counter()
                result = None
                error = None
                try:
                    result = detector.detect(request)
                except Exception as exception:
                    error = f"{type(exception).__name__}: {exception}"
                latency = perf_counter() - started
                native_duration = getattr(detector, "native_call_duration_s", None)
                with self._condition:
                    self._active_calls -= 1
                    self._active_key = None
                    self._latencies.append(latency)
                    if native_duration is not None and math.isfinite(native_duration) and native_duration >= 0:
                        self._native_call_samples.append((native_duration, error is None))
                    if error is not None:
                        self._failed += 1
                        status = DetectionCompletionStatus.FAILED
                        stale_reasons: tuple[str, ...] = ()
                    else:
                        self._completed += 1
                        record = self._gate.evaluate(result)
                        stale_reasons = record.stale_reasons
                        if record.accepted:
                            self._published += 1
                            status = DetectionCompletionStatus.PUBLISHED
                        else:
                            self._stale += 1
                            status = DetectionCompletionStatus.STALE
                    if self._completion_ack_required and not self._closing:
                        self._awaiting_ack = (request.key, request.generations)
                    elif not self._closing:
                        dirty, self._dirty = self._dirty, None
                        if dirty is not None and (
                            dirty.key != request.key or dirty.generations != request.generations
                        ):
                            self._next = (dirty, True)
                            self._latest_retries += 1
                    else:
                        self._dirty = None
                    self._dispatching_completion = True
                    self._condition.notify_all()
                completion = DetectionCompletion(
                    request_key=request.key,
                    generations=request.generations,
                    status=status,
                    result=result,
                    stale_reasons=stale_reasons,
                    error=error,
                    latency_s=latency,
                    latest_retry=is_retry,
                    request=request,
                )
                try:
                    if status is DetectionCompletionStatus.PUBLISHED and self._on_result is not None:
                        self._on_result(result)
                    if self._on_completion is not None:
                        self._on_completion(completion)
                except Exception:
                    with self._condition:
                        self._callback_failures += 1
                finally:
                    with self._condition:
                        self._dispatching_completion = False
                        self._condition.notify_all()
                if self._completion_ack_required:
                    with self._condition:
                        while self._awaiting_ack is not None and not self._closing:
                            self._condition.wait()
        finally:
            if detector is not None:
                detector.close()
            with self._condition:
                self._active_key = None
                self._next = None
                self._dirty = None
                self._awaiting_ack = None
                self._dispatching_completion = False
                self._closed = True
                self._condition.notify_all()


__all__ = [
    "DetectionCompletion",
    "DetectionCompletionStatus",
    "DetectionSchedulerClosedError",
    "DetectionSubmit",
    "DetectionTelemetry",
    "SerializedDetectionScheduler",
]
