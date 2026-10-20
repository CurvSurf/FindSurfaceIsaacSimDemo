"""Rolling wall-clock rates for actual sensor polls and published scan frames."""

from collections import deque
import math
import time


class SensorRateTracker:
    def __init__(self, window_s: float = 1.0) -> None:
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("rate window must be finite and positive")
        self._window_s = window_s
        self._runtime = None
        self._samples = deque()

    def sample(self, runtime, *, active: bool, now_s: float | None = None) -> tuple[float, float]:
        if not active or runtime is None:
            self._runtime = None
            self._samples.clear()
            return 0.0, 0.0
        telemetry = runtime.telemetry()
        now = time.monotonic() if now_s is None else now_s
        current = (now, telemetry.poll_attempts, telemetry.published_frames)
        if (runtime is not self._runtime or (self._samples and
                (now < self._samples[-1][0] or current[1] < self._samples[-1][1]
                 or current[2] < self._samples[-1][2]))):
            self._samples.clear()
        self._runtime = runtime
        self._samples.append(current)
        while len(self._samples) > 2 and self._samples[1][0] <= now - self._window_s:
            self._samples.popleft()
        first = self._samples[0]
        elapsed = now - first[0]
        if elapsed <= 0:
            return 0.0, 0.0
        return (current[1] - first[1]) / elapsed, (current[2] - first[2]) / elapsed
