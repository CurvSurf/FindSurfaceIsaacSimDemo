"""Generation-aware async operation and ordered resource ownership primitives."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class OperationToken:
    generation: int
    sequence: int


@dataclass(frozen=True)
class OperationSnapshot:
    token: OperationToken | None = None
    phase: str = "idle"
    progress: float = 0.0
    result: Any = None
    error: BaseException | None = None


class OperationTracker:
    """Publish progress/results only for the current generation and operation."""

    def __init__(self) -> None:
        self._sequence = 0
        self._snapshot = OperationSnapshot()

    @property
    def snapshot(self) -> OperationSnapshot:
        return self._snapshot

    def start(self, generation: int, phase: str = "starting") -> OperationToken:
        self._sequence += 1
        token = OperationToken(generation, self._sequence)
        self._snapshot = OperationSnapshot(token=token, phase=phase)
        return token

    def update(self, token: OperationToken, *, phase: str, progress: float) -> bool:
        if not self._is_active(token):
            return False
        self._snapshot = OperationSnapshot(
            token=token, phase=phase, progress=max(0.0, min(1.0, float(progress)))
        )
        return True

    def complete(
        self, token: OperationToken, *, result: Any = None, error: BaseException | None = None
    ) -> bool:
        if not self._is_active(token):
            return False
        self._snapshot = OperationSnapshot(
            token=token,
            phase="failed" if error is not None else "complete",
            progress=1.0,
            result=result,
            error=error,
        )
        return True

    def invalidate(self, generation: int) -> OperationToken:
        return self.start(generation, phase="invalidated")

    def _is_active(self, token: OperationToken) -> bool:
        return token == self._snapshot.token and self._snapshot.phase not in {
            "complete",
            "failed",
            "invalidated",
        }


class ShutdownStep(IntEnum):
    DETECTOR = 10
    SENSOR_PUBLICATION = 20
    SENSOR_PRODUCTS = 30
    RENDERER = 40
    STAGE_SESSION_UI = 50


CloseAction = Callable[[], None | Awaitable[None]]


class AsyncResourceOwner:
    """Close resources once, in Safe Exit phase order and reverse acquisition order."""

    def __init__(self) -> None:
        self._resources: list[tuple[ShutdownStep, int, str, CloseAction]] = []
        self._next_sequence = 0
        self._accepting_commands = True
        self._closed = False
        self._failures: tuple[tuple[str, BaseException], ...] = ()
        self._close_lock = asyncio.Lock()

    @property
    def accepting_commands(self) -> bool:
        return self._accepting_commands

    @property
    def closed(self) -> bool:
        return self._closed

    def own(self, step: ShutdownStep, name: str, close: CloseAction) -> None:
        if not self._accepting_commands:
            raise RuntimeError("cannot acquire resources after Safe Exit begins")
        self._resources.append((step, self._next_sequence, name, close))
        self._next_sequence += 1

    async def close(self) -> tuple[tuple[str, BaseException], ...]:
        self._accepting_commands = False
        async with self._close_lock:
            if self._closed:
                return self._failures
            failures: list[tuple[str, BaseException]] = []
            resources = sorted(self._resources, key=lambda item: (item[0], -item[1]))
            self._resources.clear()
            for _step, _sequence, name, close in resources:
                try:
                    outcome = close()
                    if inspect.isawaitable(outcome):
                        await outcome
                except BaseException as error:
                    failures.append((name, error))
            self._failures = tuple(failures)
            self._closed = True
            return self._failures


__all__ = [
    "AsyncResourceOwner",
    "OperationSnapshot",
    "OperationToken",
    "OperationTracker",
    "ShutdownStep",
]
