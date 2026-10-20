"""Idempotent Safe Exit ownership skeleton shared by host tests and Kit."""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable


class ShutdownPhase(Enum):
    RUNNING = auto()
    BLOCKING_COMMANDS = auto()
    RELEASING_RESOURCES = auto()
    COMPLETE = auto()


class SafeExitCoordinator:
    """Own teardown callbacks and execute them once in reverse acquisition order."""

    def __init__(self) -> None:
        self._phase = ShutdownPhase.RUNNING
        self._resources: list[tuple[str, Callable[[], None]]] = []

    @property
    def phase(self) -> ShutdownPhase:
        return self._phase

    @property
    def accepting_commands(self) -> bool:
        return self._phase is ShutdownPhase.RUNNING

    def own(self, name: str, close: Callable[[], None]) -> None:
        if not self.accepting_commands:
            raise RuntimeError("cannot acquire resources after Safe Exit begins")
        self._resources.append((name, close))

    def close(self) -> tuple[tuple[str, Exception], ...]:
        if self._phase is ShutdownPhase.COMPLETE:
            return ()
        self._phase = ShutdownPhase.BLOCKING_COMMANDS
        self._phase = ShutdownPhase.RELEASING_RESOURCES
        failures: list[tuple[str, Exception]] = []
        while self._resources:
            name, close = self._resources.pop()
            try:
                close()
            except Exception as exception:
                failures.append((name, exception))
        self._phase = ShutdownPhase.COMPLETE
        return tuple(failures)
