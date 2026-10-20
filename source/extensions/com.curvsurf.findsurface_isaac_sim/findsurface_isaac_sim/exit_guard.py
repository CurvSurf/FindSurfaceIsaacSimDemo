"""Unified async Safe Exit state shared by toolbar, shortcuts, and OS close."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .operations import AsyncResourceOwner, ShutdownStep


class ExitTrigger(str, Enum):
    TOOLBAR = "toolbar"
    ALT_F4 = "alt-f4"
    CTRL_SHIFT_Q = "ctrl-shift-q"
    WINDOW_CLOSE = "window-close"
    APP_QUIT = "app-quit"


class ExitPhase(str, Enum):
    RUNNING = "RUNNING"
    BLOCKING_COMMANDS = "BLOCKING_COMMANDS"
    RELEASING_RESOURCES = "RELEASING_RESOURCES"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class ExitSnapshot:
    phase: ExitPhase = ExitPhase.RUNNING
    trigger: ExitTrigger | None = None
    failures: tuple[tuple[str, str], ...] = ()


class UnifiedExitController:
    """Own the exact ordered path and coalesce all exit races into one task."""

    def __init__(
        self,
        post_quit: Callable[[int], None],
        *,
        owner: AsyncResourceOwner | None = None,
    ) -> None:
        self.owner = owner or AsyncResourceOwner()
        self._post_quit = post_quit
        self._snapshot = ExitSnapshot()
        self._task: asyncio.Task | None = None

    @property
    def snapshot(self) -> ExitSnapshot:
        return self._snapshot

    @property
    def accepting_commands(self) -> bool:
        return self._task is None and self.owner.accepting_commands

    def own(self, step: ShutdownStep, name: str, close) -> None:
        self.owner.own(step, name, close)

    def request(self, trigger: ExitTrigger | str) -> asyncio.Task:
        if self._task is None:
            trigger = ExitTrigger(trigger)
            self._snapshot = ExitSnapshot(ExitPhase.BLOCKING_COMMANDS, trigger)
            self._task = asyncio.ensure_future(self._close())
        return self._task

    async def wait(self) -> ExitSnapshot:
        if self._task is not None:
            await self._task
        return self._snapshot

    async def _close(self) -> None:
        self._snapshot = ExitSnapshot(ExitPhase.RELEASING_RESOURCES, self._snapshot.trigger)
        failures = await self.owner.close()
        encoded = tuple((name, f"{type(error).__name__}: {error}") for name, error in failures)
        self._snapshot = ExitSnapshot(ExitPhase.COMPLETE, self._snapshot.trigger, encoded)
        self._post_quit(0 if not failures else 1)


__all__ = ["ExitPhase", "ExitSnapshot", "ExitTrigger", "UnifiedExitController"]
