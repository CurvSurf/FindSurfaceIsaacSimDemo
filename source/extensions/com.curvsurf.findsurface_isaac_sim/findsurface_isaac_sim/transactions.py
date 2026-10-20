"""Runtime-independent commands, composite transactions, and history."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class CommandScope(Enum):
    DURABLE = "durable"
    TRANSIENT = "transient"


class Command(Protocol):
    name: str
    scope: CommandScope

    def apply(self) -> None: ...

    def undo(self) -> None: ...


@dataclass(frozen=True)
class FunctionalCommand:
    name: str
    do: Callable[[], None]
    undo_action: Callable[[], None]
    scope: CommandScope = CommandScope.DURABLE

    def apply(self) -> None:
        self.do()

    def undo(self) -> None:
        self.undo_action()


class CompositeCommand:
    """Apply or undo a group atomically using compensating actions."""

    def __init__(self, name: str, commands: tuple[Command, ...] | list[Command]) -> None:
        self.name = name
        self.commands = tuple(commands)
        self.scope = (
            CommandScope.DURABLE
            if any(command.scope is CommandScope.DURABLE for command in self.commands)
            else CommandScope.TRANSIENT
        )

    def apply(self) -> None:
        applied: list[Command] = []
        try:
            for command in self.commands:
                command.apply()
                applied.append(command)
        except Exception:
            for command in reversed(applied):
                command.undo()
            raise

    def undo(self) -> None:
        undone: list[Command] = []
        try:
            for command in reversed(self.commands):
                command.undo()
                undone.append(command)
        except Exception:
            for command in reversed(undone):
                command.apply()
            raise


class CommandHistory:
    def __init__(self) -> None:
        self._undo: list[Command] = []
        self._redo: list[Command] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_names(self) -> tuple[str, ...]:
        return tuple(command.name for command in self._undo)

    def execute(self, command: Command) -> None:
        command.apply()
        if command.scope is CommandScope.DURABLE:
            self._undo.append(command)
            self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        command = self._undo[-1]
        command.undo()
        self._undo.pop()
        self._redo.append(command)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo[-1]
        command.apply()
        self._redo.pop()
        self._undo.append(command)
        return True


__all__ = [
    "Command",
    "CommandHistory",
    "CommandScope",
    "CompositeCommand",
    "FunctionalCommand",
]
