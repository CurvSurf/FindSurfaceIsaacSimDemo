"""Shared shell interaction and capture-history state."""

from __future__ import annotations

from dataclasses import dataclass

from .analysis_pipeline import AnalysisPresentation, AnalysisSubject, evaluate_analysis, make_analysis_request
from .command_facade import CommandFacade, CommandId
from .domain import CapturedSurface, DetectionResult, GeometryMode, ScanFrame
from .scene import SceneObject


@dataclass(frozen=True)
class WorkflowSnapshot:
    geometry_type: GeometryMode
    preview_ready: bool
    selected: bool
    capture_ready: bool
    capture_count: int
    analysis: AnalysisPresentation | None
    message: str


class ShellWorkflow:
    """Legacy shell state bridge retaining its public API for existing adapters."""

    def __init__(self, facade: CommandFacade) -> None:
        self.facade = facade
        self._mode = GeometryMode.SPHERE
        self._frame: ScanFrame | None = None
        self._result: DetectionResult | None = None
        self._analysis: AnalysisPresentation | None = None
        self._selected = False
        self._capture_serial = 0
        self._message = "Choose a geometry and a point in the frozen cloud."

    @property
    def snapshot(self) -> WorkflowSnapshot:
        return WorkflowSnapshot(
            self._mode,
            self._result is not None,
            self._selected,
            self.capture_ready,
            len(self.facade.snapshot.captures),
            self._analysis,
            self._message,
        )

    @property
    def capture_ready(self) -> bool:
        return self._result is not None and self._frame is not None and self._selected

    def set_mode(self, mode: GeometryMode | str) -> None:
        mode = GeometryMode(mode)
        if mode is GeometryMode.AUTO:
            raise ValueError("choose one explicit geometry")
        command = {
            GeometryMode.PLANE: CommandId.MODE_PLANE,
            GeometryMode.SPHERE: CommandId.MODE_SPHERE,
            GeometryMode.CYLINDER: CommandId.MODE_CYLINDER,
            GeometryMode.CONE: CommandId.MODE_CONE,
            GeometryMode.TORUS: CommandId.MODE_TORUS,
        }[mode]
        self.facade.execute(command)
        self._mode = mode
        self._clear_interaction()
        self._message = f"{mode.value} selected. Choose a point."



    def capture(self):
        if not self.capture_ready:
            raise RuntimeError("preview and fixed-seed selection are required")
        self._capture_serial += 1
        surface = CapturedSurface.from_detection(
            f"capture-{self._capture_serial:06d}", self._result, self._frame
        )
        subject = AnalysisSubject.from_capture(surface)
        analysis = evaluate_analysis(
            make_analysis_request(subject, self.facade.snapshot.objects, scene_generation=0)
        )
        item = self.facade.add_capture(surface, analysis)
        self._analysis = analysis
        self._frame = None
        self._result = None
        self._selected = False
        self._message = f"Captured {item.capture_id}. Undo removes it; Redo restores the same immutable payload."
        return item

    async def undo(self) -> bool:
        changed = await self.facade.undo()
        self._sync_after_history("Undo")
        return changed

    async def redo(self) -> bool:
        changed = await self.facade.redo()
        self._sync_after_history("Redo")
        return changed

    async def reset_captures(self, transition=None) -> bool:
        changed = await self.facade.reset_captures(transition)
        self._message = "Reset complete: captured geometries cleared."
        return changed

    async def reset(self, transition=None) -> bool:
        """Compatibility entry point for the capture-only reset command."""
        return await self.reset_captures(transition)

    def close(self) -> None:
        self._clear_interaction()
        self.facade.close()

    def report_error(self, error: Exception) -> None:
        self._message = f"{type(error).__name__}: {error}"

    def invalidate(self, message: str = "Parameters changed. Choose a point again.") -> None:
        self._clear_interaction()
        self._message = str(message)

    def _designed(self) -> SceneObject:
        return next(item for item in self.facade.snapshot.objects if item.geometry_type is self._mode)

    def _clear_interaction(self) -> None:
        self._frame = None
        self._result = None
        self._analysis = None
        self._selected = False

    def _sync_after_history(self, verb: str) -> None:
        selected = self.facade.snapshot.selected_capture_id
        capture = next((item for item in self.facade.snapshot.captures if item.capture_id == selected), None)
        self._analysis = None if capture is None else capture.analysis
        self._message = f"{verb}: {self.facade.status.message}"


__all__ = ["ShellWorkflow", "WorkflowSnapshot"]
