"""Kit GLOBAL_EVENT_POST_QUIT bridge for the unified Safe Exit controller."""

from __future__ import annotations

from carb import eventdispatcher
import omni.kit.app

from .exit_guard import ExitTrigger, UnifiedExitController


class KitQuitGuard:
    def __init__(self, controller: UnifiedExitController) -> None:
        self.controller = controller
        self._allow_shutdown = False
        self._observer = eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_POST_QUIT,
            on_event=self._on_post_quit,
            observer_name="FindSurface unified async Safe Exit",
            order=-1000,
        )

    def request(self, trigger: ExitTrigger | str):
        return self.controller.request(trigger)

    def _on_post_quit(self, event) -> None:
        if self._allow_shutdown:
            return
        app = omni.kit.app.get_app()
        if not app.try_cancel_shutdown("FindSurface is releasing detector, sensor, GPU, stage, and UI resources."):
            return
        try:
            event.consume()
        except Exception:
            pass
        task = self.controller.request(ExitTrigger.APP_QUIT)
        task.add_done_callback(lambda _task: setattr(self, "_allow_shutdown", True))

    def allow_final_shutdown(self) -> None:
        self._allow_shutdown = True

    def close(self) -> None:
        self._observer = None


__all__ = ["KitQuitGuard"]
