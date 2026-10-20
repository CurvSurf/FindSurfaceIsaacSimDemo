"""OpenGLWidget construction and ordered asynchronous retirement."""

from __future__ import annotations

from dataclasses import dataclass

import carb.settings
import omni.kit.app
import omni.kit.widget.opengl as og


@dataclass(frozen=True)
class OpenGLRetirement:
    resources_released: bool
    widget_destroyed: bool
    window_destroyed: bool


def create_opengl_widget(**kwargs):
    """Use the Isaac Sim 6.0.1 Windows resize-safe resource-manager context."""

    kwargs.setdefault("context_type", og.OpenGLWidget.ContextType.RESOURCE_MANAGER_CONTEXT)
    return og.OpenGLWidget(**kwargs)


async def retire_opengl_view(
    scene,
    widget,
    window,
    *,
    release_updates: int = 60,
    retirement_updates: int = 8,
) -> OpenGLRetirement:
    """Delete GL objects while current, then retire widget and owning window."""

    app = omni.kit.app.get_app()
    settings = carb.settings.get_settings()
    renderer_wait_idle = settings.get_as_bool("/app/renderer/waitIdle")
    hydra_wait_idle = settings.get_as_bool("/app/hydraEngine/waitIdle")
    settings.set_bool("/app/renderer/waitIdle", True)
    settings.set_bool("/app/hydraEngine/waitIdle", True)
    widget_destroyed = False
    window_destroyed = False
    try:
        scene.close()
        for _ in range(max(1, int(release_updates))):
            await app.next_update_async()
            if scene.telemetry().resources_released:
                break
        resources_released = scene.telemetry().resources_released
        await app.next_update_async()
        widget.destroy()
        widget_destroyed = True
        await app.next_update_async()
        window.visible = False
        window.destroy()
        window_destroyed = True
        for _ in range(max(1, int(retirement_updates))):
            await app.next_update_async()
        return OpenGLRetirement(resources_released, widget_destroyed, window_destroyed)
    finally:
        settings.set_bool("/app/renderer/waitIdle", renderer_wait_idle)
        settings.set_bool("/app/hydraEngine/waitIdle", hydra_wait_idle)


__all__ = ["OpenGLRetirement", "create_opengl_widget", "retire_opengl_view"]
