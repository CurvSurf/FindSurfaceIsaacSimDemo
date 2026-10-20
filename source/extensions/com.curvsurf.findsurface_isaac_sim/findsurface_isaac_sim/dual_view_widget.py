"""Isaac Sim 6.0.1 host for two stable role-owned OpenGL views."""

from __future__ import annotations

from typing import Callable

import carb.input
import omni.appwindow
import omni.kit.app
import omni.ui as ui

from .camera import CameraRole, CameraSnapshot, with_viewport_aspect
from .dual_view import DualViewCompositor, ViewRect, ViewSlot, compose_role_packets, compute_dual_view_layout
from .opengl_layers_scene import LayeredOpenGLScene
from .opengl_widget import create_opengl_widget
from .render_layers import LayeredRenderPacket


class _RoleView:
    def __init__(self, role: CameraRole, widget, renderer: LayeredOpenGLScene) -> None:
        self.role = role
        self.widget = widget
        self.renderer = renderer


class DualOpenGLCompositorWidget:
    """Reparent two native widgets on swap; renderer and camera state stay role-owned."""

    def __init__(
        self,
        parent,
        *,
        margin_px: float = 16.0,
        pip_width_fraction: float = 0.30,
        pointer_fn: Callable[..., object] | None = None,
    ) -> None:
        self._margin_px = float(margin_px)
        self._pip_width_fraction = float(pip_width_fraction)
        self._role_aspects = {role: 16.0 / 9.0 for role in CameraRole}
        self._last_size = (0.0, 0.0)
        self._last_layout = None
        self._layout_apply_count = 0
        self._closed = False
        self._latest_packet = None
        self._latest_cameras = None
        self._presented_aspects = {}
        self._pointer_fn = pointer_fn
        self._layout_host = parent
        self._sensor_follows_scanner_camera = True
        self._point_depth_offset_m = 0.006
        self._pip_border = None
        self._pointer_captured = False
        self._mouse_over_primary = False
        self._last_polled_mouse = None
        self._input = None
        self._mouse = None
        self._ui_scale = 1.0
        self._init_mouse_polling()

        with parent:
            self._root = ui.ZStack(width=ui.Percent(100), height=ui.Percent(100), content_clipping=True)
            with self._root:
                ui.Rectangle(style={"background_color": 0xFF09111C})
                self._primary_placer, self._primary_frame, self._primary_input = self._build_slot(
                    ViewSlot.PRIMARY, border=False,
                )
                self._pip_placer, self._pip_frame, self._pip_input = self._build_slot(
                    ViewSlot.PIP, border=True,
                )
                # OpenGLWidget is a native interop surface.  Viewport chrome
                # built outside this compositor can be submitted before that
                # surface and disappear behind it even when omni.ui reports the
                # widgets visible.  Keep the overlay host as the final child of
                # the compositor stack so status, result, and modal chrome are
                # composited after both view slots.
                self._overlay_host = ui.Frame(
                    width=ui.Percent(100),
                    height=ui.Percent(100),
                    raster_policy=ui.RasterPolicy.NEVER,
                    horizontal_clipping=True,
                    vertical_clipping=True,
                )

        with self._primary_frame:
            scanner_widget = create_opengl_widget()
        scanner_renderer = LayeredOpenGLScene(scanner_widget)
        with self._pip_frame:
            scene_widget = create_opengl_widget()
        scene_renderer = LayeredOpenGLScene(scene_widget)
        self._views = {
            CameraRole.SCANNER: _RoleView(CameraRole.SCANNER, scanner_widget, scanner_renderer),
            CameraRole.SCENE: _RoleView(CameraRole.SCENE, scene_widget, scene_renderer),
        }
        self._apply_point_depth_presentation()
        self._compositor = DualViewCompositor(self._views)
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="FSIS dual OpenGL compositor resize")
        )

    @property
    def overlay_host(self):
        return self._overlay_host

    def _build_slot(self, slot: ViewSlot, *, border: bool):
        placer = ui.Placer(offset_x=0, offset_y=0, width=1, height=1)
        with placer:
            with ui.ZStack(width=ui.Percent(100), height=ui.Percent(100)):
                frame = ui.Frame(
                    width=ui.Percent(100),
                    height=ui.Percent(100),
                    raster_policy=ui.RasterPolicy.NEVER,
                    horizontal_clipping=True,
                    vertical_clipping=True,
                    style={"border_width": 2 if border else 0, "border_color": 0xFFE89B35},
                )
                input_surface = ui.Rectangle(
                    style={"background_color": 0x00000000},
                    opaque_for_mouse_events=True,
                    mouse_pressed_fn=lambda x, y, button, modifiers: self._route_pointer(
                        slot, "begin", x, y, button, modifiers,
                    ),
                    mouse_moved_fn=lambda x, y, *tail: self._route_pointer(
                        slot, "move", x, y, tail[-2] if len(tail) > 1 else 0,
                    ),
                    mouse_released_fn=lambda x, y, button, modifiers: self._route_pointer(
                        slot, "end", x, y, button, modifiers,
                    ),
                    mouse_wheel_fn=lambda x, y, modifiers: self._route_wheel(
                        slot, x, y, modifiers,
                    ),
                    mouse_hovered_fn=lambda hovered: self._route_pointer(
                        slot, "enter" if hovered else "leave", 0.0, 0.0, 0,
                    ),
                )
                if border:
                    # Draw after the render widget and transparent input surface
                    # so the PIP edge cannot be covered by either child.
                    self._pip_border = ui.Rectangle(
                        style={
                            "background_color": 0x00000000,
                            "border_width": 3,
                            "border_color": 0xFFE89B35,
                        },
                        opaque_for_mouse_events=False,
                    )
        return placer, frame, input_surface

    @property
    def pip_border_visible(self) -> bool:
        return self._pip_border is not None and bool(self._pip_placer.visible)

    def _route_pointer(
        self, slot: ViewSlot, action: str, x: float, y: float, button: int, modifiers: int = 0,
    ) -> None:
        # PIP is display-only by contract; its opaque surface prevents a PIP
        # gesture from leaking into the primary camera.
        if slot is not ViewSlot.PRIMARY or self._pointer_fn is None:
            return
        surface = self._primary_input
        if action == "begin":
            self._pointer_captured = True
        elif action == "end":
            self._pointer_captured = False
        elif action == "enter":
            self._mouse_over_primary = True
        elif action == "leave":
            self._mouse_over_primary = False
            self._last_polled_mouse = None
        if action == "move":
            self._mouse_over_primary = True
            self._last_polled_mouse = (float(x), float(y))
        local_x = float(x) - float(surface.screen_position_x)
        local_y = float(y) - float(surface.screen_position_y)
        self._pointer_fn(
            action,
            local_x,
            local_y,
            int(button),
            (float(surface.computed_width), float(surface.computed_height)),
            int(modifiers or 0),
        )

    def _route_wheel(
        self,
        slot: ViewSlot,
        delta_x: float,
        delta_y: float,
        modifiers: int,
    ) -> None:
        """Forward wheel deltas only from the interactive primary slot."""

        if slot is not ViewSlot.PRIMARY or self._pointer_fn is None:
            return
        surface = self._primary_input
        self._pointer_fn(
            "wheel",
            float(delta_x),
            float(delta_y),
            int(modifiers),
            (float(surface.computed_width), float(surface.computed_height)),
        )

    def _init_mouse_polling(self) -> None:
        """Mirror demo 1's polling fallback for Kit move-callback gaps."""

        try:
            app_window = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._mouse = app_window.get_mouse() if app_window else None
            self._ui_scale = max(1.0e-8, float(app_window.get_ui_scale())) if app_window else 1.0
        except Exception:
            self._input = None
            self._mouse = None
            self._ui_scale = 1.0

    def _poll_primary_pointer(self) -> None:
        if self._pointer_captured or self._pointer_fn is None:
            return
        if self._input is None or self._mouse is None:
            self._init_mouse_polling()
        if self._input is None or self._mouse is None:
            return
        try:
            x, y = self._input.get_mouse_coords_pixel(self._mouse)
            x, y = float(x) / self._ui_scale, float(y) / self._ui_scale
        except Exception:
            return
        surface = self._primary_input
        left, top = float(surface.screen_position_x), float(surface.screen_position_y)
        width, height = float(surface.computed_width), float(surface.computed_height)
        inside = left <= x < left + width and top <= y < top + height
        if inside:
            if self._last_polled_mouse != (x, y):
                self._route_pointer(ViewSlot.PRIMARY, "move", x, y, 0)
        elif self._mouse_over_primary:
            self._route_pointer(ViewSlot.PRIMARY, "leave", 0.0, 0.0, 0)

    @property
    def snapshot(self):
        return self._compositor.snapshot()

    def role_view(self, role: CameraRole) -> _RoleView:
        return self._views[CameraRole(role)]

    def presented_aspect(self, role: CameraRole) -> float | None:
        return self._presented_aspects.get(CameraRole(role))

    def logical_size_for_role(self, role: CameraRole) -> tuple[float, float]:
        slot = self._compositor.snapshot().slot_for(CameraRole(role))
        surface = self._primary_input if slot is ViewSlot.PRIMARY else self._pip_input
        return (
            max(1.0, float(surface.computed_width)),
            max(1.0, float(surface.computed_height)),
        )

    def rect_for_role(self, role: CameraRole):
        if self._last_layout is None:
            return None
        state = self._compositor.snapshot()
        if state.primary_role is role:
            return self._last_layout.primary
        return self._last_layout.pip if state.pip_visible else None

    def role_widget_ids(self) -> tuple[int, int]:
        return tuple(id(self._views[role].widget) for role in CameraRole)

    def submit(
        self,
        packet: LayeredRenderPacket,
        scanner_camera: CameraSnapshot,
        scene_camera: CameraSnapshot,
    ) -> None:
        self._role_aspects = {
            CameraRole.SCANNER: scanner_camera.intrinsics.aspect_ratio,
            CameraRole.SCENE: scene_camera.intrinsics.aspect_ratio,
        }
        self._latest_packet = packet
        self._latest_cameras = (scanner_camera, scene_camera)
        previous_layout = self._last_layout
        self.refresh_layout()
        if self._last_layout is previous_layout and previous_layout is not None:
            # Packet changes do not imply a viewport resize. Reapplying both
            # Placer rectangles here can refresh OpenGLWidget's interop texture
            # and make an unchanged camera appear to jump.
            self._present_latest(previous_layout)

    @property
    def layout_apply_count(self) -> int:
        return self._layout_apply_count

    def set_point_depth_presentation(
        self, *, sensor_follows_scanner_camera: bool, offset_m: float,
    ) -> None:
        offset_m = float(offset_m)
        if not 0.0 <= offset_m <= 1.0:
            raise ValueError("point depth offset must be between 0 and 1 metre")
        self._sensor_follows_scanner_camera = bool(sensor_follows_scanner_camera)
        self._point_depth_offset_m = offset_m
        self._apply_point_depth_presentation()

    def _apply_point_depth_presentation(self) -> None:
        for role, view in self._views.items():
            view.renderer.set_source_point_depth(
                depth_test_off=(
                    role is CameraRole.SCANNER and self._sensor_follows_scanner_camera
                ),
                offset_m=self._point_depth_offset_m,
            )

    def _present_latest(self, layout) -> None:
        if self._latest_packet is None or self._latest_cameras is None:
            return
        snapshot = self._compositor.snapshot()
        rect_by_role = {
            snapshot.primary_role: layout.primary,
            snapshot.pip_role: layout.pip,
        }
        cameras = {}
        for camera in self._latest_cameras:
            rect = rect_by_role[camera.role]
            aspect = max(1.0, rect.width) / max(1.0, rect.height)
            self._presented_aspects[camera.role] = aspect
            cameras[camera.role] = with_viewport_aspect(camera, aspect)
        packets = compose_role_packets(
            self._latest_packet,
            cameras[CameraRole.SCANNER],
            cameras[CameraRole.SCENE],
        )
        self._views[CameraRole.SCANNER].renderer.submit(packets[CameraRole.SCANNER])
        scene_view = self._views[CameraRole.SCENE]
        scene_view.renderer.submit(packets[CameraRole.SCENE])

    def swap(self):
        snapshot = self._compositor.swap()
        self._attach_roles_to_slots()
        self.refresh_layout(force=True)
        return snapshot

    def set_primary(self, role: CameraRole):
        before = self._compositor.snapshot()
        snapshot = self._compositor.set_primary(role)
        if snapshot is not before:
            self._attach_roles_to_slots()
            self.refresh_layout(force=True)
        return snapshot

    def reset(self):
        return self.set_primary(CameraRole.SCANNER)

    def set_pip_width_fraction(self, fraction: float) -> None:
        fraction = float(fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("PIP width fraction must be in (0, 1)")
        self._pip_width_fraction = fraction
        self.refresh_layout(force=True)

    def set_pip_visible(self, visible: bool):
        snapshot = self._compositor.set_pip_visible(bool(visible))
        self._attach_roles_to_slots()
        self.refresh_layout(force=True)
        return snapshot

    def _attach_roles_to_slots(self) -> None:
        primary = self._compositor.content_for(ViewSlot.PRIMARY)
        pip = self._compositor.content_for(ViewSlot.PIP)
        self._primary_frame.add_child(primary.widget)
        if pip is not None:
            self._pip_frame.add_child(pip.widget)

    def _on_update(self, _event) -> None:
        self.refresh_layout()
        self._poll_primary_pointer()

    def refresh_layout(self, *, force: bool = False) -> None:
        # Percent-sized children can retain a stale computed size for several
        # Kit frames after docking changes. The parent render host is the
        # authoritative live viewport extent.
        width = max(1.0, float(self._layout_host.computed_width))
        height = max(1.0, float(self._layout_host.computed_height))
        if not force and abs(width - self._last_size[0]) <= 0.5 and abs(height - self._last_size[1]) <= 0.5:
            return
        self._last_size = width, height
        snapshot = self._compositor.snapshot()
        layout = compute_dual_view_layout(
            width,
            height,
            primary_aspect=self._role_aspects[snapshot.primary_role],
            pip_aspect=self._role_aspects[snapshot.pip_role],
            pip_width_fraction=self._pip_width_fraction,
            margin_px=self._margin_px,
        )
        self._last_layout = layout
        self._layout_apply_count += 1
        # A stationary OS cursor has a new local coordinate after resize/dock.
        # Re-route it on the next poll so the hover seed circle stays centered.
        self._last_polled_mouse = None
        self._apply_rect(self._primary_placer, layout.primary)
        self._apply_rect(self._pip_placer, layout.pip)
        self._pip_placer.visible = snapshot.pip_visible
        self._set_role_size(snapshot.primary_role, layout.primary.width, layout.primary.height)
        self._set_role_size(snapshot.pip_role, layout.pip.width, layout.pip.height)
        self._present_latest(layout)

    def _set_role_size(self, role: CameraRole, width: float, height: float) -> None:
        view = self._views[role]
        view.renderer.set_logical_size(width, height)

    @staticmethod
    def _apply_rect(placer, rect: ViewRect) -> None:
        placer.offset_x = ui.Pixel(rect.x)
        placer.offset_y = ui.Pixel(rect.y)
        placer.width = ui.Pixel(rect.width)
        placer.height = ui.Pixel(rect.height)

    async def close(self, *, release_updates: int = 60) -> None:
        if self._closed:
            return
        self._closed = True
        self._update_subscription = None
        self._input = None
        self._mouse = None
        opengl_views = tuple(self._views.values())
        for view in opengl_views:
            view.renderer.close()
        app = omni.kit.app.get_app()
        for _ in range(max(1, int(release_updates))):
            await app.next_update_async()
            if all(view.renderer.telemetry().resources_released for view in opengl_views):
                break
        for view in reversed(opengl_views):
            view.widget.destroy()
        await app.next_update_async()


__all__ = ["DualOpenGLCompositorWidget"]
