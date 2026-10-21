"""Isaac Sim 6.0.1 classic image toolbar, property tabs, overlays, and dialogs."""

from __future__ import annotations

import asyncio
import os
from typing import Callable

import carb
import omni.kit.app
import omni.ui as ui
from findsurface.bootstrap.findsurface_window import FindSurfaceWindowCallbacks, build_findsurface_window
from findsurface.bootstrap.ui_views import BorderedGroupView

from .command_facade import CommandId
from .domain import GeometryMode
from .geometry_card import normalize_geometry_card_values
from .kit_input import kit_enum_name, kit_key_name
from .object_orientation import screen_roll_radians
from .parameter_mapping import slider_index_for_value
from .render_layers import RenderTheme, heatmap_color, style_for_theme
from .scene import MINIMUM_TORUS_SWEEP_DEG, constrain_geometry_parameter_edit
from .sensor_diagnostics import ANTI_ALIASING_LABELS
from .workspace_layout import (
    DEFAULT_PROPERTY_WIDTH, SIDE_PANEL_DOCK_RATIO, default_property_dock_ratio,
    fit_floating_bounds, fit_workspace_layout,
)
from .ui_shell import (
    ModalPhase,
    PerformanceWarningPresentation,
    PropertyTab,
    ShellSnapshot,
)


TOOLBAR_HEIGHT = 44
STATUS_BAR_HEIGHT = 28
RULER_TICK_HEIGHTS = (16, 12, 8)
RULER_TICK_WIDTHS = (2, 1, 1)
RULER_BOUNDARY_HEIGHT = 20
RULER_BOUNDARY_THICKNESS = 4
RULER_PIP_SCALE = 0.75
BUTTON_SIZE = 38
ICON_SIZE = 24
PROPERTY_WIDTH = DEFAULT_PROPERTY_WIDTH
RESULT_OVERLAY_MARGIN = 14
RESULT_OVERLAY_WIDTH = 330
RESULT_OVERLAY_ROW_HEIGHT = 18
RESULT_OVERLAY_LIVE_ROW_COUNT = 6
RESULT_OVERLAY_ANALYSIS_ROW_COUNT = 9
RESULT_OVERLAY_HISTOGRAM_WIDTH = 306
RESULT_OVERLAY_HISTOGRAM_HEIGHT = 66
RESULT_OVERLAY_BASE_HEIGHT = (
    8 + 28 + RESULT_OVERLAY_ANALYSIS_ROW_COUNT * RESULT_OVERLAY_ROW_HEIGHT
    + 5 + RESULT_OVERLAY_HISTOGRAM_HEIGHT + 8
)
HISTOGRAM_HOVER_HEIGHT = 18
RESULT_OVERLAY_HISTOGRAM_BAR_HEIGHT = 34
RESULT_OVERLAY_HISTOGRAM_SPACING = 2
GEOMETRY_CONTENT_GAP = 4
GEOMETRY_SECTION_PADDING = 6
GEOMETRY_ITEM_ROW_SPACING = 4
GEOMETRY_HEATMAP_ROW_HEIGHT = 28
GEOMETRY_FIELD_ROW_HEIGHT = 26
GEOMETRY_HISTOGRAM_TITLE_HEIGHT = 22
GEOMETRY_HISTOGRAM_TOP_GAP = 5
GEOMETRY_HEADER_HEIGHT = 28
GEOMETRY_HEADER_ICON_SIZE = 20
GEOMETRY_HEADER_ACTION_SIZE = 26
GEOMETRY_HEADER_ACTION_ICON_SIZE = 18
_GEOMETRY_ICON_BY_TYPE = {
    "Plane": "../tabler/perspective.svg",
    "Sphere": "../tabler/sphere.svg",
    "Cylinder": "../tabler/cylinder.svg",
    "Cone": "../tabler/cone.svg",
    "Torus": "detect_torus.svg",
}
_FS_RANGE_DEFAULTS = {
    "measurement_accuracy": (0.001, 0.1),
    "mean_distance": (0.01, 0.2),
    "seed_radius": (0.05, 2.0),
}
_FS_PAD_PARAMETERS = (
    "measurement_accuracy", "mean_distance", "seed_radius",
    "lateral_extension", "radial_expansion",
)
_FS_PAD_LABELS = {
    "measurement_accuracy": "Measurement Accuracy",
    "mean_distance": "Mean Distance",
    "seed_radius": "Seed Radius",
    "lateral_extension": "Lateral Extension",
    "radial_expansion": "Radial Expansion",
}
SURFACE_SIDE_SCALE_LABELS = (
    "Auto (full range)",
    "Fixed distance",
    "Sigma multiple",
)

_BUTTONS = (
    (CommandId.CAPTURE, "../tabler/capture.svg"),
    (CommandId.UNDO, "arrow-back-up.svg"),
    (CommandId.REDO, "../tabler/arrow-forward-up.svg"),
    (CommandId.RESET_CAPTURES, "restore.svg"),
    None,
    (CommandId.SENSOR_FROZEN, "../tabler/player-pause.svg"),
    (CommandId.SENSOR_LIVE_FIXED, "../tabler/radar.svg"),
    (CommandId.SENSOR_LIVE_FOLLOW, "../tabler/radar-2.svg"),
    None,
    (CommandId.PRIMARY_SCENE, "../tabler/photo.svg"),
    (CommandId.PRIMARY_SCANNER, "../tabler/photo-search.svg"),
    None,
    (CommandId.CAMERA_RESET_SCENE, "../tabler/focus-photo.svg"),
    (CommandId.CAMERA_SCENE_POS_X, "view_front.svg"),
    (CommandId.CAMERA_SCENE_POS_Y, "view_side.svg"),
    (CommandId.CAMERA_SCENE_POS_Z, "view_top.svg"),
    (CommandId.CAMERA_RESET_SCANNER, "../tabler/focus-photo-search.svg"),
    (CommandId.CAMERA_RESET_SCANNER_UP, "../tabler/rotate.svg"),
    None,
    (CommandId.MODE_INSPECT, "pointer.svg"),
    (CommandId.MODE_AUTO, "detect_auto.svg"),
    (CommandId.MODE_PLANE, "../tabler/perspective.svg"),
    (CommandId.MODE_SPHERE, "../tabler/sphere.svg"),
    (CommandId.MODE_CYLINDER, "../tabler/cylinder.svg"),
    (CommandId.MODE_CONE, "../tabler/cone.svg"),
    (CommandId.MODE_TORUS, "detect_torus.svg"),
    None,
    (CommandId.VISIBILITY_OUTLIER, "visibility_outliers_on.svg"),
    (CommandId.VISIBILITY_INLIER, "visibility_inliers_on.svg"),
    (CommandId.VISIBILITY_DESIGNED, "../tabler/cube.svg"),
    (CommandId.VISIBILITY_DETECTED, "../tabler/cube.svg"),
    (CommandId.VISIBILITY_CAPTURED, "../tabler/cube.svg"),
    None,
    (CommandId.HELP, "help.svg"),
    (CommandId.SAFE_EXIT, "../tabler/power.svg"),
)


def _set_widget_property(widget, name, value):
    if getattr(widget, name) != value:
        setattr(widget, name, value)


class _IconButton:
    def __init__(self, button, image, frame) -> None:
        self.button = button
        self.image = image
        self.frame = frame
        self._applied_state = None

    def set_state(self, *, enabled: bool, checked: bool | None, tooltip: str) -> None:
        state = (bool(enabled), bool(checked), str(tooltip))
        if state == self._applied_state:
            return
        self.button.enabled = bool(enabled)
        self.image.enabled = bool(enabled)
        # The frame remains enabled so unavailable actions retain the same border;
        # only their icon adopts the disabled grey style.
        self.frame.enabled = True
        self.button.tooltip = str(tooltip)
        self.frame.checked = bool(checked)
        self.button.checked = bool(checked)
        self._applied_state = state


class ClassicUiChrome:
    """Own non-render chrome while the compositor owns only ``render_host``."""

    def __init__(
        self,
        parent,
        initial: ShellSnapshot,
        command_fn: Callable[[CommandId], object],
        *,
        icon_directory: str | None = None,
        modal_dismiss_fn: Callable[[], object] | None = None,
        modal_cancel_fn: Callable[[], object] | None = None,
        property_fn: Callable[[str, object], object] | None = None,
        property_values: dict[str, object] | None = None,
        layout_preference: dict | None = None,
        defer_viewport_overlays: bool = False,
    ) -> None:
        self._command_fn = command_fn
        self._modal_dismiss_fn = modal_dismiss_fn
        self._modal_cancel_fn = modal_cancel_fn
        self._property_fn = property_fn or (lambda _action, _value=None: False)
        self._property_values = dict(normalize_geometry_card_values(property_values or {}))
        self._layout_preference = layout_preference
        self._syncing_properties = False
        self._pending_property_model_keys = set()
        self._icon_directory = icon_directory or default_icon_directory()
        self.buttons: dict[CommandId, _IconButton] = {}
        self.property_windows = {}
        self.property_labels = {}
        self.property_models = {}
        self.property_buttons = {}
        self._property_scrolling_frames = {}
        self._property_scroll_y = {}
        self._geometry_property_content_frame = None
        self._geometry_card_frames = {}
        self._geometry_card_controls = {}
        self._geometry_collapsed = {}
        self._geometry_hover_capture_id = None
        self._graphics_collapsible_frames = {}
        self._model_subscriptions = []
        self._range_sliders = {}
        self._pad_axes = {"x": "mean_distance", "y": "measurement_accuracy"}
        self._pad_frame = None
        self._pad_square = None
        self._pad_x_controls = None
        self._pad_column = None
        self._pad_layout_size = None
        self._pad_layout_subscription = None
        self._pad_refresh_task = None
        self._result_overlay_placer = None
        self._result_overlay_height = RESULT_OVERLAY_BASE_HEIGHT
        self._pad_surface = None
        self._pad_spacers = None
        self._pad_value_labels = {}
        self._pad_dragging = False
        self._object_pad_surfaces = {}
        self._object_pad_drag = None
        self._object_editing_keys = set()
        self._sensor_profile_entries = tuple(self._property_values.get("sensor.profile_entries", ()))
        self._sensor_profile_combo_model = None
        self._sensor_profile_description_labels = {}
        self._pending_sensor_profile_index: int | None = None
        self._sensor_profile_apply_inflight = False
        self._anti_aliasing_combo_model = None
        self._help_window = None
        self._modal_generation = -1
        self._performance_window = None
        self._docked = False
        self._dock_task = None
        self._geometry_refresh_task = None
        self._geometry_heatmap_collections = []
        self._dock_error = None
        self._dock_attempts = 0
        self._render_window = None
        self._result_overlay_state = None
        self._last_snapshot = initial
        self._viewport_overlays_mounted = False
        self.sensor_overlay = None
        self.rtx_overlay = None
        self.analysis_overlay = None
        self.result_overlay_frame = None
        self._modal_overlay = None

        with parent:
            with ui.ZStack(content_clipping=True) as self._viewport_overlay_root:
                self.render_host = ui.Frame(
                    width=ui.Percent(100), height=ui.Percent(100),
                    horizontal_clipping=True, vertical_clipping=True,
                )
                if not defer_viewport_overlays:
                    self._build_viewport_overlays()
                    self._viewport_overlays_mounted = True

        self.toolbar_window = ui.ToolBar(
            "FindSurface Toolbar", width=1280, height=TOOLBAR_HEIGHT, visible=False,
            axis=ui.ToolBarAxis.X,
            flags=_tool_flags(),
        )
        with self.toolbar_window.frame:
            self._build_toolbar(initial)
        self.status_window = ui.Window(
            "FindSurface Status", width=1280, height=STATUS_BAR_HEIGHT,
            visible=False,
            flags=(
                _tool_flags()
                | ui.WINDOW_FLAGS_NO_RESIZE
                | ui.WINDOW_FLAGS_NO_MOVE
                | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS
            ),
        )
        self.status_window.detachable = False
        with self.status_window.frame:
            with ui.HStack(height=STATUS_BAR_HEIGHT, spacing=0):
                ui.Spacer(width=6)
                self.status_label = ui.Label("", height=22)
                ui.Spacer(width=6)
        self._build_properties(initial)
        self._pad_layout_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                lambda _event: self._refresh_dynamic_layouts(),
                name="FSIS responsive chrome layout",
            )
        )
        self.refresh(initial)

    def mount_viewport_overlays(self, parent) -> bool:
        """Mount chrome after native OpenGL slots in their compositor stack."""

        if self._viewport_overlays_mounted:
            return False
        with parent:
            with ui.ZStack(
                width=ui.Percent(100), height=ui.Percent(100), content_clipping=True,
            ) as self._viewport_overlay_root:
                self._build_viewport_overlays()
        self._viewport_overlays_mounted = True
        self._result_overlay_state = None
        self.refresh(self._last_snapshot)
        return True

    def _build_viewport_overlays(self) -> None:
        with ui.Placer(offset_x=12, offset_y=12, width=410, height=54):
            self.sensor_overlay = ui.Label(
                "", word_wrap=True, opaque_for_mouse_events=False,
                style={"background_color": 0xCC10151C, "margin": 7},
            )
        with ui.Placer(offset_x=12, offset_y=72, width=410, height=54):
            self.rtx_overlay = ui.Label(
                "", word_wrap=True, opaque_for_mouse_events=False,
                style={"background_color": 0xCC10151C, "margin": 7},
            )
        # ``ui.Placer.offset_y`` is a pixel length in this Kit version; a
        # percentage there is not a bottom anchor and put this label over the
        # RTX status.  Use flexible stacks for the intended bottom-left slot.
        with ui.VStack():
            ui.Spacer()
            with ui.HStack(height=78 + RESULT_OVERLAY_MARGIN):
                ui.Spacer(width=12)
                self.analysis_overlay = ui.Label(
                    "", width=510, height=78, word_wrap=True,
                    opaque_for_mouse_events=False,
                    style={"background_color": 0xCC10151C, "margin": 7},
                )
                ui.Spacer()
            ui.Spacer(height=RESULT_OVERLAY_MARGIN)
        self._build_result_overlay()
        from .orthographic import MovableRuler
        self.scene_ruler = MovableRuler()
        self._ruler_placer = ui.Placer(width=196, height=self.scene_ruler.PANEL_HEIGHT,
                                      visible=False, opaque_for_mouse_events=False)
        with self._ruler_placer:
            self._ruler_panel = ui.ZStack(opaque_for_mouse_events=False)
            with self._ruler_panel:
                self._ruler_background = ui.Rectangle(
                    style={"background_color": 0xCC10151C, "border_radius": 6},
                    opaque_for_mouse_events=False)
                with ui.VStack(spacing=0, opaque_for_mouse_events=False):
                    ui.Spacer(height=self.scene_ruler.PADDING)
                    with ui.HStack(spacing=0, height=44, opaque_for_mouse_events=False):
                        ui.Spacer(width=self.scene_ruler.PADDING)
                        with ui.VStack(spacing=2, opaque_for_mouse_events=False):
                            with ui.HStack(height=22, spacing=0, opaque_for_mouse_events=False):
                                self._ruler_label = ui.Label("", width=0, opaque_for_mouse_events=False)
                                self._ruler_exponent = ui.Label("", width=0, height=12,
                                    alignment=ui.Alignment.LEFT_TOP, style={"font_size": 10},
                                    opaque_for_mouse_events=False)
                                ui.Label(" m", width=0, opaque_for_mouse_events=False)
                                ui.Spacer(opaque_for_mouse_events=False)
                            self._ruler_ticks = ui.Frame(height=20, opaque_for_mouse_events=False)
                        ui.Spacer(width=self.scene_ruler.PADDING)
                    ui.Spacer(height=self.scene_ruler.PADDING)
                self._ruler_handles = ui.HStack(opaque_for_mouse_events=False)
                with self._ruler_handles:
                    ui.Rectangle(width=3, style={"background_color": 0x88E8EBEF}, opaque_for_mouse_events=False)
                    ui.Spacer()
                    ui.Rectangle(width=3, style={"background_color": 0x88E8EBEF}, opaque_for_mouse_events=False)
                ui.Rectangle(style={"background_color": 0x00000000}, opaque_for_mouse_events=True,
                    tooltip="Drag the body to move; drag either end to resize (main view only)",
                    mouse_pressed_fn=lambda x, y, b, m: self._ruler_pointer("begin", x, y, b),
                    mouse_moved_fn=lambda x, y, *args: self._ruler_pointer("move", x, y, 0),
                    mouse_released_fn=lambda x, y, b, m: self._ruler_pointer("end", x, y, b))
        self._build_modal_overlay()

    def sync_scene_ruler(self, camera, rect, *, primary=False):
        from .orthographic import ruler_length_m, ruler_scale
        height = camera.intrinsics.orthographic_height_m
        visible = height is not None and rect is not None
        self._ruler_context = (camera, rect, primary)
        if not primary:
            self.scene_ruler.cancel_drag()
        self._ruler_placer.visible = visible
        self.analysis_overlay.visible = not (visible and primary)
        if not visible:
            self.scene_ruler.cancel_drag()
            return
        x, y, span = self.scene_ruler.layout((rect.width, rect.height), primary=primary)
        opacity = float(self._property_values.get(
            "graphics.ruler_main_opacity" if primary else "graphics.ruler_pip_opacity",
            0.8 if primary else 0.35))
        signature = (rect.x + x, rect.y + y, span, height, rect.height, primary, opacity)
        if signature != getattr(self, "_ruler_signature", None):
            self._ruler_placer.offset_x = rect.x + x
            self._ruler_placer.offset_y = rect.y + y
            self._ruler_placer.width = ui.Pixel(span + 2 * self.scene_ruler.PADDING)
            tick_signature = (height, rect.height, span, primary)
            if tick_signature != getattr(self, "_ruler_tick_signature", None):
                mark_scale = 1.0 if primary else RULER_PIP_SCALE
                boundary_height = RULER_BOUNDARY_HEIGHT * mark_scale
                boundary_thickness = RULER_BOUNDARY_THICKNESS * mark_scale
                scale = ruler_scale(ruler_length_m(height, rect.height, span), span)
                self._ruler_label.text = scale.significand + (" × 10" if scale.exponent else "")
                self._ruler_exponent.text = str(scale.exponent)
                self._ruler_exponent.visible = scale.exponent != 0
                self._ruler_ticks.clear()
                with self._ruler_ticks:
                    with ui.ZStack(opaque_for_mouse_events=False):
                        with ui.Placer(offset_y=boundary_height - boundary_thickness,
                                       height=boundary_thickness, opaque_for_mouse_events=False):
                            ui.Rectangle(style={"background_color": 0xFFE8EBEF}, opaque_for_mouse_events=False)
                        for tick in scale.ticks:
                            tick_height = RULER_TICK_HEIGHTS[tick.level] * mark_scale
                            tick_width = max(1.0, RULER_TICK_WIDTHS[tick.level] * mark_scale)
                            with ui.Placer(offset_x=tick.offset_px - tick_width / 2,
                                           offset_y=boundary_height - tick_height,
                                           width=tick_width, height=tick_height,
                                           opaque_for_mouse_events=False):
                                ui.Rectangle(style={"background_color": 0xFFE8EBEF},
                                             opaque_for_mouse_events=False)
                        for offset in (0.0, max(0.0, span - boundary_thickness)):
                            with ui.Placer(offset_x=offset, width=boundary_thickness,
                                           height=boundary_height,
                                           opaque_for_mouse_events=False):
                                ui.Rectangle(style={"background_color": 0xFFE8EBEF},
                                             opaque_for_mouse_events=False)
                self._ruler_tick_signature = tick_signature
            self._ruler_background.style = {"background_color": (round(opacity * 255) << 24) | 0x10151C,
                                            "border_radius": 6}
            self._ruler_handles.visible = primary
            self._ruler_signature = signature

    def _ruler_pointer(self, action, x, y, button):
        camera, rect, primary = self._ruler_context
        if self._last_snapshot.modal.input_blocked or not primary or rect is None or not self._ruler_placer.visible:
            self.scene_ruler.cancel_drag()
            return
        # Never subtract the moving ruler's computed screen position: Kit layout
        # may still reflect the previous frame while the model has already moved.
        # Demo 1 also maps events against the stationary viewport, not the ruler.
        local_x = x - self._viewport_overlay_root.screen_position_x - rect.x
        local_y = y - self._viewport_overlay_root.screen_position_y - rect.y
        self.scene_ruler.pointer(action, local_x, local_y, button)
        self.sync_scene_ruler(camera, rect, primary=primary)

    def _build_modal_overlay(self) -> None:
        """Build a lightweight in-view modal instead of a free-floating Kit window."""
        self._modal_overlay = ui.ZStack(visible=False)
        with self._modal_overlay:
            ui.Rectangle(opaque_for_mouse_events=True, style={"background_color": 0x66000000})
            with ui.VStack(spacing=0):
                ui.Spacer()
                with ui.HStack(height=190):
                    ui.Spacer()
                    with ui.ZStack(width=480):
                        ui.Rectangle(style={
                            "background_color": 0xF21B2028,
                            "border_color": 0xFF626A76,
                            "border_width": 1,
                            "border_radius": 6,
                        })
                        with ui.VStack(spacing=9, style={"margin": 16}):
                            self._modal_title = ui.Label("", height=28, style={"font_size": 18})
                            self._modal_detail = ui.Label("", word_wrap=True, height=44)
                            self._modal_progress_model = ui.SimpleFloatModel(0.0)
                            self._modal_progress = ui.ProgressBar(self._modal_progress_model, height=16)
                            self._modal_error = ui.Label(
                                "", word_wrap=True, height=34, style={"color": 0xFF7F8FFF},
                            )
                            with ui.HStack(height=30, spacing=8):
                                ui.Spacer()
                                self._modal_cancel = ui.Button(
                                    "Cancel", width=86, clicked_fn=self._cancel_modal,
                                )
                                self._modal_dismiss = ui.Button(
                                    "Close", width=86, clicked_fn=self._dismiss_modal,
                                )
                    ui.Spacer()
                ui.Spacer()

    def _build_result_overlay(self) -> None:
        """Anchor demo 1's successful-detection card to the viewport bottom-right."""

        self._result_overlay_header_icon = None
        self._result_overlay_value_labels = []
        self._result_overlay_placer = ui.Placer(
            offset_x=RESULT_OVERLAY_MARGIN,
            offset_y=RESULT_OVERLAY_MARGIN,
            width=RESULT_OVERLAY_WIDTH,
            height=RESULT_OVERLAY_BASE_HEIGHT,
            visible=False,
        )
        with self._result_overlay_placer:
            self.result_overlay_frame = ui.Frame(
                width=RESULT_OVERLAY_WIDTH,
                height=RESULT_OVERLAY_BASE_HEIGHT,
            )

    def _refresh_dynamic_layouts(self) -> None:
        self._refresh_parameter_pad_layout()
        self._refresh_result_overlay_layout()
        self._enforce_status_bar_height()

    def _refresh_result_overlay_layout(self) -> None:
        placer = self._result_overlay_placer
        root = self._viewport_overlay_root
        if placer is None or root is None:
            return
        width = float(root.computed_width)
        height = float(root.computed_height)
        if width <= 1.0 or height <= 1.0:
            return
        overlay_height = float(self._result_overlay_height)
        placer.offset_x = ui.Pixel(max(0.0, width - RESULT_OVERLAY_WIDTH - RESULT_OVERLAY_MARGIN))
        placer.offset_y = ui.Pixel(max(0.0, height - overlay_height - RESULT_OVERLAY_MARGIN))
        placer.width = ui.Pixel(min(RESULT_OVERLAY_WIDTH, width))
        placer.height = ui.Pixel(min(overlay_height, height))

    def _enforce_status_bar_height(self) -> None:
        window = self.status_window
        if window is None or not bool(getattr(window, "docked", False)):
            return
        try:
            current = float(window.frame.computed_height)
        except (AttributeError, TypeError, ValueError):
            current = -1.0
        if abs(current - STATUS_BAR_HEIGHT) > 0.5:
            # NO_RESIZE prevents ordinary splitter interaction. Keep this
            # fallback for workspace layout restoration and DPI transitions.
            window.height = ui.Pixel(STATUS_BAR_HEIGHT)

    def _refresh_result_overlay(self, state) -> None:
        if self.result_overlay_frame is None or self.analysis_overlay is None:
            return
        if state == self._result_overlay_state:
            return
        previous = self._result_overlay_state
        self._result_overlay_state = state
        self._result_overlay_placer.visible = bool(state.visible)
        self.analysis_overlay.visible = not state.visible
        if not state.visible:
            return
        height = _result_overlay_content_height(
            reserve_histogram=state.histogram_reserved,
        )
        self._result_overlay_height = height
        self.result_overlay_frame.height = ui.Pixel(height)
        self._refresh_result_overlay_layout()
        if self._result_overlay_header_icon is None:
            with self.result_overlay_frame:
                self._build_result_overlay_body(state, height)
            return
        if self._result_overlay_icon_type != state.result_type:
            self._result_overlay_header_icon.source_url = os.path.join(
                self._icon_directory, f"detect_{state.result_type.value.lower()}.svg",
            )
            self._result_overlay_icon_type = state.result_type
        if self._result_overlay_title.text != state.title:
            self._result_overlay_title.text = state.title
        if (
            previous is None
            or previous.histogram_reserved != state.histogram_reserved
            or tuple(label for label, _ in previous.rows) != tuple(label for label, _ in state.rows)
        ):
            self._result_overlay_rows_frame.clear()
            with self._result_overlay_rows_frame:
                self._build_result_overlay_rows(state)
        else:
            for widget, (_, value) in zip(self._result_overlay_value_labels, state.rows):
                if widget.text != str(value):
                    widget.text = str(value)
        self._result_overlay_rows_frame.height = ui.Pixel(
            (RESULT_OVERLAY_ANALYSIS_ROW_COUNT if state.histogram_reserved else RESULT_OVERLAY_LIVE_ROW_COUNT)
            * RESULT_OVERLAY_ROW_HEIGHT,
        )
        self._result_overlay_histogram_frame.visible = state.histogram_reserved
        self._result_overlay_histogram_frame.height = ui.Pixel(
            5 + RESULT_OVERLAY_HISTOGRAM_HEIGHT if state.histogram_reserved else 0,
        )
        if previous is None or previous.histogram != state.histogram:
            self._result_overlay_histogram_frame.clear()
            with self._result_overlay_histogram_frame:
                self._build_result_overlay_histogram(state)

    def _build_result_overlay_body(self, state, height: int) -> None:
        panel = {
            "background_color": 0x991B1E24,
            "border_color": 0x88FFFFFF,
            "border_width": 1,
            "border_radius": 5,
        }
        title_style = {"color": 0xF8FFFFFF, "font_size": 16}
        with ui.ZStack(width=RESULT_OVERLAY_WIDTH, height=ui.Percent(100)):
            ui.Rectangle(style=panel)
            with ui.VStack(width=RESULT_OVERLAY_WIDTH, height=ui.Percent(100), spacing=0):
                ui.Spacer(height=8)
                with ui.HStack(height=28, spacing=8):
                    ui.Spacer(width=10)
                    icon = os.path.join(
                        self._icon_directory,
                        f"detect_{state.result_type.value.lower()}.svg",
                    )
                    self._result_overlay_header_icon = ui.Image(icon, width=ICON_SIZE, height=ICON_SIZE)
                    self._result_overlay_icon_type = state.result_type
                    self._result_overlay_title = ui.Label(state.title, height=24, style=title_style)
                    ui.Spacer()
                self._result_overlay_rows_frame = ui.Frame(height=(
                    RESULT_OVERLAY_ANALYSIS_ROW_COUNT if state.histogram_reserved else RESULT_OVERLAY_LIVE_ROW_COUNT
                ) * RESULT_OVERLAY_ROW_HEIGHT)
                with self._result_overlay_rows_frame:
                    self._build_result_overlay_rows(state)
                self._result_overlay_histogram_frame = ui.Frame(
                    height=5 + RESULT_OVERLAY_HISTOGRAM_HEIGHT if state.histogram_reserved else 0,
                    visible=state.histogram_reserved,
                )
                with self._result_overlay_histogram_frame:
                    self._build_result_overlay_histogram(state)
                ui.Spacer()

    def _build_result_overlay_rows(self, state) -> None:
        label_style = {"color": 0xE8FFFFFF, "font_size": 13}
        dim_style = {"color": 0xB8FFFFFF, "font_size": 12}
        self._result_overlay_value_labels = []
        row_slots = RESULT_OVERLAY_ANALYSIS_ROW_COUNT if state.histogram_reserved else RESULT_OVERLAY_LIVE_ROW_COUNT
        with ui.VStack(height=row_slots * RESULT_OVERLAY_ROW_HEIGHT, spacing=0):
            for label, value in state.rows:
                with ui.HStack(height=RESULT_OVERLAY_ROW_HEIGHT, spacing=6):
                    ui.Spacer(width=12)
                    ui.Label(str(label), width=82, style=dim_style)
                    self._result_overlay_value_labels.append(
                        ui.Label(str(value), width=RESULT_OVERLAY_WIDTH - 116, style=label_style),
                    )
            if len(state.rows) < row_slots:
                ui.Spacer(height=(row_slots - len(state.rows)) * RESULT_OVERLAY_ROW_HEIGHT)

    def _build_result_overlay_histogram(self, state) -> None:
        if state.histogram is not None:
            with ui.VStack(spacing=0):
                ui.Spacer(height=5)
                self._build_result_histogram(state.histogram)

    def _build_result_histogram(self, histogram, *, hover_key=None) -> None:
        with ui.HStack(height=RESULT_OVERLAY_HISTOGRAM_HEIGHT, spacing=6):
            with ui.VStack(width=ui.Fraction(1), spacing=0):
                self._build_histogram_chart(histogram, hover_key=hover_key)
            with ui.VStack(width=158, spacing=0):
                for label, value in (
                    ("Samples", f"{histogram.used_count:,}/{histogram.source_count:,}"),
                    ("Mean", f"{histogram.mean_m:.4f} m"),
                    ("Std.dev", f"{histogram.sigma_m:.4f} m"),
                ):
                    with ui.HStack(height=18, spacing=6):
                        ui.Label(label, width=54)
                        ui.Label(value, width=ui.Fraction(1))

    def _build_histogram_chart(self, histogram, *, hover_key=None) -> None:
        bars = tuple(histogram.bars)
        if not bars:
            return
        chart_width = ui.Percent(100)
        chart_height = RESULT_OVERLAY_HISTOGRAM_HEIGHT - HISTOGRAM_HOVER_HEIGHT
        hover_label = []
        active_bin = [None]

        def set_hover(index, hovered):
            if hovered:
                active_bin[0] = index
            elif active_bin[0] == index:
                active_bin[0] = None
            else:
                # Kit can deliver new-bin enter before old-bin leave. A late
                # leave no longer owns hover and must not clear the new bin.
                return
            self._set_histogram_hover(hover_key, active_bin[0] is not None)
            if hover_label:
                selected = active_bin[0]
                value = "" if selected is None else _histogram_hover_text(
                    bars[selected].count, histogram.source_count,
                )
                if hover_label[0].text != value:
                    hover_label[0].text = value
        spacing = 0
        bar_width = ui.Fraction(1)
        axis_y = RESULT_OVERLAY_HISTOGRAM_BAR_HEIGHT + 5
        axis_color = 0x99FFFFFF
        with ui.HStack(height=chart_height, spacing=0):
            ui.Spacer(width=12)
            with ui.ZStack(width=ui.Fraction(1), height=chart_height):
                ui.Rectangle(style={
                    "background_color": 0x331B1E24,
                    "border_color": 0x66FFFFFF,
                    "border_width": 1,
                    "border_radius": 3,
                })
                with ui.VStack(width=chart_width, height=chart_height, spacing=0):
                    ui.Spacer(height=axis_y)
                    ui.Rectangle(width=chart_width, height=1, style={"background_color": axis_color})
                    ui.Spacer()
                zero_offset = ui.Percent(100 * max(0.0, min(1.0, histogram.zero_fraction)))
                with ui.HStack(width=chart_width, height=chart_height, spacing=0):
                    ui.Spacer(width=zero_offset)
                    ui.Rectangle(width=1, height=axis_y + 3, style={"background_color": axis_color})
                    ui.Spacer()
                with ui.HStack(width=chart_width, height=chart_height, spacing=spacing):
                    for bar in bars:
                        bar_height = (
                            max(2, int(RESULT_OVERLAY_HISTOGRAM_BAR_HEIGHT * bar.height_fraction))
                            if bar.count else 0
                        )
                        with ui.VStack(width=bar_width, height=chart_height, spacing=0):
                            ui.Spacer(height=RESULT_OVERLAY_HISTOGRAM_BAR_HEIGHT - bar_height + 4)
                            if bar_height:
                                ui.Rectangle(
                                    width=ui.Percent(100),
                                    height=bar_height,
                                    style={
                                        "background_color": _heatmap_ui_color(
                                            bar.heat_coordinate,
                                        )
                                    },
                                )
                            ui.Spacer()
                with ui.VStack(width=chart_width, height=chart_height, spacing=0):
                    ui.Spacer(height=axis_y + 2)
                    with ui.HStack(width=chart_width, height=6, spacing=0):
                        marker_width = 8
                        left = ui.Percent(100 * max(0.0, min(1.0, histogram.mean_fraction)))
                        ui.Spacer(width=left)
                        with ui.VStack(width=marker_width, height=6, spacing=0):
                            for width in (2, 4, 6, 8):
                                with ui.HStack(width=marker_width, height=1, spacing=0):
                                    ui.Spacer(width=(marker_width - width) * 0.5)
                                    ui.Rectangle(width=width, height=1, style={"background_color": axis_color})
                                    ui.Spacer()
                        ui.Spacer()
                    ui.Spacer()
                # Full-height columns also allow hovering empty/small bins.
                # Only the fixed footer label changes, never the window tree.
                with ui.HStack(width=chart_width, height=chart_height, spacing=spacing):
                    for index in range(len(bars)):
                        ui.Rectangle(
                            width=bar_width, height=chart_height,
                            style={"background_color": 0x0, "border_width": 0},
                            mouse_hovered_fn=lambda hovered, i=index: set_hover(i, hovered),
                        )
            ui.Spacer(width=12)
        with ui.HStack(height=HISTOGRAM_HOVER_HEIGHT, spacing=0):
            ui.Spacer(width=12)
            hover_label.append(ui.Label("", width=ui.Fraction(1), height=HISTOGRAM_HOVER_HEIGHT))
            ui.Spacer(width=12)

    def _build_toolbar(self, snapshot: ShellSnapshot) -> None:
        by_id = {item.command_id: item for item in snapshot.commands}
        self._toolbar_stack = ui.HStack(
            height=0, spacing=2, style=_toolbar_style(),
            style_type_name_override="ClassicToolbar",
        )
        with self._toolbar_stack:
            ui.Spacer(width=3)
            for descriptor in _BUTTONS:
                if descriptor is None:
                    ui.Spacer(width=3)
                    ui.Rectangle(width=1, height=BUTTON_SIZE, style={"background_color": 0xFF555A63})
                    ui.Spacer(width=3)
                    continue
                command_id, icon_name = descriptor
                item = by_id[command_id]
                self.buttons[command_id] = self._image_button(
                    icon_name,
                    f"{item.label}{' (' + item.shortcut + ')' if item.shortcut else ''}",
                    lambda value=command_id: self._dispatch(value),
                    icon_style={
                        CommandId.VISIBILITY_DESIGNED: "ClassicToolbar.DesignedIcon",
                        CommandId.VISIBILITY_DETECTED: "ClassicToolbar.DetectedIcon",
                        CommandId.VISIBILITY_CAPTURED: "ClassicToolbar.CapturedIcon",
                    }.get(command_id, "ClassicToolbar.Icon"),
                )
            ui.Spacer()

    def _image_button(self, icon_name: str, tooltip: str, clicked_fn, *, icon_style="ClassicToolbar.Icon") -> _IconButton:
        offset = (BUTTON_SIZE - ICON_SIZE) / 2
        with ui.ZStack(width=BUTTON_SIZE, height=BUTTON_SIZE, content_clipping=True):
            frame = ui.Rectangle(
                width=BUTTON_SIZE, height=BUTTON_SIZE,
                style_type_name_override="ClassicToolbar.Frame",
            )
            with ui.VStack(width=BUTTON_SIZE, height=BUTTON_SIZE):
                ui.Spacer(height=offset)
                with ui.HStack(height=ICON_SIZE):
                    ui.Spacer(width=offset)
                    image = ui.Image(
                        os.path.join(self._icon_directory, icon_name),
                        width=ICON_SIZE, height=ICON_SIZE,
                        style_type_name_override=icon_style,
                    )
                    ui.Spacer()
                ui.Spacer()
            button = ui.Button(
                " ", width=BUTTON_SIZE, height=BUTTON_SIZE, tooltip=tooltip,
                clicked_fn=clicked_fn, opaque_for_mouse_events=True,
                style_type_name_override="ClassicToolbar.Hit",
            )
        return _IconButton(button, image, frame)

    def _build_properties(self, snapshot: ShellSnapshot) -> None:
        for section in snapshot.properties:
            window = ui.Window(section.tab.value, width=PROPERTY_WIDTH, height=680, visible=False)
            self.property_windows[section.tab] = window
            with window.frame:
                if section.tab is PropertyTab.FINDSURFACE:
                    # Match demo 1: this fixed panel is placed directly in its
                    # frame, without a duplicate title or generic outer margin.
                    with ui.VStack(spacing=4):
                        self._startup_reset_button("findsurface")
                        self._build_findsurface_property(section)
                elif section.tab is PropertyTab.SETTINGS:
                    # Demo 1 builds Settings directly in the window frame: no
                    # duplicate title, status-summary rows, or generic 10 px
                    # property margin before the compact controls.
                    scrolling_frame = ui.ScrollingFrame()
                    self._property_scrolling_frames[section.tab] = scrolling_frame
                    with scrolling_frame:
                        with ui.VStack(spacing=6):
                            self._build_settings_property(section)
                elif section.tab is PropertyTab.GRAPHICS:
                    scrolling_frame = ui.ScrollingFrame()
                    self._property_scrolling_frames[section.tab] = scrolling_frame
                    with scrolling_frame:
                        with ui.VStack(height=0, spacing=6):
                            self._build_graphics_property(section)
                elif section.tab is PropertyTab.OBJECTS:
                    # Objects uses the same compact, directly framed visual
                    # language as demo 1's Settings window.
                    scrolling_frame = ui.ScrollingFrame()
                    self._property_scrolling_frames[section.tab] = scrolling_frame
                    with scrolling_frame:
                        with ui.VStack(spacing=6):
                            self._build_object_property()
                else:
                    scrolling_frame = ui.ScrollingFrame()
                    self._property_scrolling_frames[section.tab] = scrolling_frame
                    with scrolling_frame:
                        if section.tab is PropertyTab.GEOMETRIES:
                            # Keep the scrolling widget alive across content updates. Replacing
                            # it makes Kit visibly jump to y=0 before the saved offset is restored.
                            self._geometry_property_content_frame = ui.Frame(height=0)
                            with self._geometry_property_content_frame:
                                with ui.VStack(height=0, spacing=6):
                                    self._build_geometry_property(section)
                        else:
                            with ui.VStack(spacing=6):
                                if section.tab is PropertyTab.CAMERA:
                                    self._build_camera_property()
                                elif section.tab is PropertyTab.SCANNER:
                                    self._build_scanner_property(section)

    def _build_findsurface_property(self, section) -> None:
        for name, (minimum, maximum) in _FS_RANGE_DEFAULTS.items():
            minimum, maximum = self._property_values.get(f"{name}_bounds", (minimum, maximum))
            value = float(self._property_values.get(name, minimum))
            self.property_models[name] = ui.SimpleFloatModel(value)
            self.property_models[f"{name}_min"] = ui.SimpleFloatModel(minimum)
            self.property_models[f"{name}_max"] = ui.SimpleFloatModel(maximum)
            self.property_models[f"{name}_slider_index"] = ui.SimpleIntModel(
                self._range_index(value, minimum, maximum)
            )
            self._subscribe_callback(
                self.property_models[name], lambda model, current=name: self._on_range_value(current, model.as_float),
            )
            self._subscribe_callback(
                self.property_models[f"{name}_min"], lambda _model, current=name: self._on_range_bounds(current),
            )
            self._subscribe_callback(
                self.property_models[f"{name}_max"], lambda _model, current=name: self._on_range_bounds(current),
            )
            self._subscribe_callback(
                self.property_models[f"{name}_slider_index"],
                lambda model, current=name: self._on_range_slider(current, model.as_int),
            )
        for name in ("lateral_extension", "radial_expansion"):
            self.property_models[name] = ui.SimpleIntModel(int(self._property_values.get(name, 5)))
            self._subscribe_callback(
                self.property_models[name],
                lambda model, current=name: self._on_pad_integer_value(current, model.as_int),
            )
        callbacks = FindSurfaceWindowCallbacks(
            self._fs_range_float_row,
            self._fs_int_slider_row,
            self._build_parameter_pad_row_container,
        )
        build_findsurface_window(self.property_models, callbacks)

    def _fs_range_float_row(self, label, name, framed=True) -> None:
        with ui.VStack(height=76, spacing=4):
            with ui.HStack(height=24, spacing=6):
                ui.Label(f"{label.replace(chr(10), ' ')}:", width=124, height=20)
                value_field = getattr(ui, "FloatField", ui.FloatDrag)
                value_field(
                    model=self.property_models[name], step=0.0001,
                    format="%.4f", precision=4, width=76,
                )
            with ui.HStack(height=16, spacing=6):
                ui.Label("min", width=76, height=16, style={"font_size": 10})
                ui.Spacer()
                ui.Label("max", width=76, height=16, style={"font_size": 10})
            with ui.HStack(height=28, spacing=6):
                ui.FloatDrag(
                    model=self.property_models[f"{name}_min"], min=0.0001, max=1.0e12,
                    step=0.0001, format="%.4f", precision=4, width=76,
                )
                slider = ui.IntSlider(
                    model=self.property_models[f"{name}_slider_index"], min=0, max=10, step=1, format="",
                )
                self._range_sliders[name] = slider
                ui.FloatDrag(
                    model=self.property_models[f"{name}_max"], min=0.0001, max=1.0e12,
                    step=0.0001, format="%.4f", precision=4, width=76,
                )

    def _fs_int_slider_row(self, label, model, minimum=0, maximum=10, name=None, framed=True) -> None:
        with ui.HStack(height=30, spacing=8):
            ui.Label(label, width=130)
            ui.IntSlider(model=model, min=minimum, max=maximum, step=1)

    def _build_parameter_pad_row_container(self) -> None:
        # Upstream appends a flexible trailing Spacer. Give the final compound
        # control effectively all remaining height while keeping the adapter thin.
        self._pad_frame = ui.Frame(
            height=ui.Fraction(1000), horizontal_clipping=True, vertical_clipping=True,
        )
        with self._pad_frame:
            self._build_parameter_pad_row()

    def _build_parameter_pad_row(self) -> None:
        # Demo 1 frames this compound control as one compact parameter group.
        with ui.ZStack(height=ui.Percent(100)):
            ui.Rectangle(style={
                "background_color": 0x00000000,
                "border_color": 0xFFE6E6E6,
                "border_width": 1,
                "border_radius": 4,
            })
            with ui.VStack(height=ui.Percent(100), spacing=0):
                ui.Spacer(height=8)
                with ui.HStack(height=ui.Fraction(1), spacing=0):
                    ui.Spacer(width=8)
                    with ui.Frame():
                        self._build_parameter_pad_content()
                    ui.Spacer(width=8)
                ui.Spacer(height=8)

    def _build_parameter_pad_content(self) -> None:
        x_name, y_name = self._pad_axes["x"], self._pad_axes["y"]
        x_value, y_value = self._pad_normalized(x_name), self._pad_normalized(y_name)
        size, handle = float(self._pad_layout_size or 156.0), 12.0
        with ui.HStack(height=ui.Percent(100), spacing=8):
            ui.Spacer()
            with ui.VStack(width=148, spacing=0):
                with ui.VStack(height=48, spacing=6):
                    self._build_pad_axis_selector("y", width=148)
                    self._pad_value_labels["y"] = ui.Label(self._pad_value_text(y_name), height=18)
                ui.Spacer()
            with ui.VStack(width=size, height=ui.Percent(100), spacing=0) as self._pad_column:
                with ui.ZStack(width=size, height=size) as self._pad_square:
                    ui.Rectangle(style={
                        "background_color": 0x331B1E24, "border_color": 0xFF8C9098,
                        "border_width": 1, "border_radius": 4,
                    })
                    with ui.HStack(width=ui.Percent(100), height=ui.Percent(100), spacing=0):
                        ui.Spacer(width=ui.Percent(50))
                        ui.Rectangle(width=1, style={"background_color": 0x448C9098})
                        ui.Spacer()
                    with ui.VStack(width=ui.Percent(100), height=ui.Percent(100), spacing=0):
                        ui.Spacer(height=ui.Percent(50))
                        ui.Rectangle(height=1, style={"background_color": 0x448C9098})
                        ui.Spacer()
                    with ui.VStack(width=ui.Percent(100), height=ui.Percent(100), spacing=0):
                        vertical = ui.Spacer(height=(1.0 - y_value) * (size - handle))
                        with ui.HStack(height=handle, spacing=0):
                            horizontal = ui.Spacer(width=x_value * (size - handle))
                            ui.Rectangle(width=handle, height=handle, style={
                                "background_color": 0xFFEFF4FF, "border_color": 0xFF111827,
                                "border_width": 2, "border_radius": 3,
                            })
                    self._pad_surface = ui.Rectangle(
                        style={"background_color": 0x00000000}, opaque_for_mouse_events=True,
                        mouse_pressed_fn=self._on_pad_pressed,
                        mouse_moved_fn=self._on_pad_moved,
                        mouse_released_fn=self._on_pad_released,
                    )
                    self._pad_spacers = (horizontal, vertical)
                ui.Spacer(height=6)
                with ui.VStack(height=48, spacing=4) as self._pad_x_controls:
                    self._build_pad_axis_selector("x", width=ui.Percent(100))
                    self._pad_value_labels["x"] = ui.Label(self._pad_value_text(x_name), height=18)
                ui.Spacer()

    def _refresh_parameter_pad_layout(self) -> None:
        frame, square, column = self._pad_frame, self._pad_square, self._pad_column
        if frame is None or square is None or column is None:
            return
        width = float(frame.computed_width)
        height = float(frame.computed_height)
        window = self.property_windows.get(PropertyTab.FINDSURFACE)
        host = None if window is None else window.frame
        if host is not None:
            width, height = _visible_widget_extent(
                (
                    float(frame.screen_position_x), float(frame.screen_position_y),
                    width, height,
                ),
                (
                    float(host.screen_position_x), float(host.screen_position_y),
                    float(host.computed_width), float(host.computed_height),
                ),
            )
        if width <= 1.0 or height <= 1.0:
            return
        size = _parameter_pad_square_size(width, height)
        if size == self._pad_layout_size:
            return
        self._pad_layout_size = size
        square.width = ui.Pixel(size)
        square.height = ui.Pixel(size)
        column.width = ui.Pixel(size)
        x_name, y_name = self._pad_axes["x"], self._pad_axes["y"]
        handle = 12.0
        if self._pad_spacers is not None:
            horizontal, vertical = self._pad_spacers
            horizontal.width = ui.Pixel(self._pad_normalized(x_name) * max(0.0, size - handle))
            vertical.height = ui.Pixel((1.0 - self._pad_normalized(y_name)) * max(0.0, size - handle))

    def _build_pad_axis_selector(self, axis: str, *, width: float) -> None:
        options = self._pad_axis_options(axis)
        selected = self._pad_axes[axis]
        combo = ui.ComboBox(
            options.index(selected), *(_FS_PAD_LABELS[name] for name in options),
            width=width, height=24,
        )
        model = combo.model
        callback = lambda *args, current=axis, current_model=model, current_options=options: self._on_pad_axis_changed(
            current, current_model, current_options,
        )
        for method_name in ("add_item_changed_fn", "subscribe_item_changed_fn", "subscribe_value_changed_fn"):
            method = getattr(model, method_name, None)
            if method is not None:
                self._model_subscriptions.append(method(callback))
                break

    def _pad_axis_options(self, axis: str) -> tuple[str, ...]:
        other = self._pad_axes["y" if axis == "x" else "x"]
        return tuple(name for name in _FS_PAD_PARAMETERS if name != other)

    def _on_pad_axis_changed(self, axis: str, model, options) -> None:
        index = self._combo_index(model)
        self._pad_axes[axis] = options[max(0, min(len(options) - 1, index))]
        self._queue_parameter_pad_refresh()

    def _queue_parameter_pad_refresh(self) -> bool:
        """Rebuild the pad after Kit has processed the current model callback.

        Kit does not reliably re-layout an already mounted Spacer after only its
        width/height property changes. Demo 1 solves the same stale-knob issue by
        coalescing a pad-row rebuild onto the next application update.
        """

        if self._pad_dragging or self._pad_frame is None:
            return False
        task = self._pad_refresh_task
        if task is not None and not task.done():
            return False
        self._pad_refresh_task = asyncio.ensure_future(self._rebuild_pad_after_update())
        return True

    async def _rebuild_pad_after_update(self) -> None:
        try:
            await omni.kit.app.get_app().next_update_async()
            if self._pad_frame is None:
                return
            self._pad_frame.clear()
            with self._pad_frame:
                self._build_parameter_pad_row()
        finally:
            self._pad_refresh_task = None

    @staticmethod
    def _combo_index(model) -> int:
        for method_name in ("get_item_value_model", "get_value_model"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                value_model = method()
            except TypeError:
                value_model = method(None)
            return int(getattr(value_model, "as_int", 0))
        return int(getattr(model, "as_int", 0))

    def _on_pad_pressed(self, x, y, *args) -> None:
        self._pad_dragging = True
        self._apply_pad_pointer(x, y)

    def _on_pad_moved(self, x, y, *args) -> None:
        if self._pad_dragging:
            self._apply_pad_pointer(x, y)

    def _on_pad_released(self, x, y, *args) -> None:
        if self._pad_dragging:
            self._apply_pad_pointer(x, y)
        self._pad_dragging = False

    def _apply_pad_pointer(self, x: float, y: float) -> None:
        surface = self._pad_surface
        if surface is None:
            return
        width, height = max(1.0, float(surface.computed_width)), max(1.0, float(surface.computed_height))
        u = max(0.0, min(1.0, (float(x) - float(surface.screen_position_x)) / width))
        v = max(0.0, min(1.0, 1.0 - (float(y) - float(surface.screen_position_y)) / height))
        for axis, normalized in (("x", u), ("y", v)):
            name = self._pad_axes[axis]
            model = self.property_models[name]
            minimum, maximum = self._parameter_bounds(name)
            value = minimum + normalized * (maximum - minimum)
            model.set_value(int(round(value)) if name in ("lateral_extension", "radial_expansion") else value)
        self._sync_parameter_pad_from_models()

    def _parameter_bounds(self, name: str) -> tuple[float, float]:
        if name in _FS_RANGE_DEFAULTS:
            return (
                self.property_models[f"{name}_min"].as_float,
                self.property_models[f"{name}_max"].as_float,
            )
        return 0.0, 10.0

    def _pad_normalized(self, name: str) -> float:
        minimum, maximum = self._parameter_bounds(name)
        model = self.property_models[name]
        value = float(model.as_int if name in ("lateral_extension", "radial_expansion") else model.as_float)
        return 0.0 if maximum <= minimum else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    def _pad_value_text(self, name: str) -> str:
        model = self.property_models[name]
        if name in ("lateral_extension", "radial_expansion"):
            return str(model.as_int)
        return f"{model.as_float:.4f}"

    @staticmethod
    def _range_index(value: float, minimum: float, maximum: float) -> int:
        if maximum <= minimum:
            return 0
        return slider_index_for_value(float(value), minimum, maximum)

    def _on_range_value(self, name: str, value: float) -> None:
        if self._syncing_properties:
            return
        minimum, maximum = self._parameter_bounds(name)
        self._syncing_properties = True
        try:
            self.property_models[f"{name}_slider_index"].set_value(self._range_index(value, minimum, maximum))
        finally:
            self._syncing_properties = False
        self._sync_parameter_pad_from_models(ensure_visual_refresh=True)
        self._property_dispatch(f"findsurface.{name}", float(value))

    def _on_pad_integer_value(self, name: str, value: int) -> None:
        if self._syncing_properties:
            return
        self._sync_parameter_pad_from_models(ensure_visual_refresh=True)
        self._property_dispatch(f"findsurface.{name}", int(value))

    def _sync_parameter_pad_from_models(self, *, ensure_visual_refresh: bool = False) -> None:
        """Mirror every parameter model change into the active 2D-pad axes."""

        for axis in ("x", "y"):
            name = self._pad_axes[axis]
            label = self._pad_value_labels.get(axis)
            if label is not None:
                label.text = self._pad_value_text(name)
        if self._pad_spacers is None:
            return
        horizontal, vertical = self._pad_spacers
        size = float(self._pad_layout_size or 0.0)
        if self._pad_surface is not None:
            size = min(
                max(0.0, float(self._pad_surface.computed_width)),
                max(0.0, float(self._pad_surface.computed_height)),
            ) or size
        travel = max(0.0, size - 12.0)
        horizontal.width = ui.Pixel(self._pad_normalized(self._pad_axes["x"]) * travel)
        vertical.height = ui.Pixel((1.0 - self._pad_normalized(self._pad_axes["y"])) * travel)
        # Direct dimension assignment updates the model but does not reliably
        # invalidate Kit's mounted layout. Rebuild only this pad on the next
        # update, matching demo 1. Pointer drags keep their direct live path.
        if ensure_visual_refresh:
            self._queue_parameter_pad_refresh()

    def _on_range_bounds(self, name: str) -> None:
        if self._syncing_properties:
            return
        minimum, maximum = self._parameter_bounds(name)
        if minimum > maximum:
            maximum = minimum
            self.property_models[f"{name}_max"].set_value(maximum)
        slider = self._range_sliders.get(name)
        if slider is not None:
            slider.enabled = maximum > minimum
        value = max(minimum, min(maximum, self.property_models[name].as_float))
        self._property_dispatch("findsurface.ranges", tuple(self._parameter_bounds(key) for key in _FS_RANGE_DEFAULTS))
        self.property_models[name].set_value(value)
        self._sync_parameter_pad_from_models(ensure_visual_refresh=True)

    def _on_range_slider(self, name: str, index: int) -> None:
        if self._syncing_properties:
            return
        minimum, maximum = self._parameter_bounds(name)
        value = minimum if maximum <= minimum else minimum + max(0, min(10, int(index))) * (maximum - minimum) / 10.0
        self.property_models[name].set_value(value)

    def _build_geometry_property(self, _section=None) -> None:
        self._startup_reset_button("geometries")
        self._histogram_hover_basis = None
        self._geometry_hover_capture_id = None
        ui.Spacer(height=8)
        self._geometry_heatmap_collections = []
        self._geometry_card_frames = {}
        self._geometry_card_controls = {}
        entries = tuple(self._property_values.get("geometry.entries", ()))
        active_capture_ids = {card.capture_id for card in entries}
        self._geometry_collapsed = {
            capture_id: collapsed
            for capture_id, collapsed in self._geometry_collapsed.items()
            if capture_id in active_capture_ids
        }
        with ui.HStack(height=28, spacing=6):
            ui.Label(
                f"{len(entries)} captured geometr{'y' if len(entries) == 1 else 'ies'} · one per type",
                style={"color": 0xFFB7BCC5},
            )
            ui.Spacer()
            header_buttons = (
                ui.Button(
                    "M+", width=38, tooltip="Show all captured meshes",
                    clicked_fn=lambda: self._property_dispatch("geometry.set_all_mesh_visibility", True),
                ),
                ui.Button(
                    "M-", width=38, tooltip="Hide all captured meshes",
                    clicked_fn=lambda: self._property_dispatch("geometry.set_all_mesh_visibility", False),
                ),
                ui.Button(
                    "I+", width=38, tooltip="Show all captured inliers",
                    clicked_fn=lambda: self._property_dispatch("geometry.set_all_inliers_visibility", True),
                ),
                ui.Button(
                    "I-", width=38, tooltip="Hide all captured inliers",
                    clicked_fn=lambda: self._property_dispatch("geometry.set_all_inliers_visibility", False),
                ),
                ui.Button(
                    "+", width=28, tooltip="Expand all captured geometries",
                    clicked_fn=lambda: self._set_all_geometry_collapsed(False),
                ),
                ui.Button(
                    "-", width=28, tooltip="Collapse all captured geometries",
                    clicked_fn=lambda: self._set_all_geometry_collapsed(True),
                ),
            )
            for button in header_buttons:
                button.enabled = bool(entries)
        if not entries:
            with BorderedGroupView("Captured Geometries", row_heights=(48,)):
                ui.Label(
                    "No captured geometries. Freeze the sensor, select a detected point, then Capture.",
                    word_wrap=True,
                )
            ui.Spacer()
            return
        for card in entries:
            capture_id = card.capture_id
            item_height = self._geometry_item_height(card.rows, card.histograms, card.comparison_rows)
            frame = self._geometry_collapsable_frame(
                capture_id,
                card.geometry_type,
                f"{capture_id} ({card.geometry_type})",
                item_height,
                mesh_visible=bool(card.mesh_visible),
                inliers_visible=bool(card.inliers_visible),
            )
            with frame:
                with ui.VStack(height=0, spacing=0):
                    ui.Spacer(height=GEOMETRY_CONTENT_GAP)
                    with ui.HStack(height=item_height, spacing=0):
                        ui.Spacer(width=GEOMETRY_SECTION_PADDING)
                        self._build_geometry_item(
                            capture_id=capture_id,
                            rows=card.rows,
                            heatmap_basis=card.heatmap_basis,
                            histograms=card.histograms,
                            comparison=card.comparison_rows,
                            item_height=item_height,
                        )
                        ui.Spacer(width=GEOMETRY_SECTION_PADDING)
                    ui.Spacer(height=GEOMETRY_SECTION_PADDING)
        ui.Spacer()

    def _geometry_collapsable_frame(
        self,
        capture_id: str,
        geometry_type: str,
        title: str,
        item_height: int,
        *,
        mesh_visible: bool,
        inliers_visible: bool,
    ):
        frame_cls = getattr(ui, "CollapsableFrame", None)
        if frame_cls is None:
            # Isaac Sim 6.0.1 provides CollapsableFrame. This fallback keeps the
            # content usable on reduced UI test runtimes, although it cannot fold.
            return BorderedGroupView(
                title,
                row_heights=(
                    GEOMETRY_CONTENT_GAP + item_height + GEOMETRY_SECTION_PADDING,
                ),
            )

        icon_name = _GEOMETRY_ICON_BY_TYPE.get(str(geometry_type), "placeholder.svg")
        icon_path = os.path.join(self._icon_directory, icon_name)

        def build_header(collapsed: bool, header_title: str) -> None:
            triangle_alignment = (
                ui.Alignment.RIGHT_CENTER if collapsed else ui.Alignment.CENTER_BOTTOM
            )
            with ui.HStack(height=GEOMETRY_HEADER_HEIGHT, spacing=0):
                with ui.VStack(width=18, spacing=0):
                    ui.Spacer()
                    with ui.HStack(height=8, spacing=0):
                        ui.Spacer(width=3)
                        ui.Triangle(
                            alignment=triangle_alignment,
                            width=8,
                            height=8,
                            style={"background_color": 0xFFD5D9E0},
                        )
                        ui.Spacer()
                    ui.Spacer()
                ui.Spacer(width=3)
                with ui.VStack(
                    width=GEOMETRY_HEADER_ICON_SIZE,
                    height=GEOMETRY_HEADER_HEIGHT,
                    spacing=0,
                ):
                    ui.Spacer()
                    ui.Image(
                        icon_path,
                        width=GEOMETRY_HEADER_ICON_SIZE,
                        height=GEOMETRY_HEADER_ICON_SIZE,
                    )
                    ui.Spacer()
                ui.Spacer(width=5)
                ui.Label(header_title, height=20, alignment=ui.Alignment.CENTER)
                ui.Spacer()
                controls = self._geometry_card_controls.setdefault(capture_id, {})
                controls["mesh"] = self._geometry_header_icon_button(
                    "visibility_mesh_on.svg" if mesh_visible else "visibility_mesh_off.svg",
                    "Hide captured mesh" if mesh_visible else "Show captured mesh",
                    lambda value=capture_id: self._property_dispatch("geometry.toggle_mesh", value),
                    checked=mesh_visible,
                )
                ui.Spacer(width=3)
                controls["inliers"] = self._geometry_header_icon_button(
                    "visibility_inliers_on.svg" if inliers_visible else "visibility_inliers_off.svg",
                    "Hide captured inliers" if inliers_visible else "Show captured inliers",
                    lambda value=capture_id: self._property_dispatch("geometry.toggle_inliers", value),
                    checked=inliers_visible,
                )
                ui.Spacer(width=3)
                controls["delete"] = self._geometry_header_icon_button(
                    "delete.svg",
                    "Delete captured geometry",
                    lambda value=capture_id: self._property_dispatch("geometry.delete", value),
                    destructive=True,
                )
                ui.Spacer(width=4)

        frame = frame_cls(
            title,
            collapsed=bool(self._geometry_collapsed.get(capture_id, False)),
            height=0,
            build_header_fn=build_header,
            mouse_hovered_fn=lambda hovered, value=capture_id: self._set_geometry_card_hover(
                value, hovered,
            ),
        )
        if hasattr(frame, "set_collapsed_changed_fn"):
            frame.set_collapsed_changed_fn(
                lambda collapsed, value=capture_id: self._set_geometry_collapsed(value, collapsed)
            )
        self._geometry_card_frames[capture_id] = frame
        return frame

    def _geometry_header_icon_button(
        self,
        icon_name: str,
        tooltip: str,
        clicked_fn,
        *,
        checked: bool = False,
        destructive: bool = False,
    ) -> _IconButton:
        size = GEOMETRY_HEADER_ACTION_SIZE
        icon_size = GEOMETRY_HEADER_ACTION_ICON_SIZE
        offset = (size - icon_size) / 2
        frame_style = {
            "background_color": 0x402B3038,
            "border_color": 0xFF59616D,
            "border_width": 1,
            "border_radius": 3,
        }
        if destructive:
            frame_style.update({
                "background_color": 0x403A1016,
                "border_color": 0xFFDB4B57,
            })
        with ui.ZStack(
            width=size,
            height=size,
            content_clipping=True,
            opaque_for_mouse_events=True,
        ):
            frame = ui.Rectangle(width=size, height=size, style=frame_style)
            # A direct Image child of ZStack is placed at its top-left in Kit.
            # Explicit spacer layout keeps the glyph centered inside its border.
            with ui.VStack(width=size, height=size, spacing=0):
                ui.Spacer(height=offset)
                with ui.HStack(height=icon_size, spacing=0):
                    ui.Spacer(width=offset)
                    image = ui.Image(
                        os.path.join(self._icon_directory, icon_name),
                        width=icon_size,
                        height=icon_size,
                    )
                    ui.Spacer()
                ui.Spacer()
            button = ui.Button(
                " ",
                width=size,
                height=size,
                tooltip=tooltip,
                clicked_fn=clicked_fn,
                opaque_for_mouse_events=True,
                style={"background_color": 0x00000000},
            )
        wrapped = _IconButton(button, image, frame)
        wrapped.set_state(enabled=True, checked=checked, tooltip=tooltip)
        return wrapped

    @staticmethod
    def _geometry_item_height(rows, histograms, comparison=()) -> int:
        histogram_height = (
            GEOMETRY_HISTOGRAM_TOP_GAP
            + GEOMETRY_HISTOGRAM_TITLE_HEIGHT
            + RESULT_OVERLAY_HISTOGRAM_HEIGHT
        )
        heights = (
            GEOMETRY_HEATMAP_ROW_HEIGHT,
            *(GEOMETRY_FIELD_ROW_HEIGHT for _ in rows),
            *((26, *(row.height for row in comparison)) if comparison else ()),
            *(histogram_height for _ in histograms),
        )
        return sum(heights) + max(0, len(heights) - 1) * GEOMETRY_ITEM_ROW_SPACING

    def _build_geometry_item(
        self,
        *,
        capture_id: str,
        rows,
        heatmap_basis: str,
        histograms,
        item_height: int,
        comparison=(),
    ) -> None:
        with ui.VStack(height=item_height, spacing=GEOMETRY_ITEM_ROW_SPACING):
            with ui.HStack(height=GEOMETRY_HEATMAP_ROW_HEIGHT, spacing=6):
                ui.Label("Heatmap", width=96, style={"color": 0xFFB7BCC5})
                collection = ui.RadioCollection()
                detected_button = ui.RadioButton(
                    text="Detected",
                    radio_collection=collection,
                )
                original_button = ui.RadioButton(
                    text="Original",
                    radio_collection=collection,
                )
                surface_side_button = ui.RadioButton(
                    text="Surface Side",
                    radio_collection=collection,
                )
                collection.model.set_value({
                    "detected": 0,
                    "designed": 1,
                    "surface-side": 2,
                }.get(str(heatmap_basis), 0))
                detected_button.set_clicked_fn(
                    lambda value=capture_id: self._property_dispatch(
                        "geometry.heatmap_basis", (value, "detected"),
                    )
                )
                original_button.set_clicked_fn(
                    lambda value=capture_id: self._property_dispatch(
                        "geometry.heatmap_basis", (value, "designed"),
                    )
                )
                surface_side_button.set_clicked_fn(
                    lambda value=capture_id: self._property_dispatch(
                        "geometry.heatmap_basis", (value, "surface-side"),
                    )
                )
                self._geometry_heatmap_collections.append(collection)
            self._geometry_card_controls.setdefault(capture_id, {})["heatmap"] = collection
            for name, value in rows:
                with ui.HStack(height=GEOMETRY_FIELD_ROW_HEIGHT, spacing=6):
                    ui.Label(str(name), width=96, style={"color": 0xFFB7BCC5})
                    ui.Label(str(value), word_wrap=True)
            if comparison:
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Parameter", width=100)
                    for title in ("Detected", "Original", "Difference"):
                        ui.Label(title, width=ui.Fraction(1), tooltip="Original is frozen at Capture. Difference: detected - original; position: distance; axis: angle.")
                for row in comparison:
                    with ui.HStack(height=row.height, spacing=6):
                        ui.Label(row.label, width=100, style={"color": 0xFFB7BCC5})
                        for value in (row.detected, row.original, row.difference):
                            with ui.ZStack(width=ui.Fraction(1), height=row.height):
                                ui.Rectangle(style={"background_color": 0x181F232B, "border_radius": 3})
                                ui.Label(value, word_wrap=False, style={"font_size": 12, "margin": 3}, tooltip=value)
            histogram_labels = {
                "detected": "Detected Geometry Residual",
                "designed": "Original Geometry Residual",
                "surface-side": "Surface Side Distance",
            }
            histogram_height = (
                GEOMETRY_HISTOGRAM_TOP_GAP
                + GEOMETRY_HISTOGRAM_TITLE_HEIGHT
                + RESULT_OVERLAY_HISTOGRAM_HEIGHT
            )
            for basis, histogram in histograms:
                with ui.VStack(height=histogram_height, spacing=0):
                    ui.Spacer(height=GEOMETRY_HISTOGRAM_TOP_GAP)
                    ui.Label(
                        histogram_labels.get(str(basis), str(basis)),
                        height=GEOMETRY_HISTOGRAM_TITLE_HEIGHT,
                        style={"color": 0xFFB7BCC5},
                    )
                    self._build_result_histogram(
                        histogram,
                        hover_key=(str(capture_id), str(basis)),
                    )

    def _set_histogram_hover(self, key, hovered) -> None:
        if hovered:
            self._histogram_hover_basis = key
        elif getattr(self, "_histogram_hover_basis", None) == key:
            self._histogram_hover_basis = None

    @property
    def histogram_hover_basis(self):
        key = getattr(self, "_histogram_hover_basis", None)
        window = self.property_windows.get(PropertyTab.GEOMETRIES)
        if key is not None and (window is None or not window.visible
                or self._geometry_collapsed.get(key[0], False)
                or not any(card.capture_id == key[0] for card in self._property_values.get("geometry.entries", ()))):
            self._histogram_hover_basis = None
            return None
        return key

    def _set_geometry_card_hover(self, capture_id: str, hovered: bool) -> None:
        capture_id = str(capture_id)
        if hovered:
            self._geometry_hover_capture_id = capture_id
        elif self._geometry_hover_capture_id == capture_id:
            self._geometry_hover_capture_id = None

    @property
    def geometry_hover_capture_id(self) -> str | None:
        capture_id = self._geometry_hover_capture_id
        window = self.property_windows.get(PropertyTab.GEOMETRIES)
        if capture_id is not None and (
            window is None
            or not window.visible
            or not any(
                card.capture_id == capture_id
                for card in self._property_values.get("geometry.entries", ())
            )
        ):
            self._geometry_hover_capture_id = None
            return None
        return capture_id

    def _set_geometry_collapsed(self, capture_id: str, collapsed: bool) -> None:
        self._histogram_hover_basis = None
        self._geometry_collapsed[str(capture_id)] = bool(collapsed)

    def _set_all_geometry_collapsed(self, collapsed: bool) -> None:
        self._histogram_hover_basis = None
        collapsed = bool(collapsed)
        capture_ids = tuple(
            card.capture_id for card in self._property_values.get("geometry.entries", ())
        )
        needs_refresh = False
        for capture_id in capture_ids:
            self._geometry_collapsed[capture_id] = collapsed
            frame = self._geometry_card_frames.get(capture_id)
            if frame is None:
                needs_refresh = True
                continue
            try:
                frame.collapsed = collapsed
            except Exception:
                needs_refresh = True
        if needs_refresh:
            self._queue_geometry_property_refresh()

    @staticmethod
    def _geometry_structure_signature(entries) -> tuple:
        """Fields requiring card reconstruction, excluding live button/radio state."""

        return tuple(card.structure_signature for card in entries)

    def _sync_geometry_card_controls(self, entries) -> None:
        basis_indices = {"detected": 0, "designed": 1, "surface-side": 2}
        for card in entries:
            controls = self._geometry_card_controls.get(card.capture_id)
            if controls is None:
                continue
            # CollapsableFrame builds its custom header lazily. During that short
            # interval the body radio exists but the header actions do not (or
            # vice versa), so synchronize whichever controls Kit has realized.
            mesh_button = controls.get("mesh")
            if mesh_button is not None:
                mesh_button.image.source_url = os.path.join(
                    self._icon_directory,
                    "visibility_mesh_on.svg" if card.mesh_visible else "visibility_mesh_off.svg",
                )
                mesh_button.set_state(
                    enabled=True,
                    checked=bool(card.mesh_visible),
                    tooltip="Hide captured mesh" if card.mesh_visible else "Show captured mesh",
                )
            inliers_button = controls.get("inliers")
            if inliers_button is not None:
                inliers_button.image.source_url = os.path.join(
                    self._icon_directory,
                    "visibility_inliers_on.svg" if card.inliers_visible else "visibility_inliers_off.svg",
                )
                inliers_button.set_state(
                    enabled=True,
                    checked=bool(card.inliers_visible),
                    tooltip="Hide captured inliers" if card.inliers_visible else "Show captured inliers",
                )
            heatmap = controls.get("heatmap")
            if heatmap is not None:
                model = heatmap.model
                next_index = basis_indices.get(str(card.heatmap_basis), 0)
                if getattr(model, "as_int", next_index) != next_index:
                    model.set_value(next_index)

    def _queue_geometry_property_refresh(self) -> None:
        self._capture_property_scroll_position(PropertyTab.GEOMETRIES)
        task = self._geometry_refresh_task
        if task is not None and not task.done():
            return
        self._geometry_refresh_task = asyncio.ensure_future(self._refresh_geometry_property_after_update())

    async def _refresh_geometry_property_after_update(self) -> None:
        try:
            await omni.kit.app.get_app().next_update_async()
            frame = self._geometry_property_content_frame
            if frame is None:
                return
            if hasattr(frame, "clear"):
                frame.clear()
            with frame:
                with ui.VStack(height=0, spacing=6):
                    self._build_geometry_property()
            self._restore_property_scroll_position(PropertyTab.GEOMETRIES)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            carb.log_warn(f"Failed to refresh Geometries window: {error}")
        finally:
            self._geometry_refresh_task = None

    def _capture_property_scroll_position(self, tab: PropertyTab) -> None:
        value = self._read_scroll_position(self._property_scrolling_frames.get(tab))
        if value is not None:
            self._property_scroll_y[tab] = value

    def _restore_property_scroll_position(self, tab: PropertyTab) -> None:
        scrolling_frame = self._property_scrolling_frames.get(tab)
        value = self._property_scroll_y.get(tab)
        if scrolling_frame is not None and value is not None:
            self._write_scroll_position(scrolling_frame, value)

    @staticmethod
    def _read_scroll_position(scrolling_frame) -> float | None:
        if scrolling_frame is None:
            return None
        for name in ("scroll_y", "scroll_offset_y", "scroll_y_offset"):
            try:
                value = getattr(scrolling_frame, name)
            except Exception:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if hasattr(value, "as_float"):
                value = value.as_float
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _write_scroll_position(scrolling_frame, value: float) -> bool:
        for name in ("scroll_y", "scroll_offset_y", "scroll_y_offset"):
            try:
                current = getattr(scrolling_frame, name)
            except Exception:
                continue
            try:
                if hasattr(current, "set_value"):
                    current.set_value(float(value))
                else:
                    setattr(scrolling_frame, name, float(value))
                return True
            except Exception:
                continue
        return False

    def _build_object_property(self) -> None:
        self._startup_reset_button("objects")
        ui.Spacer(height=8)
        for mode in ("Plane", "Sphere", "Cylinder", "Cone", "Torus"):
            parameter_names = tuple(self._property_values.get(f"object.{mode}.parameter_names", ()))
            enabled = bool(self._property_values.get(f"object.{mode}.enabled", True))
            rows = (28, 26, 26, 26, 26, 112) + tuple(26 for _ in parameter_names) + (26,)
            with BorderedGroupView(mode, row_heights=rows):
                with ui.HStack(height=28, spacing=6):
                    ui.Label("State", width=72)
                    button = ui.Button(
                        "Enabled" if enabled else "Disabled",
                        clicked_fn=lambda value=mode: self._property_dispatch("object.toggle", value),
                    )
                    button.checked = enabled
                    self.property_buttons[f"object.{mode}"] = button
                self._build_object_vector_row(mode, "position", "Position", "position", -100.0, 100.0, 0.01)
                for axis, label in (("x", "Local X"), ("y", "Local Y"), ("z", "Local Z")):
                    self._build_object_vector_row(mode, f"axis.{axis}", label, axis, -1.0, 1.0, 0.001)
                self._build_object_rotation_pad(mode)
                for name in parameter_names:
                    key = f"object.{mode}.parameter.{name}"
                    model = ui.SimpleFloatModel(float(self._property_values.get(key, 0.0)))
                    self.property_models[key] = model
                    minimum_length = float(
                        self._property_values.get("settings.minimum_geometry_length_m", 0.01)
                    )
                    is_sweep_start = name == "sweep_start_deg"
                    is_sweep_angle = name == "sweep_angle_deg"
                    minimum = (
                        0.0 if is_sweep_start else
                        MINIMUM_TORUS_SWEEP_DEG if is_sweep_angle else
                        minimum_length
                    )
                    maximum = 360.0 if is_sweep_start or is_sweep_angle else 1000.0
                    step = 0.1 if is_sweep_start or is_sweep_angle else 0.01
                    label = {
                        "sweep_start_deg": "Sweep start (deg)",
                        "sweep_angle_deg": "Sweep angle (deg)",
                    }.get(name, name)
                    with ui.HStack(height=26, spacing=6):
                        ui.Label(label, width=150)
                        drag_options = {
                            "model": model, "min": minimum, "max": maximum, "step": step, "format": "%.4f",
                        }
                        if is_sweep_start or is_sweep_angle:
                            drag_options["format"] = "%.1f"
                        ui.FloatDrag(**drag_options)
                    self._subscribe_callback(
                        model,
                        lambda _model, current=mode, parameter=name: self._dispatch_object_models(
                            current, "parameter", edited_parameter=parameter,
                        ),
                    )
                subdivision_key = f"object.{mode}.subdivision"
                subdivision_model = ui.SimpleIntModel(
                    int(self._property_values.get(subdivision_key, 96))
                )
                self.property_models[subdivision_key] = subdivision_model
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Sensor subdivision", width=150)
                    ui.IntDrag(model=subdivision_model, min=3, max=256, step=1)
                self._subscribe_callback(
                    subdivision_model,
                    lambda _model, current=mode: self._dispatch_object_models(current, "subdivision"),
                )
        ui.Spacer()

    def _build_object_color_row(self, mode: str, group: str, label: str) -> None:
        prefix = f"object.{mode}.{group}"
        values = tuple(float(self._property_values.get(f"{prefix}.{component}", 1.0)) for component in "rgba")
        models = self._build_rgba_row(prefix, label, values, label_width=112)
        for model in models:
            self._subscribe_callback(
                model,
                lambda _model, current=mode, edited=group: self._dispatch_object_models(current, edited),
            )

    def _build_rgba_row(
        self, prefix: str, label: str, values, *, label_width: int = 112,
    ) -> tuple:
        """Build shared numeric RGB models followed by alpha and a popup ColorWidget."""

        values = tuple(float(value) for value in values)
        with ui.HStack(height=26, spacing=4):
            ui.Label(label, width=label_width)
            if hasattr(ui, "ColorWidget"):
                rgb_frame = ui.Frame(width=ui.Fraction(3))
                alpha_model = ui.SimpleFloatModel(values[3])
                ui.FloatDrag(
                    model=alpha_model, min=0.0, max=1.0, step=0.01,
                    format="%.2f", width=ui.Fraction(1),
                )
                widget = ui.ColorWidget(*values[:3], width=28, height=22)
                rgb_models = tuple(self._color_component_models(widget.model)[:3])
                if len(rgb_models) != 3:
                    raise RuntimeError("ColorWidget did not expose three RGB component models")
                with rgb_frame:
                    with ui.HStack(spacing=4):
                        for model in rgb_models:
                            ui.FloatDrag(model=model, min=0.0, max=1.0, step=0.01, format="%.2f")
                models = rgb_models + (alpha_model,)
            else:
                models = tuple(ui.SimpleFloatModel(value) for value in values)
                for model in models:
                    ui.FloatDrag(model=model, min=0.0, max=1.0, step=0.01, format="%.2f")
        for component, model in zip("rgba", models):
            self.property_models[f"{prefix}.{component}"] = model
        return models

    @staticmethod
    def _color_component_models(color_model) -> list:
        try:
            items = color_model.get_item_children()
        except TypeError:
            items = color_model.get_item_children(None)
        result = []
        for item in items:
            try:
                result.append(color_model.get_item_value_model(item))
            except TypeError:
                result.append(color_model.get_item_value_model(item, 0))
        return result

    def _dispatch_rgba_models(self, prefix: str, action: str) -> None:
        if self._syncing_properties:
            return
        self._property_dispatch(
            action,
            tuple(self.property_models[f"{prefix}.{component}"].as_float for component in "rgba"),
        )

    def _build_object_vector_row(
        self, mode: str, group: str, label: str, edited_axis: str,
        minimum: float, maximum: float, step: float,
    ) -> None:
        with ui.HStack(height=26, spacing=4):
            ui.Label(label, width=72)
            for component in "xyz":
                key = f"object.{mode}.{group}.{component}"
                model = ui.SimpleFloatModel(float(self._property_values.get(key, 0.0)))
                self.property_models[key] = model
                ui.FloatDrag(model=model, min=minimum, max=maximum, step=step, format="%.4f")
                self._subscribe_callback(
                    model,
                    lambda _model, current=mode, axis=edited_axis: self._dispatch_object_models(current, axis),
                )
                if group.startswith("axis."):
                    self._subscribe_object_edit_lifecycle(model, key, mode, edited_axis)

    def _subscribe_object_edit_lifecycle(self, model, key: str, mode: str, edited_axis: str) -> None:
        begin = getattr(model, "subscribe_begin_edit_fn", None) or getattr(model, "add_begin_edit_fn", None)
        end = getattr(model, "subscribe_end_edit_fn", None) or getattr(model, "add_end_edit_fn", None)
        if begin is not None:
            self._model_subscriptions.append(begin(lambda _model, current=key: self._object_editing_keys.add(current)))
        if end is not None:
            def finish(_model, current=key, object_mode=mode, axis=edited_axis):
                self._object_editing_keys.discard(current)
                self._dispatch_object_models(object_mode, axis)
            self._model_subscriptions.append(end(finish))

    def _build_object_rotation_pad(self, mode: str) -> None:
        with ui.HStack(height=112, spacing=6):
            ui.Label("Rotate", width=72)
            with ui.ZStack(height=104):
                ui.Rectangle(style={
                    "background_color": 0x331B1E24,
                    "border_color": 0xFF8C9098,
                    "border_width": 1,
                    "border_radius": 4,
                })
                with ui.HStack(spacing=0):
                    ui.Spacer()
                    ui.Rectangle(width=1, style={"background_color": 0x448C9098})
                    ui.Spacer()
                with ui.VStack(spacing=0):
                    ui.Spacer()
                    ui.Rectangle(height=1, style={"background_color": 0x448C9098})
                    ui.Spacer()
                ui.Label(
                    "LMB: rotate in view  |  RMB: roll", alignment=ui.Alignment.CENTER,
                    style={"color": 0xFFB7BCC5}, opaque_for_mouse_events=False,
                )
                surface = ui.Rectangle(
                    style={"background_color": 0x00000000}, opaque_for_mouse_events=True,
                    mouse_pressed_fn=lambda x, y, button, modifiers, current=mode: self._on_object_pad_pressed(
                        current, x, y, button,
                    ),
                    mouse_moved_fn=lambda x, y, buttons, modifiers, current=mode: self._on_object_pad_moved(
                        current, x, y,
                    ),
                    mouse_released_fn=lambda x, y, button, modifiers, current=mode: self._on_object_pad_released(
                        current, x, y, button,
                    ),
                )
                self._object_pad_surfaces[mode] = surface

    def _on_object_pad_pressed(self, mode: str, x: float, y: float, button: int) -> None:
        button = int(button)
        if button not in (0, 1):
            return
        self._object_pad_drag = {
            "mode": mode,
            "button": button,
            "start": (float(x), float(y)),
        }
        self._property_dispatch("object.rotate.begin", mode)

    def _on_object_pad_moved(self, mode: str, x: float, y: float) -> None:
        drag = self._object_pad_drag
        if drag is None or drag["mode"] != mode:
            return
        surface = self._object_pad_surfaces[mode]
        current = (float(x), float(y))
        if drag["button"] == 1:
            center = (
                float(surface.screen_position_x) + float(surface.computed_width) * 0.5,
                float(surface.screen_position_y) + float(surface.computed_height) * 0.5,
            )
            radians = screen_roll_radians(drag["start"], current, center)
            self._property_dispatch("object.rotate.update", {
                "mode": mode,
                "roll_radians": radians,
            })
        else:
            scale = max(1.0, float(surface.computed_width), float(surface.computed_height))
            self._property_dispatch("object.rotate.update", {
                "mode": mode,
                "drag_x": (current[0] - drag["start"][0]) / scale,
                "drag_y": (current[1] - drag["start"][1]) / scale,
            })

    def _on_object_pad_released(self, mode: str, x: float, y: float, button: int) -> None:
        drag = self._object_pad_drag
        if drag is None or drag["mode"] != mode or int(button) != drag["button"]:
            return
        self._on_object_pad_moved(mode, x, y)
        self._object_pad_drag = None
        self._property_dispatch("object.rotate.end", mode)

    def _dispatch_object_models(
        self, mode: str, edited_axis: str, *, edited_parameter: str | None = None,
    ) -> None:
        if self._syncing_properties:
            return
        parameter_names = tuple(self._property_values.get(f"object.{mode}.parameter_names", ()))
        parameters = {
            name: self.property_models[f"object.{mode}.parameter.{name}"].as_float
            for name in parameter_names
        }
        minimum_model = self.property_models.get("settings.minimum_geometry_length_m")
        minimum_length = (
            minimum_model.as_float
            if minimum_model is not None
            else float(self._property_values.get("settings.minimum_geometry_length_m", 0.01))
        )
        parameters = constrain_geometry_parameter_edit(
            GeometryMode(mode),
            parameters,
            edited_parameter=edited_parameter,
            minimum_length_m=minimum_length,
        )
        if edited_parameter is not None:
            self._syncing_properties = True
            try:
                for name, parameter in parameters.items():
                    model = self.property_models[f"object.{mode}.parameter.{name}"]
                    if model.as_float != parameter:
                        model.set_value(parameter)
            finally:
                self._syncing_properties = False
        subdivision_models = (
            self.property_models[f"object.{mode}.subdivision"],
            self.property_models[f"object.{mode}.display_subdivision"],
        )
        subdivisions = tuple(max(3, min(256, model.as_int)) for model in subdivision_models)
        self._syncing_properties = True
        try:
            for model, corrected in zip(subdivision_models, subdivisions):
                if model.as_int != corrected:
                    model.set_value(corrected)
        finally:
            self._syncing_properties = False
        self._property_dispatch("object.update", {
            "mode": mode,
            "edited_axis": edited_axis,
            "edited_parameter": edited_parameter,
            "position_m": tuple(
                self.property_models[f"object.{mode}.position.{axis}"].as_float for axis in "xyz"
            ),
            "orientation_axes": tuple(
                tuple(
                    self.property_models[f"object.{mode}.axis.{axis}.{component}"].as_float
                    for component in "xyz"
                )
                for axis in "xyz"
            ),
            "parameters": parameters,
            "subdivision": subdivisions[0],
            "display_subdivision": subdivisions[1],
            "face_color_rgba": tuple(
                self.property_models[f"object.{mode}.face_color.{component}"].as_float
                for component in "rgba"
            ),
            "wire_color_rgba": tuple(
                self.property_models[f"object.{mode}.wire_color.{component}"].as_float
                for component in "rgba"
            ),
        })

    def _build_camera_property(self) -> None:
        self._startup_reset_button("camera")
        ui.Spacer(height=8)
        for role in ("scanner", "scene"):
            rows = (26, 26, 26, 26, 30) if role == "scanner" else (26, 26, 26, 26, 26, 26, 26, 30, 30)
            with BorderedGroupView(f"{role.title()} Camera", row_heights=rows):
                with ui.HStack(height=26, spacing=4):
                    ui.Label("Position (m)", width=86)
                    for axis in "xyz":
                        key = f"camera.{role}.{axis}"
                        model = ui.SimpleFloatModel(float(self._property_values.get(key, 0.0)))
                        self.property_models[key] = model
                        ui.FloatDrag(model=model, min=-100.0, max=100.0, step=0.01, format="%.4f")
                        self._subscribe_callback(
                            model,
                            lambda _model, current=role: self._dispatch_camera_position(current),
                        )
                with ui.HStack(height=26, spacing=4):
                    ui.Label("View FOV Y (deg)", width=110, tooltip="Viewport projection only; does not change the physical sensor optics.")
                    key = f"camera.{role}.fov"
                    model = ui.SimpleFloatModel(float(self._property_values.get(key, 52.0)))
                    self.property_models[key] = model
                    ui.FloatDrag(model=model, min=1.0, max=160.0, step=0.1)
                    self._subscribe_callback(model, lambda m, current=role: self._property_dispatch(f"camera.edit.{current}", ("fov", m.as_float)))
                with ui.HStack(height=26, spacing=4):
                    ui.Label("Rotation XYZW", width=110, tooltip="World orientation quaternion; normalized on edit.")
                    for component in "xyzw":
                        key = f"camera.{role}.q{component}"
                        model = ui.SimpleFloatModel(float(self._property_values.get(key, 1.0 if component == "w" else 0.0)))
                        self.property_models[key] = model
                        ui.FloatDrag(model=model, min=-1.0, max=1.0, step=0.01)
                        self._subscribe_callback(model, lambda _m, current=role: self._dispatch_camera_vector(current, "quaternion"))
                if role == "scanner":
                    ui.Button("Use sensor FOV", height=26, clicked_fn=lambda: self._property_dispatch("camera.edit.scanner", ("sensor_fov", None)))
                if role == "scene":
                    with ui.HStack(height=26, spacing=4):
                        ui.Label("Projection", width=110)
                        collection = ui.RadioCollection()
                        self._scene_projection_collection = collection
                        for index, title in enumerate(("Perspective", "Orthographic")):
                            ui.RadioButton(text=title, radio_collection=collection,
                                clicked_fn=lambda v=index: self._property_dispatch("camera.edit.scene", ("projection", v)))
                        collection.model.set_value(int(self._property_values.get("camera.scene.projection", 0)))
                        self.property_models["camera.scene.projection"] = collection.model
                    with ui.HStack(height=26, spacing=4):
                        ui.Label("Ortho height (m)", width=110, tooltip="Vertical visible world span. Orthographic only; wheel zoom adjusts this value.")
                        model = ui.SimpleFloatModel(float(self._property_values.get("camera.scene.ortho_height", 10.0)))
                        self.property_models["camera.scene.ortho_height"] = model
                        ui.FloatDrag(model=model, min=0.0001, max=1000000.0, step=0.01, format="%.4f")
                        self._subscribe_callback(model, lambda m: self._property_dispatch("camera.edit.scene", ("ortho_height", m.as_float)))
                    with ui.HStack(height=26, spacing=4):
                        ui.Label("Orbit center (m)", width=110, tooltip="Moves the center and camera together, preserving orbit offset.")
                        for component in "xyz":
                            key = f"camera.scene.target_{component}"
                            model = ui.SimpleFloatModel(float(self._property_values.get(key, 0.0)))
                            self.property_models[key] = model
                            ui.FloatDrag(model=model, min=-100.0, max=100.0, step=0.01, format="%.4f")
                            self._subscribe_callback(model, lambda _m: self._dispatch_camera_vector("scene", "target"))
                    with ui.HStack(height=26, spacing=4):
                        ui.Label("Orbit distance (m)", width=110)
                        key = "camera.scene.orbit_distance"
                        model = ui.SimpleFloatModel(float(self._property_values.get(key, 1.0)))
                        self.property_models[key] = model
                        ui.FloatDrag(model=model, min=0.01, max=1000.0, step=0.01, format="%.4f")
                        self._subscribe_callback(model, lambda m: self._property_dispatch("camera.edit.scene", ("distance", m.as_float)))
                    with ui.HStack(height=30, spacing=4):
                        ui.Label("Preset", width=86)
                        for axis in ("x", "y", "z"):
                            ui.Button(
                                f"+{axis.upper()}",
                                clicked_fn=lambda value=axis: self._property_dispatch("camera.scene.preset", value),
                            )
                with ui.HStack(height=30, spacing=6):
                    ui.Label("View", width=86)
                    ui.Button(
                        "Reset",
                        clicked_fn=lambda value=role: self._property_dispatch(f"camera.reset.{value}", None),
                    )

        self.property_models["camera.point_depth_offset_m"] = ui.SimpleFloatModel(
            float(self._property_values.get("camera.point_depth_offset_m", 0.006))
        )
        with BorderedGroupView("Point Cloud Rendering", row_heights=(26,)):
            with ui.HStack(height=26, spacing=6):
                ui.Label("Depth offset (m)", width=120)
                ui.FloatDrag(
                    model=self.property_models["camera.point_depth_offset_m"],
                    min=0.0, max=1.0, step=0.001, format="%.4f",
                )
        self._subscribe(
            self.property_models["camera.point_depth_offset_m"],
            "camera.point_depth_offset", lambda model: model.as_float,
        )
        ui.Spacer()

    def _dispatch_camera_position(self, role: str) -> None:
        if self._syncing_properties:
            return
        value = tuple(self.property_models[f"camera.{role}.{axis}"].as_float for axis in "xyz")
        self._property_dispatch(f"camera.position.{role}", value)

    def _dispatch_camera_vector(self, role: str, field: str) -> None:
        if self._syncing_properties:
            return
        keys = [f"camera.{role}.q{axis}" for axis in "xyzw"] if field == "quaternion" else [f"camera.{role}.target_{axis}" for axis in "xyz"]
        value = tuple(self.property_models[key].as_float for key in keys)
        if field == "quaternion" and sum(v*v for v in value) < 1e-12:
            return
        self._property_dispatch(f"camera.edit.{role}", (field, value))

    def _build_scanner_property(self, section) -> None:
        self._startup_reset_button("scanner")
        ui.Spacer(height=8)
        with BorderedGroupView(
            "Status", row_heights=tuple(26 for _ in section.rows),
        ):
            self._build_compact_status_rows(section)
        axial_error_model = ui.SimpleFloatModel(
            float(self._property_values.get("scanner.axial_error_sigma_m", 0.0015))
        )
        axial_error_enabled_model = ui.SimpleBoolModel(
            bool(self._property_values.get("scanner.axial_error_enabled", False))
        )
        self.property_models["scanner.axial_error_enabled"] = axial_error_enabled_model
        self.property_models["scanner.axial_error_sigma_m"] = axial_error_model
        description_heights = (32, 26, 52, 40, 26, 78)
        with BorderedGroupView(
            "Point Cloud Sensor", row_heights=(30,) + description_heights,
        ):
            with ui.HStack(height=30, spacing=6):
                ui.Label("Profile", width=72)
                if self._sensor_profile_entries:
                    labels = tuple(str(entry[2]) for entry in self._sensor_profile_entries)
                    selected = int(self._property_values.get("sensor.profile_index", 0))
                    selected = max(0, min(len(labels) - 1, selected))
                    combo = ui.ComboBox(selected, *labels, height=30)
                    self._sensor_profile_combo_model = combo.model
                    value_model_getter = getattr(combo.model, "get_item_value_model", None)
                    if value_model_getter is None:
                        value_model_getter = combo.model.get_value_model
                    try:
                        value_model = value_model_getter()
                    except TypeError:
                        value_model = value_model_getter(None)
                    self.property_models["sensor.profile_index"] = value_model
                    callback = lambda *args, model=combo.model: self._on_sensor_profile_changed(model)
                    for method_name in (
                        "add_item_changed_fn", "subscribe_item_changed_fn", "subscribe_value_changed_fn",
                    ):
                        method = getattr(combo.model, method_name, None)
                        if method is not None:
                            self._model_subscriptions.append(method(callback))
                            break
                    self.property_buttons["sensor_profile_apply"] = ui.Button(
                        "Apply", width=58, enabled=False,
                        clicked_fn=self._apply_sensor_profile,
                    )
                else:
                    ui.Label("No point-cloud sensor available")
            initial_index = int(self._property_values.get("sensor.profile_index", 0))
            initial_index = max(0, min(len(self._sensor_profile_entries) - 1, initial_index))
            initial_rows = (
                self._sensor_profile_entries[initial_index][3]
                if self._sensor_profile_entries else ()
            )
            initial_values = dict(initial_rows)
            for name, height in zip(
                ("Name", "Type", "Function", "Profile", "Range", "Features"),
                description_heights,
            ):
                with ui.HStack(height=height, spacing=6):
                    ui.Label(name, width=72, style={"color": 0xFFB7BCC5})
                    label = ui.Label(
                        str(initial_values.get(name, "N/A")), word_wrap=True,
                    )
                    self._sensor_profile_description_labels[name] = label
        with BorderedGroupView("Target Material", row_heights=(30, 30, 30, 48)):
            self._build_sensor_material_combo("base", "Base")
            self._build_sensor_material_combo("coating", "Coating")
            self._build_sensor_material_combo("attribute", "Attribute")
            with ui.HStack(height=48, spacing=6):
                ui.Label("Recommended", width=72, style={"color": 0xFFB7BCC5})
                label = ui.Label(
                    str(self._property_values.get("sensor.material_recommendation", "N/A")),
                    word_wrap=True,
                )
                label.tooltip = (
                    "Application-defined repeatable comparison target; it is not a "
                    "sensor-manufacturer material recommendation."
                )
                self.property_labels["sensor.material_recommendation"] = label
        with BorderedGroupView("Acquisition", row_heights=(32, 30)):
            with ui.HStack(height=32, spacing=4):
                ui.Label("Mode", width=72)
                for key, label, mode in (
                    ("sensor_mode_frozen", "Frozen", "frozen"),
                    ("sensor_mode_fixed", "Fixed live", "live-fixed"),
                    ("sensor_mode_follow", "Follow live", "live-follow"),
                ):
                    self.property_buttons[key] = ui.Button(
                        label,
                        clicked_fn=lambda value=mode: self._property_dispatch("scanner.mode", value),
                    )
            with ui.HStack(height=30, spacing=6):
                ui.Label("Acquisition", width=72)
                self.property_buttons["sensor_one_shot"] = ui.Button(
                    "One-shot", clicked_fn=lambda: self._property_dispatch("scanner.one_shot", None),
                )
        with BorderedGroupView("Application Axial Error", row_heights=(26, 26, 30, 30, 30, 44)):
            with ui.HStack(height=26, spacing=6):
                checkbox = ui.CheckBox(model=axial_error_enabled_model, width=22)
                checkbox.tooltip = (
                    "Application post-process, independent of Isaac Sim Disparity Noise. "
                    "Turn off to publish points without this additional Gaussian displacement."
                )
                ui.Label("Enable additional Gaussian error")
            with ui.HStack(height=26, spacing=6):
                ui.Label("Sigma model", width=130)
                collection = ui.RadioCollection()
                self._axial_error_model_collection = collection
                for index, (model_name, title) in enumerate((("constant", "Constant"), ("distance", "Distance"))):
                    ui.RadioButton(text=title, radio_collection=collection,
                        clicked_fn=lambda mode=model_name: self._property_dispatch("scanner.axial_error_model", mode))
                collection.model.set_value(int(self._property_values.get("scanner.axial_error_model_index", 0)))
                self.property_models["scanner.axial_error_model_index"] = collection.model
            with ui.HStack(height=30, spacing=6):
                ui.Label("Constant sigma (m)", width=190)
                error_drag = ui.FloatDrag(
                    model=axial_error_model, min=0.0, max=0.1, step=0.0001, format="%.4f",
                )
                self._axial_error_constant_control = error_drag
                error_drag.tooltip = (
                    "Zero-mean Gaussian 1-sigma displacement along each DepthSensor or LiDAR ray."
                )
            distance_controls = []
            for field, label, fallback in (
                ("axial_error_intercept_m", "b: intercept at 0 m (m)", 0.0100),
                ("axial_error_slope_m_per_m", "a: per +1 m (m/m)", 0.0),
            ):
                key = f"scanner.{field}"
                model = ui.SimpleFloatModel(float(self._property_values.get(key, fallback)))
                self.property_models[key] = model
                with ui.HStack(height=30, spacing=6):
                    ui.Label(label, width=190)
                    control = ui.FloatDrag(model=model, min=-0.1 if field == "axial_error_intercept_m" else 0.0,
                                           max=0.1, step=0.0001, format="%.4f")
                    control.tooltip = "Population standard deviation, not a mean/bias. x is the sensor-to-point range before added noise."
                    distance_controls.append(control)
                self._subscribe(model, key, lambda value: value.as_float)
            self._axial_error_distance_controls = tuple(distance_controls)
            ui.Label("Distance: sigma(x) = max(0, a * x + b)\nx = sensor-to-point range. Negative sigma becomes 0.",
                     height=44, word_wrap=True, style={"color": 0xFFB7BCC5})
            enabled = bool(self._property_values.get("scanner.axial_error_enabled", False))
            distance_mode = self._property_values.get("scanner.axial_error_model", "constant") == "distance"
            error_drag.enabled = enabled and not distance_mode
            for control in distance_controls:
                control.enabled = enabled and distance_mode
        self._subscribe(
            axial_error_enabled_model,
            "scanner.axial_error_enabled",
            lambda model: model.as_bool,
        )
        self._subscribe(
            axial_error_model,
            "scanner.axial_error_sigma_m",
            lambda model: model.as_float,
        )
        ui.Spacer()

    def _build_sensor_material_combo(self, field: str, label: str) -> None:
        options = tuple(self._property_values.get(f"sensor.material_{field}_options", ()))
        with ui.HStack(height=30, spacing=6):
            ui.Label(label, width=72)
            if not options:
                ui.Label("No material values available")
                return
            selected = int(self._property_values.get(f"sensor.material_{field}_index", 0))
            selected = max(0, min(len(options) - 1, selected))
            combo = ui.ComboBox(selected, *options, height=30)
            combo.tooltip = (
                "Non-visual sensor-response material applied to every designed target."
            )
            value_model_getter = getattr(combo.model, "get_item_value_model", None)
            if value_model_getter is None:
                value_model_getter = combo.model.get_value_model
            try:
                value_model = value_model_getter()
            except TypeError:
                value_model = value_model_getter(None)
            self.property_models[f"sensor.material_{field}_index"] = value_model
            callback = (
                lambda *args, current=field, model=combo.model, choices=options:
                self._on_sensor_material_changed(current, model, choices)
            )
            for method_name in (
                "add_item_changed_fn", "subscribe_item_changed_fn", "subscribe_value_changed_fn",
            ):
                method = getattr(combo.model, method_name, None)
                if method is not None:
                    self._model_subscriptions.append(method(callback))
                    break

    def _on_sensor_material_changed(self, field: str, model, options) -> None:
        if self._syncing_properties:
            return
        index = max(0, min(len(options) - 1, self._combo_index(model)))
        self._property_dispatch("scanner.sensor_material", {
            "field": field,
            "value": options[index],
        })

    def _on_sensor_profile_changed(self, model) -> None:
        if self._syncing_properties or not self._sensor_profile_entries:
            return
        index = max(0, min(len(self._sensor_profile_entries) - 1, self._combo_index(model)))
        applied = int(self._property_values.get("sensor.profile_index", 0))
        self._pending_sensor_profile_index = None if index == applied else index
        self._show_sensor_profile_description(index)
        self._update_sensor_profile_apply_button()

    def _apply_sensor_profile(self):
        index = self._pending_sensor_profile_index
        if (
            self._sensor_profile_apply_inflight
            or index is None
            or not self._sensor_profile_entries
            or not self._sensor_profile_available(index)
        ):
            return False
        sensor_kind, profile_id, _label = self._sensor_profile_entries[index][:3]
        self._sensor_profile_apply_inflight = True
        self._update_sensor_profile_apply_button()
        result = self._property_dispatch("scanner.select_profile", {
            "sensor_kind": sensor_kind,
            "profile_id": profile_id,
        })
        if isinstance(result, asyncio.Future):
            result.add_done_callback(self._sensor_profile_apply_done)
        else:
            self._finish_sensor_profile_apply(bool(result))
        return result

    def _sensor_profile_apply_done(self, future) -> None:
        try:
            success = bool(future.result())
        except Exception as error:
            carb.log_error(f"Point-cloud sensor Apply failed: {error}")
            success = False
        self._finish_sensor_profile_apply(success)

    def _finish_sensor_profile_apply(self, success: bool) -> None:
        self._sensor_profile_apply_inflight = False
        applied = int(self._property_values.get("sensor.profile_index", 0))
        pending = self._pending_sensor_profile_index
        if not success or pending == applied:
            self._pending_sensor_profile_index = None
        display = applied if self._pending_sensor_profile_index is None else pending
        if self._sensor_profile_combo_model is not None:
            self._syncing_properties = True
            try:
                value_model_getter = getattr(
                    self._sensor_profile_combo_model, "get_item_value_model", None,
                )
                if value_model_getter is None:
                    value_model_getter = self._sensor_profile_combo_model.get_value_model
                try:
                    value_model = value_model_getter()
                except TypeError:
                    value_model = value_model_getter(None)
                value_model.set_value(display)
            finally:
                self._syncing_properties = False
        self._show_sensor_profile_description(display)
        self._update_sensor_profile_apply_button()

    def _update_sensor_profile_apply_button(self) -> None:
        button = self.property_buttons.get("sensor_profile_apply")
        if button is None:
            return
        pending = self._pending_sensor_profile_index
        available = pending is None or self._sensor_profile_available(pending)
        button.enabled = (
            pending is not None and available and not self._sensor_profile_apply_inflight
        )
        if self._sensor_profile_apply_inflight:
            button.tooltip = "Applying the selected point-cloud sensor..."
        elif not available:
            entry = self._sensor_profile_entries[pending]
            button.tooltip = (
                str(entry[5]) if len(entry) > 5 and entry[5]
                else "This point-cloud sensor profile is unavailable."
            )
        else:
            button.tooltip = (
                "Create and warm the selected sensor and apply its recommended target material."
            )

    def _sensor_profile_available(self, index: int) -> bool:
        entry = self._sensor_profile_entries[index]
        return bool(entry[4]) if len(entry) > 4 else True

    def _show_sensor_profile_description(self, index: int) -> None:
        if not self._sensor_profile_entries:
            return
        index = max(0, min(len(self._sensor_profile_entries) - 1, int(index)))
        values = dict(self._sensor_profile_entries[index][3])
        for name, label in self._sensor_profile_description_labels.items():
            label.text = str(values.get(name, "N/A"))

    def _startup_reset_button(self, scope: str) -> None:
        ui.Button("Reset to Startup", height=26,
                  clicked_fn=lambda: self._property_dispatch("reset.startup", scope))

    def _build_graphics_property(self, _section=None) -> None:
        ui.Spacer(height=8)
        ruler_frame = ui.CollapsableFrame("Scene Ruler", height=0, collapsed=True)
        self._graphics_collapsible_frames["Scene Ruler"] = ruler_frame
        with ruler_frame:
            with BorderedGroupView("Background Opacity (0 = transparent, 1 = opaque)", row_heights=(26, 26)):
                for name, label, default in (("ruler_main_opacity", "Main View", 0.8),
                                             ("ruler_pip_opacity", "PIP", 0.35)):
                    key = f"graphics.{name}"
                    model = ui.SimpleFloatModel(float(self._property_values.get(key, default)))
                    self.property_models[key] = model
                    with ui.HStack(height=26, spacing=6):
                        ui.Label(label, width=96)
                        ui.FloatSlider(model=model, min=0.0, max=1.0, step=0.01)
                    self._subscribe(model, key, lambda value: value.as_float)
        for mode in ("Plane", "Sphere", "Cylinder", "Cone", "Torus"):
            frame = ui.CollapsableFrame(f"{mode} Mesh", height=0, collapsed=True)
            self._graphics_collapsible_frames[mode] = frame
            with frame:
                with ui.VStack(height=0, spacing=4):
                    with BorderedGroupView("Original", row_heights=(26, 26, 26)):
                        key = f"object.{mode}.display_subdivision"
                        model = ui.SimpleIntModel(int(self._property_values.get(key, 96)))
                        self.property_models[key] = model
                        with ui.HStack(height=26, spacing=6):
                            ui.Label("Subdivision", width=112)
                            ui.IntDrag(model=model, min=3, max=256, step=1)
                        self._subscribe_callback(
                            model,
                            lambda _model, current=mode: self._dispatch_object_models(
                                current, "display_subdivision",
                            ),
                        )
                        self._build_object_color_row(mode, "face_color", "Face RGBA")
                        self._build_object_color_row(mode, "wire_color", "Outline RGBA")
                    self._build_geometry_mesh_appearance(mode, "detected", "Detected")
                    self._build_geometry_mesh_appearance(mode, "captured", "Captured")

        point_size = ui.SimpleFloatModel(
            float(self._property_values.get("settings.point_size", 2.5))
        )
        inlier_size = ui.SimpleFloatModel(
            float(self._property_values.get("settings.inlier_size", 2.5))
        )
        self.property_models["settings.point_size"] = point_size
        self.property_models["settings.inlier_size"] = inlier_size
        point_frame = ui.CollapsableFrame("Point Clouds", height=0, collapsed=True)
        self._graphics_collapsible_frames["Point Clouds"] = point_frame
        with point_frame:
            with BorderedGroupView("Point Rendering", row_heights=(26, 26, 26, 26, 26, 26)):
                self._settings_float_row("Outlier Size", point_size, minimum=1.0, maximum=16.0)
                point_color_models = []
                for prefix, label, value_key in (
                    ("settings.outlier_color", "Outlier RGBA", "outlier_color_rgba"),
                    ("settings.inlier_negative", "Inlier - RGBA", "inlier_negative_rgba"),
                    ("settings.inlier_zero", "Inlier 0 RGBA", "inlier_zero_rgba"),
                    ("settings.inlier_positive", "Inlier + RGBA", "inlier_positive_rgba"),
                ):
                    values = tuple(
                        float(self._property_values.get(f"{prefix}.{component}", 1.0))
                        for component in "rgba"
                    )
                    models = self._build_rgba_row(prefix, label, values, label_width=112)
                    point_color_models.append((prefix, value_key, models))
                self._settings_float_row("Inlier Size", inlier_size, minimum=1.0, maximum=16.0)
        self._subscribe(point_size, "settings.point_size", lambda model: model.as_float)
        self._subscribe(inlier_size, "settings.inlier_size", lambda model: model.as_float)
        for prefix, value_key, models in point_color_models:
            for model in models:
                self._subscribe_callback(
                    model,
                    lambda _model, current=prefix, action=value_key: self._dispatch_rgba_models(
                        current, f"settings.{action}",
                    ),
                )
        ui.Spacer()

    def _build_geometry_mesh_appearance(self, mode: str, kind: str, title: str) -> None:
        prefix = f"graphics.{mode}.{kind}"
        subdivision = ui.SimpleIntModel(
            int(self._property_values.get(f"{prefix}.subdivision", 96))
        )
        self.property_models[f"{prefix}.subdivision"] = subdivision
        with BorderedGroupView(title, row_heights=(26, 26, 26)):
            with ui.HStack(height=26, spacing=6):
                ui.Label("Subdivision", width=112)
                ui.IntDrag(model=subdivision, min=3, max=256, step=1)
            face = self._build_rgba_row(
                f"{prefix}.face", "Face RGBA",
                tuple(float(self._property_values.get(f"{prefix}.face.{c}", 1.0)) for c in "rgba"),
            )
            wire = self._build_rgba_row(
                f"{prefix}.wire", "Outline RGBA",
                tuple(float(self._property_values.get(f"{prefix}.wire.{c}", 1.0)) for c in "rgba"),
            )
        for model in (subdivision, *face, *wire):
            self._subscribe_callback(
                model,
                lambda _model, current_mode=mode, current_kind=kind:
                    self._dispatch_geometry_mesh_appearance(current_mode, current_kind),
            )

    def _dispatch_geometry_mesh_appearance(self, mode: str, kind: str) -> None:
        if self._syncing_properties:
            return
        prefix = f"graphics.{mode}.{kind}"
        subdivision_model = self.property_models[f"{prefix}.subdivision"]
        subdivision = max(3, min(256, subdivision_model.as_int))
        if subdivision_model.as_int != subdivision:
            self._syncing_properties = True
            try:
                subdivision_model.set_value(subdivision)
            finally:
                self._syncing_properties = False
        self._property_dispatch("graphics.mesh_appearance", {
            "mode": mode,
            "kind": kind,
            "subdivision": subdivision,
            "face_rgba": tuple(
                self.property_models[f"{prefix}.face.{component}"].as_float
                for component in "rgba"
            ),
            "wire_rgba": tuple(
                self.property_models[f"{prefix}.wire.{component}"].as_float
                for component in "rgba"
            ),
        })

    def _build_settings_property(self, section) -> None:
        with ui.CollapsableFrame("Settings Defaults", height=0, collapsed=False):
            with ui.VStack(height=0, spacing=4):
                ui.Button("Restore Settings Defaults", height=26,
                          clicked_fn=lambda: self._property_dispatch("settings.restore_defaults", None),
                          tooltip="Restore packaged camera, object, sensor, FindSurface and Settings defaults. Keep captured geometries. Undo is available.")
                ui.Button("Reset Window Layout", height=26,
                          clicked_fn=lambda: self._property_dispatch("settings.reset_window_layout", None),
                          tooltip="Restore the default dock positions, tab order, and property window sizes; forget the saved layout.")
        self.property_models["settings.probe_radius_px"] = ui.SimpleFloatModel(
            float(self._property_values.get("settings.probe_radius_px", 6.0))
        )
        self.property_models["settings.pick_nearest"] = ui.SimpleBoolModel(
            bool(self._property_values.get("settings.pick_nearest", True))
        )
        self.property_models["settings.sticky_auto_preview"] = ui.SimpleBoolModel(
            bool(self._property_values.get("settings.sticky_auto_preview", True))
        )
        self.property_models["settings.sticky_timeout_ms"] = ui.SimpleIntModel(
            int(self._property_values.get("settings.sticky_timeout_ms", 500))
        )
        self.property_models["settings.detection_average_window"] = ui.SimpleIntModel(
            int(self._property_values.get("settings.detection_average_window", 10)), min=1, max=120,
        )
        self.property_models["settings.minimum_geometry_length_m"] = ui.SimpleFloatModel(
            float(self._property_values.get("settings.minimum_geometry_length_m", 0.01))
        )
        self.property_models["settings.surface_side_fixed_scale_m"] = ui.SimpleFloatModel(
            float(self._property_values.get("settings.surface_side_fixed_scale_m", 0.01))
        )
        self.property_models["settings.surface_side_sigma_multiplier"] = ui.SimpleFloatModel(
            float(self._property_values.get("settings.surface_side_sigma_multiplier", 3.0))
        )
        self.property_models["settings.depth_disparity_noise"] = ui.SimpleBoolModel(
            bool(self._property_values.get("settings.depth_disparity_noise", True))
        )
        self.property_models["settings.depth_disparity_noise_mean_px"] = ui.SimpleFloatModel(
            float(self._property_values.get("settings.depth_disparity_noise_mean_px", 0.25))
        )
        self.property_models["settings.depth_disparity_noise_sigma_px"] = ui.SimpleFloatModel(
            float(self._property_values.get("settings.depth_disparity_noise_sigma_px", 0.25))
        )
        self.property_models["settings.depth_outlier_removal"] = ui.SimpleBoolModel(
            bool(self._property_values.get("settings.depth_outlier_removal", True))
        )
        for name in (
            "smart_convert_cone_to_cylinder",
            "smart_convert_torus_to_sphere",
            "smart_convert_torus_to_cylinder",
        ):
            self.property_models[f"settings.{name}"] = ui.SimpleBoolModel(
                bool(self._property_values.get(f"settings.{name}", True))
            )
        self.property_models["pip_width"] = ui.SimpleFloatModel(float(self._property_values.get("pip_width", 0.32)))

        with BorderedGroupView(
            "Isaac Sim Depth Sensor (Temporary)",
            row_heights=(30, 26, 30, 30, 26, 30, 34),
        ):
            with ui.HStack(height=30, spacing=6):
                ui.Label("Anti-aliasing", width=130)
                selected = int(self._property_values.get("settings.anti_aliasing_mode", 1))
                selected = max(0, min(len(ANTI_ALIASING_LABELS) - 1, selected))
                combo = ui.ComboBox(selected, *ANTI_ALIASING_LABELS, height=30)
                self._anti_aliasing_combo_model = combo.model
                value_model_getter = getattr(combo.model, "get_item_value_model", None)
                if value_model_getter is None:
                    value_model_getter = combo.model.get_value_model
                try:
                    value_model = value_model_getter()
                except TypeError:
                    value_model = value_model_getter(None)
                self.property_models["settings.anti_aliasing_mode"] = value_model
                callback = lambda *args, model=combo.model: self._on_anti_aliasing_changed(model)
                for method_name in (
                    "add_item_changed_fn", "subscribe_item_changed_fn", "subscribe_value_changed_fn",
                ):
                    method = getattr(combo.model, method_name, None)
                    if method is not None:
                        self._model_subscriptions.append(method(callback))
                        break
            with ui.HStack(height=26, spacing=6):
                checkbox = ui.CheckBox(
                    model=self.property_models["settings.depth_disparity_noise"], width=22,
                )
                checkbox.tooltip = (
                    "Isaac Sim SingleViewDepthCameraSensor disparity-space noise. "
                    "The next live or one-shot sensor frame uses the new value."
                )
                ui.Label("Disparity Noise (Isaac Sim)")
            with ui.HStack(height=30, spacing=6):
                ui.Label("Disparity Noise Mean", width=150)
                mean_drag = ui.FloatDrag(
                    model=self.property_models["settings.depth_disparity_noise_mean_px"],
                    min=-10.0, max=10.0, step=0.01, format="%.2f",
                )
                mean_drag.tooltip = (
                    "Isaac Sim noiseMean in pixels. Higher values reduce depth resolution."
                )
            with ui.HStack(height=30, spacing=6):
                ui.Label("Disparity Noise Sigma", width=150)
                sigma_drag = ui.FloatDrag(
                    model=self.property_models["settings.depth_disparity_noise_sigma_px"],
                    min=-10.0, max=10.0, step=0.01, format="%.2f",
                )
                sigma_drag.tooltip = (
                    "Isaac Sim noiseSigma in pixels. Higher magnitude widens disparity variation."
                )
            with ui.HStack(height=26, spacing=6):
                ui.CheckBox(model=self.property_models["settings.depth_outlier_removal"], width=22)
                ui.Label("Outlier Removal")
            with ui.HStack(height=30, spacing=6):
                ui.Label("Frozen cloud", width=130)
                export_enabled = bool(self._property_values.get("settings.export_xyz_enabled", False))
                button = ui.Button(
                    "Export .xyz",
                    enabled=export_enabled,
                    clicked_fn=lambda: self._property_dispatch("settings.export_xyz", None),
                )
                button.tooltip = str(self._property_values.get(
                    "settings.export_xyz_tooltip", "Freeze the sensor before exporting.",
                ))
                self.property_buttons["settings.export_xyz"] = button
            label = ui.Label(
                str(self._property_values.get("settings.export_xyz_status", "No export yet")),
                word_wrap=True,
                height=34,
                style={"color": 0xFFB7BCC5},
            )
            self.property_labels["settings.export_xyz_status"] = label
        ui.Spacer(height=8)
        with BorderedGroupView("FindSurface Timing", row_heights=(26, 26, 38)):
            model = self.property_models["settings.detection_average_window"]
            with ui.HStack(height=26, spacing=4):
                ui.Label("Call window", width=140)
                ui.Button("-", width=26, clicked_fn=lambda m=model: m.set_value(max(1, m.as_int - 1)))
                ui.IntField(model=model, width=52)
                ui.Button("+", width=26, clicked_fn=lambda m=model: m.set_value(min(120, m.as_int + 1)))
                ui.Label("calls (1-120)", width=100)
            with ui.HStack(height=26, spacing=4):
                ui.Label("Quick set", width=140)
                for count in (5, 10, 15, 30, 60, 120):
                    ui.Button(str(count), width=32, clicked_fn=lambda n=count, m=model: m.set_value(n))
            ui.Label("Mean of successes in the last N SDK calls. Failures advance the window; idle holds it.",
                     height=38, word_wrap=True, style={"color": 0xFFB7BCC5})
        with BorderedGroupView("Picking", row_heights=(26, 26, 26, 26)):
            self._settings_float_row(
                "Probe px", self.property_models["settings.probe_radius_px"], minimum=1.0, maximum=80.0,
            )
            with ui.HStack(height=26, spacing=6):
                ui.CheckBox(model=self.property_models["settings.pick_nearest"], width=22)
                ui.Label("Pick Nearest On Probe Miss")
            with ui.HStack(height=26, spacing=6):
                ui.CheckBox(model=self.property_models["settings.sticky_auto_preview"], width=22)
                ui.Label("Sticky Auto Preview")
            with ui.HStack(height=26, spacing=6):
                ui.Label("Sticky Preview Timeout", width=150)
                ui.IntDrag(
                    model=self.property_models["settings.sticky_timeout_ms"],
                    min=100, max=60000, step=1, width=60,
                )
                ui.Label("ms", width=24)
        with BorderedGroupView("Surface Side Heatmap", row_heights=(30, 30, 30)):
            with ui.HStack(height=30, spacing=6):
                ui.Label("Scale", width=96)
                selected = int(self._property_values.get("settings.surface_side_scale_mode", 0))
                selected = max(0, min(len(SURFACE_SIDE_SCALE_LABELS) - 1, selected))
                combo = ui.ComboBox(selected, *SURFACE_SIDE_SCALE_LABELS, height=30)
                value_model_getter = getattr(combo.model, "get_item_value_model", None)
                if value_model_getter is None:
                    value_model_getter = combo.model.get_value_model
                try:
                    value_model = value_model_getter()
                except TypeError:
                    value_model = value_model_getter(None)
                self.property_models["settings.surface_side_scale_mode"] = value_model
                callback = lambda *args, model=combo.model: self._on_surface_side_scale_changed(model)
                for method_name in (
                    "add_item_changed_fn", "subscribe_item_changed_fn", "subscribe_value_changed_fn",
                ):
                    method = getattr(combo.model, method_name, None)
                    if method is not None:
                        self._model_subscriptions.append(method(callback))
                        break
            with ui.HStack(height=30, spacing=6):
                ui.Label("Fixed range", width=96)
                ui.Label("±", width=12)
                ui.FloatDrag(
                    model=self.property_models["settings.surface_side_fixed_scale_m"],
                    min=0.000001,
                    max=100.0,
                    step=0.0001,
                    format="%.4f",
                )
                ui.Label("m", width=24)
            with ui.HStack(height=30, spacing=6):
                ui.Label("Sigma range", width=96)
                ui.Label("±", width=12)
                ui.FloatDrag(
                    model=self.property_models["settings.surface_side_sigma_multiplier"],
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    format="%.1f sigma",
                )
        with BorderedGroupView("Smart Conversion (Temporary)", row_heights=(26, 26, 26)):
            for name, label in (
                ("smart_convert_cone_to_cylinder", "Cone -> Cylinder"),
                ("smart_convert_torus_to_sphere", "Torus -> Sphere"),
                ("smart_convert_torus_to_cylinder", "Torus -> Cylinder"),
            ):
                with ui.HStack(height=26, spacing=6):
                    ui.CheckBox(model=self.property_models[f"settings.{name}"], width=22)
                    ui.Label(label)
        with BorderedGroupView("Geometry Editing", row_heights=(26,)):
            with ui.HStack(height=26, spacing=6):
                ui.Label("Minimum length (m)", width=150)
                ui.FloatDrag(
                    model=self.property_models["settings.minimum_geometry_length_m"],
                    min=0.0001, max=100.0, step=0.001, format="%.4f",
                )
        with BorderedGroupView("PIP", row_heights=(26, 30)):
            self._settings_float_row(
                "Width", self.property_models["pip_width"], minimum=0.15, maximum=0.6,
            )
            with ui.HStack(height=30, spacing=6):
                ui.Button("PIP on/off", clicked_fn=lambda: self._property_dispatch("settings.pip_toggle", None))
        ui.Spacer()

        self._subscribe(
            self.property_models["settings.probe_radius_px"],
            "settings.probe_radius_px", lambda model: model.as_float,
        )
        self._subscribe(
            self.property_models["settings.pick_nearest"],
            "settings.pick_nearest", lambda model: model.as_bool,
        )
        self._subscribe(
            self.property_models["settings.sticky_auto_preview"],
            "settings.sticky_auto_preview", lambda model: model.as_bool,
        )
        self._subscribe(
            self.property_models["settings.sticky_timeout_ms"],
            "settings.sticky_timeout_ms", lambda model: model.as_int,
        )
        self._subscribe(
            self.property_models["settings.minimum_geometry_length_m"],
            "settings.minimum_geometry_length_m", lambda model: model.as_float,
        )
        self._subscribe(self.property_models["settings.detection_average_window"],
                        "settings.detection_average_window", lambda model: model.as_int)
        self._subscribe(
            self.property_models["settings.surface_side_fixed_scale_m"],
            "settings.surface_side_fixed_scale_m", lambda model: model.as_float,
        )
        self._subscribe(
            self.property_models["settings.surface_side_sigma_multiplier"],
            "settings.surface_side_sigma_multiplier", lambda model: model.as_float,
        )
        self._subscribe(
            self.property_models["settings.depth_disparity_noise"],
            "settings.depth_disparity_noise", lambda model: model.as_bool,
        )
        self._subscribe(
            self.property_models["settings.depth_disparity_noise_mean_px"],
            "settings.depth_disparity_noise_mean_px", lambda model: model.as_float,
        )
        self._subscribe(
            self.property_models["settings.depth_disparity_noise_sigma_px"],
            "settings.depth_disparity_noise_sigma_px", lambda model: model.as_float,
        )
        self._subscribe(
            self.property_models["settings.depth_outlier_removal"],
            "settings.depth_outlier_removal", lambda model: model.as_bool,
        )
        for name in (
            "smart_convert_cone_to_cylinder",
            "smart_convert_torus_to_sphere",
            "smart_convert_torus_to_cylinder",
        ):
            self._subscribe(
                self.property_models[f"settings.{name}"],
                f"settings.{name}", lambda model: model.as_bool,
            )
        self._subscribe(
            self.property_models["pip_width"],
            "settings.pip_width", lambda model: model.as_float,
        )

    def _on_anti_aliasing_changed(self, model) -> None:
        if self._syncing_properties:
            return
        self._property_dispatch("settings.anti_aliasing_mode", self._combo_index(model))

    def _on_surface_side_scale_changed(self, model) -> None:
        if self._syncing_properties:
            return
        self._property_dispatch("settings.surface_side_scale_mode", self._combo_index(model))

    @staticmethod
    def _settings_float_row(label, model, *, minimum, maximum) -> None:
        """Demo-1 Settings float row: 26 px high with a 72 px label."""

        with ui.HStack(height=26, spacing=6):
            ui.Label(label, width=72)
            ui.FloatDrag(model=model, min=minimum, max=maximum, step=0.01)

    def _build_compact_status_rows(self, section) -> None:
        for name, value in section.rows:
            with ui.HStack(height=26, spacing=6):
                ui.Label(name, width=96, style={"color": 0xFFB7BCC5})
                label = ui.Label(value, word_wrap=True)
                self.property_labels[(section.tab, name)] = label

    def _subscribe(self, model, action: str, converter) -> None:
        callback = lambda changed: None if self._syncing_properties else self._property_dispatch(action, converter(changed))
        self._subscribe_callback(model, callback)

    def _subscribe_callback(self, model, callback) -> None:
        if hasattr(model, "subscribe_value_changed_fn"):
            subscription = model.subscribe_value_changed_fn(callback)
        else:
            subscription = model.add_value_changed_fn(callback)
        self._model_subscriptions.append(subscription)

    def _property_dispatch(self, action: str, value):
        # Model synchronization is presentation, never a new user transaction.
        # Direct subscriptions (not just _subscribe) also pass this boundary.
        if self._syncing_properties:
            return None
        result = self._property_fn(action, value)
        if asyncio.iscoroutine(result):
            return asyncio.ensure_future(result)
        return result

    def sync_property_values(self, values: dict[str, object]) -> None:
        values = normalize_geometry_card_values(values)
        # Kit refreshes every frame. Do not overwrite an in-progress text/drag edit
        # until the backing application state has actually changed.
        pending_model_keys = getattr(self, "_pending_property_model_keys", set())
        self._pending_property_model_keys = pending_model_keys
        pending_model_keys.intersection_update(self.property_models)
        ready_model_keys = pending_model_keys - self._object_editing_keys
        if values == self._property_values and not ready_model_keys:
            return
        values = dict(values)
        previous_values = self._property_values
        previous_geometry_entries = tuple(self._property_values.get("geometry.entries", ()))
        geometry_entries = tuple(values.get("geometry.entries", ()))
        pad_state_keys = set(self._pad_axes.values())
        pad_state_keys.update(
            f"{name}_bounds" for name in self._pad_axes.values() if name in _FS_RANGE_DEFAULTS
        )
        pad_state_changed = any(
            values.get(key) != self._property_values.get(key) for key in pad_state_keys
        ) or bool(ready_model_keys & pad_state_keys)
        geometry_structure_changed = self._geometry_structure_signature(
            geometry_entries
        ) != self._geometry_structure_signature(previous_geometry_entries)
        self._property_values = values
        incoming_profile_index = int(values.get("sensor.profile_index", 0))
        if self._pending_sensor_profile_index == incoming_profile_index:
            self._pending_sensor_profile_index = None
        display_profile_index = (
            incoming_profile_index
            if self._pending_sensor_profile_index is None
            else self._pending_sensor_profile_index
        )
        self._syncing_properties = True
        try:
            for key, model in self.property_models.items():
                if key not in values:
                    continue
                changed = values[key] != previous_values.get(key) or key in pending_model_keys
                if key in self._object_editing_keys:
                    if changed:
                        pending_model_keys.add(key)
                    continue
                if changed:
                    model.set_value(
                        display_profile_index if key == "sensor.profile_index" else values[key]
                    )
                    pending_model_keys.discard(key)
        finally:
            self._syncing_properties = False
        error_enabled = bool(values.get("scanner.axial_error_enabled", False))
        distance_error = values.get("scanner.axial_error_model", "constant") == "distance"
        constant_control = getattr(self, "_axial_error_constant_control", None)
        if constant_control is not None:
            constant_control.enabled = error_enabled and not distance_error
        for control in getattr(self, "_axial_error_distance_controls", ()):
            control.enabled = error_enabled and distance_error
        if pad_state_changed:
            self._sync_parameter_pad_from_models(ensure_visual_refresh=pad_state_changed)
        if (
            "sensor.profile_index" in values
            and values["sensor.profile_index"] != previous_values.get("sensor.profile_index")
        ):
            self._show_sensor_profile_description(display_profile_index)
            self._update_sensor_profile_apply_button()
        for mode in ("Plane", "Sphere", "Cylinder", "Cone", "Torus"):
            button = self.property_buttons.get(f"object.{mode}")
            if button is not None:
                enabled = bool(values.get(f"object.{mode}.enabled", False))
                _set_widget_property(button, "text", "Enabled" if enabled else "Disabled")
                _set_widget_property(button, "checked", enabled)
        export_button = self.property_buttons.get("settings.export_xyz")
        if export_button is not None:
            _set_widget_property(export_button, "enabled", bool(values.get("settings.export_xyz_enabled", False)))
            _set_widget_property(export_button, "tooltip", str(values.get(
                "settings.export_xyz_tooltip", "Freeze the sensor before exporting.",
            )))
        export_label = self.property_labels.get("settings.export_xyz_status")
        if export_label is not None:
            _set_widget_property(export_label, "text", str(values.get("settings.export_xyz_status", "No export yet")))
        recommendation = self.property_labels.get("sensor.material_recommendation")
        if recommendation is not None:
            _set_widget_property(recommendation, "text", str(values.get("sensor.material_recommendation", "N/A")))
        if geometry_structure_changed:
            self._queue_geometry_property_refresh()
        elif geometry_entries != previous_geometry_entries:
            # Visibility, selection and heatmap-basis changes do not alter card
            # layout. Update their controls in place so Kit never destroys the
            # scrolling subtree merely because a button was clicked.
            self._sync_geometry_card_controls(geometry_entries)

    def dock(self, render_window):
        """Defer initial docking until Kit has registered every workspace window."""
        if self._docked or self._dock_task is not None or render_window is None:
            return self._dock_task
        self._render_window = render_window
        self._dock_task = asyncio.ensure_future(self._dock_after_updates(render_window))
        return self._dock_task

    async def _dock_after_updates(self, render_window) -> None:
        state = self.dock_state()
        try:
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()
            # This extension can load before the application reports ready. Retry the same
            # first-demo layout until the workspace accepts it, but never loop indefinitely.
            for attempt in range(1, 21):
                self._dock_attempts = attempt
                dock_space = ui.Workspace.get_window("DockSpace")
                if dock_space is None:
                    await omni.kit.app.get_app().next_update_async()
                    continue
                render_window.visible = True
                if not bool(getattr(render_window, "docked", False)):
                    render_window.dock_in(dock_space, ui.DockPosition.SAME)
                    for _ in range(2):
                        await omni.kit.app.get_app().next_update_async()
                if not bool(getattr(render_window, "docked", False)):
                    continue
                self._dock_now(render_window)
                for _ in range(2):
                    await omni.kit.app.get_app().next_update_async()
                self._hide_fixed_tab_bars(render_window)
                state = self.dock_state()
                if all(state.values()):
                    self._docked = True
                    self._dock_error = None
                    return
            raise RuntimeError(f"workspace rejected initial docking after 20 attempts: {state}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._dock_error = f"{type(error).__name__}: {error}"
            carb.log_error(f"FindSurface initial docking failed: {self._dock_error}")
        finally:
            self._dock_task = None

    def _dock_now(self, render_window) -> None:
        self.toolbar_window.visible = True
        self.status_window.visible = True
        for window in self.property_windows.values():
            window.visible = True
        # Match the first demo: fixed chrome around the viewport, properties tabbed at right.
        self.toolbar_window.dock_in(render_window, ui.DockPosition.TOP, 0.08)
        self.status_window.dock_in(render_window, ui.DockPosition.BOTTOM, 0.035)
        self.status_window.height = ui.Pixel(STATUS_BAR_HEIGHT)
        preference = self._layout_preference or {}
        by_title = {tab.value: window for tab, window in self.property_windows.items()}
        ordered = tuple(by_title[title] for title in preference.get("tab_order", by_title))
        first = ordered[0]
        ratio = (preference["side_ratio"] if "side_ratio" in preference
                 else default_property_dock_ratio(float(render_window.width)))
        first.dock_in(render_window, ui.DockPosition.RIGHT, ratio)
        for window in ordered[1:]:
            window.visible = True
            window.dock_in(first, ui.DockPosition.SAME, 0.5)
        self._hide_fixed_tab_bars(render_window)
        by_title.get(preference.get("active_tab"), first).focus()

    async def restore_floating_layout(self) -> None:
        floating = (self._layout_preference or {}).get("floating", {})
        for tab, window in self.property_windows.items():
            bounds = floating.get(tab.value)
            if bounds is None:
                continue
            window.undock()
            await omni.kit.app.get_app().next_update_async()
            x, y, width, height = fit_floating_bounds(bounds, self.workspace_content_size())
            window.width, window.height = width, height
            window.position_x, window.position_y = x, y
            window.visible = True
        # Let Kit settle undocking/sizing before correcting against the actual resulting size.
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()
        for tab, window in self.property_windows.items():
            if tab.value in floating:
                x, y, width, height = fit_floating_bounds(
                    [window.position_x, window.position_y, window.width, window.height],
                    self.workspace_content_size(),
                )
                window.width, window.height = width, height
                window.position_x, window.position_y = x, y

    @staticmethod
    def workspace_content_size() -> tuple[float, float]:
        # Window.position/width/height are points, not physical OS pixels. These
        # APIs return the same client-coordinate space, including the current UI DPI.
        return (float(ui.Workspace.get_main_window_width()),
                float(ui.Workspace.get_main_window_height()))

    def capture_layout(self, render_window) -> dict:
        windows = tuple(self.property_windows.values())
        docked = [window for window in windows if window.docked]
        reference = next((window for window in docked if window.is_selected_in_dock()), None)
        reference = reference or (docked[0] if docked else None)
        previous = self._layout_preference or {}
        ratio = previous.get("side_ratio", SIDE_PANEL_DOCK_RATIO)
        if reference is not None:
            total = float(render_window.width) + float(reference.width)
            if total > 0:
                measured = float(reference.width) / total
                if 0.20 <= measured <= 0.50:
                    ratio = measured
        original_order = {tab.value: index for index, tab in enumerate(self.property_windows)}
        order = sorted(
            self.property_windows,
            key=lambda tab: (
                not self.property_windows[tab].docked,
                int(self.property_windows[tab].dock_order or 0),
                original_order[tab.value],
            ),
        )
        active = next((tab.value for tab, window in self.property_windows.items()
                       if window.docked and window.is_selected_in_dock()), None)
        floating = {
            tab.value: [float(window.position_x), float(window.position_y),
                        float(window.width), float(window.height)]
            for tab, window in self.property_windows.items() if not window.docked
        }
        return fit_workspace_layout(
            {"side_ratio": ratio, "tab_order": [tab.value for tab in order],
             "active_tab": active, "floating": floating}, self.workspace_content_size(),
        )

    async def reset_docking(self, render_window) -> None:
        """Rebuild the product dock tree without Kit's global workspace clear."""
        self._docked = False
        self._layout_preference = None
        for _ in range(20):
            dock_space = ui.Workspace.get_window("DockSpace")
            if dock_space is None:
                await omni.kit.app.get_app().next_update_async()
                continue
            render_window.visible = True
            render_window.dock_in(dock_space, ui.DockPosition.SAME)
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
            self._dock_now(render_window)
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
            if all(self.dock_state().values()):
                self._docked = True
                self._dock_error = None
                return
        raise RuntimeError(f"Could not restore product docking: {self.dock_state()}")

    def _hide_fixed_tab_bars(self, render_window) -> None:
        self.toolbar_window.dock_tab_bar_visible = False
        self.status_window.dock_tab_bar_visible = False
        render_window.dock_tab_bar_visible = False

    def dock_state(self) -> dict[str, bool]:
        result = {
            "render": bool(getattr(self._render_window, "docked", False)),
            "toolbar": _toolbar_is_docked(self.toolbar_window),
            "status": bool(getattr(self.status_window, "docked", False)),
        }
        result.update({tab.value: bool(getattr(window, "docked", False)) for tab, window in self.property_windows.items()})
        return result

    @property
    def dock_error(self) -> str | None:
        return self._dock_error

    @property
    def dock_attempts(self) -> int:
        return self._dock_attempts

    def refresh(self, snapshot: ShellSnapshot) -> None:
        self._last_snapshot = snapshot
        by_id = {item.command_id: item for item in snapshot.commands}
        for item in snapshot.commands:
            button = self.buttons.get(item.command_id)
            if button is not None:
                button.set_state(enabled=item.enabled, checked=item.checked, tooltip=item.tooltip)
        for key, command_id in (
            ("sensor_mode_frozen", CommandId.SENSOR_FROZEN),
            ("sensor_mode_fixed", CommandId.SENSOR_LIVE_FIXED),
            ("sensor_mode_follow", CommandId.SENSOR_LIVE_FOLLOW),
        ):
            button = self.property_buttons.get(key)
            item = by_id.get(command_id)
            if button is not None and item is not None:
                _set_widget_property(button, "enabled", item.enabled)
                _set_widget_property(button, "checked", bool(item.checked))
                _set_widget_property(button, "tooltip", item.tooltip)
        for section in snapshot.properties:
            for name, value in section.rows:
                label = self.property_labels.get((section.tab, name))
                if label is not None:
                    _set_widget_property(label, "text", value)
        overlays = snapshot.overlays
        if self._viewport_overlays_mounted:
            _set_widget_property(self.sensor_overlay, "text", overlays.sensor)
            _set_widget_property(self.rtx_overlay, "text", overlays.rtx)
            _set_widget_property(self.analysis_overlay, "text", f"{overlays.detection}\n{overlays.analysis}")
            self._refresh_result_overlay(overlays.result)
        _set_widget_property(self.status_label, "text", "Input blocked by modal operation" if snapshot.modal.input_blocked else "Ready")
        if self._viewport_overlays_mounted:
            self.sync_modal(snapshot)

    def sync_modal(self, snapshot: ShellSnapshot) -> None:
        if self._modal_overlay is None:
            return
        modal = snapshot.modal
        if not modal.visible:
            _set_widget_property(self._modal_overlay, "visible", False)
            return
        self._modal_generation = modal.generation
        _set_widget_property(self._modal_overlay, "visible", True)
        _set_widget_property(self._modal_title, "text", modal.title)
        _set_widget_property(self._modal_detail, "text", modal.detail)
        if self._modal_progress_model.as_float != modal.progress:
            self._modal_progress_model.set_value(modal.progress)
        _set_widget_property(self._modal_error, "text", modal.error or "")
        _set_widget_property(self._modal_cancel, "visible", modal.cancellable and modal.phase is ModalPhase.ACTIVE)
        _set_widget_property(self._modal_cancel, "enabled", modal.cancellable and modal.phase is ModalPhase.ACTIVE)
        _set_widget_property(self._modal_dismiss, "enabled", modal.phase is not ModalPhase.ACTIVE)

    def show_help(self, snapshot: ShellSnapshot) -> None:
        if self._help_window is None:
            self._help_window = ui.Window("Controls & Help", width=620, height=520)
            with self._help_window.frame:
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=8, style={"margin": 16}):
                        ui.Label("Shortcuts", height=28, style={"font_size": 18})
                        for shortcut, action in snapshot.help.shortcuts:
                            ui.Label(f"{shortcut:24}  {action}", height=22)
                        ui.Separator(height=6)
                        ui.Label(snapshot.help.capture_contract, word_wrap=True, height=66)
                        ui.Label(snapshot.help.camera_contract, word_wrap=True, height=44)
                        ui.Label(snapshot.help.exit_contract, word_wrap=True, height=44)
        self._help_window.visible = True
        self._help_window.set_top_modal()
        self._help_window.focus()

    def show_performance_warning(
        self,
        presentation: PerformanceWarningPresentation,
        *,
        keep_fn: Callable[[], None],
        switch_fn: Callable[[], None],
    ) -> None:
        self._destroy_performance_window()
        self._performance_window = ui.Window(
            presentation.title, width=680, height=330,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_COLLAPSE,
        )
        window = self._performance_window
        with window.frame:
            with ui.VStack(spacing=10, style={"margin": 16}):
                ui.Label(presentation.rates, word_wrap=True, height=52)
                ui.Label(presentation.point_status, word_wrap=True, height=38)
                ui.Label(presentation.sensor_cost_notice, word_wrap=True, height=48)
                ui.Label(presentation.transition_notice, word_wrap=True, height=48)
                with ui.HStack(height=34, spacing=8):
                    ui.Spacer()
                    ui.Button(presentation.keep_label, width=110, clicked_fn=lambda: self._choose_performance(keep_fn))
                    ui.Button(presentation.switch_label, width=160, clicked_fn=lambda: self._choose_performance(switch_fn))
        window.set_visibility_changed_fn(lambda visible: None if visible else self._choose_performance(keep_fn))
        window.set_top_modal()
        window.focus()

    def _dispatch(self, command_id: CommandId) -> None:
        result = self._command_fn(command_id)
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    def _dismiss_modal(self) -> None:
        if self._modal_dismiss_fn is not None:
            self._modal_dismiss_fn()
        if self._modal_overlay is not None:
            self._modal_overlay.visible = False

    def _cancel_modal(self) -> None:
        if self._modal_cancel_fn is not None:
            self._modal_cancel_fn()

    def _destroy_modal(self) -> None:
        if self._modal_overlay is not None:
            self._modal_overlay.visible = False

    def _choose_performance(self, callback) -> None:
        if self._performance_window is None:
            return
        self._destroy_performance_window()
        result = callback()
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    def _destroy_performance_window(self) -> None:
        window, self._performance_window = self._performance_window, None
        if window is not None:
            window.set_visibility_changed_fn(None)
            window.destroy()

    def close(self) -> None:
        self._pad_layout_subscription = None
        task, self._dock_task = self._dock_task, None
        if task is not None and not task.done():
            task.cancel()
        task, self._geometry_refresh_task = self._geometry_refresh_task, None
        if task is not None and not task.done():
            task.cancel()
        task, self._pad_refresh_task = self._pad_refresh_task, None
        if task is not None and not task.done():
            task.cancel()
        self._render_window = None
        self._destroy_modal()
        self._destroy_performance_window()
        window, self._help_window = self._help_window, None
        if window is not None:
            window.destroy()
        for window in self.property_windows.values():
            window.destroy()
        self.property_windows.clear()
        self.property_labels.clear()
        self.property_models.clear()
        self.property_buttons.clear()
        self._property_scrolling_frames.clear()
        self._property_scroll_y.clear()
        self._geometry_property_content_frame = None
        self._geometry_card_frames.clear()
        self._geometry_card_controls.clear()
        self._graphics_collapsible_frames.clear()
        self._geometry_collapsed.clear()
        self._geometry_hover_capture_id = None
        self._model_subscriptions.clear()
        self._sensor_profile_combo_model = None
        self._sensor_profile_description_labels.clear()
        self._pending_sensor_profile_index = None
        self._sensor_profile_apply_inflight = False
        self._pad_frame = self._pad_surface = self._pad_spacers = None
        self._object_pad_surfaces.clear()
        self._object_pad_drag = None
        self._object_editing_keys.clear()
        toolbar, self.toolbar_window = self.toolbar_window, None
        status, self.status_window = self.status_window, None
        if toolbar is not None:
            toolbar.destroy()
        if status is not None:
            status.destroy()


class KitShortcutBridge:
    """Route documented shortcuts without importing input APIs into the shell model."""

    def __init__(
        self, dispatch_fn, *, camera_key_fn=None, modifier_fn=None,
        input_interface=None, keyboard=None,
    ) -> None:
        self._dispatch = dispatch_fn
        self._camera_key_fn = camera_key_fn
        self._modifier_fn = modifier_fn
        self._input = input_interface
        self._keyboard = keyboard
        self._subscription = None
        self._pressed_modifiers = set()
        self._key_inputs = {}

    def start(self) -> None:
        if self._subscription is not None:
            return
        if self._input is None or self._keyboard is None:
            import carb.input
            import omni.appwindow
            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("default Kit app window is unavailable")
            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
        keyboard_input = getattr(carb.input, "KeyboardInput", object())
        self._key_inputs = {
            key: getattr(keyboard_input, key)
            for key in ("W", "A", "S", "D", "UP", "LEFT", "DOWN", "RIGHT", "Q", "E", "Z", "C", "R", "P", "F4")
            if hasattr(keyboard_input, key)
        }
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self.on_keyboard_event)

    def on_keyboard_event(self, event) -> bool:
        value = getattr(event, "input", None)
        key = next((name for name, enum_value in self._key_inputs.items() if value == enum_value), None)
        key = key or kit_key_name(value)
        kind = kit_enum_name(getattr(event, "type", None))
        if kind not in ("KEY_PRESS", "KEY_REPEAT", "KEY_RELEASE"):
            return True
        pressed = kind in ("KEY_PRESS", "KEY_REPEAT")
        modifier = _modifier(key)
        if modifier is not None:
            if pressed:
                self._pressed_modifiers.add(modifier)
            else:
                self._pressed_modifiers.discard(modifier)
            if self._modifier_fn is not None:
                self._modifier_fn(modifier, pressed)
            return True
        handled = self._camera_key_fn is not None and self._camera_key_fn(
            key, pressed, repeat=kind == "KEY_REPEAT",
        )
        if handled:
            self._consume(event)
            return True
        if not pressed or kind == "KEY_REPEAT":
            return True
        handled = self._dispatch(
            key,
            ctrl="ctrl" in self._pressed_modifiers,
            shift="shift" in self._pressed_modifiers,
            alt="alt" in self._pressed_modifiers,
        )
        if handled:
            self._consume(event)
        return True

    @property
    def alt_pressed(self) -> bool:
        return "alt" in self._pressed_modifiers

    @staticmethod
    def _consume(event) -> None:
        try:
            event.consume()
        except (AttributeError, RuntimeError):
            pass

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None
        self._pressed_modifiers.clear()
        self._key_inputs.clear()


def _toolbar_is_docked(toolbar) -> bool:
    if toolbar is None:
        return False
    if toolbar.docked:
        return True
    # Kit 110.1.2 ToolBar can report docked=False while its workspace node
    # contains the toolbar and is attached to a parent split. Verify membership
    # rather than accepting a nonzero (possibly stale) dock_id alone.
    dock_id = toolbar.dock_id
    return bool(dock_id and ui.Workspace.get_parent_dock_id(dock_id)
                and any(member.title == toolbar.title
                        for member in ui.Workspace.get_docked_windows(dock_id)))


def default_icon_directory() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "icons", "findsurface"))


def _modifier(key: str) -> str | None:
    if "CTRL" in key or "CONTROL" in key:
        return "ctrl"
    if "SHIFT" in key:
        return "shift"
    if "ALT" in key:
        return "alt"
    return None


def _tool_flags() -> int:
    flags = ui.WINDOW_FLAGS_NO_SCROLLBAR
    if hasattr(ui, "WINDOW_FLAGS_NO_TITLE_BAR"):
        flags |= ui.WINDOW_FLAGS_NO_TITLE_BAR
    return flags


def _toolbar_style() -> dict:
    icon = 0xFFFFFFFF
    disabled = 0xFF686D76
    render_style = style_for_theme(RenderTheme.EDITOR_DARK)
    def packed_color(rgba):
        r, g, b, a = (round(value * 255) for value in rgba)
        return (a << 24) | (b << 16) | (g << 8) | r
    return {
        "ClassicToolbar": {"margin_height": 2},
        "ClassicToolbar.DesignedIcon": {"color": packed_color(render_style.designed_wire)},
        "ClassicToolbar.DetectedIcon": {"color": packed_color(render_style.detected_wire)},
        # This toolbar identity color intentionally remains the captured default,
        # independent of per-geometry Graphics-window customization.
        "ClassicToolbar.CapturedIcon": {"color": packed_color((1.0, 0.78, 0.16, 1.0))},
        "ClassicToolbar.DesignedIcon:disabled": {"color": disabled},
        "ClassicToolbar.DetectedIcon:disabled": {"color": disabled},
        "ClassicToolbar.CapturedIcon:disabled": {"color": disabled},
        "ClassicToolbar.Frame": {"background_color": 0x0, "border_color": 0xFF60656E, "border_width": 1},
        "ClassicToolbar.Frame:checked": {"background_color": 0xFF222831, "border_color": 0xFFE89B35},
        "ClassicToolbar.Frame:disabled": {"border_color": 0xFF3E4249},
        "ClassicToolbar.Hit": {"background_color": 0x0, "border_width": 0},
        "ClassicToolbar.Hit.Label": {"color": 0x0},
        "ClassicToolbar.Icon": {"color": icon},
        "ClassicToolbar.Icon:disabled": {"color": disabled},
    }


def _histogram_hover_text(count: int, source_count: int) -> str:
    percent = 100.0 * count / source_count if source_count > 0 else 0.0
    return f"Count: {count:,} pts ({percent:.2f}%)"


def _result_overlay_content_height(*, reserve_histogram: bool) -> int:
    """Keep a stable card height per sensor mode, including pending analysis."""

    if reserve_histogram:
        return RESULT_OVERLAY_BASE_HEIGHT
    return 8 + 28 + RESULT_OVERLAY_LIVE_ROW_COUNT * RESULT_OVERLAY_ROW_HEIGHT + 8


def _parameter_pad_square_size(width: float, height: float) -> int:
    """Largest unclipped pad square after the frame, Y controls and X row."""

    available_width = float(width) - 16.0 - 148.0 - 8.0
    available_height = float(height) - 16.0 - 48.0 - 6.0
    return max(1, int(min(available_width, available_height)))


def _visible_widget_extent(widget_rect, clip_rect) -> tuple[float, float]:
    """Return the visible width/height of a widget inside its clipping host."""

    x, y, width, height = (float(value) for value in widget_rect)
    clip_x, clip_y, clip_width, clip_height = (float(value) for value in clip_rect)
    visible_width = max(0.0, min(x + width, clip_x + clip_width) - max(x, clip_x))
    visible_height = max(0.0, min(y + height, clip_y + clip_height) - max(y, clip_y))
    return visible_width, visible_height


def _fraction_offset(fraction: float, width: int) -> int:
    fraction = max(0.0, min(1.0, float(fraction)))
    return max(0, min(width - 1, int(fraction * (width - 1) + 0.5)))


def _heatmap_ui_color(coordinate: float) -> int:
    """Match the renderer's negative-blue/zero-yellow/positive-red heatmap."""

    rgba = heatmap_color(style_for_theme(RenderTheme.EDITOR_DARK), coordinate)
    channels = [max(0, min(255, int(round(value * 255.0)))) for value in rgba]
    return (channels[3] << 24) | (channels[2] << 16) | (channels[1] << 8) | channels[0]


__all__ = [
    "BUTTON_SIZE", "ClassicUiChrome", "ICON_SIZE", "KitShortcutBridge", "PROPERTY_WIDTH",
    "SIDE_PANEL_DOCK_RATIO", "STATUS_BAR_HEIGHT", "TOOLBAR_HEIGHT", "default_icon_directory",
]
