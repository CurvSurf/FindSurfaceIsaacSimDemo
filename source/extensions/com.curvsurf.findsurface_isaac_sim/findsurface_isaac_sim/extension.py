"""Production classic shell and ordered application lifecycle for Isaac Sim 6.0.1."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import carb
import omni.ext
import omni.appwindow
import omni.kit.app
import omni.ui as ui
import omni.usd

from .analysis_pipeline import (
    AnalysisCompletionStatus,
    AnalysisPresentation,
    AnalysisSubject,
    ResidualBasis,
    SerializedAnalysisScheduler,
    SurfaceSideScaleMode,
    detected_scene_object,
    evaluate_analysis,
    make_analysis_request,
)
from .camera import (
    CameraIntrinsics, CameraRole, CameraSnapshot, ScenePreset,
    aligned_orbit_target, scene_preset, with_viewport_aspect,
)
from .camera_input import CameraInputController, ShortcutIntent, update_event_dt_seconds
from .capabilities import collect_capabilities
from .classic_ui import ClassicUiChrome, KitShortcutBridge
from .command_facade import CommandFacade, CommandId, SensorMode
from .config import load_app_config
from .depth_sensor import DEFAULT_DEPTH_PROFILE_ID, DEPTH_PROFILES
from .dual_view_widget import DualOpenGLCompositorWidget
from .exit_guard import ExitTrigger, UnifiedExitController
from .findsurface_adapter import create_domain_detector
from .shell_workflow import ShellWorkflow
from .geometry import build_mesh
from .geometry_card import GeometryCardPresentation
from .domain import GenerationSnapshot, GeometryMode, Pose, SensorKind
from .kit_quit_guard import KitQuitGuard
from .kit_input import kit_modifier_has_alt
from .operations import ShutdownStep
from .object_orientation import corrected_orientation, orientation_axes, roll_in_view, rotate_in_view
from .picking import PickingController
from .preview_capture import CaptureAvailability, CaptureBlockReason, PreviewCaptureController, PreviewPhase
from .point_cloud_export import export_points_xyz
from .render_layers import (
    LayeredRenderPacket, RenderLayer, RenderTheme, build_mesh_layer,
    merge_point_layers, style_for_theme,
)
from .render_cache import RenderPayloadCache
from .parameter_mapping import slider_index_for_value
from .ui_presentation_cache import (
    UiPresentationCache, identity_dependency, interaction_property_dependencies,
)
from .result_overlay import (
    ResultOverlayPresentation,
    build_capture_result_overlay,
    build_result_overlay,
)
from .scene import SceneCatalog, SceneState, constrain_geometry_parameter_edit
from .lidar_sensor import (
    DEFAULT_LIDAR_PROFILE_ID, SCANNER_LIDAR_PROFILES, lidar_profile,
    lidar_profile_description_rows, lidar_profile_display_label,
)
from .sensor_material import (
    DEPTH_RECOMMENDED_MATERIAL,
    SENSOR_MATERIAL_ATTRIBUTES,
    SENSOR_MATERIAL_BASES,
    SENSOR_MATERIAL_COATINGS,
    SensorMaterialSetting,
)
from .sensor_control import SensorControlPhase, SensorCoordinator, SensorSelection
from .sensor_rates import SensorRateTracker
from .sensor_diagnostics import AntiAliasingMode, DepthSensorDiagnostics, diagnostics_for_reset
from .ui_shell import ClassicShellModel, ModalKind, SensorUiStatus, ShellCommandController
from .user_preferences import (
    PERSISTENT_SETTING_PATH,
    depth_diagnostics_from_config,
    encode_user_preferences,
    load_user_preferences,
)
from .workspace_layout import (
    PERSISTENT_WORKSPACE_PATH, decode_workspace_layout, encode_workspace_layout,
)
from .usd_scene import UsdSceneSession


SETTING_ROOT = "/exts/com.curvsurf.findsurface_isaac_sim"
_PROPERTY_ACTION_REFRESH = object()


class FindSurfaceIsaacSimExtension(omni.ext.IExt):
    def on_startup(self, _extension_id: str) -> None:
        self._window = self._chrome = self._compositor = self._keyboard = None
        self._standard_viewport_window = ui.Workspace.get_window("Viewport")
        if self._standard_viewport_window is not None:
            self._standard_viewport_window.visible = False
        self._scene_session = self._scene_state = self._scene_task = self._smoke_task = None
        self._refresh_subscription = None
        self._teardown_events = []
        self._ui_probe_report = None
        self._render_packet = None
        self._render_revision = 22
        self._render_payloads = RenderPayloadCache()
        self._ui_presentation_cache = UiPresentationCache()
        self._sensor_coordinator = None
        self._sensor_rates = SensorRateTracker()
        self._object_change_task = None
        self._depth_diagnostics_refresh_task = None
        self._depth_diagnostics_revision = 0
        self._xyz_export_task = None
        self._xyz_export_running = False
        self._xyz_export_status = "No export yet"
        self._object_rotation_gesture = None
        self._sensor_polling = True
        self._rendered_sensor_frame = None
        self._pip_visible = True
        self._pip_width = 0.32
        self._camera_controllers = {}
        self._pointer_role = None
        self._pointer_retroactive_pick = False
        self._keyboard_camera_before = None
        self._hover_preview = None
        self._hover_frame_identity = None
        self._hover_frame_revision = 0
        self._hover_options_signature = None
        self._hover_options_revision = 0
        self._analysis_scheduler = None
        self._hover_analysis: AnalysisPresentation | None = None
        self._hover_analysis_current_key = None
        self._hover_analysis_pending_key = None
        self._hover_analysis_error = None
        self._hover_analysis_source = None
        self._user_preferences_enabled = False
        self._last_user_preferences_payload = None
        self._workspace_layout_task = None
        self._workspace_layout_cache = None
        self._default_workspace_layout = None
        self._workspace_layout_resetting = False
        self._last_workspace_capture_at = 0.0

        self._exit = UnifiedExitController(self._post_quit)
        self._quit_guard = KitQuitGuard(self._exit)
        self._window = ui.Window(
            "Viewport",
            width=1180,
            height=760,
            raster_policy=ui.RasterPolicy.NEVER,
        )
        self._window.set_visibility_changed_fn(self._on_window_visibility_changed)
        self._exit.own(ShutdownStep.STAGE_SESSION_UI, "main-window", self._destroy_window)

        report = collect_capabilities()
        settings = carb.settings.get_settings()
        self._user_preferences_enabled = bool(
            settings.get_as_bool(f"{SETTING_ROOT}/userPreferencesEnabled")
        )
        self._saved_workspace_layout = (
            decode_workspace_layout(settings.get_as_string(PERSISTENT_WORKSPACE_PATH))
            if self._user_preferences_enabled else None
        )
        if self._user_preferences_enabled and self._saved_workspace_layout is None:
            # Discard the old global-workspace format, including broken dumps.
            settings.set(PERSISTENT_WORKSPACE_PATH, "")
        if self._user_preferences_enabled:
            self._exit.own(ShutdownStep.DETECTOR, "workspace-layout", self._persist_workspace_layout)
        try:
            self._anti_aliasing_mode = AntiAliasingMode(settings.get_as_int("/rtx/post/aa/op"))
        except ValueError:
            self._anti_aliasing_mode = AntiAliasingMode.TAA
        self._depth_diagnostics = DepthSensorDiagnostics()
        self._findsurface_library_path = settings.get_as_string(f"{SETTING_ROOT}/findSurfaceLibraryPath")
        defaults_override = settings.get_as_string(f"{SETTING_ROOT}/appDefaultsPath")
        config_result = load_app_config(defaults_override or None)
        self._app_config = config_result.config
        # Packaged/appDefaultsPath values are the reset baseline; persistent
        # user preferences are overlaid only on the startup state below.
        self._settings_defaults = config_result.config
        preference_result = None
        if self._user_preferences_enabled:
            preference_result = load_user_preferences(
                self._app_config, settings.get_as_string(PERSISTENT_SETTING_PATH),
            )
            self._app_config = preference_result.config
        self._depth_diagnostics = depth_diagnostics_from_config(self._app_config)
        self._startup_depth_diagnostics = self._depth_diagnostics
        self._startup_display = (self._anti_aliasing_mode, self._pip_visible, self._pip_width)
        for diagnostic in config_result.diagnostics:
            carb.log_warn(f"App defaults fallback ({diagnostic.code}) at {diagnostic.path}: {diagnostic.message}")
        if preference_result is not None and preference_result.diagnostic is not None:
            diagnostic = preference_result.diagnostic
            carb.log_warn(f"User preferences fallback ({diagnostic.code}): {diagnostic.message}")

        self._facade = CommandFacade(self._app_config)
        self._workflow = ShellWorkflow(self._facade)
        self._analysis_scheduler = SerializedAnalysisScheduler()
        self._exit.own(ShutdownStep.DETECTOR, "preview-detection-workflow", self._close_workflow)
        if self._user_preferences_enabled:
            if preference_result is not None and preference_result.loaded:
                self._last_user_preferences_payload = encode_user_preferences(
                    self._facade.snapshot, self._depth_diagnostics,
                )
            self._exit.own(
                ShutdownStep.DETECTOR, "user-preferences", self._persist_user_preferences,
            )
        self._exit.own(ShutdownStep.SENSOR_PUBLICATION, "sensor-publication", self._close_sensor_publication)
        self._exit.own(ShutdownStep.SENSOR_PRODUCTS, "sensor-products", self._close_sensor_products)
        self._shell_model = ClassicShellModel(self._facade)
        self._shell_snapshot = self._build_shell_snapshot()
        self._shell_commands = ShellCommandController(
            self._shell_model,
            lambda: self._shell_snapshot,
            {command_id: lambda value=command_id: self._execute_command(value) for command_id in CommandId},
        )

        self._chrome = ClassicUiChrome(
            self._window.frame,
            self._shell_snapshot,
            self._execute_command,
            modal_dismiss_fn=self._dismiss_modal,
            modal_cancel_fn=self._cancel_modal,
            property_fn=self._handle_property_or_layout_action,
            property_values=self._property_values(),
            layout_preference=self._saved_workspace_layout,
            defer_viewport_overlays=True,
        )
        self._exit.own(ShutdownStep.STAGE_SESSION_UI, "classic-ui", self._close_chrome)
        self._compositor = DualOpenGLCompositorWidget(
            self._chrome.render_host, pointer_fn=self._handle_viewport_pointer,
        )
        self._chrome.mount_viewport_overlays(self._compositor.overlay_host)
        self._exit.own(ShutdownStep.RENDERER, "dual-view-compositor", self._close_compositor)

        self._keyboard = KitShortcutBridge(
            self._dispatch_shortcut,
            camera_key_fn=self._handle_camera_key,
            modifier_fn=self._handle_modifier,
        )
        self._keyboard.start()
        self._exit.own(ShutdownStep.STAGE_SESSION_UI, "shortcut-input", self._close_keyboard)
        self._exit.own(ShutdownStep.STAGE_SESSION_UI, "usd-scene-session", self._close_scene_session)
        dock_task = self._chrome.dock(self._window)
        self._workspace_layout_task = asyncio.ensure_future(
            self._restore_workspace_after_dock(dock_task)
        )

        self._scene_task = asyncio.ensure_future(self._initialize_scene())
        self._refresh_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="FindSurface classic shell refresh"
            )
        )
        self._exit.own(ShutdownStep.STAGE_SESSION_UI, "shell-refresh", self._close_refresh)

        carb.log_info("FindSurfaceIsaacSimDemo classic shell startup complete")


    def _persist_user_preferences(self) -> None:
        """Write the current durable subset to Kit's per-user persistent settings."""

        if not self._user_preferences_enabled or not hasattr(self, "_facade"):
            return
        try:
            payload = encode_user_preferences(self._facade.snapshot, self._depth_diagnostics)
            if payload == self._last_user_preferences_payload:
                return
            carb.settings.get_settings().set(PERSISTENT_SETTING_PATH, payload)
            self._last_user_preferences_payload = payload
        except Exception as error:
            # Preference persistence must never abort Safe Exit or a successful
            # sensor/property transaction.
            carb.log_warn(f"Could not persist user preferences: {type(error).__name__}: {error}")

    async def _restore_workspace_after_dock(self, dock_task) -> None:
        try:
            if dock_task is not None:
                await dock_task
            if self._chrome is None or self._window is None:
                return
            if not all(self._chrome.dock_state().values()):
                raise RuntimeError("initial docking is incomplete; restoring hardcoded default layout")
            self._default_workspace_layout = self._chrome.capture_layout(self._window)
            layout = self._saved_workspace_layout
            if layout is not None:
                await self._chrome.restore_floating_layout()
            self._capture_workspace_layout()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            carb.log_warn(f"Could not restore window layout: {type(error).__name__}: {error}")
            if self._chrome is not None and self._window is not None:
                # Never reuse the failed preference or a previously captured broken layout.
                self._saved_workspace_layout = None
                self._workspace_layout_cache = None
                if self._user_preferences_enabled:
                    carb.settings.get_settings().set(PERSISTENT_WORKSPACE_PATH, "")
                try:
                    await self._chrome.reset_docking(self._window)
                    self._capture_workspace_layout()
                    self._default_workspace_layout = self._workspace_layout_cache
                except Exception as fallback_error:
                    carb.log_error(
                        "Could not recover default window layout: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )

    def _capture_workspace_layout(self) -> None:
        if self._chrome is None or self._window is None or not self._window.visible:
            return
        layout = self._chrome.capture_layout(self._window)
        encode_workspace_layout(layout)
        self._workspace_layout_cache = layout

    def _maybe_capture_workspace_layout(self) -> None:
        if not self._user_preferences_enabled:
            return
        now = time.monotonic()
        if now - self._last_workspace_capture_at < 1.0:
            return
        self._last_workspace_capture_at = now
        if self._default_workspace_layout is None or (
            self._workspace_layout_task is not None and not self._workspace_layout_task.done()
        ) or self._workspace_layout_resetting:
            return
        try:
            self._capture_workspace_layout()
        except Exception as error:
            carb.log_warn(f"Could not capture window layout: {type(error).__name__}: {error}")

    def _persist_workspace_layout(self) -> None:
        if not self._user_preferences_enabled:
            return
        try:
            # A title-bar close has already hidden Viewport. Use the last intact
            # snapshot captured during the update loop in that case.
            if self._window is not None and self._window.visible:
                try:
                    self._capture_workspace_layout()
                except Exception as error:
                    # Minimized/tearing-down OS windows can report zero client extent.
                    # Keep the last bounded snapshot rather than saving transient geometry.
                    carb.log_warn(f"Using last valid window layout during shutdown: {error}")
            if self._workspace_layout_cache is not None:
                carb.settings.get_settings().set(
                    PERSISTENT_WORKSPACE_PATH,
                    encode_workspace_layout(self._workspace_layout_cache),
                )
        except Exception as error:
            carb.log_warn(f"Could not persist window layout: {type(error).__name__}: {error}")

    async def _reset_window_layout(self) -> None:
        self._workspace_layout_resetting = True
        try:
            task = self._workspace_layout_task
            if task is not None and not task.done():
                await task
            if self._chrome is None or self._window is None:
                raise RuntimeError("product workspace is not ready")
            await self._chrome.reset_docking(self._window)
            self._saved_workspace_layout = None
            self._workspace_layout_cache = None
            carb.settings.get_settings().set(PERSISTENT_WORKSPACE_PATH, "")
            self._capture_workspace_layout()
        except Exception as error:
            carb.log_error(f"Could not reset window layout: {type(error).__name__}: {error}")
        finally:
            self._workspace_layout_resetting = False

    async def _initialize_scene(self) -> None:
        await self._wait_for_sensor_startup_window()
        if not self._exit.accepting_commands:
            return
        context = omni.usd.get_context()
        created_stage = context.get_stage() is None
        if created_stage:
            await context.new_stage_async()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("USD context did not provide a stage")
        if created_stage or self._is_empty_kit_stage(stage):
            # Camera local -Z and app meshes use the canonical Z-up frame.
            # Kit may already have created an anonymous stage with only its
            # protected render products/cameras before our startup task runs.
            from pxr import UsdGeom

            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        catalog = SceneCatalog.from_config(self._app_config["objects"])
        session = UsdSceneSession(
            stage, catalog, self._facade.snapshot.scanner_pose,
            scene_pose=self._facade.snapshot.scene_pose,
            sensor_material=self._facade.snapshot.sensor.target_material,
        )
        self._scene_session = session
        self._scene_state = SceneState(
            catalog,
            object_writer=session.write_object,
            event_sink=self._on_scene_object_change,
        )
        scanner, scene = self._camera_snapshots(catalog, prefer_controllers=False)
        self._camera_controllers = {
            CameraRole.SCANNER: CameraInputController(scanner),
            CameraRole.SCENE: CameraInputController(scene),
        }
        self._sync_render_from_state()
        setting = self._facade.snapshot.sensor
        coordinator = SensorCoordinator(
            camera_pose_world=scanner.pose_world,
            follows_camera=setting.follows_camera,
            depth_diagnostics=self._depth_diagnostics,
            object_values=lambda: self._facade.snapshot.objects,
        )
        self._sensor_coordinator = coordinator
        if not self._exit.accepting_commands or coordinator.snapshot.phase is SensorControlPhase.CLOSED:
            return
        sensor = await coordinator.start(
            self._sensor_selection(setting),
            disabled=not setting.enabled,
        )
        if sensor.phase is SensorControlPhase.ERROR:
            raise RuntimeError(sensor.error or "point-cloud sensor initialization failed")
        self._sync_scanner_sensor_intrinsics()
        if not hasattr(self, "_startup_camera_views"):
            self._startup_camera_views = {role: controller.snapshot for role, controller in self._camera_controllers.items()}
            self._startup_scanner_manual_fov = self._facade.snapshot.scanner_manual_fov
            self._facade.bind_startup_camera_views(self._startup_camera_views.values())
        self._sync_render_from_state()
        carb.log_info("FindSurface production scene session initialized")

    @staticmethod
    def _is_empty_kit_stage(stage) -> bool:
        """Claim only an unsaved stage containing Kit's initial infrastructure."""
        root, session = stage.GetRootLayer(), stage.GetSessionLayer()
        if (not root.anonymous or root.subLayerPaths or session.subLayerPaths
                or root.customLayerData or session.customLayerData or stage.GetDefaultPrim()):
            return False
        if any(layer not in (root, session) for layer in stage.GetUsedLayers()):
            return False
        kit_paths = {"/Render", "/OmniverseKit_Persp", "/OmniverseKit_Front",
                     "/OmniverseKit_Top", "/OmniverseKit_Right"}
        for prim in stage.GetPseudoRoot().GetAllChildren():
            if (str(prim.GetPath()) not in kit_paths
                    or not prim.GetMetadata("hide_in_stage_window")
                    or not prim.GetMetadata("no_delete")):
                return False
        kit_paths |= {"/Render/OmniverseKit", "/Render/OmniverseKit/HydraTextures",
                      "/Render/OmniverseGlobalRenderSettings", "/Render/Vars",
                      "/Render/Vars/LdrColor"}
        for prim in stage.TraverseAll():
            if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
                return False
            path = str(prim.GetPath())
            if path not in kit_paths:
                if (str(prim.GetParent().GetPath()) != "/Render/OmniverseKit/HydraTextures"
                        or not prim.GetName().startswith("omni_kit_widget_viewport_ViewportTexture_")
                        or prim.GetTypeName() != "RenderProduct"
                        or not prim.GetMetadata("no_delete")):
                    return False
        return True

    async def _wait_for_sensor_startup_window(self) -> None:
        """Avoid the app-ready Hydra/Fabric attach race reproduced by the G4 gates."""

        app = omni.kit.app.get_app()
        started = asyncio.get_running_loop().time()
        updates = 0
        while updates < 30 or asyncio.get_running_loop().time() - started < 2.0:
            await app.next_update_async()
            updates += 1

    @staticmethod
    def _sensor_selection(setting) -> SensorSelection:
        profile_id = setting.profile_id
        if profile_id == "default":
            profile_id = (DEFAULT_DEPTH_PROFILE_ID if setting.sensor_kind is SensorKind.DEPTH
                          else DEFAULT_LIDAR_PROFILE_ID)
        return SensorSelection(setting.sensor_kind, profile_id)

    @staticmethod
    def _recommended_sensor_material(sensor_kind: SensorKind, profile_id: str):
        return (DEPTH_RECOMMENDED_MATERIAL if sensor_kind is SensorKind.DEPTH
                else lidar_profile(profile_id).recommended_material)

    def _camera_snapshots(
        self, catalog: SceneCatalog, *, prefer_controllers: bool = True,
    ) -> tuple[CameraSnapshot, CameraSnapshot]:
        if prefer_controllers and len(self._camera_controllers) == len(CameraRole):
            return tuple(self._camera_controllers[role].snapshot for role in CameraRole)
        intrinsics = CameraIntrinsics(52.0, 16.0 / 9.0, 0.05, 100.0)
        center = catalog.bounds.center_m
        state = self._facade.snapshot
        scanner_fallback = CameraSnapshot(
            CameraRole.SCANNER,
            state.scanner_pose,
            intrinsics,
            aligned_orbit_target(state.scanner_pose, center),
        )
        scene_fallback = CameraSnapshot(
            CameraRole.SCENE,
            state.scene_pose,
            intrinsics,
            aligned_orbit_target(state.scene_pose, center),
        )
        return (
            next((replace(v, pose_world=state.scanner_pose) for v in state.camera_views if v.role is CameraRole.SCANNER), scanner_fallback),
            next((replace(v, pose_world=state.scene_pose) for v in state.camera_views if v.role is CameraRole.SCENE), scene_fallback),
        )


    def _capture_availability(self) -> CaptureAvailability:
        if self._hover_preview is not None:
            return self._hover_preview.snapshot.capture_availability
        workflow = self._workflow.snapshot
        reasons = []
        if self._facade.snapshot.sensor.enabled:
            reasons.append(CaptureBlockReason.SENSOR_ACTIVE)
        if not workflow.selected:
            reasons.append(CaptureBlockReason.SELECTION_REQUIRED)
        if not workflow.preview_ready:
            reasons.append(CaptureBlockReason.NO_SUCCESSFUL_RESULT)
        enabled = not reasons and workflow.capture_ready
        text = {
            CaptureBlockReason.SENSOR_ACTIVE: "Disable the sensor before capturing.",
            CaptureBlockReason.SELECTION_REQUIRED: "Select a point in the fixed cloud before capturing.",
            CaptureBlockReason.NO_SUCCESSFUL_RESULT: "No successful current detection is available.",
        }
        tooltip = "Capture the latest selected result." if enabled else " ".join(text[item] for item in reasons)
        return CaptureAvailability(enabled, tuple(reasons), tooltip)

    def _build_shell_snapshot(self):
        workflow = self._workflow.snapshot
        analysis = workflow.analysis
        live_hover = self._facade.snapshot.sensor.enabled
        if (
            live_hover
            and analysis is not None
            and analysis.key.subject_id == "preview"
        ):
            analysis = None
        result_overlay = ResultOverlayPresentation()
        analysis_text = (
            "Live hover: histogram, heatmap and inliers disabled"
            if live_hover
            else "No current analysis"
        )
        if analysis is not None and analysis.available:
            histogram = analysis.analysis.histogram
            analysis_text = f"histogram {histogram.used_count}/{histogram.source_count}"
            workflow_result = getattr(self._workflow, "_result", None)
            if workflow_result is not None and workflow_result.request_key == analysis.key.request_key:
                analysis_text = f"Fit RMS {workflow_result.rms_error_m:.4f} m · {analysis_text}"
        coordinator = self._sensor_coordinator
        sensor_snapshot = None if coordinator is None else coordinator.snapshot
        frame = None if coordinator is None else coordinator.latest_frame()
        active_profile = None if sensor_snapshot is None else sensor_snapshot.active
        active_profile_id = (
            self._facade.snapshot.sensor.profile_id
            if active_profile is None
            else active_profile.profile_id
        )
        detection_status = workflow.message
        if self._hover_preview is not None:
            hover = self._hover_preview.snapshot
            if hover.phase is PreviewPhase.PENDING:
                detection_status = "Hover detection pending"
            elif hover.phase is PreviewPhase.READY and hover.result is not None:
                detection_status = (
                    f"Hover: {hover.result.geometry_type.value} · "
                    f"{len(hover.result.inlier_indices):,} inliers"
                )
            elif hover.display_result_sticky and hover.display_result is not None:
                detection_status = (
                    f"Sticky preview: {hover.display_result.geometry_type.value} · "
                    f"{len(hover.display_result.inlier_indices):,} inliers"
                )
            elif hover.phase is PreviewPhase.ERROR:
                detection_status = f"Hover detection failed: {hover.error}"
            display_result = hover.display_result
            if display_result is not None:
                if not live_hover:
                    analysis_text = f"Fit RMS {display_result.rms_error_m:.4f} m"
                hover_analysis = None if live_hover else self._hover_analysis
                if (
                    hover_analysis is None
                    or hover_analysis.key.request_key != display_result.request_key
                ):
                    hover_analysis = None
                if hover_analysis is not None and hover_analysis.available:
                    histogram = hover_analysis.analysis.histogram
                    analysis_text = (
                        f"Fit RMS {display_result.rms_error_m:.4f} m · "
                        f"histogram {histogram.used_count}/{histogram.source_count}"
                    )
                title = "Preview"
                pending = (
                    not live_hover
                    and self._hover_analysis_pending_key is not None
                    and self._hover_analysis_pending_key.request_key == display_result.request_key
                )
                error = None if live_hover else self._hover_analysis_error
                result_overlay = self._ui_presentations().resolve(
                    "hover-overlay",
                    (identity_dependency(display_result), identity_dependency(hover_analysis),
                     title, pending, error, live_hover),
                    lambda: build_result_overlay(
                        display_result, hover_analysis, title=title, pending=pending, error=error,
                        reserve_histogram=not live_hover,
                    ),
                )
            else:
                self._ui_presentations().retain("hover-overlay", ())
        update_hz, complete_scan_hz = self._sensor_ui_rates()
        return self._shell_model.snapshot(
            capture=self._capture_availability(),
            sensor=SensorUiStatus(
                ("CREATING" if sensor_snapshot is None else sensor_snapshot.phase.value),
                active_profile_id,
                None if frame is None else frame.sequence,
                0 if frame is None else frame.compacted_count,
                update_hz=update_hz,
                complete_scan_hz=complete_scan_hz,
                findsurface_ms=self._findsurface_ui_duration_ms(),
                follows_camera=True if sensor_snapshot is None else sensor_snapshot.pose.follows_camera,
                mode=self._facade.snapshot.sensor.mode,
                no_returns=False if sensor_snapshot is None else sensor_snapshot.no_returns,
            ),
            detection_status=detection_status,
            analysis_status=analysis_text,
            result_overlay=result_overlay,
        )

    def _sensor_ui_rates(self) -> tuple[float, float]:
        coordinator = self._sensor_coordinator
        tracker = getattr(self, "_sensor_rates", None)
        if tracker is None:
            self._sensor_rates = tracker = SensorRateTracker()
        return tracker.sample(
            None if coordinator is None else coordinator.active_runtime,
            active=coordinator is not None and coordinator.snapshot.phase is SensorControlPhase.RUNNING,
        )

    def _findsurface_ui_duration_ms(self) -> float | None:
        preview = self._hover_preview
        if preview is None:
            return None
        return preview.successful_call_mean_ms(self._facade.snapshot.interaction.detection_average_window)

    def _execute_command(self, command_id: CommandId):
        command_id = CommandId(command_id)
        if not self._shell_snapshot.command(command_id).enabled:
            return False
        try:
            if command_id is CommandId.CAPTURE:
                if (
                    self._hover_preview is not None
                    and self._hover_preview.snapshot.capture_availability.enabled
                ):
                    entry = self._hover_preview.capture()
                    surface = entry.surface
                    capture_ids = {
                        item.capture_id for item in self._facade.snapshot.captures
                    }
                    if surface.capture_id in capture_ids:
                        serial = 1
                        while f"capture-{serial:06d}" in capture_ids:
                            serial += 1
                        surface = replace(
                            surface,
                            capture_id=f"capture-{serial:06d}",
                        )
                    request = make_analysis_request(
                        AnalysisSubject.from_capture(surface),
                        self._facade.snapshot.objects,
                        scene_generation=0,
                    )
                    self._sync_render_from_state()
                    self._refresh_ui()
                    return self._schedule_action(
                        self._complete_capture_analysis(surface, request)
                    )
                else:
                    self._workflow.capture()
            elif command_id is CommandId.UNDO:
                return self._schedule_action(self._workflow.undo())
            elif command_id is CommandId.REDO:
                return self._schedule_action(self._workflow.redo())
            elif command_id is CommandId.RESET_CAPTURES:
                return self._schedule_capture_reset()
            elif command_id in (CommandId.PRIMARY_SCENE, CommandId.PRIMARY_SCANNER):
                self._facade.execute(command_id)
                role = CameraRole.SCENE if command_id is CommandId.PRIMARY_SCENE else CameraRole.SCANNER
                self._compositor.set_primary(role)
            elif command_id in {
                CommandId.MODE_PLANE, CommandId.MODE_SPHERE, CommandId.MODE_CYLINDER,
                CommandId.MODE_CONE, CommandId.MODE_TORUS,
            }:
                mode = {
                    CommandId.MODE_PLANE: GeometryMode.PLANE,
                    CommandId.MODE_SPHERE: GeometryMode.SPHERE,
                    CommandId.MODE_CYLINDER: GeometryMode.CYLINDER,
                    CommandId.MODE_CONE: GeometryMode.CONE,
                    CommandId.MODE_TORUS: GeometryMode.TORUS,
                }[command_id]
                self._workflow.set_mode(mode)
            elif command_id in {
                CommandId.MODE_INSPECT, CommandId.MODE_AUTO,
                CommandId.VISIBILITY_INLIER, CommandId.VISIBILITY_DESIGNED,
                CommandId.VISIBILITY_DETECTED, CommandId.VISIBILITY_CAPTURED,
                CommandId.VISIBILITY_OUTLIER,
            }:
                self._facade.execute(command_id)
                if command_id is CommandId.MODE_INSPECT:
                    self._pause_hover_detection()
                    self._workflow.invalidate("Detection paused: inspecting geometries.")
                elif command_id is CommandId.MODE_AUTO:
                    self._workflow.invalidate()
            elif command_id in {
                CommandId.CAMERA_RESET_SCENE, CommandId.CAMERA_SCENE_POS_X,
                CommandId.CAMERA_SCENE_POS_Y, CommandId.CAMERA_SCENE_POS_Z,
                CommandId.CAMERA_RESET_SCANNER, CommandId.CAMERA_RESET_SCANNER_UP,
            }:
                self._apply_camera_command(command_id)
            elif command_id in {
                CommandId.SENSOR_FROZEN,
                CommandId.SENSOR_LIVE_FIXED,
                CommandId.SENSOR_LIVE_FOLLOW,
            }:
                mode = {
                    CommandId.SENSOR_FROZEN: SensorMode.FROZEN,
                    CommandId.SENSOR_LIVE_FIXED: SensorMode.LIVE_FIXED,
                    CommandId.SENSOR_LIVE_FOLLOW: SensorMode.LIVE_FOLLOW,
                }[command_id]
                return self._schedule_sensor_mode(mode)
            elif command_id is CommandId.HELP:
                self._chrome.show_help(self._shell_snapshot)
            elif command_id is CommandId.SAFE_EXIT:
                return self.request_safe_exit(ExitTrigger.TOOLBAR)
        except Exception as error:
            self._workflow.report_error(error)
            carb.log_warn(f"Classic shell command failed: {error}")
        self._persist_user_preferences()
        self._sync_render_from_state()
        self._refresh_ui()
        return True

    async def _complete_capture_analysis(self, surface, request) -> bool:
        """Build both capture residual presentations without blocking Kit UI."""

        try:
            analysis = await asyncio.to_thread(evaluate_analysis, request)
            self._facade.add_capture(surface, analysis)
            return True
        except Exception as error:
            self._workflow.report_error(error)
            carb.log_error(f"Capture analysis failed: {type(error).__name__}: {error}")
            return False

    def _apply_camera_command(self, command_id: CommandId) -> None:
        defaults = CommandFacade(self._app_config).snapshot
        state = self._facade.snapshot
        view = None
        manual = None
        if command_id is CommandId.CAMERA_RESET_SCENE:
            pose, role = defaults.scene_pose, "scene"
            view = self._startup_camera_views[CameraRole.SCENE]
        elif command_id is CommandId.CAMERA_RESET_SCANNER:
            pose, role = defaults.scanner_pose, "scanner"
            view = self._startup_camera_views[CameraRole.SCANNER]
            manual = self._startup_scanner_manual_fov
        elif command_id is CommandId.CAMERA_RESET_SCANNER_UP:
            camera = self._camera_controllers[CameraRole.SCANNER].snapshot
            view = CameraInputController(camera).reset_scanner_up()
            pose, role = view.pose_world, "scanner"
        else:
            preset = {
                CommandId.CAMERA_SCENE_POS_X: ScenePreset.POSITIVE_X,
                CommandId.CAMERA_SCENE_POS_Y: ScenePreset.POSITIVE_Y,
                CommandId.CAMERA_SCENE_POS_Z: ScenePreset.POSITIVE_Z,
            }[command_id]
            camera = self._camera_controllers[CameraRole.SCENE].snapshot
            view = scene_preset(camera, preset)
            pose, role = view.pose_world, "scene"
        if view is None:
            view = CameraSnapshot(CameraRole(role), pose, CameraIntrinsics(52.0, 16.0 / 9.0, 0.05, 100.0), SceneCatalog(state.objects).bounds.center_m)
            manual = False if role == "scanner" else None
        self._facade.replace_camera(role, pose, view=view, manual_fov=manual)
        self._restore_camera_controller(role)
        self._sync_sensor_camera_pose(CameraRole(role), self._camera_controllers[CameraRole(role)].snapshot.pose_world)

    def _restore_camera_controller(self, role: str) -> None:
        if self._scene_state is None or not self._camera_controllers:
            return
        role_enum = CameraRole(role)
        catalog = SceneCatalog(self._facade.snapshot.objects)
        snapshots = self._camera_snapshots(catalog, prefer_controllers=False)
        snapshot = snapshots[0] if role_enum is CameraRole.SCANNER else snapshots[1]
        self._camera_controllers[role_enum].restore(snapshot)
        if role_enum is CameraRole.SCANNER:
            self._sync_scanner_sensor_intrinsics()

    def _restore_all_camera_controllers(self) -> None:
        if len(self._camera_controllers) != len(CameraRole):
            return
        for role in CameraRole:
            self._restore_camera_controller(role.value)
        self._sync_scanner_sensor_intrinsics()

    def _sync_scanner_sensor_intrinsics(self) -> bool:
        """Apply the active D455 optical calibration without changing camera pose."""

        if self._facade.snapshot.scanner_manual_fov:
            return False
        coordinator = self._sensor_coordinator
        controller = self._camera_controllers.get(CameraRole.SCANNER)
        if coordinator is None or controller is None:
            return False
        depth = coordinator.active_depth_intrinsics()
        if depth is None:
            return False
        centered = (
            abs(depth.principal_x_px - depth.width * 0.5) <= 1.0e-4
            and abs(depth.principal_y_px - depth.height * 0.5) <= 1.0e-4
        )
        if not centered:
            carb.log_warn(
                "Active depth camera has an off-centre principal point; "
                "the symmetric OpenGL projection cannot represent it exactly"
            )
        before = controller.snapshot
        calibrated = CameraIntrinsics.from_centered_pinhole(
            depth.width,
            depth.height,
            depth.focal_x_px,
            depth.focal_y_px,
            near_m=before.intrinsics.near_m,
            far_m=before.intrinsics.far_m,
        )
        if calibrated == before.intrinsics:
            return False
        controller.restore(replace(before, intrinsics=calibrated))
        return True

    def _present_camera_controllers(self, *, sync_continuous_sensor: bool = False) -> None:
        if self._render_packet is None or len(self._camera_controllers) != len(CameraRole):
            return
        if sync_continuous_sensor:
            self._sync_sensor_camera_pose(
                CameraRole.SCANNER,
                self._camera_controllers[CameraRole.SCANNER].snapshot.pose_world,
            )
        cleared_scan = (
            sync_continuous_sensor
            and self._rendered_sensor_frame is not None
            and self._sensor_coordinator is not None
            and self._sensor_coordinator.latest_frame() is None
        )
        if self._hover_preview is not None or cleared_scan:
            # Re-pick a stationary cursor and remove a superseded point layer.
            self._sync_render_from_state()
        else:
            self._compositor.submit(
                self._render_packet,
                self._camera_controllers[CameraRole.SCANNER].snapshot,
                self._camera_controllers[CameraRole.SCENE].snapshot,
            )

    def _handle_viewport_pointer(self, action, x, y, button, logical_size, modifiers=0) -> None:
        if not self._camera_controllers or self._shell_snapshot.modal.input_blocked:
            return
        scene = self._camera_controllers[CameraRole.SCENE].snapshot
        if (self._compositor.snapshot.primary_role is CameraRole.SCENE
                and self._pointer_role is None
                and scene.intrinsics.orthographic_height_m is not None
                and hasattr(self._chrome, "scene_ruler")
                and self._chrome.scene_ruler.pointer(action, x, y, button)):
            if self._hover_preview is not None:
                self._hover_preview.clear_hover()
                self._sync_render_from_state(sync_hover_context=False)
            return
        if action == "wheel":
            if self._pointer_role is not None:
                return
            role = self._compositor.snapshot.primary_role
            result = self._camera_controllers[role].wheel_dolly(y)
            if result.command is not None:
                controller = self._camera_controllers[role]
                self._facade.replace_camera(role.value, controller.snapshot.pose_world, view=controller.snapshot)
                self._sync_render_from_state()
                self._refresh_ui()
            return
        if action == "leave":
            if self._hover_preview is not None:
                self._hover_preview.clear_hover()
                self._sync_render_from_state(sync_hover_context=False)
                self._refresh_ui()
            return
        if action == "enter":
            return
        if action == "move" and self._pointer_role is None:
            self._update_hover((x, y), logical_size)
            return
        if action == "begin":
            self._pointer_role = self._compositor.snapshot.primary_role
            self._pointer_retroactive_pick = bool(
                int(button) == 0
                and (
                    kit_modifier_has_alt(modifiers, carb.input)
                    or bool(self._keyboard and self._keyboard.alt_pressed)
                )
            )
        role = self._pointer_role
        if role is None:
            return
        controller = self._camera_controllers[role]
        if action == "begin":
            controller.pointer_begin(button, x, y, logical_size)
            self._sync_render_from_state(sync_hover_context=False)
        elif action == "move":
            result = controller.pointer_move(x, y)
            if result.captured:
                self._present_camera_controllers(sync_continuous_sensor=True)
        elif action == "end":
            result = controller.pointer_end(button, x, y)
            self._pointer_role = None
            retroactive_pick = self._pointer_retroactive_pick
            self._pointer_retroactive_pick = False
            if result.pick_click is not None:
                self._select_hover_seed(
                    result.pick_click,
                    logical_size,
                    retroactive=retroactive_pick,
                )
            if result.command is not None:
                self._facade.replace_camera(role.value, controller.snapshot.pose_world, view=controller.snapshot)
                self._sync_sensor_camera_pose(role, controller.snapshot.pose_world)
                self._sync_render_from_state()
                self._refresh_ui()
            elif result.pick_click is None:
                self._sync_render_from_state(sync_hover_context=False)

    def _hover_options(self):
        current = self._facade.snapshot.detection
        signature = replace(current, revision=0)
        if self._hover_options_signature is None:
            self._hover_options_signature = signature
        elif signature != self._hover_options_signature:
            self._hover_options_signature = signature
            self._hover_options_revision += 1
        return replace(current, revision=self._hover_options_revision)

    def _hover_generations(self, frame) -> GenerationSnapshot:
        if frame.identity != self._hover_frame_identity:
            self._hover_frame_identity = frame.identity
            self._hover_frame_revision += 1
        selection_revision = (
            0
            if self._hover_preview is None
            else self._hover_preview.picking.state.generations.selection_revision
        )
        return GenerationSnapshot(
            session=1,
            sensor=frame.generation,
            frame=self._hover_frame_revision,
            options_revision=self._hover_options_revision,
            selection_revision=selection_revision,
        )

    def _sync_hover_context(self, frame, camera, logical_size) -> None:
        preview = self._hover_preview
        if preview is None or frame is None or self._facade.snapshot.inspect_mode:
            return
        options = self._hover_options()
        presented_aspect = self._compositor.presented_aspect(camera.role)
        if presented_aspect is None:
            presented_aspect = max(1.0, float(logical_size[0])) / max(1.0, float(logical_size[1]))
        camera = with_viewport_aspect(camera, presented_aspect)
        interaction = self._facade.snapshot.interaction
        preview.configure_sticky(
            enabled=interaction.sticky_auto_preview,
            timeout_s=interaction.sticky_preview_timeout_ms / 1000.0,
        )
        preview.picking.update_settings(
            probe_radius_logical_px=interaction.probe_radius_px,
            pick_nearest_on_probe_miss=interaction.pick_nearest_on_probe_miss,
        )
        preview.update_context(
            frame,
            camera,
            logical_size,
            self._hover_generations(frame),
            options,
            sensor_active=(
                self._sensor_coordinator is not None
                and self._sensor_coordinator.snapshot.phase is SensorControlPhase.RUNNING
            ),
        )

    def _sync_hover_analysis(self, result, frame) -> None:
        """Submit the current successful preview to one latest-only analysis worker."""

        scheduler = self._analysis_scheduler
        if scheduler is None or result is None or frame is None:
            if scheduler is not None:
                scheduler.update_current(None)
            self._hover_analysis = None
            self._hover_analysis_current_key = None
            self._hover_analysis_pending_key = None
            self._hover_analysis_error = None
            self._hover_analysis_source = None
            return
        if (
            result.request_key.scan_generation != frame.generation
            or result.request_key.scan_sequence != frame.sequence
        ):
            self._sync_hover_analysis(None, None)
            return
        state = self._facade.snapshot
        scene_generation = 0 if self._scene_state is None else self._scene_state.scene_generation
        source = (
            id(result),
            frame.identity,
            scene_generation,
            state.detection.measurement_accuracy_m,
            state.objects,
        )
        if source == self._hover_analysis_source:
            return
        self._hover_analysis_source = source
        try:
            subject = AnalysisSubject.from_preview(
                result,
                frame,
                measurement_accuracy_m=state.detection.measurement_accuracy_m,
            )
            request = make_analysis_request(
                subject,
                state.objects,
                scene_generation=scene_generation,
                # Hover meshes are already supplied by RenderPayloadCache.
                # Avoid rebuilding them in the analysis worker, where the
                # Python mesh loops would still contend with Kit for the GIL.
                include_mesh_layers=False,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._hover_analysis = None
            self._hover_analysis_pending_key = None
            self._hover_analysis_error = f"{type(error).__name__}: {error}"
            return
        scheduler.update_current(request.key)
        if request.key == self._hover_analysis_current_key:
            return
        self._hover_analysis = None
        self._hover_analysis_error = None
        self._hover_analysis_current_key = request.key
        self._hover_analysis_pending_key = request.key
        scheduler.submit(request)

    def _drain_hover_analysis_completion(self) -> bool:
        scheduler = self._analysis_scheduler
        completion = None if scheduler is None else scheduler.drain_completion()
        if completion is None:
            return False
        if completion.key != self._hover_analysis_current_key:
            return False
        if completion.status is AnalysisCompletionStatus.PUBLISHED:
            self._hover_analysis = completion.presentation
            self._hover_analysis_pending_key = None
            self._hover_analysis_error = None
            return True
        if completion.status is AnalysisCompletionStatus.FAILED:
            self._hover_analysis = None
            self._hover_analysis_pending_key = None
            lines = tuple(line.strip() for line in (completion.error or "analysis failed").splitlines() if line.strip())
            self._hover_analysis_error = lines[-1] if lines else "analysis failed"
            carb.log_warn(f"Hover analysis failed: {self._hover_analysis_error}")
            return True
        return False

    def _create_hover_detector(self):
        library_path = self._findsurface_library_path or None
        return create_domain_detector(
            library_path=library_path,
            allow_sdk_download=library_path is None,
        )

    def _select_hover_seed(self, cursor, logical_size, *, retroactive: bool = False) -> None:
        if self._facade.snapshot.inspect_mode:
            return
        preview = self._hover_preview
        if preview is None:
            self._update_hover(cursor, logical_size)
            preview = self._hover_preview
        if preview is None:
            return
        frame = None if self._sensor_coordinator is None else self._sensor_coordinator.latest_frame()
        if frame is None:
            return
        self._sync_hover_context(
            frame,
            self._camera_controllers[self._compositor.snapshot.primary_role].snapshot,
            logical_size,
        )
        if retroactive:
            transition = preview.select_last_successful()
        else:
            transition = preview.select_click(cursor)
        if transition.state.selection is None:
            # Demo 1 immediately restores ordinary hover after an empty click
            # clears a locked seed.
            preview.update_hover(cursor)
        self._sync_render_from_state(sync_hover_context=False)
        self._refresh_ui()

    def _update_hover(self, cursor, logical_size) -> None:
        if self._facade.snapshot.inspect_mode:
            return
        coordinator = self._sensor_coordinator
        frame = None if coordinator is None else coordinator.latest_frame()
        if frame is None or not self._camera_controllers:
            return
        camera = self._camera_controllers[self._compositor.snapshot.primary_role].snapshot
        try:
            presented_aspect = self._compositor.presented_aspect(camera.role)
            if presented_aspect is None:
                presented_aspect = max(1.0, float(logical_size[0])) / max(1.0, float(logical_size[1]))
            camera = with_viewport_aspect(camera, presented_aspect)
            if self._hover_preview is None:
                options = self._hover_options()
                interaction = self._facade.snapshot.interaction
                picking = PickingController(
                    frame,
                    camera,
                    logical_size,
                    self._hover_generations(frame),
                    sensor_active=coordinator.snapshot.phase is SensorControlPhase.RUNNING,
                    probe_radius_logical_px=interaction.probe_radius_px,
                    pick_nearest_on_probe_miss=interaction.pick_nearest_on_probe_miss,
                )
                self._hover_preview = PreviewCaptureController.create(
                    picking,
                    self._create_hover_detector,
                    frame,
                    options,
                    sticky_timeout_s=(
                        interaction.sticky_preview_timeout_ms / 1000.0
                        if interaction.sticky_auto_preview else None
                    ),
                )
            else:
                self._sync_hover_context(frame, camera, logical_size)
            self._hover_preview.update_hover(cursor)
            self._sync_render_from_state(sync_hover_context=False)
            self._refresh_ui()
        except Exception as error:
            self._workflow.report_error(error)
            carb.log_warn(f"Hover detection update failed: {error}")

    def _pause_hover_detection(self) -> None:
        """Enter Demo 1's pointer/inspect mode without waiting for FindSurface."""

        preview = self._hover_preview
        if preview is not None:
            preview.pause_detection()
        self._sync_hover_analysis(None, None)
        self._render_payloads.clear_classified()
        self._render_payloads.clear_detected()

    def _handle_camera_key(self, key: str, pressed: bool, *, repeat: bool = False) -> bool:
        controllers = getattr(self, "_camera_controllers", None)
        if not controllers or self._shell_snapshot.modal.input_blocked:
            return False
        if key.upper() in ('ESC', 'ESCAPE'):
            if pressed and self._hover_preview is not None:
                self._hover_preview.cancel_selection()
                self._sync_render_from_state()
                self._refresh_ui()
            return True
        role = self._compositor.snapshot.primary_role
        controller = controllers[role]
        movement = key in controller.MOVEMENT_KEYS
        first_press = movement and pressed and key not in controller.pressed_keys
        if movement and pressed and self._keyboard_camera_before is None:
            self._keyboard_camera_before = (role, controller.snapshot)
        intent = controller.key_event(key, pressed, repeat=repeat)
        # A normal tap can press and release between two Kit update frames.
        # Match demo 1's press-driven shortcuts with one deterministic movement
        # step, then retain held-key frame updates for smooth motion.
        if first_press:
            before = controller.snapshot
            controller.update(1.0 / 30.0)
            if controller.snapshot != before:
                self._present_camera_controllers(sync_continuous_sensor=True)
        if intent is ShortcutIntent.RESET_PRIMARY_CAMERA:
            command = CommandId.CAMERA_RESET_SCANNER if role is CameraRole.SCANNER else CommandId.CAMERA_RESET_SCENE
            self._execute_command(command)
        elif intent is ShortcutIntent.TOGGLE_PRIMARY_VIEW:
            command = CommandId.PRIMARY_SCENE if role is CameraRole.SCANNER else CommandId.PRIMARY_SCANNER
            self._execute_command(command)
        if movement and not pressed and not controller.pressed_keys and self._keyboard_camera_before is not None:
            before_role, before = self._keyboard_camera_before
            self._keyboard_camera_before = None
            if before_role is role and before != controller.snapshot:
                self._facade.replace_camera(role.value, controller.snapshot.pose_world, view=controller.snapshot)
                self._sync_sensor_camera_pose(role, controller.snapshot.pose_world)
                self._refresh_ui()
        return movement or intent is not None

    def _handle_modifier(self, modifier: str, _pressed: bool) -> None:
        if modifier in ("ctrl", "shift"):
            for controller in getattr(self, "_camera_controllers", {}).values():
                controller.set_modifier(modifier, _pressed)
        if modifier == "alt" and self._hover_preview is not None:
            self._sync_render_from_state(sync_hover_context=False)

    def _schedule_action(self, awaitable):
        async def run():
            try:
                return await awaitable
            finally:
                self._persist_user_preferences()
                if self._facade.snapshot.inspect_mode:
                    self._pause_hover_detection()
                self._restore_all_camera_controllers()
                self._sync_sensor_camera_pose(
                    CameraRole.SCANNER,
                    self._camera_controllers[CameraRole.SCANNER].snapshot.pose_world,
                )
                self._sync_render_from_state()
                self._refresh_ui()
        return asyncio.ensure_future(run())

    def _schedule_capture_reset(self):
        async def reset_captures():
            changed = await self._workflow.reset_captures()
            if changed and self._hover_preview is not None:
                self._hover_preview.clear_captures()
            return changed

        return self._schedule_action(reset_captures())

    def _schedule_sensor_mode(self, mode: SensorMode | str):
        coordinator = self._sensor_coordinator
        if coordinator is None:
            return False
        mode = SensorMode(mode)
        current = self._facade.snapshot.sensor
        target = current.with_mode(mode)
        if target == current:
            return True
        token = self._shell_model.modal.begin(
            ModalKind.PROGRESS,
            "Point-cloud sensor",
            {
                SensorMode.FROZEN: "Freezing pose and stopping updates",
                SensorMode.LIVE_FIXED: "Starting updates at the fixed pose",
                SensorMode.LIVE_FOLLOW: "Synchronizing with the scanner camera",
            }[mode],
        )
        self._refresh_ui()

        async def apply_setting(setting):
            # Pose mode is applied before enabling. Entering LIVE_FOLLOW thus
            # copies the latest scanner camera pose into the sensor slot before
            # the first resumed frame can be published.
            if setting.follows_camera:
                coordinator.follow_sensor()
            else:
                coordinator.freeze_sensor()
            if setting.enabled and coordinator.snapshot.phase is not SensorControlPhase.RUNNING:
                snapshot = await coordinator.enable()
            elif not setting.enabled and coordinator.snapshot.phase is not SensorControlPhase.DISABLED:
                snapshot = coordinator.disable()
            else:
                snapshot = coordinator.snapshot
            if snapshot.phase is SensorControlPhase.ERROR:
                raise RuntimeError(snapshot.error or "sensor transition failed")

        async def transition(before, after):
            self._shell_model.modal.update(token, detail="Applying sensor mode", progress=0.5)
            try:
                await apply_setting(after)
            except Exception:
                try:
                    await apply_setting(before)
                except Exception as rollback_error:
                    carb.log_error(f"Sensor mode rollback failed: {rollback_error}")
                raise

        async def run():
            try:
                changed = await self._facade.select_sensor(target, transition)
                if not changed:
                    raise RuntimeError(self._facade.status.error or "sensor state change rolled back")
                self._sync_render_from_state()
                self._shell_model.modal.succeed(token, {
                    SensorMode.FROZEN: "Sensor pose and point cloud frozen",
                    SensorMode.LIVE_FIXED: "Point cloud updating at fixed pose",
                    SensorMode.LIVE_FOLLOW: "Sensor synchronized with scanner camera",
                }[mode])
                await self._dismiss_modal_after_updates(token, 6)
                return True
            except Exception as error:
                self._shell_model.modal.fail(token, error)
                self._workflow.report_error(error)
                return False
            finally:
                self._refresh_ui()

        return asyncio.ensure_future(run())

    def _schedule_sensor_profile(self, sensor_kind: SensorKind, profile_id: str):
        coordinator = self._sensor_coordinator
        if coordinator is None:
            return False
        current = self._facade.snapshot.sensor
        if current.sensor_kind is sensor_kind and self._sensor_selection(current).profile_id == profile_id:
            return True
        recommendation = self._recommended_sensor_material(sensor_kind, profile_id)
        token = self._shell_model.modal.begin(
            ModalKind.PROGRESS,
            "Point-cloud sensor",
            f"Preparing {sensor_kind.value}/{profile_id}",
        )
        self._refresh_ui()

        async def transition(before, after):
            before_selection = self._sensor_selection(before)
            target_selection = self._sensor_selection(after)
            self._shell_model.modal.update(
                token,
                detail="Stopping current sensor, then testing selected profile",
                progress=0.5,
            )
            carb.log_info(
                "Exclusive sensor switch requested: "
                f"{before_selection.sensor_kind.value}/{before_selection.profile_id} -> "
                f"{target_selection.sensor_kind.value}/{target_selection.profile_id}"
            )
            try:
                await self._apply_target_material(after.target_material, rescan_frozen=False)
                snapshot = await coordinator.switch(target_selection)
                if snapshot.error is not None or snapshot.active != target_selection:
                    raise RuntimeError(snapshot.error or "sensor profile transition failed")
                self._sync_scanner_sensor_intrinsics()
                carb.log_info(
                    "Exclusive sensor switch completed: "
                    f"{target_selection.sensor_kind.value}/{target_selection.profile_id}"
                )
            except Exception as switch_error:
                active = coordinator.snapshot.active
                active_label = (
                    "none"
                    if active is None
                    else f"{active.sensor_kind.value}/{active.profile_id}"
                )
                carb.log_warn(
                    "Exclusive sensor switch failed: "
                    f"target={target_selection.sensor_kind.value}/{target_selection.profile_id}, "
                    f"restored={active_label}, error={switch_error}"
                )
                try:
                    await self._apply_target_material(before.target_material, rescan_frozen=False)
                    if coordinator.snapshot.phase is SensorControlPhase.DISABLED:
                        await coordinator.scan_once()
                except Exception as material_error:
                    carb.log_error(f"Sensor material rollback failed: {material_error}")
                raise

        async def run():
            target = replace(
                current,
                sensor_kind=sensor_kind,
                profile_id=profile_id,
                target_material=recommendation,
            )
            try:
                changed = await self._facade.select_sensor(target, transition)
                if not changed:
                    raise RuntimeError(self._facade.status.error or "sensor profile change rolled back")
                self._persist_user_preferences()
                self._sync_render_from_state()
                if coordinator.snapshot.no_returns:
                    detail = (
                        f"Selected {profile_id}. Sensor is ready, but no object is returning points "
                        "within the valid range and the current field of view."
                    )
                else:
                    detail = f"Selected {profile_id}"
                self._shell_model.modal.succeed(token, detail)
                await self._dismiss_modal_after_updates(token, 6)
                return True
            except Exception as error:
                self._shell_model.modal.fail(token, error)
                self._workflow.report_error(error)
                return False
            finally:
                self._refresh_ui()

        return asyncio.ensure_future(run())

    async def _apply_target_material(
        self, setting: SensorMaterialSetting, *, rescan_frozen: bool = True,
    ):
        if self._scene_session is None:
            raise RuntimeError("scene session is not ready")
        self._scene_session.apply_sensor_material(setting)
        app = omni.kit.app.get_app()
        # Let Hydra/Fabric consume the material opinion before asking a frozen
        # sensor for its replacement point cloud.
        await app.next_update_async()
        await app.next_update_async()
        coordinator = self._sensor_coordinator
        if (
            rescan_frozen
            and coordinator is not None
            and coordinator.snapshot.phase is SensorControlPhase.DISABLED
        ):
            frame = await coordinator.scan_once()
            if frame is None and not coordinator.snapshot.no_returns:
                raise RuntimeError("target-material one-shot scan produced no valid frame")
            return frame
        return None

    def _schedule_sensor_material(self, payload: dict[str, object]):
        field = str(payload.get("field", ""))
        value = str(payload.get("value", ""))
        field_options = {
            "base": SENSOR_MATERIAL_BASES,
            "coating": SENSOR_MATERIAL_COATINGS,
            "attribute": SENSOR_MATERIAL_ATTRIBUTES,
        }
        if field not in field_options or value not in field_options[field]:
            raise ValueError(f"unsupported target material selection: {field}={value}")
        before_material = self._facade.snapshot.sensor.target_material
        target_material = replace(before_material, **{field: value})
        if target_material == before_material:
            return True

        async def transition(before, after):
            try:
                await self._apply_target_material(after.target_material)
            except Exception:
                try:
                    await self._apply_target_material(before.target_material)
                except Exception as rollback_error:
                    carb.log_error(f"Target material rollback failed: {rollback_error}")
                raise

        async def run():
            try:
                changed = await self._facade.set_sensor_material(target_material, transition)
                if not changed:
                    raise RuntimeError(self._facade.status.error or "target material change rolled back")
                self._persist_user_preferences()
                self._workflow.invalidate(
                    "Target material changed. Point cloud refreshed; run Preview again."
                )
                self._sync_render_from_state()
                return True
            except Exception as error:
                self._workflow.report_error(error)
                return False
            finally:
                self._refresh_ui()

        return asyncio.ensure_future(run())

    async def _dismiss_modal_after_updates(self, token, update_count: int) -> None:
        for _ in range(max(0, int(update_count))):
            await omni.kit.app.get_app().next_update_async()
        if self._shell_model.modal.dismiss(token):
            self._refresh_ui()

    def _schedule_sensor_one_shot(self):
        coordinator = self._sensor_coordinator
        if coordinator is None or coordinator.snapshot.phase is not SensorControlPhase.DISABLED:
            return False

        async def run():
            frame = await coordinator.scan_once()
            if frame is not None:
                self._sync_render_from_state()
            self._refresh_ui()
            return frame is not None

        return asyncio.ensure_future(run())

    def _schedule_depth_diagnostics_refresh(self):
        """Coalesce Frozen diagnostic edits to one latest native scan."""

        self._depth_diagnostics_revision += 1
        if (
            self._depth_diagnostics_refresh_task is not None
            and not self._depth_diagnostics_refresh_task.done()
        ):
            return self._depth_diagnostics_refresh_task

        async def run():
            refreshed = False
            while True:
                await omni.kit.app.get_app().next_update_async()
                requested_revision = self._depth_diagnostics_revision
                coordinator = self._sensor_coordinator
                if (
                    coordinator is None
                    or coordinator.snapshot.phase is not SensorControlPhase.DISABLED
                    or coordinator.snapshot.active is None
                ):
                    break
                frame = await coordinator.scan_once()
                refreshed = refreshed or frame is not None
                if frame is not None:
                    self._sync_render_from_state()
                if requested_revision == self._depth_diagnostics_revision:
                    break
            self._refresh_ui()
            return refreshed

        self._depth_diagnostics_refresh_task = asyncio.ensure_future(run())
        return self._depth_diagnostics_refresh_task

    def _sync_sensor_camera_pose(
        self, role: CameraRole, pose: Pose, *, invalidate_frame: bool = True,
    ) -> None:
        coordinator = self._sensor_coordinator
        if role is CameraRole.SCANNER and coordinator is not None:
            coordinator.set_camera_pose(pose, invalidate_frame=invalidate_frame)

    def _dismiss_modal(self) -> None:
        self._shell_model.modal.dismiss()
        self._refresh_ui()

    def _schedule_settings_defaults(self, *, scope=None):
        token = self._shell_model.modal.begin(
            ModalKind.PROGRESS, "Restore settings" if scope is None else f"Reset {scope} to startup",
            "Restoring settings; please wait",
        )
        self._refresh_ui()
        original = self._facade.snapshot
        original_diagnostics = self._depth_diagnostics
        defaults_diagnostics = diagnostics_for_reset(
            original_diagnostics,
            depth_diagnostics_from_config(self._settings_defaults) if scope is None else self._startup_depth_diagnostics,
            scope,
        )
        original_display = (self._anti_aliasing_mode, self._pip_visible, self._pip_width)
        default_display = ((AntiAliasingMode.TAA, True, 0.32) if scope is None else
                           self._startup_display if scope == "all" else original_display)

        async def apply(state, diagnostics, display):
            coordinator = self._sensor_coordinator
            if coordinator is None or self._scene_state is None:
                raise RuntimeError("Sensor and scene must be ready before restoring settings")
            if coordinator.snapshot.phase is SensorControlPhase.RUNNING:
                coordinator.disable()
            coordinator.freeze_sensor()
            self._depth_diagnostics = diagnostics
            coordinator.configure_depth_diagnostics(diagnostics)
            self._restoring_settings = True
            try:
                for item in state.objects:
                    current_item = self._scene_state.catalog.object(item.geometry_type)
                    if current_item.sensor_signature != item.sensor_signature:
                        self._scene_state.replace_object(item)
                    elif current_item != item:
                        self._scene_state.replace_presentation(item)
            finally:
                self._restoring_settings = False
            await self._apply_target_material(state.sensor.target_material, rescan_frozen=False)
            coordinator.set_camera_pose(state.scanner_pose)
            if state.sensor.follows_camera:
                coordinator.follow_sensor()
            selection = self._sensor_selection(state.sensor)
            if coordinator.snapshot.active != selection:
                status = await coordinator.switch(selection)
                if status.error or status.active != selection:
                    raise RuntimeError(status.error or "Sensor default profile could not be restored")
            if state.sensor.enabled:
                status = await coordinator.enable()
                if status.error:
                    raise RuntimeError(status.error)
            else:
                frame = await coordinator.scan_once()
                if frame is None and not coordinator.snapshot.no_returns:
                    raise RuntimeError(coordinator.snapshot.error or "Default settings scan failed")
            self._chrome.set_theme(state.theme)
            self._anti_aliasing_mode, self._pip_visible, self._pip_width = display
            settings = carb.settings.get_settings()
            settings.set_int("/rtx/post/aa/op", int(self._anti_aliasing_mode))
            settings.set_int("/rtx-defaults/post/aa/op", int(self._anti_aliasing_mode))
            self._compositor.set_pip_visible(self._pip_visible)
            self._compositor.set_pip_width_fraction(self._pip_width)
            self._workflow.invalidate("Settings restored; captures retained.")
            if self._hover_preview is not None:
                self._hover_preview.cancel_selection()

        async def transition(before, after):
            target_diagnostics = original_diagnostics if after is original else defaults_diagnostics
            prior_diagnostics = self._depth_diagnostics
            target_display = original_display if after is original else default_display
            prior_display = (self._anti_aliasing_mode, self._pip_visible, self._pip_width)
            try:
                await apply(after, target_diagnostics, target_display)
            except Exception:
                await apply(before, prior_diagnostics, prior_display)
                raise

        async def run():
            try:
                for task in (self._object_change_task, self._depth_diagnostics_refresh_task):
                    if task is not None and not task.done():
                        await task
                changed = (await self._facade.restore_settings(self._settings_defaults, transition)
                           if scope is None else
                           await self._facade.restore_startup_scope(scope, transition))
                if not changed:
                    raise RuntimeError(self._facade.status.error or "Settings restoration failed")
                self._persist_user_preferences()
                self._restore_all_camera_controllers()
                self._sync_render_from_state()
                self._shell_model.modal.succeed(token, "Restoration complete.")
                await self._dismiss_modal_after_updates(token, 6)
                return True
            except Exception as error:
                self._shell_model.modal.fail(token, error)
                return False
            finally:
                self._refresh_ui()
        return asyncio.ensure_future(run())

    def _cancel_modal(self) -> None:
        self._shell_model.modal.cancel_current()
        self._refresh_ui()

    def _ui_presentations(self):
        cache = getattr(self, "_ui_presentation_cache", None)
        if cache is None:
            self._ui_presentation_cache = cache = UiPresentationCache()
        return cache

    def _property_values(self) -> dict[str, object]:
        state = self._facade.snapshot
        coordinator = self._sensor_coordinator
        export_frame = None if coordinator is None else coordinator.latest_frame()
        export_running = bool(self._xyz_export_running)
        export_enabled = bool(
            coordinator is not None
            and coordinator.snapshot.phase is SensorControlPhase.DISABLED
            and export_frame is not None
            and not export_running
        )
        if export_running:
            export_tooltip = "An XYZ export is currently running."
        elif export_enabled:
            export_tooltip = "Export the current frozen world-metre point cloud beside repo.bat."
        else:
            export_tooltip = "Freeze the sensor and acquire a point cloud before exporting."
        cache = self._ui_presentations()
        active = None if coordinator is None else coordinator.snapshot.active
        profile = (
            state.sensor.sensor_kind if active is None else active.sensor_kind,
            self._sensor_selection(state.sensor).profile_id if active is None else active.profile_id,
        )
        sections = [
            cache.resolve("detection", (state.detection, state.interaction.fs_parameter_ranges),
                          lambda: self._detection_property_values(state)),
            cache.resolve("interaction", (interaction_property_dependencies(state.interaction),
                                          state.point_depth_offset_m),
                          lambda: self._interaction_property_values(state)),
            cache.resolve("diagnostics", (self._depth_diagnostics, self._anti_aliasing_mode),
                          self._diagnostic_property_values),
            cache.resolve("material", (state.sensor.target_material,),
                          lambda: self._material_property_values(state)),
            cache.resolve("export", (export_enabled, export_running, self._xyz_export_status),
                          lambda: {"settings.export_xyz_enabled": export_enabled,
                                   "settings.export_xyz_tooltip": export_tooltip,
                                   "settings.export_xyz_status": self._xyz_export_status}),
            cache.resolve("display", (self._pip_width,), lambda: {"pip_width": self._pip_width}),
            cache.resolve("sensor-profile", profile,
                          lambda: self._sensor_profile_property_values(state)),
            cache.resolve("geometry", (
                tuple(identity_dependency(item) for item in state.captures),
                state.selected_capture_id, state.interaction.surface_side_scale,
            ), lambda: self._geometry_property_values(state)),
        ]
        for role, pose in (("scanner", state.scanner_pose), ("scene", state.scene_pose)):
            camera = self._camera_controllers.get(CameraRole(role))
            view = None if camera is None else camera.snapshot
            sections.append(cache.resolve(
                f"camera.{role}", (pose, view),
                lambda role=role, pose=pose: self._camera_property_values(role, pose),
            ))
        for item in state.objects:
            sections.append(cache.resolve(
                f"object.{item.geometry_type.value}", (item,),
                lambda item=item: self._object_property_values(item),
            ))
        return cache.merge(sections)

    def _detection_property_values(self, state):
        detection = state.detection
        values = {
            "measurement_accuracy": detection.measurement_accuracy_m,
            "mean_distance": detection.mean_distance_m,
            "seed_radius": detection.seed_radius_m,
            "lateral_extension": detection.lateral_extension,
            "radial_expansion": detection.radial_expansion,
            "settings.smart_convert_cone_to_cylinder": detection.smart_convert_cone_to_cylinder,
            "settings.smart_convert_torus_to_sphere": detection.smart_convert_torus_to_sphere,
            "settings.smart_convert_torus_to_cylinder": detection.smart_convert_torus_to_cylinder,
        }
        for name, bounds in zip(("measurement_accuracy", "mean_distance", "seed_radius"), state.interaction.fs_parameter_ranges):
            minimum, maximum = bounds
            values[f"{name}_bounds"] = bounds
            values[f"{name}_min"], values[f"{name}_max"] = minimum, maximum
            values[f"{name}_slider_index"] = slider_index_for_value(values[name], minimum, maximum)
        return values

    def _interaction_property_values(self, state):
        values = {
            "camera.point_depth_offset_m": state.point_depth_offset_m,
            "settings.point_size": state.interaction.point_rendering_size,
            "graphics.ruler_main_opacity": state.interaction.ruler_main_opacity,
            "graphics.ruler_pip_opacity": state.interaction.ruler_pip_opacity,
            "settings.inlier_size": state.interaction.inlier_rendering_size,
            "settings.probe_radius_px": state.interaction.probe_radius_px,
            "settings.pick_nearest": state.interaction.pick_nearest_on_probe_miss,
            "settings.sticky_auto_preview": state.interaction.sticky_auto_preview,
            "settings.sticky_timeout_ms": state.interaction.sticky_preview_timeout_ms,
            "settings.detection_average_window": state.interaction.detection_average_window,
            "settings.minimum_geometry_length_m": state.interaction.minimum_geometry_length_m,
            "settings.surface_side_scale_mode": (
                SurfaceSideScaleMode.AUTO,
                SurfaceSideScaleMode.FIXED,
                SurfaceSideScaleMode.SIGMA,
            ).index(state.interaction.surface_side_scale_mode),
            "settings.surface_side_fixed_scale_m": (
                state.interaction.surface_side_fixed_scale_m
            ),
            "settings.surface_side_sigma_multiplier": (
                state.interaction.surface_side_sigma_multiplier
            ),
        }
        for prefix, color in (
            ("settings.outlier_color", state.interaction.outlier_color_rgba),
            ("settings.inlier_negative", state.interaction.inlier_negative_rgba),
            ("settings.inlier_zero", state.interaction.inlier_zero_rgba),
            ("settings.inlier_positive", state.interaction.inlier_positive_rgba),
            ("settings.detected_preview_face", state.interaction.detected_preview_face_rgba),
            ("settings.detected_preview_wire", state.interaction.detected_preview_wire_rgba),
        ):
            for component, coordinate in zip("rgba", color):
                values[f"{prefix}.{component}"] = coordinate
        values["settings.detected_preview_subdivision"] = (
            state.interaction.detected_preview_subdivision
        )
        for mode in (
            GeometryMode.PLANE, GeometryMode.SPHERE, GeometryMode.CYLINDER,
            GeometryMode.CONE, GeometryMode.TORUS,
        ):
            for kind in ("detected", "captured"):
                appearance = state.interaction.mesh_appearance(kind, mode)
                prefix = f"graphics.{mode.value}.{kind}"
                values[f"{prefix}.subdivision"] = appearance.subdivision
                for name, color in (("face", appearance.face_rgba), ("wire", appearance.wire_rgba)):
                    for component, coordinate in zip("rgba", color):
                        values[f"{prefix}.{name}.{component}"] = coordinate
        return values

    def _diagnostic_property_values(self):
        values = {
            "settings.anti_aliasing_mode": int(self._anti_aliasing_mode),
            "settings.depth_disparity_noise": self._depth_diagnostics.disparity_noise_enabled,
            "settings.depth_disparity_noise_mean_px": self._depth_diagnostics.disparity_noise_mean_px,
            "settings.depth_disparity_noise_sigma_px": self._depth_diagnostics.disparity_noise_sigma_px,
            "settings.depth_outlier_removal": self._depth_diagnostics.outlier_removal_enabled,
            "scanner.axial_error_enabled": self._depth_diagnostics.axial_error_enabled,
            "scanner.axial_error_sigma_m": self._depth_diagnostics.axial_error_sigma_m,
            "scanner.axial_error_model": self._depth_diagnostics.axial_error_model,
            "scanner.axial_error_model_index": int(self._depth_diagnostics.axial_error_model == "distance"),
            "scanner.axial_error_slope_m_per_m": self._depth_diagnostics.axial_error_slope_m_per_m,
            "scanner.axial_error_intercept_m": self._depth_diagnostics.axial_error_intercept_m,
        }
        return values

    def _material_property_values(self, state):
        values = {
            "sensor.material_base_index": SENSOR_MATERIAL_BASES.index(
                state.sensor.target_material.base
            ),
            "sensor.material_coating_index": SENSOR_MATERIAL_COATINGS.index(
                state.sensor.target_material.coating
            ),
            "sensor.material_attribute_index": SENSOR_MATERIAL_ATTRIBUTES.index(
                state.sensor.target_material.attribute
            ),
            "sensor.material_base_options": SENSOR_MATERIAL_BASES,
            "sensor.material_coating_options": SENSOR_MATERIAL_COATINGS,
            "sensor.material_attribute_options": SENSOR_MATERIAL_ATTRIBUTES,
        }
        return values

    def _geometry_property_values(self, state):
        cache = self._ui_presentations()
        values = {}
        geometry_entries = []
        for item in state.captures:
            slot = f"capture-overlay.{item.capture_id}"
            overlay = cache.resolve(
                slot,
                (identity_dependency(item.surface), identity_dependency(item.analysis),
                 state.interaction.surface_side_scale),
                lambda item=item: build_capture_result_overlay(
                    item.surface, item.analysis,
                    surface_side_scale=state.interaction.surface_side_scale,
                ),
            )
            geometry_entries.append(GeometryCardPresentation(
                capture_id=item.capture_id,
                geometry_type=item.surface.geometry_type.value,
                mesh_visible=item.mesh_visible,
                inliers_visible=item.inliers_visible,
                selected=item.capture_id == state.selected_capture_id,
                rows=(("Name", item.capture_id), ("Type", item.surface.geometry_type.value))
                + tuple(row for row in overlay.rows if row[0] not in {"Pos", "Normal", "Axis", "Size"}),
                heatmap_basis=item.heatmap_basis.value,
                histograms=tuple((entry.basis.value, entry.histogram) for entry in overlay.histograms),
                comparison_rows=overlay.comparison,
                mesh_appearance=(
                    item.mesh_subdivision,
                    item.mesh_face_rgba,
                    item.mesh_wire_rgba,
                ),
            ))
        values["geometry.entries"] = tuple(geometry_entries)
        cache.retain("capture-overlay.", (f"capture-overlay.{item.capture_id}" for item in state.captures))
        return values

    def _sensor_catalog_property_values(self):
        depth_profile_entries = tuple(
            (
                SensorKind.DEPTH.value,
                profile.profile_id,
                f"DepthSensor · {profile.label}",
                (
                    ("Name", "Intel RealSense D455"),
                    ("Type", "Single-view stereo depth camera"),
                    ("Function", "Dense forward point-cloud acquisition for surface reconstruction."),
                    ("Profile", f"{profile.width}×{profile.height} · target {profile.target_point_count:,} points · {profile.tick_rate_hz:g} Hz"),
                    ("Range", "Defined by the official D455 RTX camera asset"),
                    ("Features", "Isaac Sim disparity noise and outlier-removal controls are available in Settings."),
                ),
                True,
                "",
            )
            for profile in DEPTH_PROFILES
        )
        lidar_profile_entries = tuple(
            (SensorKind.RTX_LIDAR.value, profile.profile_id,
             f"RTX LiDAR · {lidar_profile_display_label(profile)}",
             lidar_profile_description_rows(profile), True, "")
            for profile in SCANNER_LIDAR_PROFILES
        )
        return depth_profile_entries + lidar_profile_entries

    def _sensor_profile_property_values(self, state):
        values = {}
        coordinator = self._sensor_coordinator
        profile_entries = self._ui_presentations().resolve(
            "sensor-catalog", (), self._sensor_catalog_property_values,
        )
        active = None if coordinator is None else coordinator.snapshot.active
        active_kind = state.sensor.sensor_kind if active is None else active.sensor_kind
        active_profile_id = self._sensor_selection(state.sensor).profile_id if active is None else active.profile_id
        values["sensor.profile_entries"] = profile_entries
        values["sensor.profile_index"] = next(
            (
                index
                for index, entry in enumerate(profile_entries)
                if entry[0] == active_kind.value and entry[1] == active_profile_id
            ),
            0,
        )
        selected_entry = profile_entries[values["sensor.profile_index"]]
        selected_material = self._recommended_sensor_material(
            SensorKind(selected_entry[0]), selected_entry[1])
        values["sensor.material_recommendation"] = (
            f"App recommendation for {selected_entry[2]}: {selected_material.summary}"
        )
        return values

    def _camera_property_values(self, role, pose):
        values = {}
        camera = self._camera_controllers.get(CameraRole(role))
        if camera is not None:
            view = camera.snapshot
            values[f"camera.{role}.fov"] = view.intrinsics.vertical_fov_degrees
            if role == "scene":
                from .orthographic import matching_height
                values["camera.scene.projection"] = int(view.intrinsics.orthographic_height_m is not None)
                values["camera.scene.ortho_height"] = view.intrinsics.orthographic_height_m or matching_height(view)
            for component, coordinate in zip("xyzw", view.pose_world.orientation_xyzw):
                values[f"camera.{role}.q{component}"] = coordinate
            for component, coordinate in zip("xyz", view.target_world_m):
                values[f"camera.{role}.target_{component}"] = coordinate
            values[f"camera.{role}.orbit_distance"] = math.sqrt(sum((a-b)**2 for a,b in zip(view.pose_world.position_m, view.target_world_m)))
        for axis, value in zip("xyz", pose.position_m):
            values[f"camera.{role}.{axis}"] = value
        return values

    def _object_property_values(self, item):
        values = {}
        mode = item.geometry_type.value
        values[f"object.{mode}.enabled"] = item.enabled
        values[f"object.{mode}.parameter_names"] = tuple(item.parameter_dict)
        values[f"object.{mode}.subdivision"] = item.subdivision
        values[f"object.{mode}.display_subdivision"] = item.display_subdivision
        for component, value in zip("rgba", item.face_color_rgba):
            values[f"object.{mode}.face_color.{component}"] = value
        for component, value in zip("rgba", item.wire_color_rgba):
            values[f"object.{mode}.wire_color.{component}"] = value
        for axis, coordinate in zip("xyz", item.pose_world_m.position_m):
            values[f"object.{mode}.position.{axis}"] = coordinate
        for axis, direction in zip("xyz", orientation_axes(item.pose_world_m.orientation_xyzw)):
            for component, coordinate in zip("xyz", direction):
                values[f"object.{mode}.axis.{axis}.{component}"] = coordinate
        for name, parameter in item.parameters:
            values[f"object.{mode}.parameter.{name}"] = parameter
        return values

    def _handle_property_or_layout_action(self, action: str, value=None):
        if action == "settings.reset_window_layout":
            if self._shell_model.modal.snapshot.input_blocked:
                return False
            return asyncio.ensure_future(self._reset_window_layout())
        return self._handle_property_action(action, value)

    def _handle_property_action(self, action: str, value=None):
        if self._shell_model.modal.snapshot.input_blocked:
            return False
        if action == "reset.startup":
            return self._schedule_settings_defaults(scope=str(value))
        try:
            outcome = _PROPERTY_ACTION_REFRESH
            domain = action.partition(".")[0]
            if domain == "findsurface":
                outcome = self._handle_findsurface_property(action, value)
            elif domain == "object":
                outcome = self._handle_object_property(action, value)
            elif domain == "camera":
                outcome = self._handle_camera_property(action, value)
            elif domain == "geometry":
                outcome = self._handle_geometry_property(action, value)
            elif domain == "scanner":
                outcome = self._handle_scanner_property(action, value)
            elif domain == "graphics":
                outcome = self._handle_graphics_property(action, value)
            elif domain == "settings":
                outcome = self._handle_settings_and_graphics_property(action, value)
            if outcome is not _PROPERTY_ACTION_REFRESH:
                return outcome
            self._persist_user_preferences()
            self._sync_render_from_state()
            self._refresh_ui()
            return True
        except Exception as error:
            carb.log_error(
                f"Property action {action!r} failed for value {value!r}: "
                f"{type(error).__name__}: {error}"
            )
            self._workflow.report_error(error)
            self._refresh_ui()
            return False

    def _handle_findsurface_property(self, action: str, value):
        if action.startswith("findsurface.") and action.split(".")[-1] in {
            "measurement_accuracy", "mean_distance", "seed_radius", "lateral_extension", "radial_expansion",
        }:
            field = action.split(".")[-1]
            names = {
                "measurement_accuracy": "measurement_accuracy_m",
                "mean_distance": "mean_distance_m",
                "seed_radius": "seed_radius_m",
            }
            key = names.get(field, field)
            options = replace(self._facade.snapshot.detection, **{key: value})
            self._facade.replace_detection(options)
            self._workflow.invalidate()
        elif action == "findsurface.mode":
            command = {
                "Auto": CommandId.MODE_AUTO, "Plane": CommandId.MODE_PLANE,
                "Sphere": CommandId.MODE_SPHERE, "Cylinder": CommandId.MODE_CYLINDER,
                "Cone": CommandId.MODE_CONE, "Torus": CommandId.MODE_TORUS,
            }[str(value)]
            return self._execute_command(command)
        elif action == "findsurface.ranges":
            self._facade.set_interaction(fs_parameter_ranges=value)
        return _PROPERTY_ACTION_REFRESH

    def _handle_object_property(self, action: str, value):
        if action == "object.toggle":
            mode = GeometryMode(str(value))
            current = next(item for item in self._facade.snapshot.objects if item.geometry_type is mode)
            updated = replace(current, enabled=not current.enabled)
            self._facade.replace_object(updated)
            if self._scene_state is not None:
                self._scene_state.replace_object(updated)
            self._workflow.invalidate("Object changed. Selection cleared; run Preview again.")
        elif action == "object.update":
            payload = dict(value)
            mode = GeometryMode(payload["mode"])
            current = next(item for item in self._facade.snapshot.objects if item.geometry_type is mode)
            edited_axis = str(payload.get("edited_axis", "x"))
            edited_parameter = payload.get("edited_parameter")
            if edited_parameter is None:
                current_parameters = current.parameter_dict
                changed_parameters = tuple(
                    name
                    for name, parameter in dict(payload["parameters"]).items()
                    if float(parameter) != current_parameters.get(name)
                )
                edited_parameter = changed_parameters[0] if len(changed_parameters) == 1 else None
            parameters = constrain_geometry_parameter_edit(
                mode,
                payload["parameters"],
                edited_parameter=edited_parameter,
                minimum_length_m=self._facade.snapshot.interaction.minimum_geometry_length_m,
            )
            orientation, _axes = corrected_orientation(
                tuple(tuple(axis) for axis in payload["orientation_axes"]),
                current.pose_world_m.orientation_xyzw,
                edited_axis=edited_axis if edited_axis in "xyz" else "x",
            )
            updated = type(current)(
                mode,
                current.enabled,
                Pose(payload["position_m"], orientation),
                parameters,
                int(payload["subdivision"]),
                int(payload["display_subdivision"]),
                payload["face_color_rgba"],
                payload["wire_color_rgba"],
            )
            if updated != current:
                self._facade.replace_object(updated)
                if updated.sensor_signature != current.sensor_signature:
                    if self._scene_state is not None:
                        self._scene_state.replace_object(updated)
                    self._workflow.invalidate("Object changed. Selection cleared; run Preview again.")
                elif self._scene_state is not None:
                    self._scene_state.replace_presentation(updated)
        elif action == "object.rotate.begin":
            mode = GeometryMode(str(value))
            current = next(item for item in self._facade.snapshot.objects if item.geometry_type is mode)
            camera = self._camera_controllers[self._compositor.snapshot.primary_role].snapshot
            self._object_rotation_gesture = {
                "mode": mode,
                "before": self._facade.snapshot,
                "object": current,
                "right": camera.right,
                "up": camera.up,
                "forward": camera.forward,
            }
            self._workflow.invalidate("Object rotation started. Selection cleared.")
        elif action == "object.rotate.update":
            payload = dict(value)
            gesture = self._object_rotation_gesture
            mode = GeometryMode(payload["mode"])
            if gesture is None or gesture["mode"] is not mode:
                raise RuntimeError("object rotation update has no matching gesture")
            original = gesture["object"]
            if "roll_radians" in payload:
                orientation = roll_in_view(
                    original.pose_world_m.orientation_xyzw,
                    camera_forward=gesture["forward"],
                    # A positive screen-space angle is counter-clockwise to
                    # the viewer, while positive rotation about a camera
                    # forward axis points away from the viewer. Negate it
                    # so the object follows the pointer like a grabbed item.
                    radians=-float(payload["roll_radians"]),
                )
            else:
                preview = original
                orientation = rotate_in_view(
                    original.pose_world_m.orientation_xyzw,
                    camera_right=gesture["right"], camera_up=gesture["up"],
                    drag_x=float(payload["drag_x"]), drag_y=float(payload["drag_y"]),
                )
            updated = replace(original, pose_world_m=Pose(original.pose_world_m.position_m, orientation))
            self._facade.preview_object(updated)
            if self._scene_session is not None:
                self._scene_session.write_object(updated)
        elif action == "object.rotate.end":
            gesture, self._object_rotation_gesture = self._object_rotation_gesture, None
            mode = GeometryMode(str(value))
            if gesture is None or gesture["mode"] is not mode:
                raise RuntimeError("object rotation end has no matching gesture")
            current = next(item for item in self._facade.snapshot.objects if item.geometry_type is mode)
            self._facade.commit_object_preview(gesture["before"], mode)
            if self._scene_state is not None and current != gesture["object"]:
                self._scene_state.replace_object(current)
        return _PROPERTY_ACTION_REFRESH

    def _handle_camera_property(self, action: str, value):
        if action.startswith("camera.edit."):
            role = action.rsplit(".", 1)[-1]
            before = self._camera_controllers[CameraRole(role)].snapshot
            field, edited = value
            changed = before
            manual = None
            if field == "projection" and role == "scene":
                from .orthographic import matching_height
                height = (before.intrinsics.orthographic_height_m or matching_height(before)) if int(edited) else None
                changed = replace(before, intrinsics=replace(before.intrinsics, orthographic_height_m=height))
            elif field == "ortho_height" and role == "scene":
                if before.intrinsics.orthographic_height_m is None:
                    return
                changed = replace(before, intrinsics=replace(before.intrinsics, orthographic_height_m=float(edited)))
            elif field == "fov":
                changed = replace(before, intrinsics=replace(before.intrinsics, vertical_fov_degrees=float(edited)))
                manual = True if role == "scanner" else None
            elif field == "quaternion":
                changed = replace(before, pose_world=Pose(before.pose_world.position_m, tuple(edited)))
                if role == "scene":
                    distance = math.sqrt(sum((p-t)**2 for p,t in zip(before.pose_world.position_m, before.target_world_m)))
                    changed = replace(changed, target_world_m=tuple(p+distance*f for p,f in zip(changed.pose_world.position_m, changed.forward)))
            elif field == "target":
                delta = tuple(float(v)-old for v, old in zip(edited, before.target_world_m))
                changed = replace(before, target_world_m=tuple(edited), pose_world=Pose(tuple(p+d for p,d in zip(before.pose_world.position_m, delta)), before.pose_world.orientation_xyzw))
            elif field == "distance":
                distance = float(edited)
                if not math.isfinite(distance) or distance < 0.01:
                    raise ValueError("orbit distance must be at least 0.01 m")
                offset = tuple(p-t for p,t in zip(before.pose_world.position_m, before.target_world_m))
                length = math.sqrt(sum(v*v for v in offset))
                direction = tuple(v/length for v in offset) if length > 1e-9 else tuple(-v for v in before.forward)
                changed = replace(before, pose_world=Pose(tuple(t+distance*v for t,v in zip(before.target_world_m, direction)), before.pose_world.orientation_xyzw))
            elif field == "sensor_fov":
                manual = False
            self._facade.replace_camera(role, changed.pose_world, view=changed, manual_fov=manual)
            self._camera_controllers[CameraRole(role)].restore(changed)
            if field == "sensor_fov":
                self._sync_scanner_sensor_intrinsics()
            self._sync_sensor_camera_pose(CameraRole(role), changed.pose_world)
        elif action.startswith("camera.position."):
            role = action.rsplit(".", 1)[-1]
            before = self._camera_controllers[CameraRole(role)].snapshot
            delta = tuple(float(v)-p for v,p in zip(value, before.pose_world.position_m))
            changed = replace(before, pose_world=Pose(tuple(value), before.pose_world.orientation_xyzw), target_world_m=tuple(t+d for t,d in zip(before.target_world_m, delta)))
            self._facade.replace_camera(role, changed.pose_world, view=changed)
            self._restore_camera_controller(role)
            self._sync_sensor_camera_pose(CameraRole(role), self._camera_controllers[CameraRole(role)].snapshot.pose_world)
        elif action.startswith("camera.reset."):
            role = action.rsplit(".", 1)[-1]
            command = CommandId.CAMERA_RESET_SCANNER if role == "scanner" else CommandId.CAMERA_RESET_SCENE
            return self._execute_command(command)
        elif action == "camera.scene.preset":
            command = {"x": CommandId.CAMERA_SCENE_POS_X, "y": CommandId.CAMERA_SCENE_POS_Y, "z": CommandId.CAMERA_SCENE_POS_Z}[str(value)]
            return self._execute_command(command)
        elif action == "camera.point_depth_offset":
            self._facade.set_point_depth_offset(float(value))
        return _PROPERTY_ACTION_REFRESH

    def _handle_geometry_property(self, action: str, value):
        if action == "geometry.toggle_selected":
            capture_id = self._facade.snapshot.selected_capture_id
            if capture_id:
                item = next(item for item in self._facade.snapshot.captures if item.capture_id == capture_id)
                self._facade.set_capture_visibility(capture_id, not item.visible)
        elif action == "geometry.delete_selected":
            capture_id = self._facade.snapshot.selected_capture_id
            if capture_id:
                self._facade.delete_capture(capture_id)
        elif action == "geometry.select":
            capture_id = str(value)
            self._facade.select_capture(capture_id)
        elif action == "geometry.toggle":
            capture_id = str(value)
            item = next(
                item for item in self._facade.snapshot.captures
                if item.capture_id == capture_id
            )
            self._facade.set_capture_visibility(capture_id, not item.visible)
        elif action == "geometry.toggle_mesh":
            capture_id = str(value)
            item = next(
                item for item in self._facade.snapshot.captures
                if item.capture_id == capture_id
            )
            self._facade.set_capture_mesh_visibility(capture_id, not item.mesh_visible)
        elif action == "geometry.toggle_inliers":
            capture_id = str(value)
            item = next(
                item for item in self._facade.snapshot.captures
                if item.capture_id == capture_id
            )
            self._facade.set_capture_inliers_visibility(capture_id, not item.inliers_visible)
        elif action == "geometry.heatmap_basis":
            capture_id, basis = value
            self._facade.set_capture_heatmap_basis(
                str(capture_id), ResidualBasis(str(basis)),
            )
        elif action == "geometry.mesh_appearance":
            payload = dict(value)
            self._facade.set_capture_mesh_appearance(
                str(payload["capture_id"]),
                subdivision=max(3, min(256, int(payload["subdivision"]))),
                face_rgba=payload["face_rgba"],
                wire_rgba=payload["wire_rgba"],
            )
        elif action == "geometry.delete":
            self._facade.delete_capture(str(value))
        elif action == "geometry.set_all_visibility":
            self._facade.set_all_capture_visibility(bool(value))
        elif action == "geometry.set_all_mesh_visibility":
            self._facade.set_all_capture_mesh_visibility(bool(value))
        elif action == "geometry.set_all_inliers_visibility":
            self._facade.set_all_capture_inliers_visibility(bool(value))
        return _PROPERTY_ACTION_REFRESH

    def _handle_scanner_property(self, action: str, value):
        if action == "scanner.mode":
            command = {
                SensorMode.FROZEN.value: CommandId.SENSOR_FROZEN,
                SensorMode.LIVE_FIXED.value: CommandId.SENSOR_LIVE_FIXED,
                SensorMode.LIVE_FOLLOW.value: CommandId.SENSOR_LIVE_FOLLOW,
            }[SensorMode(value).value]
            return self._execute_command(command)
        elif action == "scanner.one_shot":
            return self._schedule_sensor_one_shot()
        elif action == "scanner.select_profile":
            payload = dict(value)
            return self._schedule_sensor_profile(
                SensorKind(payload["sensor_kind"]), str(payload["profile_id"]),
            )
        elif action == "scanner.sensor_material":
            return self._schedule_sensor_material(dict(value))
        elif action == "scanner.axial_error_enabled":
            self._depth_diagnostics = replace(
                self._depth_diagnostics, axial_error_enabled=bool(value),
            )
            self._persist_user_preferences()
            if self._sensor_coordinator is not None:
                self._sensor_coordinator.configure_depth_diagnostics(self._depth_diagnostics)
                if self._sensor_coordinator.snapshot.phase is SensorControlPhase.DISABLED:
                    self._refresh_ui()
                    return self._schedule_depth_diagnostics_refresh()
        elif action in ("scanner.axial_error_sigma_m", "scanner.axial_error_model",
                        "scanner.axial_error_slope_m_per_m", "scanner.axial_error_intercept_m"):
            field = action.removeprefix("scanner.")
            self._depth_diagnostics = replace(
                self._depth_diagnostics, **{field: str(value) if field == "axial_error_model" else float(value)},
            )
            self._persist_user_preferences()
            if self._sensor_coordinator is not None:
                self._sensor_coordinator.configure_depth_diagnostics(self._depth_diagnostics)
                if (
                    self._depth_diagnostics.axial_error_enabled
                    and self._sensor_coordinator.snapshot.phase is SensorControlPhase.DISABLED
                ):
                    self._refresh_ui()
                    return self._schedule_depth_diagnostics_refresh()
        return _PROPERTY_ACTION_REFRESH

    def _handle_graphics_property(self, action: str, value):
        if action in ("graphics.ruler_main_opacity", "graphics.ruler_pip_opacity"):
            self._facade.set_interaction(**{action.split(".")[1]: float(value)})
        elif action == "graphics.mesh_appearance":
            payload = dict(value)
            kind = str(payload["kind"])
            self._facade.set_geometry_mesh_appearance(
                kind,
                GeometryMode(str(payload["mode"])),
                subdivision=max(3, min(256, int(payload["subdivision"]))),
                face_rgba=tuple(payload["face_rgba"]),
                wire_rgba=tuple(payload["wire_rgba"]),
            )
            if kind == "detected":
                self._render_payloads.clear_detected()
        return _PROPERTY_ACTION_REFRESH

    def _handle_settings_and_graphics_property(self, action: str, value):
        if action == "settings.restore_defaults":
            return self._schedule_settings_defaults()
        elif action == "settings.theme":
            self._facade.set_theme(str(value))
            self._chrome.set_theme(str(value))
        elif action == "settings.inlier_size":
            self._facade.set_interaction(inlier_rendering_size=float(value))
        elif action == "settings.point_size":
            self._facade.set_interaction(point_rendering_size=float(value))
        elif action == "settings.outlier_color_rgba":
            self._facade.set_interaction(outlier_color_rgba=tuple(value))
        elif action == "settings.inlier_negative_rgba":
            self._facade.set_interaction(inlier_negative_rgba=tuple(value))
        elif action == "settings.inlier_zero_rgba":
            self._facade.set_interaction(inlier_zero_rgba=tuple(value))
        elif action == "settings.inlier_positive_rgba":
            self._facade.set_interaction(inlier_positive_rgba=tuple(value))
        elif action == "settings.detected_preview_subdivision":
            self._facade.set_interaction(
                detected_preview_subdivision=max(3, min(256, int(value))),
            )
            self._render_payloads.clear_detected()
        elif action == "settings.detected_preview_face_rgba":
            self._facade.set_interaction(detected_preview_face_rgba=tuple(value))
            self._render_payloads.clear_detected()
        elif action == "settings.detected_preview_wire_rgba":
            self._facade.set_interaction(detected_preview_wire_rgba=tuple(value))
            self._render_payloads.clear_detected()
        elif action == "settings.probe_radius_px":
            self._facade.set_interaction(probe_radius_px=float(value))
        elif action == "settings.pick_nearest":
            self._facade.set_interaction(pick_nearest_on_probe_miss=bool(value))
        elif action == "settings.sticky_auto_preview":
            self._facade.set_interaction(sticky_auto_preview=bool(value))
        elif action == "settings.sticky_timeout_ms":
            self._facade.set_interaction(sticky_preview_timeout_ms=int(value))
        elif action == "settings.detection_average_window":
            self._facade.set_interaction(detection_average_window=max(1, min(120, int(value))))
        elif action == "settings.minimum_geometry_length_m":
            self._facade.set_interaction(minimum_geometry_length_m=float(value))
        elif action == "settings.surface_side_scale_mode":
            modes = (
                SurfaceSideScaleMode.AUTO,
                SurfaceSideScaleMode.FIXED,
                SurfaceSideScaleMode.SIGMA,
            )
            index = max(0, min(len(modes) - 1, int(value)))
            self._facade.set_interaction(surface_side_scale_mode=modes[index])
        elif action == "settings.surface_side_fixed_scale_m":
            self._facade.set_interaction(
                surface_side_fixed_scale_m=float(value),
            )
        elif action == "settings.surface_side_sigma_multiplier":
            self._facade.set_interaction(
                surface_side_sigma_multiplier=float(value),
            )
        elif action == "settings.anti_aliasing_mode":
            mode = AntiAliasingMode(int(value))
            settings = carb.settings.get_settings()
            settings.set_int("/rtx/post/aa/op", int(mode))
            settings.set_int("/rtx-defaults/post/aa/op", int(mode))
            self._anti_aliasing_mode = mode
        elif action == "settings.depth_disparity_noise":
            self._depth_diagnostics = replace(
                self._depth_diagnostics, disparity_noise_enabled=bool(value),
            )
            self._persist_user_preferences()
            if self._sensor_coordinator is not None:
                self._sensor_coordinator.configure_depth_diagnostics(self._depth_diagnostics)
                if self._sensor_coordinator.snapshot.phase is SensorControlPhase.DISABLED:
                    self._refresh_ui()
                    return self._schedule_depth_diagnostics_refresh()
        elif action in {
            "settings.depth_disparity_noise_mean_px",
            "settings.depth_disparity_noise_sigma_px",
        }:
            field = action.removeprefix("settings.depth_")
            self._depth_diagnostics = replace(
                self._depth_diagnostics, **{field: float(value)},
            )
            self._persist_user_preferences()
            if self._sensor_coordinator is not None:
                self._sensor_coordinator.configure_depth_diagnostics(self._depth_diagnostics)
                if (
                    self._depth_diagnostics.disparity_noise_enabled
                    and self._sensor_coordinator.snapshot.phase is SensorControlPhase.DISABLED
                ):
                    self._refresh_ui()
                    return self._schedule_depth_diagnostics_refresh()
        elif action == "settings.depth_outlier_removal":
            self._depth_diagnostics = replace(
                self._depth_diagnostics, outlier_removal_enabled=bool(value),
            )
            self._persist_user_preferences()
            if self._sensor_coordinator is not None:
                self._sensor_coordinator.configure_depth_diagnostics(self._depth_diagnostics)
                if self._sensor_coordinator.snapshot.phase is SensorControlPhase.DISABLED:
                    self._refresh_ui()
                    return self._schedule_depth_diagnostics_refresh()
        elif action == "settings.export_xyz":
            return self._schedule_xyz_export()
        elif action.startswith("settings.smart_convert_"):
            field = action.removeprefix("settings.")
            if field not in {
                "smart_convert_cone_to_cylinder",
                "smart_convert_torus_to_sphere",
                "smart_convert_torus_to_cylinder",
            }:
                raise KeyError(action)
            options = replace(self._facade.snapshot.detection, **{field: bool(value)})
            self._facade.replace_detection(options)
            self._workflow.invalidate()
        elif action == "settings.pip_toggle":
            self._pip_visible = not self._pip_visible
            self._compositor.set_pip_visible(self._pip_visible)
        elif action == "settings.pip_width":
            self._pip_width = float(value)
            self._compositor.set_pip_width_fraction(self._pip_width)
        return _PROPERTY_ACTION_REFRESH

    def _schedule_xyz_export(self):
        if self._xyz_export_task is None or self._xyz_export_task.done():
            self._xyz_export_task = asyncio.ensure_future(self._export_frozen_cloud_xyz())
        return self._xyz_export_task

    async def _export_frozen_cloud_xyz(self):
        coordinator = self._sensor_coordinator
        if coordinator is None or coordinator.snapshot.phase is not SensorControlPhase.DISABLED:
            self._xyz_export_status = "Export blocked: sensor is not Frozen"
            self._refresh_ui()
            return None
        frame = coordinator.latest_frame()
        if frame is None:
            self._xyz_export_status = "Export blocked: no current point cloud"
            self._refresh_ui()
            return None
        self._xyz_export_running = True
        self._xyz_export_status = f"Exporting {frame.compacted_count:,} points..."
        self._refresh_ui()
        try:
            destination = await asyncio.to_thread(export_points_xyz, frame.points_world_m)
        except Exception as error:
            self._xyz_export_status = f"Export failed: {type(error).__name__}: {error}"
            carb.log_error(self._xyz_export_status)
            return None
        else:
            self._xyz_export_status = f"Saved {destination.name}"
            carb.log_info(f"Frozen point cloud exported to {destination}")
            return destination
        finally:
            self._xyz_export_running = False
            self._refresh_ui()

    def _on_scene_object_change(self, event) -> None:
        """Coalesce object edits and execute the sensor's latest one-shot demand."""

        if getattr(self, "_restoring_settings", False):
            return
        coordinator = self._sensor_coordinator
        if coordinator is None or not coordinator.notify_object_change(event):
            return
        if self._object_change_task is None or self._object_change_task.done():
            self._object_change_task = asyncio.ensure_future(self._flush_scene_object_changes())

    async def _flush_scene_object_changes(self) -> None:
        try:
            # Float drags can emit several edits in one Kit update. Let the
            # coordinator retain just their latest sequence before scanning.
            await omni.kit.app.get_app().next_update_async()
            coordinator = self._sensor_coordinator
            while coordinator is not None and coordinator.snapshot.phase not in {
                SensorControlPhase.RUNNING, SensorControlPhase.DISABLED,
                SensorControlPhase.ERROR, SensorControlPhase.CLOSED,
            }:
                await omni.kit.app.get_app().next_update_async()
            while coordinator is not None and coordinator.has_pending_object_change:
                # One trailing rescan after the final slider/text edit. Changes
                # during an in-flight scan remain pending for the next pass.
                revision = coordinator.snapshot.object_events
                quiet_since = time.monotonic()
                while time.monotonic() - quiet_since < 0.15:
                    await omni.kit.app.get_app().next_update_async()
                    if coordinator.snapshot.object_events != revision:
                        revision = coordinator.snapshot.object_events
                        quiet_since = time.monotonic()
                await coordinator.flush_object_changes()
                self._sync_render_from_state()
                self._refresh_ui()
                if coordinator.snapshot.phase in {SensorControlPhase.ERROR, SensorControlPhase.CLOSED}:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._workflow.report_error(error)
            carb.log_warn(f"Object-change sensor refresh failed: {error}")
        finally:
            self._object_change_task = None

    def _sync_render_from_state(self, *, sync_hover_context: bool = True) -> None:
        if self._compositor is None or self._scene_session is None:
            return
        state = self._facade.snapshot
        live_hover = state.sensor.enabled
        catalog = SceneCatalog(state.objects)
        designed_layer = self._render_payloads.designed(state.objects)
        analysis = self._workflow.snapshot.analysis
        if (
            live_hover
            and analysis is not None
            and analysis.key.subject_id == "preview"
        ):
            analysis = None
        preview_meshes = ()
        if analysis is not None and analysis.key.subject_id == "preview":
            if analysis.detected_snapshot is not None:
                preview_appearance = state.interaction.mesh_appearance(
                    "detected", analysis.detected_snapshot.geometry_type,
                )
                detected = replace(
                    analysis.detected_snapshot,
                    subdivision=preview_appearance.subdivision,
                    display_subdivision=preview_appearance.subdivision,
                )
                def build_analysis_preview():
                    layer = build_mesh_layer(
                        RenderLayer.DETECTED_MESH,
                        (build_mesh(detected).transformed(detected.pose_world_m),),
                        sphere_pole_center_world_m=(
                            detected.pose_world_m.position_m if detected.geometry_type is GeometryMode.SPHERE else None
                        ),
                    )
                    return type(layer)(
                        layer.layer,
                        tuple(replace(
                            draw,
                            face_color=preview_appearance.face_rgba,
                            wire_color=preview_appearance.wire_rgba,
                        ) for draw in layer.draws),
                    )
                preview_meshes = (self._render_payloads.analysis_detected(
                    (
                        id(analysis),
                        preview_appearance.subdivision,
                        preview_appearance.face_rgba,
                        preview_appearance.wire_rgba,
                    ),
                    build_analysis_preview,
                ),)
            else:
                preview_mode = state.detection.geometry_mode
                if preview_mode is GeometryMode.AUTO:
                    preview_mode = GeometryMode.PLANE
                preview_appearance = state.interaction.mesh_appearance("detected", preview_mode)
                preview_meshes = tuple(
                    type(layer)(
                        layer.layer,
                        tuple(replace(
                            draw,
                            face_color=preview_appearance.face_rgba,
                            wire_color=preview_appearance.wire_rgba,
                        ) for draw in layer.draws),
                    )
                    for layer in analysis.mesh_layers if layer.layer is RenderLayer.DETECTED_MESH
                )
        capture_mesh_layer, capture_inlier_layer = self._render_payloads.captures(
            state.captures,
            hover_basis=self._chrome.histogram_hover_basis if self._chrome is not None else None,
            highlighted_capture_id=(
                self._chrome.geometry_hover_capture_id if self._chrome is not None else None
            ),
            surface_side_scale=state.interaction.surface_side_scale,
        )
        capture_meshes = () if capture_mesh_layer is None else (capture_mesh_layer,)
        sensor_frame = None if self._sensor_coordinator is None else self._sensor_coordinator.latest_frame()
        if sensor_frame is None and self._rendered_sensor_frame is not None and self._hover_preview is not None:
            self._hover_preview.pause_detection()
        scanner, scene = self._camera_snapshots(catalog)
        if sync_hover_context:
            primary = self._compositor.snapshot.primary_role
            logical_size = self._compositor.logical_size_for_role(primary)
            self._sync_hover_context(sensor_frame, scanner if primary is CameraRole.SCANNER else scene, logical_size)
        point_layers = self._render_payloads.source(sensor_frame)
        workflow_preview_inlier = (
            None
            if analysis is None or analysis.key.subject_id != "preview"
            else analysis.inlier_layer
        )
        if workflow_preview_inlier is not None:
            point_layers += (workflow_preview_inlier,)
        if capture_inlier_layer is not None:
            point_layers += (capture_inlier_layer,)
        line_layers = ()
        hover = None if state.inspect_mode else self._hover_preview
        classification_hover = None
        if hover is not None:
            overlay = hover.pick_overlay(
                retroactive=bool(
                    self._pointer_retroactive_pick
                    or (self._keyboard and self._keyboard.alt_pressed)
                ),
                show_hover_radius=self._pointer_role is None,
            )
            seed_layer = overlay.seed
            hover_snapshot = hover.snapshot
            if seed_layer is not None:
                point_layers = tuple(
                    layer for layer in point_layers if layer.layer is not RenderLayer.SEED
                ) + (seed_layer,)
            line_layers = (overlay.guide,)
            if overlay.seed_radius is not None:
                line_layers += (overlay.seed_radius,)
            hover_result = hover_snapshot.display_result
            if hover_result is not None:
                try:
                    if live_hover:
                        # A continuously changing cloud makes hover residual analysis
                        # obsolete before it can be read. Keep the detected mesh, but
                        # do not submit histogram/heatmap work or classify hover inliers.
                        self._sync_hover_analysis(None, None)
                        self._render_payloads.clear_classified()
                        self._render_payloads.clear_pending_hover()
                    else:
                        self._sync_hover_analysis(hover_result, sensor_frame)
                    if (
                        not live_hover
                        and sensor_frame is not None
                        and hover_result.request_key.scan_generation == sensor_frame.generation
                        and hover_result.request_key.scan_sequence == sensor_frame.sequence
                    ):
                        classification_hover = hover_result
                    else:
                        self._render_payloads.clear_classified()
                    def build_detected_layer(result):
                        preview_appearance = state.interaction.mesh_appearance(
                            "detected", result.geometry_type,
                        )
                        detected = detected_scene_object(result.geometry_type, result.parameters)
                        detected = replace(
                            detected,
                            subdivision=preview_appearance.subdivision,
                            display_subdivision=preview_appearance.subdivision,
                        )
                        layer = build_mesh_layer(
                            RenderLayer.DETECTED_MESH,
                            (build_mesh(detected).transformed(detected.pose_world_m),),
                            sphere_pole_center_world_m=(
                                detected.pose_world_m.position_m if detected.geometry_type is GeometryMode.SPHERE else None
                            ),
                        )
                        return type(layer)(
                            layer.layer,
                            tuple(replace(
                                draw,
                                face_color=preview_appearance.face_rgba,
                                wire_color=preview_appearance.wire_rgba,
                            ) for draw in layer.draws),
                        )

                    detected_layer = self._render_payloads.detected(
                        hover_result,
                        build_detected_layer,
                        (
                            state.interaction.mesh_appearance("detected", hover_result.geometry_type).subdivision,
                            state.interaction.mesh_appearance("detected", hover_result.geometry_type).face_rgba,
                            state.interaction.mesh_appearance("detected", hover_result.geometry_type).wire_rgba,
                        ),
                    )
                    preview_meshes = () if detected_layer is None else (detected_layer,)
                except (KeyError, TypeError, ValueError) as error:
                    carb.log_warn(f"Hover result presentation failed: {error}")
            else:
                self._sync_hover_analysis(None, None)
                self._render_payloads.clear_classified()
                self._render_payloads.clear_detected()
        else:
            self._sync_hover_analysis(None, None)
            self._render_payloads.clear_classified()
        if sensor_frame is not None and (state.captures or classification_hover is not None):
            classified = self._render_payloads.capture_classified(
                sensor_frame,
                state.captures,
                classification_hover,
                outlier_only=True,
            )
            if classified:
                hover_inlier = None
                if (
                    classification_hover is not None
                    and self._hover_analysis is not None
                    and self._hover_analysis.key.request_key == classification_hover.request_key
                    and self._hover_analysis.inlier_layer is not None
                ):
                    self._render_payloads.clear_pending_hover()
                    hover_inlier = self._hover_analysis.inlier_layer
                elif classification_hover is not None:
                    hover_inlier = self._render_payloads.pending_hover_inlier(
                        sensor_frame,
                        classification_hover,
                    )
                else:
                    self._render_payloads.clear_pending_hover()
                combined_inlier = merge_point_layers(
                    RenderLayer.INLIER_POINTS,
                    (capture_inlier_layer, hover_inlier),
                )
                point_layers = tuple(
                    layer for layer in point_layers
                    if layer.layer not in {
                        RenderLayer.SOURCE_POINTS,
                        RenderLayer.INLIER_POINTS,
                        RenderLayer.OUTLIER_POINTS,
                    }
                ) + tuple(
                    layer for layer in classified
                    if layer.layer is RenderLayer.OUTLIER_POINTS
                )
                if combined_inlier is not None:
                    point_layers += (combined_inlier,)
        else:
            self._render_payloads.clear_capture_classified()
            self._render_payloads.clear_pending_hover()
        current_inlier_layers = tuple(
            layer for layer in point_layers if layer.layer is RenderLayer.INLIER_POINTS
        )
        merged_inliers = merge_point_layers(RenderLayer.INLIER_POINTS, current_inlier_layers)
        point_layers = tuple(
            layer for layer in point_layers if layer.layer is not RenderLayer.INLIER_POINTS
        )
        if merged_inliers is not None:
            point_layers += (merged_inliers,)
        self._render_revision += 1
        style = style_for_theme(
            RenderTheme.IVORY_LIGHT if state.theme == "classic-light" else RenderTheme.EDITOR_DARK
        )
        point_size = state.interaction.point_rendering_size
        style = replace(
            style,
            point_size_dip=point_size,
            inlier_size_dip=state.interaction.inlier_rendering_size,
            seed_size_dip=max(point_size + 3.0, 7.0),
            source_fill=state.interaction.outlier_color_rgba,
            outlier_fill=state.interaction.outlier_color_rgba,
            heatmap_zero=state.interaction.inlier_negative_rgba,
            heatmap_middle=state.interaction.inlier_zero_rgba,
            heatmap_high=state.interaction.inlier_positive_rgba,
        )
        packet = LayeredRenderPacket(
            self._render_revision,
            mesh_layers=(designed_layer,) + capture_meshes + preview_meshes,
            point_layers=point_layers,
            line_layers=line_layers,
            style=style,
        )
        visibility = {
            RenderLayer.DESIGNED_MESH: state.visibility.designed,
            RenderLayer.DETECTED_MESH: state.visibility.detected,
            RenderLayer.CAPTURE: state.visibility.captured,
            RenderLayer.INLIER_POINTS: state.visibility.inlier,
            RenderLayer.OUTLIER_POINTS: state.visibility.outlier,
            RenderLayer.SOURCE_POINTS: state.visibility.outlier,
        }
        for layer, visible in visibility.items():
            packet = packet.with_visibility(layer, visible)
        self._render_packet = packet
        self._rendered_sensor_frame = None if sensor_frame is None else sensor_frame.identity
        follows_scanner = (
            True
            if self._sensor_coordinator is None
            else self._sensor_coordinator.snapshot.pose.follows_camera
        )
        self._compositor.set_point_depth_presentation(
            sensor_follows_scanner_camera=follows_scanner,
            offset_m=state.point_depth_offset_m,
        )
        self._compositor.submit(packet, scanner, scene)

    def _dispatch_shortcut(self, key: str, *, ctrl: bool, shift: bool, alt: bool):
        key = str(key).upper()
        if alt and key == "F4":
            self.request_safe_exit(ExitTrigger.ALT_F4)
            return True
        if ctrl and shift and key == "Q":
            self.request_safe_exit(ExitTrigger.CTRL_SHIFT_Q)
            return True
        return self._shell_commands.dispatch_shortcut(key, ctrl=ctrl, shift=shift, alt=alt) is not False

    def _on_update(self, _event) -> None:
        # UI-only hover does not dispatch a durable command or a sensor frame.
        # Refresh render packets on entry AND exit without rebuilding properties.
        if self._chrome is not None:
            hover_basis = self._chrome.histogram_hover_basis
            geometry_hover = getattr(self._chrome, "geometry_hover_capture_id", None)
            if (
                hover_basis != getattr(self, "_rendered_histogram_hover_basis", None)
                or geometry_hover != getattr(self, "_rendered_geometry_hover_capture_id", None)
            ):
                self._rendered_histogram_hover_basis = hover_basis
                self._rendered_geometry_hover_capture_id = geometry_hover
                self._sync_render_from_state(sync_hover_context=False)
        self._maybe_capture_workspace_layout()
        if self._hover_preview is not None:
            preview = self._hover_preview
            try:
                if preview.drain_completion():
                    # Present the completion without feeding the unchanged
                    # context back into a failed request and retrying forever.
                    self._sync_render_from_state(sync_hover_context=False)
                elif preview.expire_sticky():
                    self._sync_render_from_state(sync_hover_context=False)
            except Exception as error:
                self._workflow.report_error(error)
            finally:
                # The worker may start the one latest retry only after the
                # completed result has been handed to the renderer state.
                preview.acknowledge_completion()
        if self._drain_hover_analysis_completion():
            self._sync_render_from_state(sync_hover_context=False)
        if self._camera_controllers:
            dt_s = update_event_dt_seconds(_event)
            role = self._compositor.snapshot.primary_role
            controller = self._camera_controllers[role]
            before = controller.snapshot
            controller.update(max(0.0, dt_s))
            if controller.snapshot != before:
                self._present_camera_controllers(sync_continuous_sensor=True)
        coordinator = self._sensor_coordinator
        if (
            self._sensor_polling
            and coordinator is not None
            and coordinator.snapshot.phase is SensorControlPhase.RUNNING
        ):
            try:
                before_scan_state = (
                    coordinator.snapshot.latest_frame_identity,
                    coordinator.snapshot.no_returns,
                )
                frame = coordinator.poll()
                after_scan_state = (
                    coordinator.snapshot.latest_frame_identity,
                    coordinator.snapshot.no_returns,
                )
                if (
                    (frame is not None and frame.identity != self._rendered_sensor_frame)
                    or after_scan_state != before_scan_state
                ):
                    self._sync_render_from_state()
            except Exception as error:
                self._sensor_polling = False
                self._workflow.report_error(error)
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        if self._chrome is not None:
            self._shell_snapshot = self._build_shell_snapshot()
            self._chrome.refresh(self._shell_snapshot)
            self._chrome.sync_property_values(self._property_values())
            if self._camera_controllers and hasattr(self._chrome, "scene_ruler"):
                self._chrome.sync_scene_ruler(
                    self._camera_controllers[CameraRole.SCENE].snapshot,
                    self._compositor.rect_for_role(CameraRole.SCENE),
                    primary=self._compositor.snapshot.primary_role is CameraRole.SCENE,
                )





    def request_safe_exit(self, trigger: ExitTrigger | str = ExitTrigger.TOOLBAR):
        return self._quit_guard.request(trigger)

    def _on_window_visibility_changed(self, visible: bool) -> None:
        if not visible and self._exit.accepting_commands:
            self.request_safe_exit(ExitTrigger.WINDOW_CLOSE)

    async def _close_scene_session(self) -> None:
        task = self._scene_task
        if task is not None and not task.done():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        session, self._scene_session = self._scene_session, None
        if session is not None:
            session.close()
        self._teardown_events.append("usd-scene-session")

    def _close_workflow(self) -> None:
        scheduler, self._analysis_scheduler = self._analysis_scheduler, None
        if scheduler is not None:
            scheduler.update_current(None)
            scheduler.close()
        self._hover_analysis = None
        self._hover_analysis_current_key = None
        self._hover_analysis_pending_key = None
        self._hover_analysis_source = None
        hover, self._hover_preview = self._hover_preview, None
        if hover is not None:
            hover.close()
        self._workflow.close()
        self._teardown_events.append("preview-detection-workflow")

    def _close_sensor_publication(self) -> None:
        self._sensor_polling = False
        for name in ("_object_change_task", "_depth_diagnostics_refresh_task"):
            task = getattr(self, name, None)
            setattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()
        coordinator = self._sensor_coordinator
        if coordinator is not None and coordinator.snapshot.phase is SensorControlPhase.RUNNING:
            coordinator.disable()
        self._teardown_events.append("sensor-publication")

    def _close_sensor_products(self) -> None:
        coordinator, self._sensor_coordinator = self._sensor_coordinator, None
        if coordinator is not None:
            coordinator.close()
        self._teardown_events.append("sensor-products")

    async def _close_compositor(self) -> None:
        await self._compositor.close()
        self._render_payloads.clear()
        self._render_packet = None
        self._teardown_events.append("dual-view-compositor")

    def _close_refresh(self) -> None:
        self._refresh_subscription = None
        self._teardown_events.append("shell-refresh")

    def _close_keyboard(self) -> None:
        keyboard, self._keyboard = self._keyboard, None
        if keyboard is not None:
            keyboard.close()
        self._teardown_events.append("shortcut-input")

    def _close_chrome(self) -> None:
        task, self._workspace_layout_task = self._workspace_layout_task, None
        if task is not None and not task.done():
            task.cancel()
        self._ui_presentations().clear()
        chrome, self._chrome = self._chrome, None
        if chrome is not None:
            chrome.close()
        self._teardown_events.append("classic-ui")

    def _destroy_window(self) -> None:
        window, self._window = self._window, None
        if window is not None:
            window.set_visibility_changed_fn(None)
            window.visible = False
            window.destroy()
        self._standard_viewport_window = None
        self._teardown_events.append("main-window")

    def _post_quit(self, exit_code: int) -> None:
        self._quit_guard.allow_final_shutdown()
        omni.kit.app.get_app().post_quit(int(exit_code))

    def on_shutdown(self) -> None:
        self._persist_workspace_layout()
        self._persist_user_preferences()
        self._quit_guard.close()
        for name in (
            "_smoke_task", "_scene_task", "_xyz_export_task", "_depth_diagnostics_refresh_task",
        ):
            task = getattr(self, name, None)
            setattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()
        carb.log_info("FindSurfaceIsaacSimDemo shutdown complete")
