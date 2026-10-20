"""Deferred, stage-bound RTX ViewportWidget scene backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import import_module
import re
import math
from typing import Callable

import carb.settings
import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Gf, Usd, UsdGeom

from .render_layers import LayeredRenderPacket, RenderLayer, flatten_mesh_layer
from .rtx_policy import (
    EffectiveSceneRenderer,
    RtxPointOverlayCapability,
    RtxPointOverlayStatus,
    SceneRendererMode,
    probe_rtx_point_overlay,
)


RTX_EXTENSIONS = ("omni.hydra.rtx", "omni.kit.widget.viewport")
PRESENTATION_PATH = "/World/FindSurfaceIsaacSim/RtxPresentation"
STORE_CAMERA_SETTINGS = "/app/omni.usd/storeCameraSettingsToUsdStage"
SAVE_RENDER_SETTINGS = "/app/hydra/renderSettings/saveUsdAttributes"
USE_FABRIC_SCENE_DELEGATE = "/app/useFabricSceneDelegate"


@dataclass(frozen=True)
class RtxSceneTelemetry:
    packet_revision: int = -1
    mesh_layer_count: int = 0
    mesh_vertex_count: int = 0
    line_vertex_count: int = 0
    point_overlay_status: RtxPointOverlayStatus = RtxPointOverlayStatus.UNAVAILABLE
    point_overlay_reason: str | None = "not evaluated"
    first_frame_rendered: bool = False
    closed: bool = False


@dataclass
class RtxSceneView:
    widget: object
    viewport_widget: object
    requested_mode: SceneRendererMode
    point_overlay: RtxPointOverlayCapability
    stage: object
    usd_context_name: str
    settings: object
    settings_before: tuple[bool, bool, bool]
    edit_target_before: object
    telemetry_state: RtxSceneTelemetry
    presented_mesh_layers: dict = field(default_factory=dict)
    presented_line_layers: dict = field(default_factory=dict)
    presented_style: object | None = None

    @property
    def viewport_api(self):
        return self.viewport_widget.viewport_api

    def telemetry(self) -> RtxSceneTelemetry:
        return self.telemetry_state

    def mark_first_frame(self) -> None:
        self.telemetry_state = replace(self.telemetry_state, first_frame_rendered=True)

    def set_resolution(self, width: float, height: float) -> None:
        if width > 0.0 and height > 0.0:
            self.viewport_widget.resolution = (max(1, round(width)), max(1, round(height)))

    def present_camera(self, camera) -> None:
        """Mirror the canonical GL camera into the session-owned USD camera."""
        if camera == getattr(self, "_last_camera", None):
            return
        from .domain import StageMetadata
        from .scene import StageSpace
        from .usd_scene import SCENE_CAMERA_PATH
        unit = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        space = StageSpace(StageMetadata(str(UsdGeom.GetStageUpAxis(self.stage)), unit))
        pose = space.to_stage_pose(camera.pose_world)
        x, y, z, w = pose.orientation_xyzw
        transform = Gf.Matrix4d(1.0)
        transform.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        transform.SetTranslateOnly(Gf.Vec3d(*pose.position_stage_units))
        intrinsics = camera.intrinsics
        height = intrinsics.orthographic_height_m
        # USD camera apertures are in tenths of stage units (also in ortho).
        aperture = height * 10.0 / unit if height is not None else 100.0 * math.tan(intrinsics.vertical_fov_radians / 2.0)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            usd_camera = UsdGeom.Camera(self.stage.GetPrimAtPath(SCENE_CAMERA_PATH))
            usd_camera.MakeMatrixXform().Set(transform)
            usd_camera.CreateProjectionAttr('orthographic' if height is not None else 'perspective')
            usd_camera.CreateVerticalApertureAttr(aperture)
            usd_camera.CreateHorizontalApertureAttr(aperture * intrinsics.aspect_ratio)
            usd_camera.CreateFocalLengthAttr(50.0)
            usd_camera.CreateClippingRangeAttr(Gf.Vec2f(intrinsics.near_m / unit, intrinsics.far_m / unit))
        self._last_camera = camera

    def present_packet(self, packet: LayeredRenderPacket) -> None:
        """Author only mesh/line presentation; ScanFrame points never become USD prims."""

        if self.telemetry_state.closed:
            return
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        scale = 1.0 / meters_per_unit
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            root = UsdGeom.Xform.Define(self.stage, PRESENTATION_PATH)
            root.GetPrim().SetCustomDataByKey("findSurfaceSessionOwned", True)
            mesh_by_layer = {layer.layer: layer for layer in packet.mesh_layers}
            for removed in set(self.presented_mesh_layers) - set(mesh_by_layer):
                self.stage.RemovePrim(f"{PRESENTATION_PATH}/{_prim_name(removed.value)}")
            mesh_vertices = 0
            for layer in packet.mesh_layers:
                path = f"{PRESENTATION_PATH}/{_prim_name(layer.layer.value)}"
                changed = self.presented_mesh_layers.get(layer.layer) is not layer
                per_draw_style = any(
                    draw.face_color is not None or draw.wire_color is not None
                    for draw in layer.draws
                )
                if changed and per_draw_style:
                    if self.stage.GetPrimAtPath(path):
                        self.stage.RemovePrim(path)
                    UsdGeom.Xform.Define(self.stage, path)
                    for draw in layer.draws:
                        draw_name = _prim_name(draw.draw_id)
                        geometry = flatten_mesh_layer(replace(layer, draws=(draw,)))
                        mesh = _author_rtx_mesh(
                            self.stage, f"{path}/{draw_name}_Face", geometry, scale,
                        )
                        _set_display(mesh, packet, layer.layer, draw.face_color)
                        _author_rtx_wire(
                            self.stage,
                            f"{path}/{draw_name}_Outline",
                            geometry,
                            scale,
                            draw.wire_color or _wire_color(packet, layer.layer),
                        )
                elif changed:
                    geometry = flatten_mesh_layer(layer)
                    mesh = _author_rtx_mesh(self.stage, path, geometry, scale)
                elif not per_draw_style:
                    mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(path))
                if not per_draw_style and (changed or self.presented_style != packet.style):
                    _set_display(mesh, packet, layer.layer)
                mesh_vertices += layer.vertex_count
            line_by_layer = {
                layer.layer: layer for layer in packet.line_layers
                if layer.layer is RenderLayer.CAMERA_FRUSTUM
            }
            for removed in set(self.presented_line_layers) - set(line_by_layer):
                self.stage.RemovePrim(f"{PRESENTATION_PATH}/{_prim_name(removed.value)}")
            line_vertices = 0
            for layer in line_by_layer.values():
                path = f"{PRESENTATION_PATH}/{_prim_name(layer.layer.value)}"
                changed = self.presented_line_layers.get(layer.layer) is not layer
                if changed:
                    points = np.frombuffer(layer.vertices_f32, dtype=np.float32).reshape((-1, 3))
                    curves = UsdGeom.BasisCurves.Define(self.stage, path)
                    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
                    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
                    curves.CreateCurveVertexCountsAttr([2] * (len(points) // 2))
                    curves.CreatePointsAttr([Gf.Vec3f(*(float(value) * scale for value in point)) for point in points])
                    curves.CreateWidthsAttr([0.012 * scale])
                    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                else:
                    curves = UsdGeom.BasisCurves(self.stage.GetPrimAtPath(path))
                if changed or self.presented_style != packet.style:
                    curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                        [Gf.Vec3f(*packet.style.frustum_edge[:3])]
                    )
                line_vertices += layer.vertex_count
        self.presented_mesh_layers = mesh_by_layer
        self.presented_line_layers = line_by_layer
        self.presented_style = packet.style
        self.telemetry_state = replace(
            self.telemetry_state,
            packet_revision=packet.revision,
            mesh_layer_count=len(packet.mesh_layers),
            mesh_vertex_count=mesh_vertices,
            line_vertex_count=line_vertices,
        )

    def close(self) -> None:
        if self.telemetry_state.closed:
            return
        store_before, save_before, fabric_before = self.settings_before
        try:
            try:
                self.viewport_widget.destroy()
            finally:
                with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                    self.stage.RemovePrim(PRESENTATION_PATH)
                context = omni.usd.get_context(self.usd_context_name)
                if context is not None and context.get_stage() is self.stage:
                    self.stage.SetEditTarget(self.edit_target_before)
                self.widget.destroy()
        finally:
            self.settings.set_bool(STORE_CAMERA_SETTINGS, store_before)
            self.settings.set_bool(SAVE_RENDER_SETTINGS, save_before)
            self.settings.set_bool(USE_FABRIC_SCENE_DELEGATE, fabric_before)
            self.presented_mesh_layers.clear()
            self.presented_line_layers.clear()
            self.presented_style = None
            self.telemetry_state = replace(self.telemetry_state, closed=True)
            self.viewport_widget = self.widget = self.stage = self.settings = None


@dataclass(frozen=True)
class RtxSceneAttempt:
    view: RtxSceneView | None
    requested: SceneRendererMode
    effective: EffectiveSceneRenderer
    fallback_reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.view is not None


def try_create_rtx_scene_view(
    parent,
    *,
    requested: SceneRendererMode,
    camera_path: str,
    usd_context_name: str = "",
    resolution: tuple[int, int] = (512, 288),
    extension_manager=None,
    module_importer: Callable[[str], object] = import_module,
) -> RtxSceneAttempt:
    requested = SceneRendererMode(requested)
    if requested is SceneRendererMode.OPENGL:
        return RtxSceneAttempt(None, requested, EffectiveSceneRenderer.OPENGL, None)
    manager = extension_manager or omni.kit.app.get_app().get_extension_manager()
    settings = carb.settings.get_settings()
    before = (
        settings.get_as_bool(STORE_CAMERA_SETTINGS),
        settings.get_as_bool(SAVE_RENDER_SETTINGS),
        settings.get_as_bool(USE_FABRIC_SCENE_DELEGATE),
    )
    container = viewport = stage = edit_target = None
    try:
        settings.set_bool(STORE_CAMERA_SETTINGS, False)
        settings.set_bool(SAVE_RENDER_SETTINGS, False)
        settings.set_bool(USE_FABRIC_SCENE_DELEGATE, False)
        for extension in RTX_EXTENSIONS:
            if not manager.set_extension_enabled_immediate(extension, True):
                raise RuntimeError(f"failed to enable {extension}")
        context = omni.usd.get_context(usd_context_name)
        stage = context.get_stage() if context is not None else None
        if stage is None:
            raise RuntimeError(f'USD context "{usd_context_name}" has no stage')
        if not stage.GetPrimAtPath(camera_path):
            raise RuntimeError(f"RTX scene camera does not exist: {camera_path}")
        edit_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        viewport_module = module_importer("omni.kit.widget.viewport")
        with parent:
            container = ui.Frame(width=ui.Percent(100), height=ui.Percent(100))
        with container:
            viewport = viewport_module.ViewportWidget(
                usd_context_name=usd_context_name,
                camera_path=camera_path,
                resolution=resolution,
                hd_engine="rtx",
            )
            viewport.resolution_uses_dpi = False
        settings.set_bool(USE_FABRIC_SCENE_DELEGATE, before[2])
        point_overlay = probe_rtx_point_overlay(viewport.viewport_api)
        telemetry = RtxSceneTelemetry(
            point_overlay_status=point_overlay.status,
            point_overlay_reason=point_overlay.reason,
        )
        view = RtxSceneView(
            container, viewport, requested, point_overlay, stage, usd_context_name,
            settings, before, edit_target, telemetry,
        )
        return RtxSceneAttempt(view, requested, EffectiveSceneRenderer.RTX, None)
    except Exception as exception:
        if viewport is not None:
            viewport.destroy()
        if container is not None:
            container.destroy()
        if stage is not None and edit_target is not None:
            stage.SetEditTarget(edit_target)
        settings.set_bool(STORE_CAMERA_SETTINGS, before[0])
        settings.set_bool(SAVE_RENDER_SETTINGS, before[1])
        settings.set_bool(USE_FABRIC_SCENE_DELEGATE, before[2])
        return RtxSceneAttempt(
            None, requested, EffectiveSceneRenderer.OPENGL, f"{type(exception).__name__}: {exception}"
        )


def _prim_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _author_rtx_mesh(stage, path: str, geometry, scale: float):
    points = np.frombuffer(geometry.vertices_f32, dtype=np.float32).reshape((-1, 3))
    triangles = np.frombuffer(geometry.triangle_indices_u32, dtype=np.uint32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*(float(value) * scale for value in point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * (len(triangles) // 3))
    mesh.CreateFaceVertexIndicesAttr([int(value) for value in triangles])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def _wire_color(packet: LayeredRenderPacket, layer: RenderLayer):
    style = packet.style
    return {
        RenderLayer.DESIGNED_MESH: style.designed_wire,
        RenderLayer.DETECTED_MESH: style.detected_wire,
        RenderLayer.PREVIEW: style.preview_wire,
        RenderLayer.CAPTURE: style.capture_wire,
        RenderLayer.CAMERA_FRUSTUM_FACE: style.frustum_edge,
    }[layer]


def _author_rtx_wire(stage, path: str, geometry, scale: float, color) -> None:
    points = np.frombuffer(geometry.vertices_f32, dtype=np.float32).reshape((-1, 3))
    edges = np.frombuffer(geometry.edge_indices_u32, dtype=np.uint32)
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreateCurveVertexCountsAttr([2] * (len(edges) // 2))
    curves.CreatePointsAttr([
        Gf.Vec3f(*(float(value) * scale for value in points[int(index)]))
        for index in edges
    ])
    curves.CreateWidthsAttr([0.003 * scale])
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color[:3])])
    curves.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(color[3])])


def _set_display(mesh, packet: LayeredRenderPacket, layer: RenderLayer, override=None) -> None:
    style = packet.style
    colors = {
        RenderLayer.DESIGNED_MESH: style.designed_face,
        RenderLayer.DETECTED_MESH: style.detected_face,
        RenderLayer.PREVIEW: style.preview_face,
        RenderLayer.CAPTURE: style.capture_face,
        RenderLayer.CAMERA_FRUSTUM_FACE: style.frustum_face,
    }
    color = override or colors.get(layer, style.designed_face)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color[:3])])
    opacity = color[3] if override is not None else max(color[3], 0.18)
    mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(opacity)])


__all__ = [
    "PRESENTATION_PATH", "RTX_EXTENSIONS", "RtxSceneAttempt", "RtxSceneTelemetry",
    "RtxSceneView", "try_create_rtx_scene_view",
]
