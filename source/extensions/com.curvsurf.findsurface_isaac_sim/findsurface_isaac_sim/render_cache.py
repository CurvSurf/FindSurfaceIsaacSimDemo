"""Provenance-keyed immutable render payload caching."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from .analysis_pipeline import SurfaceSideScale, detected_scene_object
from .domain import DetectionResult, FrameIdentity, GeometryMode, Pose, ScanFrame
from .geometry import build_mesh
from .render_layers import (
    MeshDrawPacket,
    MeshGeometryPacket,
    MeshLayerPacket,
    PointLayerPacket,
    RenderLayer,
    build_mesh_geometry,
    build_mesh_layer,
    build_classified_point_layers,
    build_outlier_layer,
    build_pending_inlier_layer,
    emphasize_inlier_layer,
    hover_context_color,
    hover_emphasis_color,
    build_point_layers,
    merge_point_layers,
    mesh_model_matrix,
)
from .scene import SceneObject


class RenderPayloadCache:
    """Retain expensive immutable layers until their source provenance changes."""

    def __init__(self) -> None:
        self._designed_key: tuple[SceneObject, ...] | None = None
        self._designed_layer: MeshLayerPacket | None = None
        self._designed_geometries: dict[str, tuple[object, MeshGeometryPacket]] = {}
        self._source_identity: FrameIdentity | None = None
        self._source_layers: tuple[PointLayerPacket, ...] = ()
        self._classified_key: tuple[FrameIdentity, DetectionResult] | None = None
        self._classified_layers: tuple[PointLayerPacket, ...] = ()
        self._pending_hover_key: tuple[FrameIdentity, DetectionResult] | None = None
        self._pending_hover_inlier: PointLayerPacket | None = None
        self._detected_result: DetectionResult | None = None
        self._detected_presentation_key = None
        self._detected_layer: MeshLayerPacket | None = None
        self._analysis_detected_key = None
        self._analysis_detected_layer: MeshLayerPacket | None = None
        self._capture_mesh_key: tuple | None = None
        self._capture_inlier_key: tuple | None = None
        self._capture_mesh_layer: MeshLayerPacket | None = None
        self._capture_inlier_layer: PointLayerPacket | None = None
        self._capture_highlight_mesh_key: tuple | None = None
        self._capture_highlight_inlier_key: tuple | None = None
        self._capture_highlight_mesh_layer: MeshLayerPacket | None = None
        self._capture_highlight_inlier_layer: PointLayerPacket | None = None
        self._capture_indices_key: tuple | None = None
        self._capture_indices: tuple[int, ...] = ()
        self._capture_classified_key: tuple | None = None
        self._capture_classified_layers: tuple[PointLayerPacket, ...] = ()
        self.designed_builds = 0
        self.designed_geometry_builds = 0
        self.source_builds = 0
        self.classified_builds = 0
        self.pending_hover_builds = 0
        self.detected_builds = 0
        self.capture_builds = 0
        self.capture_highlight_mesh_builds = 0
        self.capture_highlight_inlier_builds = 0
        self.capture_membership_builds = 0

    def designed(self, objects) -> MeshLayerPacket:
        key = tuple(item for item in objects if item.enabled)
        if self._designed_key != key or self._designed_layer is None:
            draws = tuple(self._designed_draw(item) for item in key)
            self._designed_layer = MeshLayerPacket(RenderLayer.DESIGNED_MESH, draws)
            self._designed_key = key
            self.designed_builds += 1
        return self._designed_layer

    def _designed_draw(self, item: SceneObject) -> MeshDrawPacket:
        parameters = item.parameter_dict
        if item.geometry_type is GeometryMode.PLANE:
            geometry_key = (GeometryMode.PLANE, "unit", item.display_subdivision)
            mesh_item = SceneObject(
                GeometryMode.PLANE, True, Pose(), {"width_m": 1.0, "height_m": 1.0},
                item.display_subdivision,
            )
            scale = (parameters["width_m"], parameters["height_m"], 1.0)
        elif item.geometry_type is GeometryMode.SPHERE:
            geometry_key = (GeometryMode.SPHERE, "unit", item.display_subdivision)
            mesh_item = SceneObject(
                GeometryMode.SPHERE, True, Pose(), {"radius_m": 1.0}, item.display_subdivision,
            )
            radius = parameters["radius_m"]
            scale = (radius, radius, radius)
        elif item.geometry_type is GeometryMode.CYLINDER:
            geometry_key = (GeometryMode.CYLINDER, "unit", item.display_subdivision)
            mesh_item = SceneObject(
                GeometryMode.CYLINDER, True, Pose(), {"radius_m": 1.0, "length_m": 1.0},
                item.display_subdivision,
            )
            scale = (parameters["radius_m"], parameters["radius_m"], parameters["length_m"])
        else:
            # Cone taper and torus sweep/radius ratios are topology-local values.
            # Rebuild them when parameters or subdivision change; pose remains a matrix.
            geometry_key = (item.geometry_type, item.parameters, item.display_subdivision)
            mesh_item = replace(item, subdivision=item.display_subdivision)
            scale = (1.0, 1.0, 1.0)
        cached = self._designed_geometries.get(item.object_id)
        if cached is None or cached[0] != geometry_key:
            geometry = build_mesh_geometry((build_mesh(mesh_item),))
            self._designed_geometries[item.object_id] = (geometry_key, geometry)
            self.designed_geometry_builds += 1
        else:
            geometry = cached[1]
        return MeshDrawPacket(
            item.object_id,
            geometry,
            mesh_model_matrix(item.pose_world_m, scale),
            item.face_color_rgba,
            item.wire_color_rgba,
        )

    def source(self, frame: ScanFrame | None) -> tuple[PointLayerPacket, ...]:
        if frame is None:
            self._source_identity = None
            self._source_layers = ()
        elif self._source_identity != frame.identity:
            source_layers = build_point_layers(frame)
            self._source_identity = frame.identity
            self._source_layers = source_layers
            self.source_builds += 1
        return self._source_layers

    def classified(
        self, frame: ScanFrame, result: DetectionResult,
    ) -> tuple[PointLayerPacket, ...]:
        """Cache mutually exclusive inlier/outlier payloads for one detection result."""

        if (
            result.request_key.scan_generation != frame.generation
            or result.request_key.scan_sequence != frame.sequence
        ):
            raise ValueError("classification result does not belong to the supplied frame")
        key = (frame.identity, result)
        if self._classified_key != key:
            layers = build_classified_point_layers(frame, inlier_indices=result.inlier_indices)
            self._classified_key = key
            self._classified_layers = layers
            self.classified_builds += 1
        return self._classified_layers

    def clear_classified(self) -> None:
        self._classified_key = None
        self._classified_layers = ()

    def pending_hover_inlier(
        self, frame: ScanFrame, result: DetectionResult,
    ) -> PointLayerPacket | None:
        """Cache only the hover inliers while residual analysis is pending.

        Durable capture inliers are rendered by :meth:`captures`; they must not
        be copied out of the capture-membership classification packet because
        that packet intentionally carries neutral heat coordinates.
        """

        if (
            result.request_key.scan_generation != frame.generation
            or result.request_key.scan_sequence != frame.sequence
        ):
            raise ValueError("hover result does not belong to the supplied frame")
        key = (frame.identity, result)
        if self._pending_hover_key != key:
            indices = tuple(int(index) for index in result.inlier_indices)
            if (
                len(set(indices)) != len(indices)
                or any(index < 0 or index >= frame.compacted_count for index in indices)
            ):
                raise ValueError("hover inlier indices must be unique and inside the frame")
            self._pending_hover_inlier = (
                None
                if not indices
                else build_pending_inlier_layer(frame.points_world_m[list(indices)])
            )
            self._pending_hover_key = key
            self.pending_hover_builds += 1
        return self._pending_hover_inlier

    def clear_pending_hover(self) -> None:
        self._pending_hover_key = None
        self._pending_hover_inlier = None

    def detected(
        self,
        result: DetectionResult | None,
        builder: Callable[[DetectionResult], MeshLayerPacket],
        presentation_key=None,
    ) -> MeshLayerPacket | None:
        if result is None:
            self._detected_result = None
            self._detected_presentation_key = None
            self._detected_layer = None
        elif self._detected_result is not result or self._detected_presentation_key != presentation_key:
            detected_layer = builder(result)
            self._detected_result = result
            self._detected_presentation_key = presentation_key
            self._detected_layer = detected_layer
            self.detected_builds += 1
        return self._detected_layer

    def clear_detected(self) -> None:
        self._detected_result = None
        self._detected_presentation_key = None
        self._detected_layer = None

    def analysis_detected(self, key, builder: Callable[[], MeshLayerPacket]) -> MeshLayerPacket:
        if key != self._analysis_detected_key:
            layer = builder()
            self._analysis_detected_key = key
            self._analysis_detected_layer = layer
            self.detected_builds += 1
        return self._analysis_detected_layer

    def captures(
        self,
        captures,
        *,
        surface_side_scale: SurfaceSideScale = SurfaceSideScale(),
        hover_basis: tuple[str, str] | None = None,
        highlighted_capture_id: str | None = None,
    ) -> tuple[MeshLayerPacket | None, PointLayerPacket | None]:
        """Cache all visible captured meshes/inliers as one packet per render role."""

        mesh_key = tuple(
            (
                item.capture_id,
                bool(item.mesh_visible),
                id(item.surface),
                item.mesh_subdivision,
                item.mesh_face_rgba,
                item.mesh_wire_rgba,
            )
            for item in captures
        )
        inlier_key = (surface_side_scale,) + tuple(
            (
                item.capture_id,
                bool(item.inliers_visible),
                hover_basis[1] if hover_basis and hover_basis[0] == item.capture_id else item.heatmap_basis.value,
                id(item.analysis),
            )
            for item in captures
        )
        rebuilt = False
        if mesh_key != self._capture_mesh_key:
            draws = []
            for item in captures:
                if item.mesh_visible:
                    detected = detected_scene_object(
                        item.surface.geometry_type,
                        item.surface.parameters,
                    )
                    detected = replace(
                        detected,
                        subdivision=item.mesh_subdivision,
                        display_subdivision=item.mesh_subdivision,
                        face_color_rgba=item.mesh_face_rgba,
                        wire_color_rgba=item.mesh_wire_rgba,
                    )
                    mesh_layer = build_mesh_layer(
                        RenderLayer.CAPTURE,
                        (build_mesh(detected).transformed(detected.pose_world_m),),
                        sphere_pole_center_world_m=(
                            detected.pose_world_m.position_m if detected.geometry_type is GeometryMode.SPHERE else None
                        ),
                    )
                    draws.extend(
                        replace(
                            draw,
                            draw_id=f"{item.capture_id}:{draw.draw_id}",
                            face_color=item.mesh_face_rgba,
                            wire_color=item.mesh_wire_rgba,
                        )
                        for draw in mesh_layer.draws
                    )
            self._capture_mesh_layer = (
                MeshLayerPacket(RenderLayer.CAPTURE, tuple(draws)) if draws else None
            )
            self._capture_mesh_key = mesh_key
            rebuilt = True
        if inlier_key != self._capture_inlier_key:
            inlier_layers = []
            for item in captures:
                residual = item.analysis.residual(
                    hover_basis[1] if hover_basis and hover_basis[0] == item.capture_id else item.heatmap_basis,
                    surface_side_scale,
                )
                if item.inliers_visible and residual is not None and residual.inlier_layer is not None:
                    inlier_layers.append(residual.inlier_layer)
            self._capture_inlier_layer = merge_point_layers(
                RenderLayer.INLIER_POINTS,
                inlier_layers,
            )
            self._capture_inlier_key = inlier_key
            rebuilt = True
        if rebuilt:
            self.capture_builds += 1
        mesh_layer = self._capture_mesh_layer
        inlier_layer = self._capture_inlier_layer
        if highlighted_capture_id is None or not captures:
            self._clear_capture_highlight()
            return mesh_layer, inlier_layer

        highlight_mesh_key = (mesh_key, highlighted_capture_id)
        if highlight_mesh_key != self._capture_highlight_mesh_key:
            highlighted_mesh = self._highlight_capture_mesh(mesh_layer, highlighted_capture_id)
            self._capture_highlight_mesh_layer = highlighted_mesh
            self._capture_highlight_mesh_key = highlight_mesh_key
            self.capture_highlight_mesh_builds += 1
        highlight_inlier_key = (inlier_key, highlighted_capture_id)
        if highlight_inlier_key != self._capture_highlight_inlier_key:
            highlighted_inliers = self._highlight_capture_inliers(
                captures, surface_side_scale, hover_basis, highlighted_capture_id,
            )
            self._capture_highlight_inlier_layer = highlighted_inliers
            self._capture_highlight_inlier_key = highlight_inlier_key
            self.capture_highlight_inlier_builds += 1
        return self._capture_highlight_mesh_layer, self._capture_highlight_inlier_layer

    @staticmethod
    def _highlight_capture_mesh(
        layer: MeshLayerPacket | None, capture_id: str,
    ) -> MeshLayerPacket | None:
        if layer is None:
            return None
        prefix = f"{capture_id}:"
        return MeshLayerPacket(
            RenderLayer.CAPTURE,
            tuple(
                replace(
                    draw,
                    face_color=hover_emphasis_color(draw.face_color),
                    wire_color=hover_emphasis_color(draw.wire_color),
                )
                if draw.draw_id.startswith(prefix)
                else replace(
                    draw,
                    face_color=hover_context_color(draw.face_color),
                    wire_color=hover_context_color(draw.wire_color),
                )
                for draw in layer.draws
            ),
        )

    @staticmethod
    def _highlight_capture_inliers(
        captures, surface_side_scale: SurfaceSideScale,
        hover_basis: tuple[str, str] | None, capture_id: str,
    ) -> PointLayerPacket | None:
        emphasized_layers = []
        for item in captures:
            residual = item.analysis.residual(
                hover_basis[1] if hover_basis and hover_basis[0] == item.capture_id else item.heatmap_basis,
                surface_side_scale,
            )
            if item.inliers_visible and residual is not None and residual.inlier_layer is not None:
                emphasized_layers.append(
                    emphasize_inlier_layer(
                        residual.inlier_layer, 1.0 if item.capture_id == capture_id else -1.0,
                    )
                )
        return merge_point_layers(RenderLayer.INLIER_POINTS, emphasized_layers)

    def _clear_capture_highlight(self) -> None:
        self._capture_highlight_mesh_key = None
        self._capture_highlight_inlier_key = None
        self._capture_highlight_mesh_layer = None
        self._capture_highlight_inlier_layer = None

    def capture_classified(
        self,
        frame: ScanFrame,
        captures,
        hover_result: DetectionResult | None = None,
        *,
        outlier_only: bool = False,
    ) -> tuple[PointLayerPacket, ...]:
        """Classify the current cloud by the union of durable captures and hover.

        Captures preserve compact membership bound to their exact source frame.
        Legacy/direct captures without that membership fall back to a cached,
        byte-identical float32 coordinate lookup. Replacing/deleting a capture
        changes the union, so released points return to the outlier packet without
        modifying ``ScanFrame``.

        The production renderer requests only outliers; the default keeps the
        legacy inlier/outlier pair for callers that need both.
        """

        if hover_result is not None and (
            hover_result.request_key.scan_generation != frame.generation
            or hover_result.request_key.scan_sequence != frame.sequence
        ):
            raise ValueError("hover result does not belong to the scan frame")
        applicable = tuple(
            item.surface
            for item in captures
            if item.surface.request_key.scan_generation == frame.generation
            and item.surface.request_key.scan_sequence == frame.sequence
        )
        key = (
            frame.identity,
            tuple(id(surface) for surface in applicable),
            hover_result,
            outlier_only,
        )
        if key != self._capture_classified_key:
            membership_key = (frame.identity, tuple(id(surface) for surface in applicable))
            if membership_key != self._capture_indices_key:
                indices = set()
                fallback = []
                for surface in applicable:
                    if surface.source_inlier_indices:
                        if any(
                            index >= frame.compacted_count
                            for index in surface.source_inlier_indices
                        ):
                            raise ValueError("captured inlier membership is outside the scan frame")
                        indices.update(surface.source_inlier_indices)
                    elif surface.inlier_points_world_m.shape[0]:
                        fallback.append(surface.inlier_points_world_m)
                if fallback:
                    source = np.ascontiguousarray(frame.points_world_m, dtype=np.float32)
                    captured = np.ascontiguousarray(
                        np.concatenate(tuple(fallback), axis=0),
                        dtype=np.float32,
                    )
                    row_type = np.dtype((np.void, source.dtype.itemsize * source.shape[1]))
                    source_rows = source.view(row_type).reshape(-1)
                    captured_rows = captured.view(row_type).reshape(-1)
                    indices.update(
                        int(index)
                        for index in np.flatnonzero(np.isin(source_rows, captured_rows))
                    )
                self._capture_indices = tuple(sorted(indices))
                self._capture_indices_key = membership_key
                self.capture_membership_builds += 1
            indices = set(self._capture_indices)
            if hover_result is not None:
                indices.update(hover_result.inlier_indices)
            if indices:
                ordered_indices = tuple(sorted(indices))
                self._capture_classified_layers = (
                    (build_outlier_layer(frame, inlier_indices=ordered_indices),)
                    if outlier_only
                    else build_classified_point_layers(frame, inlier_indices=ordered_indices)
                )
            else:
                # Captures from an older frame have no membership in the new
                # immutable cloud. Keep every new point visible as an outlier.
                source = self.source(frame)[0]
                self._capture_classified_layers = (
                    replace(source, layer=RenderLayer.OUTLIER_POINTS),
                )
            self._capture_classified_key = key
            self.classified_builds += 1
        return self._capture_classified_layers

    def clear_capture_classified(self) -> None:
        self._capture_indices_key = None
        self._capture_indices = ()
        self._capture_classified_key = None
        self._capture_classified_layers = ()

    def clear(self) -> None:
        self._designed_key = None
        self._designed_layer = None
        self._designed_geometries.clear()
        self._source_identity = None
        self._source_layers = ()
        self.clear_classified()
        self.clear_pending_hover()
        self.clear_detected()
        self._analysis_detected_key = None
        self._analysis_detected_layer = None
        self._capture_mesh_key = None
        self._capture_inlier_key = None
        self._capture_mesh_layer = None
        self._capture_inlier_layer = None
        self._clear_capture_highlight()
        self.clear_capture_classified()


__all__ = ["RenderPayloadCache"]
