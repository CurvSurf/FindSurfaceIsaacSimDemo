"""Layered OpenGLWidget renderer for mesh, point, and guide packets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time

import omni.kit.widget.opengl as og

from .opengl_runtime import GpuResource, GpuResourceKind, GpuResourceOwner
from .render_layers import (
    IDENTITY_MATRIX,
    LayerPayloadTracker,
    LayeredRenderPacket,
    LatestLayeredPacket,
    PointLayerPacket,
    RenderLayer,
    RenderStyle,
    TRANSLUCENT_MESH_LAYERS,
)


@dataclass(frozen=True)
class LayeredOpenGLTelemetry:
    initialize_count: int = 0
    context_generation: int = 0
    draw_count: int = 0
    upload_count: int = 0
    mesh_buffer_upload_count: int = 0
    packet_revision: int = -1
    framebuffer_width: int = 0
    framebuffer_height: int = 0
    logical_width: float = 0.0
    logical_height: float = 0.0
    dpi_scale: float = 1.0
    mesh_vertex_count: int = 0
    mesh_triangle_index_count: int = 0
    mesh_edge_index_count: int = 0
    point_count: int = 0
    source_point_depth_mode: str = "OFFSET"
    source_point_depth_offset_m: float = 0.006
    line_vertex_count: int = 0
    visible_draws: tuple[tuple[str, int], ...] = ()
    gpu_slot_count: int = 0
    gpu_resource_count: int = 0
    release_count: int = 0
    stale_handle_discard_count: int = 0
    resources_released: bool = False
    snapshot_count: int = 0
    snapshot_draw_framebuffer: int = -1
    snapshot_read_framebuffer: int = -1
    last_pack_ms: float = 0.0
    last_upload_ms: float = 0.0
    last_draw_ms: float = 0.0
    gl_error_count: int = 0
    last_gl_error: int | None = None
    error: str | None = None


@dataclass
class _MeshGpu:
    vao: int
    vertices: int
    triangles: int
    edges: int
    triangle_count: int = 0
    edge_count: int = 0


@dataclass
class _PointGpu:
    vao: int
    positions: int
    heatmap: int
    emphasis: int
    count: int = 0
    uploaded_positions: bytes | None = None
    uploaded_heatmap: bytes | None = None
    uploaded_emphasis: bytes | None = None


@dataclass
class _LineGpu:
    vao: int
    vertices: int
    count: int = 0


class LayeredOpenGLScene(og.OpenGLScene):
    """Consume immutable layer packets without reading live domain state."""

    _MESH_VERTEX_SHADER = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 viewProjection;
        uniform mat4 model;
        void main() {
            vec4 clipPosition = viewProjection * model * vec4(position, 1.0);
            // OpenGLWidget exposes a bottom-left framebuffer as a top-left
            // omni.ui texture. Keep camera/picking math conventional and flip
            // exactly once at this presentation boundary.
            clipPosition.y = -clipPosition.y;
            gl_Position = clipPosition;
        }
    """
    _MESH_FRAGMENT_SHADER = """
        #version 330 core
        uniform vec4 layerColor;
        out vec4 fragmentColor;
        void main() { fragmentColor = layerColor; }
    """
    _POINT_VERTEX_SHADER = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in float heatmapCoordinate;
        layout(location = 2) in float hoverEmphasis;
        uniform mat4 viewProjection;
        uniform float pointSizePx;
        uniform vec3 cameraEyeWorld;
        uniform int sourcePointDepthMode;
        uniform float sourcePointDepthOffset;
        out float heatValue;
        out float emphasisValue;
        void main() {
            vec4 clipPosition = viewProjection * vec4(position, 1.0);
            if (sourcePointDepthMode == 2 && clipPosition.w > 0.000001) {
                float pointDistance = distance(position, cameraEyeWorld);
                if (pointDistance > 0.000001 && sourcePointDepthOffset > 0.0) {
                    float safeOffset = min(
                        sourcePointDepthOffset,
                        max(0.0, pointDistance - 0.0001)
                    );
                    vec3 comparisonPosition = mix(
                        position,
                        cameraEyeWorld,
                        safeOffset / pointDistance
                    );
                    vec4 comparisonClip = viewProjection * vec4(comparisonPosition, 1.0);
                    if (comparisonClip.w > 0.000001) {
                        clipPosition.z = (comparisonClip.z / comparisonClip.w) * clipPosition.w;
                    }
                }
            }
            clipPosition.y = -clipPosition.y;
            gl_Position = clipPosition;
            gl_PointSize = pointSizePx;
            heatValue = heatmapCoordinate;
            emphasisValue = hoverEmphasis;
        }
    """
    _POINT_FRAGMENT_SHADER = """
        #version 330 core
        in float heatValue;
        in float emphasisValue;
        uniform vec4 fillColor;
        uniform vec4 outlineColor;
        uniform vec4 heatZero;
        uniform vec4 heatMiddle;
        uniform vec4 heatHigh;
        uniform int useHeatmap;
        uniform int useOutline;
        out vec4 fragmentColor;
        void main() {
            vec2 offset = gl_PointCoord * 2.0 - vec2(1.0);
            float radiusSquared = dot(offset, offset);
            if (radiusSquared > 1.0) discard;
            float radius = sqrt(radiusSquared);
            vec4 interior = fillColor;
            if (useHeatmap == 1) {
                float value = clamp(heatValue, 0.0, 1.0);
                interior = value < 0.5
                    ? mix(heatZero, heatMiddle, value * 2.0)
                    : mix(heatMiddle, heatHigh, (value - 0.5) * 2.0);
            }
            vec4 pointColor = (useOutline == 1 && radius > 0.70) ? outlineColor : interior;
            if (emphasisValue > 0.5) {
                pointColor.rgb = mix(pointColor.rgb, vec3(1.0), 0.24);
            } else if (emphasisValue < -0.5) {
                pointColor.rgb *= 0.62;
            }
            fragmentColor = pointColor;
        }
    """

    def __init__(self, widget, *, error_hook=None) -> None:
        self._packets = LatestLayeredPacket()
        self._resources = GpuResourceOwner()
        self._lock = threading.Lock()
        self._telemetry = LayeredOpenGLTelemetry()
        self._error_hook = error_hook
        self._mesh_program = 0
        self._point_program = 0
        self._mesh_uniforms: dict[str, int] = {}
        self._point_uniforms: dict[str, int] = {}
        self._mesh_gpu: dict[tuple[RenderLayer, str], _MeshGpu] = {}
        self._point_gpu: dict[RenderLayer, _PointGpu] = {}
        self._line_gpu: dict[RenderLayer, _LineGpu] = {}
        self._uploaded_payloads = LayerPayloadTracker()
        self._uploaded_mesh_geometry = {}
        self._uploaded_submission = -1
        self._close_requested = False
        self._recreate_requested = False
        self._snapshot_requested = False
        self._snapshot: tuple[int, int, bytes] | None = None
        self._source_point_depth_mode = 2
        self._source_point_depth_offset_m = 0.006
        super().__init__(widget)

    def submit(self, packet: LayeredRenderPacket) -> int:
        return self._packets.submit(packet)

    def set_source_point_depth(self, *, depth_test_off: bool, offset_m: float) -> None:
        offset_m = float(offset_m)
        if not 0.0 <= offset_m <= 1.0:
            raise ValueError("source point depth offset must be between 0 and 1 metre")
        with self._lock:
            self._source_point_depth_mode = 1 if depth_test_off else 2
            self._source_point_depth_offset_m = offset_m
            self._telemetry = replace(
                self._telemetry,
                source_point_depth_mode="DEPTH_TEST_OFF" if depth_test_off else "OFFSET",
                source_point_depth_offset_m=offset_m,
            )

    def set_logical_size(self, width: float, height: float) -> None:
        width, height = max(0.0, float(width)), max(0.0, float(height))
        current = self.telemetry()
        scales = [
            current.framebuffer_width / width if width else 1.0,
            current.framebuffer_height / height if height else 1.0,
        ]
        self._update(logical_width=width, logical_height=height, dpi_scale=max(scales))

    def request_context_recreate(self) -> None:
        if not self._close_requested:
            self._recreate_requested = True

    def request_snapshot(self) -> None:
        self._snapshot_requested = True

    def snapshot(self) -> tuple[int, int, bytes] | None:
        with self._lock:
            return self._snapshot

    def telemetry(self) -> LayeredOpenGLTelemetry:
        with self._lock:
            return self._telemetry

    def initialize_gl(self, viewport_width: int, viewport_height: int, first_init: bool) -> None:
        try:
            from OpenGL import GL

            self._set_framebuffer_size(viewport_width, viewport_height)
            if self._close_requested:
                self._release_gl(GL)
                return
            if self._mesh_program and not first_init:
                current = self.telemetry()
                self._update(initialize_count=current.initialize_count + 1)
                return
            lost = bool(self._mesh_program and first_init)
            if lost:
                self._clear_handles()
            generation = self._resources.begin_context(previous_context_lost=lost)
            self._create_programs(GL)
            current = self.telemetry()
            self._update(
                initialize_count=current.initialize_count + 1,
                context_generation=generation,
                stale_handle_discard_count=self._resources.stale_discard_count,
                resources_released=False,
                error=None,
            )
            self._check_errors(GL)
        except Exception as error:
            self._update(error=f"{type(error).__name__}: {error}")

    def draw_gl(self, viewport_width: int, viewport_height: int) -> None:
        started = time.perf_counter()
        try:
            from OpenGL import GL

            self._set_framebuffer_size(viewport_width, viewport_height)
            if self._close_requested:
                self._release_gl(GL)
                return
            if self._recreate_requested:
                self._recreate_requested = False
                self._destroy_gl(GL)
                generation = self._resources.begin_context()
                self._create_programs(GL)
                self._update(context_generation=generation, resources_released=False)
            if not self._mesh_program or viewport_width <= 0 or viewport_height <= 0:
                return
            submission, packet = self._packets.snapshot()
            if packet is None:
                return
            if submission != self._uploaded_submission:
                upload_started = time.perf_counter()
                self._upload_packet(GL, packet)
                self._uploaded_submission = submission
                current = self.telemetry()
                self._update(
                    upload_count=current.upload_count + 1,
                    packet_revision=packet.revision,
                    last_upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                )
            self._draw_packet(GL, packet)
            if self._snapshot_requested:
                self._capture_snapshot(GL, viewport_width, viewport_height)
            self._check_errors(GL)
            current = self.telemetry()
            self._update(
                draw_count=current.draw_count + 1,
                last_draw_ms=(time.perf_counter() - started) * 1000.0,
                gpu_slot_count=len(self._mesh_gpu) + len(self._point_gpu) + len(self._line_gpu),
                gpu_resource_count=self._resources.resource_count,
                error=None,
            )
        except Exception as error:
            self._update(error=f"{type(error).__name__}: {error}")

    def close(self) -> None:
        if not self.telemetry().resources_released:
            self._close_requested = True

    def _set_framebuffer_size(self, width: int, height: int) -> None:
        width, height = max(0, int(width)), max(0, int(height))
        current = self.telemetry()
        scale = max(
            width / current.logical_width if current.logical_width else 1.0,
            height / current.logical_height if current.logical_height else 1.0,
        )
        self._update(framebuffer_width=width, framebuffer_height=height, dpi_scale=scale)

    def _create_programs(self, GL) -> None:
        self._mesh_program = self._build_program(GL, self._MESH_VERTEX_SHADER, self._MESH_FRAGMENT_SHADER)
        self._point_program = self._build_program(GL, self._POINT_VERTEX_SHADER, self._POINT_FRAGMENT_SHADER)
        self._mesh_uniforms = {
            name: GL.glGetUniformLocation(self._mesh_program, name)
            for name in ("viewProjection", "model", "layerColor")
        }
        self._point_uniforms = {
            name: GL.glGetUniformLocation(self._point_program, name)
            for name in (
                "viewProjection", "pointSizePx", "fillColor", "outlineColor",
                "heatZero", "heatMiddle", "heatHigh", "useHeatmap", "useOutline",
                "cameraEyeWorld", "sourcePointDepthMode", "sourcePointDepthOffset",
            )
        }
        self._uploaded_submission = -1

    def _build_program(self, GL, vertex_source: str, fragment_source: str) -> int:
        shaders: list[int] = []
        try:
            for shader_type, source in (
                (GL.GL_VERTEX_SHADER, vertex_source),
                (GL.GL_FRAGMENT_SHADER, fragment_source),
            ):
                shader = self._resources.own(GpuResourceKind.SHADER, GL.glCreateShader(shader_type))
                shaders.append(shader)
                GL.glShaderSource(shader, source)
                GL.glCompileShader(shader)
                if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                    raise RuntimeError(self._text(GL.glGetShaderInfoLog(shader)))
            program = self._resources.own(GpuResourceKind.PROGRAM, GL.glCreateProgram())
            for shader in shaders:
                GL.glAttachShader(program, shader)
            GL.glLinkProgram(program)
            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                raise RuntimeError(self._text(GL.glGetProgramInfoLog(program)))
            return program
        finally:
            for shader in shaders:
                if self._resources.is_current(GpuResourceKind.SHADER, shader):
                    GL.glDeleteShader(shader)
                    self._resources.forget(GpuResourceKind.SHADER, shader)

    def _upload_packet(self, GL, packet: LayeredRenderPacket) -> None:
        delta = self._uploaded_payloads.update(packet)
        mesh_buffer_uploads = 0
        current_mesh_geometry = {
            (layer.layer, draw.draw_id): draw.geometry
            for layer in packet.mesh_layers
            for draw in layer.draws
        }
        for key in self._mesh_gpu.keys() - current_mesh_geometry.keys():
            self._release_mesh_gpu(GL, key)
        for layer in delta.point_removed:
            gpu = self._point_gpu.get(layer)
            if gpu is not None:
                gpu.count = 0
        for layer in delta.line_removed:
            gpu = self._line_gpu.get(layer)
            if gpu is not None:
                gpu.count = 0
        for key, geometry in current_mesh_geometry.items():
            if self._uploaded_mesh_geometry.get(key) is geometry:
                gpu = self._mesh_gpu[key]
                gpu.triangle_count = geometry.triangle_index_count
                gpu.edge_count = geometry.edge_index_count
                continue
            gpu = self._ensure_mesh_gpu(GL, key)
            GL.glBindVertexArray(gpu.vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, gpu.vertices)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                len(geometry.vertices_f32),
                geometry.vertices_f32,
                GL.GL_STATIC_DRAW,
            )
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, gpu.triangles)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                len(geometry.triangle_indices_u32),
                geometry.triangle_indices_u32,
                GL.GL_STATIC_DRAW,
            )
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, gpu.edges)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                len(geometry.edge_indices_u32),
                geometry.edge_indices_u32,
                GL.GL_STATIC_DRAW,
            )
            gpu.triangle_count = geometry.triangle_index_count
            gpu.edge_count = geometry.edge_index_count
            mesh_buffer_uploads += 1
            self._uploaded_mesh_geometry[key] = geometry
        for layer in packet.point_layers:
            gpu = self._ensure_point_gpu(GL, layer.layer)
            self._upload_point_attributes(GL, gpu, layer)
        for layer in delta.line_changed:
            gpu = self._ensure_line_gpu(GL, layer.layer)
            GL.glBindVertexArray(gpu.vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, gpu.vertices)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(layer.vertices_f32), layer.vertices_f32, GL.GL_DYNAMIC_DRAW)
            gpu.count = layer.vertex_count
        GL.glBindVertexArray(0)
        current = self.telemetry()
        self._update(
            mesh_buffer_upload_count=current.mesh_buffer_upload_count + mesh_buffer_uploads,
            mesh_vertex_count=sum(layer.vertex_count for layer in packet.mesh_layers),
            mesh_triangle_index_count=sum(layer.triangle_index_count for layer in packet.mesh_layers),
            mesh_edge_index_count=sum(layer.edge_index_count for layer in packet.mesh_layers),
            point_count=sum(layer.point_count for layer in packet.point_layers),
            line_vertex_count=sum(layer.vertex_count for layer in packet.line_layers),
        )

    @staticmethod
    def _upload_point_attributes(GL, gpu: _PointGpu, layer: PointLayerPacket) -> None:
        for buffer, payload, cached_name in (
            (gpu.positions, layer.positions_f32, "uploaded_positions"),
            (gpu.heatmap, layer.heatmap_f32, "uploaded_heatmap"),
            (gpu.emphasis, layer.emphasis_f32, "uploaded_emphasis"),
        ):
            previous = getattr(gpu, cached_name)
            if previous is payload:
                continue
            if previous is not None and previous == payload:
                setattr(gpu, cached_name, payload)
                continue
            GL.glBindVertexArray(gpu.vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(payload), payload, GL.GL_STREAM_DRAW)
            setattr(gpu, cached_name, payload)
        gpu.count = layer.point_count

    def _ensure_mesh_gpu(self, GL, key: tuple[RenderLayer, str]) -> _MeshGpu:
        gpu = self._mesh_gpu.get(key)
        if gpu is not None:
            if all((gpu.vao, gpu.vertices, gpu.triangles, gpu.edges)):
                return gpu
            self._release_mesh_gpu(GL, key)
        vao = self._own_vao(GL)
        vertices, triangles, edges = self._own_buffers(GL, 3)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertices)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)
        gpu = _MeshGpu(vao, vertices, triangles, edges)
        self._mesh_gpu[key] = gpu
        return gpu

    def _release_mesh_gpu(self, GL, key: tuple[RenderLayer, str]) -> None:
        gpu = self._mesh_gpu.get(key)
        if gpu is None:
            return
        self._uploaded_mesh_geometry.pop(key, None)
        gpu.triangle_count = gpu.edge_count = 0
        for name, kind in (
            ("edges", GpuResourceKind.BUFFER),
            ("triangles", GpuResourceKind.BUFFER),
            ("vertices", GpuResourceKind.BUFFER),
            ("vao", GpuResourceKind.VERTEX_ARRAY),
        ):
            handle = getattr(gpu, name)
            if self._resources.is_current(kind, handle):
                if kind is GpuResourceKind.VERTEX_ARRAY:
                    GL.glDeleteVertexArrays(1, [handle])
                else:
                    GL.glDeleteBuffers(1, [handle])
                self._resources.forget(kind, handle)
                current = self.telemetry()
                self._update(release_count=current.release_count + 1)
            setattr(gpu, name, 0)
        del self._mesh_gpu[key]

    def _ensure_point_gpu(self, GL, layer: RenderLayer) -> _PointGpu:
        gpu = self._point_gpu.get(layer)
        if gpu is not None:
            return gpu
        vao = self._own_vao(GL)
        positions, heatmap, emphasis = self._own_buffers(GL, 3)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, positions)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, heatmap)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 1, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, emphasis)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 1, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)
        gpu = _PointGpu(vao, positions, heatmap, emphasis)
        self._point_gpu[layer] = gpu
        return gpu

    def _ensure_line_gpu(self, GL, layer: RenderLayer) -> _LineGpu:
        gpu = self._line_gpu.get(layer)
        if gpu is not None:
            return gpu
        vao = self._own_vao(GL)
        vertices = self._own_buffers(GL, 1)[0]
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertices)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)
        gpu = _LineGpu(vao, vertices)
        self._line_gpu[layer] = gpu
        return gpu

    def _own_vao(self, GL) -> int:
        return self._resources.own(GpuResourceKind.VERTEX_ARRAY, GL.glGenVertexArrays(1))

    def _own_buffers(self, GL, count: int) -> tuple[int, ...]:
        values = GL.glGenBuffers(count)
        if count == 1:
            values = (values,)
        return tuple(self._resources.own(GpuResourceKind.BUFFER, int(value)) for value in values)

    def _draw_packet(self, GL, packet: LayeredRenderPacket) -> None:
        style = packet.style
        GL.glViewport(0, 0, self.telemetry().framebuffer_width, self.telemetry().framebuffer_height)
        GL.glClearColor(*style.background)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        draw_counts: dict[str, int] = {}

        GL.glUseProgram(self._mesh_program)
        GL.glUniformMatrix4fv(self._mesh_uniforms["viewProjection"], 1, GL.GL_FALSE, packet.view_projection)
        for layer_packet in packet.mesh_layers:
            if not packet.visibility.is_visible(layer_packet.layer):
                continue
            face, wire = self._mesh_colors(style, layer_packet.layer)
            late_overlay = layer_packet.layer in {RenderLayer.PREVIEW, RenderLayer.CAPTURE}
            # Keep depth testing so nearer opaque content still wins, but never
            # let a translucent comparison surface hide later meshes or points.
            writes_depth = layer_packet.layer not in (
                TRANSLUCENT_MESH_LAYERS | {RenderLayer.CAMERA_FRUSTUM_FACE}
            )
            culls_back_faces = layer_packet.layer in TRANSLUCENT_MESH_LAYERS
            for draw_packet in layer_packet.draws:
                draw_face = draw_packet.face_color or face
                draw_wire = draw_packet.wire_color or wire
                gpu = self._mesh_gpu[(layer_packet.layer, draw_packet.draw_id)]
                GL.glUniformMatrix4fv(
                    self._mesh_uniforms["model"], 1, GL.GL_FALSE, draw_packet.model_matrix,
                )
                if culls_back_faces:
                    # Isolate each object, not the whole semantic layer, so an
                    # unrelated object cannot win this object's depth pre-pass.
                    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
                    GL.glEnable(GL.GL_DEPTH_TEST)
                    GL.glEnable(GL.GL_CULL_FACE)
                    # The presentation-space Y reflection reverses winding
                    # before rasterization, so local CCW fronts arrive as CW.
                    GL.glFrontFace(GL.GL_CW)
                    GL.glCullFace(GL.GL_BACK)
                elif late_overlay:
                    GL.glDisable(GL.GL_DEPTH_TEST)
                GL.glBindVertexArray(gpu.vao)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, gpu.triangles)
                if culls_back_faces:
                    GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
                    GL.glDepthMask(GL.GL_TRUE)
                    GL.glDrawElements(GL.GL_TRIANGLES, gpu.triangle_count, GL.GL_UNSIGNED_INT, None)
                    GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glDepthMask(GL.GL_TRUE if writes_depth else GL.GL_FALSE)
                GL.glUniform4f(self._mesh_uniforms["layerColor"], *draw_face)
                GL.glDrawElements(GL.GL_TRIANGLES, gpu.triangle_count, GL.GL_UNSIGNED_INT, None)
                GL.glDisable(GL.GL_BLEND)
                GL.glUniform4f(self._mesh_uniforms["layerColor"], *draw_wire)
                GL.glLineWidth(style.wire_width_dip * self.telemetry().dpi_scale)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, gpu.edges)
                GL.glDrawElements(GL.GL_LINES, gpu.edge_count, GL.GL_UNSIGNED_INT, None)
                GL.glDepthMask(GL.GL_TRUE)
                if culls_back_faces:
                    GL.glDisable(GL.GL_CULL_FACE)
                    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
                if late_overlay:
                    GL.glEnable(GL.GL_DEPTH_TEST)
                draw_counts[layer_packet.layer.value] = draw_counts.get(layer_packet.layer.value, 0) + 2

        GL.glUseProgram(self._point_program)
        GL.glUniformMatrix4fv(self._point_uniforms["viewProjection"], 1, GL.GL_FALSE, packet.view_projection)
        GL.glUniform3f(self._point_uniforms["cameraEyeWorld"], *packet.camera_eye_world_m)
        with self._lock:
            source_depth_mode = self._source_point_depth_mode
            source_depth_offset_m = self._source_point_depth_offset_m
        GL.glUniform1f(self._point_uniforms["sourcePointDepthOffset"], source_depth_offset_m)
        for layer_packet in packet.point_layers:
            if not packet.visibility.is_visible(layer_packet.layer):
                continue
            gpu = self._point_gpu[layer_packet.layer]
            fill, size, heatmap, outline = self._point_style(style, layer_packet)
            source_cloud = layer_packet.layer in {
                RenderLayer.SOURCE_POINTS, RenderLayer.INLIER_POINTS, RenderLayer.OUTLIER_POINTS,
            }
            point_depth_mode = source_depth_mode if source_cloud else 0
            GL.glUniform1i(self._point_uniforms["sourcePointDepthMode"], point_depth_mode)
            GL.glUniform1f(self._point_uniforms["pointSizePx"], size * self.telemetry().dpi_scale)
            GL.glUniform4f(self._point_uniforms["fillColor"], *fill)
            GL.glUniform4f(self._point_uniforms["outlineColor"], *style.inlier_outline)
            GL.glUniform4f(self._point_uniforms["heatZero"], *style.heatmap_zero)
            GL.glUniform4f(self._point_uniforms["heatMiddle"], *style.heatmap_middle)
            GL.glUniform4f(self._point_uniforms["heatHigh"], *style.heatmap_high)
            GL.glUniform1i(self._point_uniforms["useHeatmap"], int(heatmap))
            GL.glUniform1i(self._point_uniforms["useOutline"], int(outline))
            GL.glBindVertexArray(gpu.vao)
            if point_depth_mode == 1:
                GL.glDisable(GL.GL_DEPTH_TEST)
                GL.glDepthMask(GL.GL_FALSE)
            GL.glDrawArrays(GL.GL_POINTS, 0, gpu.count)
            if point_depth_mode == 1:
                GL.glDepthMask(GL.GL_TRUE)
                GL.glEnable(GL.GL_DEPTH_TEST)
            draw_counts[layer_packet.layer.value] = draw_counts.get(layer_packet.layer.value, 0) + 1

        GL.glUseProgram(self._mesh_program)
        GL.glUniformMatrix4fv(self._mesh_uniforms["viewProjection"], 1, GL.GL_FALSE, packet.view_projection)
        GL.glUniformMatrix4fv(self._mesh_uniforms["model"], 1, GL.GL_FALSE, IDENTITY_MATRIX)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glLineWidth(style.wire_width_dip * self.telemetry().dpi_scale)
        for layer_packet in packet.line_layers:
            if not packet.visibility.is_visible(layer_packet.layer):
                continue
            depth_aware = layer_packet.layer is RenderLayer.CAMERA_FRUSTUM
            (GL.glEnable if depth_aware else GL.glDisable)(GL.GL_DEPTH_TEST)
            color = style.frustum_edge if depth_aware else style.guide_color
            if layer_packet.layer is RenderLayer.SEED_RADIUS:
                color = style.seed_radius_color
            GL.glUniform4f(self._mesh_uniforms["layerColor"], *color)
            gpu = self._line_gpu[layer_packet.layer]
            GL.glBindVertexArray(gpu.vao)
            GL.glDrawArrays(GL.GL_LINES, 0, gpu.count)
            draw_counts[layer_packet.layer.value] = draw_counts.get(layer_packet.layer.value, 0) + 1
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        self._update(visible_draws=tuple(sorted(draw_counts.items())))

    @staticmethod
    def _mesh_colors(style: RenderStyle, layer: RenderLayer):
        return {
            RenderLayer.DESIGNED_MESH: (style.designed_face, style.designed_wire),
            RenderLayer.DETECTED_MESH: (style.detected_face, style.detected_wire),
            RenderLayer.PREVIEW: (style.preview_face, style.preview_wire),
            RenderLayer.CAPTURE: (style.capture_face, style.capture_wire),
            RenderLayer.CAMERA_FRUSTUM_FACE: (style.frustum_face, style.frustum_edge),
        }[layer]

    @staticmethod
    def _point_style(style: RenderStyle, packet):
        return {
            RenderLayer.SOURCE_POINTS: (style.source_fill, style.point_size_dip, False, False),
            RenderLayer.OUTLIER_POINTS: (style.outlier_fill, style.point_size_dip, False, False),
            # A fixed-colour outline covers every fragment of a one-pixel point
            # on common DPI scales, hiding the signed heatmap as uniform yellow.
            # Inliers therefore use the heatmap for the complete circular point.
            RenderLayer.INLIER_POINTS: (style.heatmap_zero, style.inlier_size_dip, True, False),
            RenderLayer.SEED: (
                style.seed_fallback_fill if packet.used_fallback else style.seed_fill,
                style.seed_size_dip,
                False,
                True,
            ),
        }[packet.layer]

    def _capture_snapshot(self, GL, width: int, height: int) -> None:
        GL.glFinish()
        draw_framebuffer = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        read_framebuffer = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        if draw_framebuffer != read_framebuffer:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, draw_framebuffer)
        if draw_framebuffer:
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        pixels = bytes(GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE))
        if draw_framebuffer != read_framebuffer:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_framebuffer)
        with self._lock:
            self._snapshot = (width, height, pixels)
            self._telemetry = replace(
                self._telemetry,
                snapshot_count=self._telemetry.snapshot_count + 1,
                snapshot_draw_framebuffer=draw_framebuffer,
                snapshot_read_framebuffer=read_framebuffer,
            )
        self._snapshot_requested = False

    def _release_gl(self, GL) -> None:
        released = self._destroy_gl(GL)
        current = self.telemetry()
        self._close_requested = False
        self._update(
            release_count=current.release_count + released,
            resources_released=True,
            gpu_resource_count=0,
        )

    def _destroy_gl(self, GL) -> int:
        GL.glFinish()

        def delete(resource: GpuResource) -> None:
            if resource.kind is GpuResourceKind.PROGRAM:
                GL.glDeleteProgram(resource.handle)
            elif resource.kind is GpuResourceKind.SHADER:
                GL.glDeleteShader(resource.handle)
            elif resource.kind is GpuResourceKind.VERTEX_ARRAY:
                GL.glDeleteVertexArrays(1, [resource.handle])
            elif resource.kind is GpuResourceKind.BUFFER:
                GL.glDeleteBuffers(1, [resource.handle])
            elif resource.kind is GpuResourceKind.TEXTURE:
                GL.glDeleteTextures(1, [resource.handle])
            elif resource.kind is GpuResourceKind.FRAMEBUFFER:
                GL.glDeleteFramebuffers(1, [resource.handle])

        released = self._resources.release(delete)
        GL.glFinish()
        self._clear_handles()
        return released

    def _clear_handles(self) -> None:
        self._mesh_program = self._point_program = 0
        self._mesh_uniforms.clear()
        self._point_uniforms.clear()
        self._mesh_gpu.clear()
        self._point_gpu.clear()
        self._line_gpu.clear()
        self._uploaded_payloads.reset()
        self._uploaded_mesh_geometry.clear()
        self._uploaded_submission = -1

    def _check_errors(self, GL) -> None:
        errors = []
        while len(errors) < 16:
            error = int(GL.glGetError())
            if error == int(GL.GL_NO_ERROR):
                break
            errors.append(error)
        if errors:
            if self._error_hook is not None:
                for error in errors:
                    self._error_hook(error)
            current = self.telemetry()
            self._update(
                gl_error_count=current.gl_error_count + len(errors),
                last_gl_error=errors[-1],
            )

    def _update(self, **changes) -> None:
        with self._lock:
            self._telemetry = replace(self._telemetry, **changes)

    @staticmethod
    def _text(value) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


__all__ = ["LayeredOpenGLScene", "LayeredOpenGLTelemetry"]
