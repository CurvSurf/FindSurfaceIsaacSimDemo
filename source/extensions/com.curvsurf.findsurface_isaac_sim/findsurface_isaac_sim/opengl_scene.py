"""Isaac Sim OpenGLWidget adapter with generation-safe GPU ownership."""

from __future__ import annotations

import ctypes
import time

import omni.kit.widget.opengl as og

from .opengl_runtime import (
    GpuResource,
    GpuResourceKind,
    OpenGLRenderPacket,
    OpenGLRuntimeState,
    OpenGLTelemetry,
)


class OpenGLScene(og.OpenGLScene):
    """OpenGL foundation consumed by the render layers."""

    _VERTEX_SHADER = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform float pointPass;
        out vec3 color;
        void main() {
            gl_Position = vec4(position, 1.0);
            gl_PointSize = 7.0;
            color = pointPass > 0.5 ? vec3(0.10, 0.85, 1.0) : vec3(0.98, 0.62, 0.18);
        }
    """
    _FRAGMENT_SHADER = """
        #version 330 core
        in vec3 color;
        uniform float pointPass;
        out vec4 fragmentColor;
        void main() {
            if (pointPass > 0.5) {
                vec2 offset = gl_PointCoord * 2.0 - vec2(1.0);
                if (dot(offset, offset) > 1.0) discard;
            }
            fragmentColor = vec4(color, 1.0);
        }
    """

    def __init__(self, widget, *, error_hook=None) -> None:
        self._state = OpenGLRuntimeState(error_hook=error_hook)
        self._program = 0
        self._vao = 0
        self._triangle_vbo = 0
        self._triangle_ibo = 0
        self._point_vbo = 0
        self._point_pass_location = -1
        self._uploaded_revision = -1
        super().__init__(widget)

    def submit(self, packet: OpenGLRenderPacket) -> None:
        self._state.packets.submit(packet)

    def set_logical_size(self, width: float, height: float) -> None:
        self._state.set_logical_size(width, height)

    def request_context_recreate(self) -> None:
        self._state.request_recreate()

    def telemetry(self) -> OpenGLTelemetry:
        return self._state.telemetry()

    def initialize_gl(self, viewport_width: int, viewport_height: int, first_init: bool) -> None:
        try:
            from OpenGL import GL

            self._state.set_framebuffer_size(viewport_width, viewport_height)
            if self._state.close_requested:
                self._release_gl(GL)
                return
            if self._program and not first_init:
                telemetry = self.telemetry()
                self._state.update_telemetry(initialize_count=telemetry.initialize_count + 1)
                return
            context_lost = bool(self._program and first_init)
            if context_lost:
                self._clear_local_handles()
            generation = self._state.resources.begin_context(previous_context_lost=context_lost)
            self._create_gl(GL)
            self._uploaded_revision = -1
            telemetry = self.telemetry()
            self._state.update_telemetry(
                initialize_count=telemetry.initialize_count + 1,
                context_generation=generation,
                stale_handle_discard_count=self._state.resources.stale_discard_count,
                resources_released=False,
                error=None,
            )
            self._check_gl_errors(GL)
        except Exception as error:
            self._state.update_telemetry(error=f"{type(error).__name__}: {error}")

    def draw_gl(self, viewport_width: int, viewport_height: int) -> None:
        draw_started = time.perf_counter()
        try:
            from OpenGL import GL

            self._state.set_framebuffer_size(viewport_width, viewport_height)
            if self._state.close_requested:
                self._release_gl(GL)
                return
            if self._state.consume_recreate_request():
                self._destroy_gl(GL)
                generation = self._state.resources.begin_context()
                self._create_gl(GL)
                self._uploaded_revision = -1
                telemetry = self.telemetry()
                self._state.update_telemetry(
                    context_generation=generation,
                    recreate_count=telemetry.recreate_count + 1,
                    resources_released=False,
                )
            if not self._program or viewport_width <= 0 or viewport_height <= 0:
                return
            packet = self._state.packets.snapshot()
            telemetry = self.telemetry()
            if packet.revision != self._uploaded_revision:
                upload_started = time.perf_counter()
                self._upload(GL, packet)
                self._uploaded_revision = packet.revision
                telemetry = self.telemetry()
                self._state.update_telemetry(
                    upload_count=telemetry.upload_count + 1,
                    last_upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                )

            GL.glViewport(0, 0, int(viewport_width), int(viewport_height))
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glClearColor(0.035, 0.05, 0.075, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glUseProgram(self._program)
            GL.glBindVertexArray(self._vao)
            if packet.triangle_vertex_count:
                GL.glUniform1f(self._point_pass_location, 0.0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._triangle_vbo)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._triangle_ibo)
                GL.glDrawElements(
                    GL.GL_TRIANGLES,
                    packet.triangle_vertex_count,
                    GL.GL_UNSIGNED_INT,
                    None,
                )
            if packet.point_count:
                GL.glUniform1f(self._point_pass_location, 1.0)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._point_vbo)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                GL.glDrawArrays(GL.GL_POINTS, 0, packet.point_count)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            self._check_gl_errors(GL)
            telemetry = self.telemetry()
            self._state.update_telemetry(
                draw_count=telemetry.draw_count + 1,
                packet_revision=packet.revision,
                triangle_vertex_count=packet.triangle_vertex_count,
                point_count=packet.point_count,
                last_draw_ms=(time.perf_counter() - draw_started) * 1000.0,
                error=None,
            )
        except Exception as error:
            self._state.update_telemetry(error=f"{type(error).__name__}: {error}")

    def close(self) -> None:
        if not self.telemetry().resources_released:
            self._state.request_close()

    def _create_gl(self, GL) -> None:
        shaders: list[int] = []
        try:
            for shader_type, source in (
                (GL.GL_VERTEX_SHADER, self._VERTEX_SHADER),
                (GL.GL_FRAGMENT_SHADER, self._FRAGMENT_SHADER),
            ):
                shader = self._state.resources.own(
                    GpuResourceKind.SHADER, GL.glCreateShader(shader_type)
                )
                shaders.append(shader)
                GL.glShaderSource(shader, source)
                GL.glCompileShader(shader)
                if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                    raise RuntimeError(self._log_text(GL.glGetShaderInfoLog(shader)))
            self._program = self._state.resources.own(
                GpuResourceKind.PROGRAM, GL.glCreateProgram()
            )
            for shader in shaders:
                GL.glAttachShader(self._program, shader)
            GL.glLinkProgram(self._program)
            if not GL.glGetProgramiv(self._program, GL.GL_LINK_STATUS):
                raise RuntimeError(self._log_text(GL.glGetProgramInfoLog(self._program)))
            self._point_pass_location = GL.glGetUniformLocation(self._program, "pointPass")
            self._vao = self._state.resources.own(
                GpuResourceKind.VERTEX_ARRAY, GL.glGenVertexArrays(1)
            )
            buffers = GL.glGenBuffers(3)
            self._triangle_vbo, self._triangle_ibo, self._point_vbo = (
                self._state.resources.own(GpuResourceKind.BUFFER, int(value))
                for value in buffers
            )
            GL.glBindVertexArray(self._vao)
            GL.glEnableVertexAttribArray(0)
            GL.glBindVertexArray(0)
        finally:
            for shader in shaders:
                if self._state.resources.is_current(GpuResourceKind.SHADER, shader):
                    GL.glDeleteShader(shader)
                    self._state.resources.forget(GpuResourceKind.SHADER, shader)

    def _upload(self, GL, packet: OpenGLRenderPacket) -> None:
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._triangle_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            len(packet.triangle_vertices_f32),
            packet.triangle_vertices_f32,
            GL.GL_DYNAMIC_DRAW,
        )
        indices = (ctypes.c_uint * packet.triangle_vertex_count)(
            *range(packet.triangle_vertex_count)
        )
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._triangle_ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(indices), indices, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._point_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            len(packet.point_vertices_f32),
            packet.point_vertices_f32,
            GL.GL_STREAM_DRAW,
        )
        GL.glBindVertexArray(0)

    def _release_gl(self, GL) -> None:
        released = self._destroy_gl(GL)
        self._state.release_complete(released)

    def _destroy_gl(self, GL) -> int:
        GL.glFinish()

        def delete(resource: GpuResource) -> None:
            if resource.kind is GpuResourceKind.SHADER:
                GL.glDeleteShader(resource.handle)
            elif resource.kind is GpuResourceKind.PROGRAM:
                GL.glDeleteProgram(resource.handle)
            elif resource.kind is GpuResourceKind.VERTEX_ARRAY:
                GL.glDeleteVertexArrays(1, [resource.handle])
            elif resource.kind is GpuResourceKind.BUFFER:
                GL.glDeleteBuffers(1, [resource.handle])
            elif resource.kind is GpuResourceKind.TEXTURE:
                GL.glDeleteTextures(1, [resource.handle])
            elif resource.kind is GpuResourceKind.FRAMEBUFFER:
                GL.glDeleteFramebuffers(1, [resource.handle])

        released = self._state.resources.release(delete)
        GL.glFinish()
        self._clear_local_handles()
        return released

    def _clear_local_handles(self) -> None:
        self._program = self._vao = 0
        self._triangle_vbo = self._triangle_ibo = self._point_vbo = 0
        self._point_pass_location = -1
        self._uploaded_revision = -1

    def _check_gl_errors(self, GL) -> None:
        errors: list[int] = []
        while True:
            error = int(GL.glGetError())
            if error == int(GL.GL_NO_ERROR):
                break
            errors.append(error)
            if len(errors) >= 16:
                break
        if errors:
            if self._state.error_hook is not None:
                for error in errors:
                    self._state.error_hook(error)
            telemetry = self.telemetry()
            self._state.update_telemetry(
                gl_error_count=telemetry.gl_error_count + len(errors),
                last_gl_error=errors[-1],
            )

    @staticmethod
    def _log_text(value) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


__all__ = ["OpenGLScene"]
