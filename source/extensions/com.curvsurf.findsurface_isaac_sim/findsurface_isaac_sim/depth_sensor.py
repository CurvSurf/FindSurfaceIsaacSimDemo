"""Isaac Sim 6.0.1 SingleViewDepthCameraSensor profile and runtime adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import importlib.util
import math
import time
from typing import Any, Awaitable, Callable, Protocol, Sequence

import numpy as np

from .domain import Pose, ScanFrame, SensorKind, SensorLifecycle, SensorProfileDescriptor, SensorStatus, StageMetadata
from .scan import NativePointPayload, ScanDispatcher
from .scene import StagePose, StageSpace
from .sensor_diagnostics import DepthSensorDiagnostics
from .sensors import ProviderTelemetry, SensorOperationToken, SensorProviderStateMachine


DEPTH_EXTENSION_VERSION = "1.4.6"
D455_ASSET_URI = "/Isaac/Sensors/RealSense/D455/rsd455.usd"
DEFAULT_DEPTH_PROFILE_ID = "d455-16x9-50k"
DEPTH_ANNOTATOR = "depth_sensor_point_cloud_position"
DEPTH_DISTANCE_ANNOTATOR = "depth_sensor_distance"
DEPTH_RENDER_TRIGGER_ANNOTATOR = "distance_to_image_plane"

# Validated native XYZ encoding boundary for the shipped 6.0.1 D455 path.
# Other unit/clip combinations are not supported by this fixed-axis adapter.
D455_NATIVE_CLIP_STAGE_UNITS = (0.01, 1_000_000.0)


def validate_d455_native_space(metadata: StageMetadata, clipping_range=None) -> None:
    if metadata.up_axis != "Z" or not math.isclose(
        metadata.meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Unsupported D455 native XYZ stage: requires Z-up and 1 metre/unit; "
            f"got {metadata.up_axis}-up and {metadata.meters_per_unit} metre/unit. "
            "The existing stage has not been modified."
        )
    if clipping_range is not None and (
        len(clipping_range) != 2 or any(
            not math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-8)
            for actual, expected in zip(clipping_range, D455_NATIVE_CLIP_STAGE_UNITS)
        )
    ):
        raise RuntimeError(
            "Unsupported D455 native XYZ camera clippingRange: "
            f"requires {D455_NATIVE_CLIP_STAGE_UNITS}, got {tuple(clipping_range)}. "
            "No fallback or point correction has been applied."
        )


@dataclass(frozen=True)
class DepthIntrinsics:
    width: int
    height: int
    focal_x_px: float
    focal_y_px: float
    principal_x_px: float
    principal_y_px: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsics resolution must be positive")
        values = (self.focal_x_px, self.focal_y_px, self.principal_x_px, self.principal_y_px)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("intrinsics must be finite")
        if self.focal_x_px <= 0.0 or self.focal_y_px <= 0.0:
            raise ValueError("intrinsics focal lengths must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "focal_x_px": self.focal_x_px,
            "focal_y_px": self.focal_y_px,
            "principal_x_px": self.principal_x_px,
            "principal_y_px": self.principal_y_px,
        }


@dataclass(frozen=True)
class DepthProfile:
    profile_id: str
    label: str
    target_point_count: int
    height: int
    width: int
    tick_rate_hz: float = 30.0
    asset_uri: str = D455_ASSET_URI

    def __post_init__(self) -> None:
        if not self.profile_id or not self.label:
            raise ValueError("depth profile id and label cannot be empty")
        if self.target_point_count <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("depth profile point count and resolution must be positive")
        if self.maximum_pixel_count > 100_000:
            raise ValueError("depth profile cannot exceed 100,000 pixels")
        if not math.isfinite(self.tick_rate_hz) or self.tick_rate_hz <= 0.0:
            raise ValueError("tick_rate_hz must be finite and positive")

    @property
    def resolution(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def maximum_pixel_count(self) -> int:
        return self.height * self.width

    @property
    def descriptor(self) -> SensorProfileDescriptor:
        return SensorProfileDescriptor(
            self.profile_id,
            SensorKind.DEPTH,
            model={
                "sensor_model": "Intel RealSense D455",
                "resolution_height_width": self.resolution,
                "target_point_count": self.target_point_count,
                "maximum_pixel_count": self.maximum_pixel_count,
                "tick_rate_hz": self.tick_rate_hz,
                "asset_uri": self.asset_uri,
                "annotator": DEPTH_ANNOTATOR,
                "supporting_annotator": DEPTH_DISTANCE_ANNOTATOR,
            },
            calibration_provenance={
                "runtime": "Isaac Sim 6.0.1",
                "extension": f"isaacsim.sensors.experimental.rtx {DEPTH_EXTENSION_VERSION}",
                "asset_registry": "rtx_camera_configs.py",
                "native_xyz_metres_per_unit": (0.01, 0.01, 0.0001),
                "native_to_usd_camera_axes": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
                "application_axial_error_configurable": True,
            },
        )


_PROFILE_ROWS = (
    ("10k", 10_000, 72, 128),
    ("20k", 20_000, 108, 192),
    ("30k", 30_000, 130, 231),
    ("40k", 40_000, 150, 267),
    ("50k", 50_000, 168, 298),
    ("60k", 60_000, 184, 326),
    ("70k", 70_000, 199, 352),
    ("80k", 80_000, 212, 377),
    ("90k", 90_000, 225, 400),
    ("100k", 100_000, 237, 421),
)

DEPTH_PROFILES = tuple(
    DepthProfile(f"d455-16x9-{key}", f"D455 ~{key}", target, height, width)
    for key, target, height, width in _PROFILE_ROWS
)


def depth_profile(profile_id: str) -> DepthProfile:
    for profile in DEPTH_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(f"unknown depth profile: {profile_id}")


@dataclass(frozen=True)
class DepthCapability:
    available: bool
    reason: str
    extension_version: str = DEPTH_EXTENSION_VERSION
    asset_uri: str = D455_ASSET_URI


def probe_depth_capability() -> DepthCapability:
    module = "isaacsim.sensors.experimental.rtx"
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None:
        return DepthCapability(False, f"missing Python API: {module}")
    return DepthCapability(True, "SingleViewDepthCameraSensor API is importable")


class DepthBackend(Protocol):
    async def start(self) -> None: ...
    def read_positions(self) -> Any | None: ...
    def intrinsics(self) -> DepthIntrinsics: ...
    def optical_pose_world_m(self) -> Pose: ...
    def configure_diagnostics(self, diagnostics: DepthSensorDiagnostics) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def quiesce_native_updates(self) -> None: ...
    def resume_native_updates(self) -> None: ...
    def close(self) -> None: ...


DepthBackendFactory = Callable[[DepthProfile, str], DepthBackend]


@dataclass(frozen=True)
class DepthRuntimeTelemetry:
    backend_creations: int = 0
    backend_closes: int = 0
    poll_attempts: int = 0
    no_data_payloads: int = 0
    malformed_payloads: int = 0
    complete_payloads: int = 0
    published_frames: int = 0
    cadence_skips: int = 0
    pose_discarded_payloads: int = 0
    last_returned_count: int = 0
    last_valid_count: int = 0
    last_intrinsics: DepthIntrinsics | None = None
    last_error: str | None = None


class DepthSensorRuntime:
    """Profile-driven orchestration over the common generation-safe provider."""

    def __init__(
        self,
        *,
        stage_metadata: StageMetadata = StageMetadata(),
        dispatcher: ScanDispatcher | None = None,
        backend_factory: DepthBackendFactory | None = None,
        next_update: Callable[[], Awaitable[None]] | None = None,
        generation_seed: int = 0,
        diagnostics: DepthSensorDiagnostics = DepthSensorDiagnostics(),
    ) -> None:
        self._profiles = {profile.profile_id: profile for profile in DEPTH_PROFILES}
        self.provider = SensorProviderStateMachine(
            adapter_name="isaacsim-6.0.1-single-view-depth",
            profiles=tuple(profile.descriptor for profile in DEPTH_PROFILES),
            dispatcher=dispatcher,
            stage_metadata=stage_metadata,
            generation_seed=generation_seed,
        )
        self._backend_factory = backend_factory or IsaacD455Backend
        self._next_update = next_update or self._kit_next_update
        self._backend: DepthBackend | None = None
        self._runtime_path: str | None = None
        self._active_profile: DepthProfile | None = None
        self._last_publication_clock_s: float | None = None
        self._next_publication_clock_s: float | None = None
        self._native_frame_serial = 0
        self._discard_payloads = 0
        self._no_returns = False
        self._telemetry = DepthRuntimeTelemetry()
        self._operation_lock = asyncio.Lock()
        self._diagnostics = diagnostics

    @property
    def dispatcher(self) -> ScanDispatcher:
        return self.provider.dispatcher

    def list_profiles(self) -> Sequence[SensorProfileDescriptor]:
        return self.provider.list_profiles()

    def latest_status(self) -> SensorStatus:
        return self.provider.latest_status()

    def latest_frame(self) -> ScanFrame | None:
        return self.provider.latest_frame()

    def has_no_returns(self) -> bool:
        return self._no_returns

    def telemetry(self) -> DepthRuntimeTelemetry:
        return self._telemetry

    def provider_telemetry(self) -> ProviderTelemetry:
        return self.provider.telemetry()

    @property
    def diagnostics(self) -> DepthSensorDiagnostics:
        return self._diagnostics

    def configure_diagnostics(self, diagnostics: DepthSensorDiagnostics) -> None:
        """Apply diagnostic switches now and retain them across sensor recreation."""

        self._diagnostics = diagnostics
        if self._backend is not None:
            self._backend.configure_diagnostics(diagnostics)

    def _reset_cadence(self) -> None:
        """Drop the publication phase after a sensor lifecycle or clock reset."""

        self._last_publication_clock_s = None
        self._next_publication_clock_s = None

    async def start(
        self,
        profile_id: str,
        pose_world: Pose,
        runtime_path: str,
        *,
        maximum_warmup_updates: int = 300,
    ) -> SensorStatus:
        if maximum_warmup_updates <= 0:
            raise ValueError("maximum_warmup_updates must be positive")
        async with self._operation_lock:
            profile = self._profiles[profile_id]
            generation = self.provider.start(profile.descriptor, pose_world)
            self._no_returns = False
            self._runtime_path = runtime_path
            self._active_profile = profile
            self._reset_cadence()
            backend = self._backend_factory(profile, runtime_path)
            self._backend = backend
            self._telemetry = replace(
                self._telemetry,
                backend_creations=self._telemetry.backend_creations + 1,
                last_error=None,
            )
            try:
                await backend.start()
                backend.configure_diagnostics(self._diagnostics)
                if not self.provider.resource_created(generation):
                    raise RuntimeError("depth creation completed for a stale generation")
                frame = await self._warmup(generation, maximum_warmup_updates)
                if frame is None and not self._no_returns:
                    raise RuntimeError(
                        f"{profile_id} produced no non-empty native position frame in "
                        f"{maximum_warmup_updates} updates"
                    )
            except Exception as error:
                self._close_backend()
                self.provider.fail(generation, str(error))
                self._telemetry = replace(self._telemetry, last_error=str(error))
            return self.provider.latest_status()

    def poll(
        self,
        *,
        timestamp_s: float | None = None,
        operation: SensorOperationToken | None = None,
        enforce_cadence: bool = True,
    ) -> ScanFrame | None:
        backend = self._backend
        profile = self._active_profile
        status = self.provider.latest_status()
        if backend is None or profile is None:
            return None
        clock = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(clock) or clock < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self._last_publication_clock_s is None or clock < self._last_publication_clock_s:
            self._reset_cadence()
        if (enforce_cadence and self._next_publication_clock_s is not None
                and clock < self._next_publication_clock_s - 1.0e-6):
            self._telemetry = replace(
                self._telemetry,
                cadence_skips=self._telemetry.cadence_skips + 1,
            )
            return None
        self._telemetry = replace(
            self._telemetry,
            poll_attempts=self._telemetry.poll_attempts + 1,
        )
        positions = backend.read_positions()
        if positions is None:
            self._telemetry = replace(
                self._telemetry,
                no_data_payloads=self._telemetry.no_data_payloads + 1,
            )
            return None
        if self._discard_payloads > 0:
            self._discard_payloads -= 1
            self._telemetry = replace(
                self._telemetry,
                pose_discarded_payloads=self._telemetry.pose_discarded_payloads + 1,
            )
            return None
        array = np.asarray(positions)
        if array.size != profile.maximum_pixel_count * 3:
            self._telemetry = replace(
                self._telemetry,
                malformed_payloads=self._telemetry.malformed_payloads + 1,
                last_returned_count=int(array.size // 3),
            )
            return None
        array = array.reshape((-1, 3))
        intrinsics = backend.intrinsics()
        if (intrinsics.height, intrinsics.width) != profile.resolution:
            self._telemetry = replace(
                self._telemetry,
                malformed_payloads=self._telemetry.malformed_payloads + 1,
                last_error="runtime intrinsics resolution differs from active profile",
            )
            return None
        self._native_frame_serial += 1
        native = NativePointPayload(
            points_native=array,
            timestamp_s=clock,
            native_frame_id=f"depth-{status.generation}-{self._native_frame_serial}",
            native_units="D455 render-product XYZ: lateral centimetres, forward 0.1 millimetres",
            native_axes="render-product +X/+Y with camera looking along native +Z",
            metres_per_native_axis=(0.01, 0.01, 0.0001),
            axis_transform_to_sensor=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
            minimum_range_m=0.5,
            maximum_range_m=1000.0,
            axial_error_sigma_m=self._diagnostics.effective_axial_error_sigma_m,
            axial_error_model=self._diagnostics.axial_error_model,
            axial_error_slope_m_per_m=self._diagnostics.effective_distance_error_parameters[0],
            axial_error_intercept_m=self._diagnostics.effective_distance_error_parameters[1],
            axial_error_seed=(status.generation << 32) | self._native_frame_serial,
            details={
                "annotator": DEPTH_ANNOTATOR,
                "resolution_height_width": profile.resolution,
                "intrinsics": intrinsics.to_dict(),
                "optical_frame": "USD camera local frame (+X right, +Y up, -Z forward)",
                "standard_depth_fallback_used": False,
                "isaac_sim_disparity_noise_enabled": self._diagnostics.disparity_noise_enabled,
                "isaac_sim_disparity_noise_configured_mean_px": (
                    self._diagnostics.disparity_noise_mean_px
                ),
                "isaac_sim_disparity_noise_configured_sigma_px": (
                    self._diagnostics.disparity_noise_sigma_px
                ),
                "isaac_sim_disparity_noise_effective_mean_px": (
                    self._diagnostics.effective_disparity_noise_parameters_px[0]
                ),
                "isaac_sim_disparity_noise_effective_sigma_px": (
                    self._diagnostics.effective_disparity_noise_parameters_px[1]
                ),
                "isaac_sim_outlier_removal_enabled": self._diagnostics.outlier_removal_enabled,
                "application_axial_error_enabled": self._diagnostics.axial_error_enabled,
                "application_axial_error_configured_sigma_m": self._diagnostics.axial_error_sigma_m,
                "application_axial_error_configured_model": self._diagnostics.axial_error_model,
                "application_axial_error_configured_slope_m_per_m": self._diagnostics.axial_error_slope_m_per_m,
                "application_axial_error_configured_intercept_m": self._diagnostics.axial_error_intercept_m,
                "application_error_model": (
                    "zero-mean Gaussian displacement along the sensor ray"
                    if self._diagnostics.axial_error_enabled
                    else "disabled"
                ),
            },
        )
        empty_before = self.provider.telemetry().empty_payloads
        frame = self.provider.ingest_native(
            native,
            generation=status.generation,
            operation=operation,
            sensor_pose_world=backend.optical_pose_world_m(),
        )
        accepted_empty = self.provider.telemetry().empty_payloads != empty_before
        if frame is not None or accepted_empty:
            self._no_returns = accepted_empty
        self._telemetry = replace(
            self._telemetry,
            complete_payloads=self._telemetry.complete_payloads + 1,
            published_frames=self._telemetry.published_frames + (1 if frame is not None else 0),
            last_returned_count=profile.maximum_pixel_count,
            last_valid_count=frame.compacted_count if frame is not None else 0,
            last_intrinsics=intrinsics,
        )
        if frame is not None or accepted_empty:
            period_s = 1.0 / profile.tick_rate_hz
            deadline_s = self._next_publication_clock_s
            if not enforce_cadence or deadline_s is None:
                deadline_s = clock
            # Preserve the nominal phase instead of adding a period to each
            # late Kit update. Missed slots are dropped, never queued.
            elapsed_slots = max(1, math.floor((clock - deadline_s + 1.0e-6) / period_s) + 1)
            self._next_publication_clock_s = deadline_s + elapsed_slots * period_s
            self._last_publication_clock_s = clock
        return frame

    def disable(self) -> SensorStatus:
        backend = self._require_backend()
        backend.pause()
        self.provider.disable()
        self._reset_cadence()
        return self.provider.latest_status()

    def suspend_for_switch(self) -> None:
        """Quiesce native rendering while retaining this runtime for rollback."""

        backend = self._require_backend()
        if self.provider.latest_status().lifecycle is SensorLifecycle.RUNNING:
            backend.pause()
        backend.quiesce_native_updates()
        self._reset_cadence()

    def resume_after_failed_switch(self) -> None:
        """Resume only a runtime that was live before the attempted switch."""

        backend = self._require_backend()
        backend.resume_native_updates()
        if self.provider.latest_status().lifecycle is SensorLifecycle.RUNNING:
            self._discard_payloads = max(self._discard_payloads, 1)
            backend.resume()
        self._reset_cadence()

    def invalidate_pose(self) -> None:
        """Hide the previous pose's cloud until a post-pose render is ready."""

        self.provider.dispatcher.clear_generation(self.provider.latest_status().generation)
        self._no_returns = False
        self._discard_payloads = max(self._discard_payloads, 1)
        require_fresh = getattr(self._backend, "require_fresh_capture", None)
        if require_fresh is not None:
            require_fresh()
        self._reset_cadence()

    def invalidate_scene(self) -> None:
        """Fence the next publication against a completed post-edit render."""
        self._require_backend().require_fresh_capture()
        self._reset_cadence()

    async def enable(self, *, maximum_warmup_updates: int = 300) -> SensorStatus:
        async with self._operation_lock:
            backend = self._require_backend()
            # A paused RTX backend retains its render product.  Never publish the
            # payload cached immediately before the pause as the resumed frame.
            self._discard_payloads = max(self._discard_payloads, 1)
            backend.resume()
            self.provider.enable()
            self._reset_cadence()
            generation = self.provider.latest_status().generation
            if await self._warmup(generation, maximum_warmup_updates) is None and not self._no_returns:
                message = "DepthSensor enable produced no non-empty frame"
                self.provider.fail(generation, message)
                self._telemetry = replace(self._telemetry, last_error=message)
            return self.provider.latest_status()

    async def scan_once(self, *, maximum_warmup_updates: int = 300) -> ScanFrame | None:
        async with self._operation_lock:
            backend = self._require_backend()
            token = self.provider.scan_once()
            self._discard_payloads = max(self._discard_payloads, 1)
            backend.resume()
            self._reset_cadence()
            try:
                return await self._warmup(
                    token.generation,
                    maximum_warmup_updates,
                    operation=token,
                )
            finally:
                backend.pause()

    async def replace(
        self,
        profile_id: str,
        *,
        maximum_warmup_updates: int = 300,
    ) -> SensorStatus:
        async with self._operation_lock:
            profile = self._profiles[profile_id]
            was_disabled = self.provider.latest_status().lifecycle is SensorLifecycle.DISABLED
            generation = self.provider.replace(profile.descriptor)
            self._no_returns = False
            self._close_backend()
            await self._next_update()
            await self._next_update()
            if not self.provider.resource_stopped(generation):
                return self.provider.latest_status()
            runtime_path = self._runtime_path
            if runtime_path is None:
                raise RuntimeError("depth runtime path is unavailable")
            backend = self._backend_factory(profile, runtime_path)
            self._backend = backend
            self._active_profile = profile
            self._reset_cadence()
            self._telemetry = replace(
                self._telemetry,
                backend_creations=self._telemetry.backend_creations + 1,
                last_error=None,
            )
            try:
                await backend.start()
                backend.configure_diagnostics(self._diagnostics)
                self.provider.resource_created(generation)
                if await self._warmup(generation, maximum_warmup_updates) is None and not self._no_returns:
                    raise RuntimeError(f"{profile_id} produced no non-empty frame after recreate")
                if was_disabled:
                    backend.pause()
                    self.provider.disable()
            except Exception as error:
                self._close_backend()
                self.provider.fail(generation, str(error))
                self._telemetry = replace(self._telemetry, last_error=str(error))
            return self.provider.latest_status()

    def close(self) -> None:
        self._close_backend()
        self.provider.close()
        self._active_profile = None
        self._runtime_path = None
        self._reset_cadence()
        self._discard_payloads = 0

    async def _warmup(
        self,
        generation: int,
        maximum_updates: int,
        *,
        operation: SensorOperationToken | None = None,
    ) -> ScanFrame | None:
        for _ in range(maximum_updates):
            if generation != self.provider.latest_status().generation:
                return None
            frame = self.poll(operation=operation, enforce_cadence=False)
            if frame is not None or self._no_returns:
                return frame
            await self._next_update()
        return None

    def _close_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.close()
            finally:
                self._telemetry = replace(
                    self._telemetry,
                    backend_closes=self._telemetry.backend_closes + 1,
                )

    def _require_backend(self) -> DepthBackend:
        if self._backend is None:
            raise RuntimeError("depth backend is unavailable")
        return self._backend

    @staticmethod
    async def _kit_next_update() -> None:
        import omni.kit.app

        await omni.kit.app.get_app().next_update_async()


def _remove_owned_prim_spec(layer: Any, path: Any) -> bool:
    spec = layer.GetPrimAtPath(path)
    if spec is None:
        return False
    del spec.nameParent.nameChildren[spec.name]
    return True


class IsaacD455Backend:
    """Kit-only D455 backend with anonymous-layer ownership and ordered teardown."""

    def __init__(self, profile: DepthProfile, runtime_path: str) -> None:
        self._profile = profile
        self._runtime_path = runtime_path
        self._stage: Any | None = None
        self._runtime_layer: Any | None = None
        self._previous_edit_target: Any | None = None
        self._camera: Any | None = None
        self._camera_prim_path: str | None = None
        self._sensor: Any | None = None
        self._sensor_class: Any | None = None
        self._retired_render_products: set[str] = set()
        self._render_observer = None
        self._completed_render = -1
        self._fresh_render_count = 0
        self._fresh_render_required = False

    async def start(self) -> None:
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from isaacsim.sensors.experimental.rtx import RtxCamera, SingleViewDepthCameraSensor
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("a USD stage is required before starting DepthSensor")
        validate_d455_native_space(StageMetadata(
            str(UsdGeom.GetStageUpAxis(stage)), float(UsdGeom.GetStageMetersPerUnit(stage)),
        ))
        parent_path = Sdf.Path(self._runtime_path).GetParentPath()
        if not stage.GetPrimAtPath(parent_path):
            raise RuntimeError(f"DepthSensor owner path is missing: {parent_path}")
        if stage.GetPrimAtPath(self._runtime_path):
            raise RuntimeError(f"DepthSensor runtime path already exists: {self._runtime_path}")
        runtime_layer = Sdf.Layer.CreateAnonymous("FindSurfaceDepthRuntime.usda")
        self._stage = stage
        self._runtime_layer = runtime_layer
        self._previous_edit_target = stage.GetEditTarget()
        session_layer = stage.GetSessionLayer()
        session_layer.subLayerPaths.append(runtime_layer.identifier)
        with Usd.EditContext(stage, runtime_layer):
            self._camera = RtxCamera.create(
                self._runtime_path,
                config=self._profile.asset_uri,
                tick_rate=self._profile.tick_rate_hz,
                reset_xform_op_properties=False,
            )
            root_prim = stage.GetPrimAtPath(self._runtime_path)
            root_prim.SetCustomDataByKey("findSurfaceDepthRuntimeOwned", True)
            # Cancel the D455 asset camera mount so this child denotes the optical pose.
            UsdGeom.Xformable(root_prim).AddOrientOp().Set(
                Gf.Quatf(0.5, Gf.Vec3f(-0.5, 0.5, 0.5))
            )
            # A profile switch runs on an already-playing timeline. Disable the
            # referenced asset's rigid bodies before yielding even one update;
            # disabling them after warmup preserves any gravity-induced drift.
            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()
        self._camera_prim_path = str(self._camera.paths[0])
        self.native_output_contract()
        self._validate_optical_mount()
        self._sensor_class = SingleViewDepthCameraSensor
        self._create_sensor()
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()

    def _validate_optical_mount(self) -> None:
        """Reject a moving or misaligned sensor before publishing its first scan."""
        from pxr import Usd, UsdGeom, UsdPhysics

        root = self._stage.GetPrimAtPath(self._runtime_path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                raise RuntimeError(f"DepthSensor asset still has an enabled rigid body: {prim.GetPath()}")
        expected = UsdGeom.Xformable(root.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        actual = UsdGeom.Xformable(self._camera_prim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if not np.allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=1.0e-6):
            raise RuntimeError("DepthSensor optical mount does not match the requested sensor pose")

    def read_positions(self) -> Any | None:
        if self._sensor is None:
            return None
        if self._fresh_render_required:
            return None
        self.native_output_contract()
        # Keep the standard depth render variable live, as in NVIDIA's D455
        # smoke path. Its payload is deliberately not used as a point fallback.
        self._sensor.get_data(DEPTH_RENDER_TRIGGER_ANNOTATOR)
        payload, _info = self._sensor.get_data(DEPTH_ANNOTATOR)
        if payload is None:
            return None
        return np.asarray(payload.numpy()).reshape((-1, 3))

    def native_output_contract(self) -> dict:
        from pxr import UsdGeom

        metadata = StageMetadata(
            str(UsdGeom.GetStageUpAxis(self._stage)),
            float(UsdGeom.GetStageMetersPerUnit(self._stage)),
        )
        clipping = tuple(float(value) for value in
                         UsdGeom.Camera(self._camera_prim()).GetClippingRangeAttr().Get())
        validate_d455_native_space(metadata, clipping)
        return {"stage_up_axis": metadata.up_axis,
                "stage_metres_per_unit": metadata.meters_per_unit,
                "camera_clipping_range_stage_units": clipping}

    def intrinsics(self) -> DepthIntrinsics:
        from pxr import UsdGeom

        camera = UsdGeom.Camera(self._camera_prim())
        focal = float(camera.GetFocalLengthAttr().Get())
        horizontal = float(camera.GetHorizontalApertureAttr().Get())
        vertical = float(camera.GetVerticalApertureAttr().Get())
        return DepthIntrinsics(
            self._profile.width,
            self._profile.height,
            focal * self._profile.width / horizontal,
            focal * self._profile.height / vertical,
            self._profile.width * 0.5,
            self._profile.height * 0.5,
        )

    def optical_pose_world_m(self) -> Pose:
        from pxr import Usd, UsdGeom

        matrix = UsdGeom.Xformable(self._camera_prim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        stage = self._stage
        if stage is None:
            raise RuntimeError("DepthSensor stage is unavailable")
        metadata = StageMetadata(
            str(UsdGeom.GetStageUpAxis(stage)),
            float(UsdGeom.GetStageMetersPerUnit(stage)),
        )
        return StageSpace(metadata).to_world_pose(
            StagePose(
                tuple(float(translation[index]) for index in range(3)),
                tuple(float(imaginary[index]) for index in range(3)) + (float(quaternion.GetReal()),),
            )
        )

    def configure_diagnostics(self, diagnostics: DepthSensorDiagnostics) -> None:
        """Change only renderer-owned depth post-processing attributes."""

        sensor = self._sensor
        if sensor is None:
            raise RuntimeError("DepthSensor diagnostics require an active render product")
        sensor.set_enabled_outlier_removal(diagnostics.outlier_removal_enabled)
        noise = diagnostics.effective_disparity_noise_parameters_px
        sensor.set_sensor_noise_parameters(noise_mean=noise[0], noise_sigma=noise[1])

    def pause(self) -> None:
        sensor = self._sensor
        if sensor is not None:
            sensor.set_enabled_post_processing(False)

    def resume(self) -> None:
        if self._camera is None:
            raise RuntimeError("cannot resume a closed DepthSensor")
        if self._sensor is None:
            self._create_sensor()
        else:
            self._sensor.set_enabled_post_processing(True)
        self.require_fresh_capture()

    def require_fresh_capture(self) -> None:
        """Ignore an in-flight render and wait for the next completed render.

        Count distinct renderer frames, not get_data calls or Kit updates.
        """
        if self._render_observer is None:
            import omni.hydratexture
            from carb.eventdispatcher import get_eventdispatcher
            texture = self._sensor._hydra_texture
            texture = getattr(texture, 'hydra_texture', texture)

            def completed(event):
                number = texture.get_frame_info(event['result_handle']).get('frame_number')
                self._record_completed_render(number)

            self._render_observer = get_eventdispatcher().observe_event(
                observer_name='FindSurface depth freshness',
                event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                on_event=completed, filter=texture.get_event_key(),
            )
        self._fresh_render_count = 0
        self._fresh_render_required = True

    def _record_completed_render(self, number) -> None:
        if number is None or number <= self._completed_render:
            return
        self._completed_render = number
        if self._fresh_render_required:
            self._fresh_render_count += 1
            if self._fresh_render_count >= 2:
                self._fresh_render_required = False

    def quiesce_native_updates(self) -> None:
        self._set_native_updates_enabled(False)

    def resume_native_updates(self) -> None:
        self._set_native_updates_enabled(True)

    def _set_native_updates_enabled(self, enabled: bool) -> None:
        sensor = self._sensor
        hydra_texture = None if sensor is None else getattr(sensor, "_hydra_texture", None)
        if hydra_texture is not None:
            native_texture = getattr(hydra_texture, "hydra_texture", hydra_texture)
            native_texture.set_updates_enabled(enabled)

    def close(self) -> None:
        from pxr import Sdf

        self._render_observer = None
        sensor, self._sensor = self._sensor, None
        if sensor is not None:
            self._invalidate_sensor(sensor)
        self._camera = None
        self._camera_prim_path = None
        stage, self._stage = self._stage, None
        runtime_layer, self._runtime_layer = self._runtime_layer, None
        if stage is None or runtime_layer is None:
            return
        self._remove_retired_render_product_targets(stage)
        self._retired_render_products.clear()
        session_layer = stage.GetSessionLayer()
        if self._previous_edit_target is not None:
            stage.SetEditTarget(self._previous_edit_target)
        self._previous_edit_target = None
        paths = list(session_layer.subLayerPaths)
        if runtime_layer.identifier in paths:
            paths.remove(runtime_layer.identifier)
            session_layer.subLayerPaths = paths
        runtime_path = Sdf.Path(self._runtime_path)
        _remove_owned_prim_spec(session_layer, runtime_path)
        _remove_owned_prim_spec(stage.GetRootLayer(), runtime_path)

    def _camera_prim(self) -> Any:
        if self._stage is None or self._camera_prim_path is None:
            raise RuntimeError("DepthSensor camera is unavailable")
        prim = self._stage.GetPrimAtPath(self._camera_prim_path)
        if not prim:
            raise RuntimeError("DepthSensor camera prim is invalid")
        return prim

    def _create_sensor(self) -> None:
        if self._sensor_class is None or self._camera is None:
            raise RuntimeError("DepthSensor API is unavailable")
        if self._stage is None or self._runtime_layer is None:
            raise RuntimeError("DepthSensor owned layer is unavailable")
        # Let Replicator author its render product in the stage's current edit
        # target. Forcing a different target here prevents the host annotator
        # pipeline from resolving the product in Isaac Sim 6.0.1.
        self._sensor = self._sensor_class(
            self._camera,
            resolution=self._profile.resolution,
            annotators=[DEPTH_DISTANCE_ANNOTATOR, DEPTH_ANNOTATOR, DEPTH_RENDER_TRIGGER_ANNOTATOR],
        )
        self._sensor.set_enabled_post_processing(True)

    def _invalidate_sensor(self, sensor: Any) -> None:
        self._retired_render_products.add(str(sensor.render_product.GetPath()))
        sensor._invalidate_sensor()
        if self._stage is not None:
            self._remove_retired_render_product_targets(self._stage)

    def _remove_retired_render_product_targets(self, stage: Any) -> None:
        if not self._retired_render_products:
            return
        from pxr import Sdf, Usd

        retired = {Sdf.Path(path) for path in self._retired_render_products}
        for prim in stage.Traverse():
            for relationship in prim.GetRelationships():
                targets = list(relationship.GetTargets())
                filtered = [target for target in targets if target not in retired]
                if filtered != targets:
                    with Usd.EditContext(stage, stage.GetRootLayer()):
                        relationship.SetTargets(filtered)
        for target_layer in (stage.GetSessionLayer(), stage.GetRootLayer()):
            for path in retired:
                _remove_owned_prim_spec(target_layer, path)

        # Replicator may author inert ``over`` ancestors in the user root
        # while targeting the session-owned sensor prim. Remove only empty
        # ancestors of this backend's exact runtime path.
        root_layer = stage.GetRootLayer()
        ancestor = Sdf.Path(self._runtime_path).GetParentPath()
        while ancestor != Sdf.Path.absoluteRootPath:
            spec = root_layer.GetPrimAtPath(ancestor)
            if spec is None or spec.nameChildren or spec.properties:
                break
            parent = ancestor.GetParentPath()
            _remove_owned_prim_spec(root_layer, ancestor)
            ancestor = parent


__all__ = [
    "DEFAULT_DEPTH_PROFILE_ID",
    "DEPTH_ANNOTATOR",
    "DEPTH_DISTANCE_ANNOTATOR",
    "DEPTH_RENDER_TRIGGER_ANNOTATOR",
    "DEPTH_PROFILES",
    "D455_ASSET_URI",
    "DepthCapability",
    "DepthIntrinsics",
    "DepthProfile",
    "DepthRuntimeTelemetry",
    "DepthSensorRuntime",
    "IsaacD455Backend",
    "depth_profile",
    "probe_depth_capability",
]
