"""Atomic DepthSensor/LiDAR switching, scanner pose, and object-rescan control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import Enum
from typing import Awaitable, Callable, Protocol, Sequence

from .depth_sensor import DEPTH_PROFILES, DepthIntrinsics, DepthSensorRuntime
from .lidar_sensor import SCANNER_LIDAR_PROFILES, RtxLidarRuntime
from .domain import (
    FrameIdentity,
    Pose,
    ScanFrame,
    SensorKind,
    SensorLifecycle,
    SensorProfileDescriptor,
    StageMetadata,
)
from .scan import ScanDispatcher
from .scene import ObjectChangeEvent, SceneObject, SENSOR_RIG_PATH, StageSpace
from .sensor_diagnostics import DepthSensorDiagnostics


class SensorControlPhase(str, Enum):
    IDLE = "IDLE"
    CREATING = "CREATING"
    WARMING = "WARMING"
    STOPPING = "STOPPING"
    RUNNING = "RUNNING"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class SensorSelection:
    """Durable command payload; intentionally contains no point-cloud data."""

    sensor_kind: SensorKind
    profile_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_kind", SensorKind(self.sensor_kind))
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")


@dataclass(frozen=True)
class ScannerPoseState:
    camera_pose_world: Pose = Pose()
    sensor_pose_world: Pose = Pose()
    follows_camera: bool = True
    revision: int = 0


@dataclass(frozen=True)
class SensorControlSnapshot:
    phase: SensorControlPhase = SensorControlPhase.IDLE
    active_generation: int = 0
    active: SensorSelection | None = None
    pending: SensorSelection | None = None
    pose: ScannerPoseState = ScannerPoseState()
    latest_frame_identity: FrameIdentity | None = None
    no_returns: bool = False
    error: str | None = None
    switch_attempts: int = 0
    switch_completions: int = 0
    rollbacks: int = 0
    object_events: int = 0
    object_rescans: int = 0


class SensorRuntime(Protocol):
    provider: object
    dispatcher: ScanDispatcher

    def list_profiles(self) -> Sequence[SensorProfileDescriptor]: ...
    async def start(self, profile_id: str, pose_world: Pose, runtime_path: str, *, maximum_warmup_updates: int): ...
    def latest_status(self): ...
    def latest_frame(self) -> ScanFrame | None: ...
    def has_no_returns(self) -> bool: ...
    def poll(self) -> ScanFrame | None: ...
    def disable(self): ...
    def suspend_for_switch(self) -> None: ...
    def resume_after_failed_switch(self) -> None: ...
    async def enable(self, *, maximum_warmup_updates: int): ...
    async def scan_once(self, *, maximum_warmup_updates: int) -> ScanFrame | None: ...
    def close(self) -> None: ...


class RuntimeSlot(Protocol):
    runtime_path: str

    def set_pose(self, pose_world: Pose) -> None: ...
    def close(self) -> None: ...


RuntimeFactory = Callable[[SensorKind, int], SensorRuntime]
SlotFactory = Callable[[int, Pose], RuntimeSlot]
SelectionClear = Callable[[int], None]
UpdateWait = Callable[[], Awaitable[None]]


async def _default_update_wait() -> None:
    """Yield one Kit update, with a host-test fallback."""

    try:
        import omni.kit.app
    except ImportError:
        await asyncio.sleep(0)
        return
    try:
        app = omni.kit.app.get_app()
    except RuntimeError:
        # Isaac Sim's Python environment can import omni.kit.app even when a
        # host-only unit test has not acquired IApp.
        await asyncio.sleep(0)
        return
    await app.next_update_async()


class MemoryRuntimeSlot:
    """Framework-free slot used by component tests and non-USD adapters."""

    def __init__(self, generation: int, pose_world: Pose, *, root_path: str = SENSOR_RIG_PATH) -> None:
        self.runtime_path = f"{root_path}/Candidate_{generation:06d}/Sensor"
        self.pose_world = pose_world
        self.closed = False

    def set_pose(self, pose_world: Pose) -> None:
        if self.closed:
            raise RuntimeError("closed runtime slot cannot move")
        self.pose_world = pose_world

    def close(self) -> None:
        self.closed = True


class IsaacSensorRuntimeSlot:
    """Anonymous-layer parent xform for a DepthSensor runtime."""

    def __init__(self, generation: int, pose_world: Pose, *, root_path: str = SENSOR_RIG_PATH) -> None:
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("a USD stage is required for a sensor runtime slot")
        if not stage.GetPrimAtPath(root_path):
            raise RuntimeError(f"sensor rig path is missing: {root_path}")
        self._stage = stage
        self._layer = Sdf.Layer.CreateAnonymous("FindSurfaceSensorSlot.usda")
        self._previous_edit_target = stage.GetEditTarget()
        stage.GetSessionLayer().subLayerPaths.append(self._layer.identifier)
        self._slot_path = f"{root_path}/Candidate_{generation:06d}"
        self.runtime_path = f"{self._slot_path}/Sensor"
        self._closed = False
        with Usd.EditContext(stage, self._layer):
            slot = UsdGeom.Xform.Define(stage, self._slot_path)
            slot.GetPrim().SetCustomDataByKey("findSurfaceSensorSlotOwned", True)
        self.set_pose(pose_world)

    def set_pose(self, pose_world: Pose) -> None:
        from pxr import Gf, Usd, UsdGeom

        if self._closed:
            raise RuntimeError("closed runtime slot cannot move")
        stage = self._stage
        metadata = StageMetadata(
            str(UsdGeom.GetStageUpAxis(stage)), float(UsdGeom.GetStageMetersPerUnit(stage))
        )
        pose = StageSpace(metadata).to_stage_pose(pose_world)
        with Usd.EditContext(stage, self._layer):
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(self._slot_path))
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pose.position_stage_units))
            x, y, z, w = pose.orientation_xyzw
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))

    def close(self) -> None:
        from pxr import Sdf

        if self._closed:
            return
        self._closed = True
        stage, layer = self._stage, self._layer
        if self._previous_edit_target is not None:
            stage.SetEditTarget(self._previous_edit_target)
        self._previous_edit_target = None
        paths = list(stage.GetSessionLayer().subLayerPaths)
        if layer.identifier in paths:
            paths.remove(layer.identifier)
            stage.GetSessionLayer().subLayerPaths = paths
        for target_layer in (stage.GetSessionLayer(), stage.GetRootLayer()):
            spec = target_layer.GetPrimAtPath(Sdf.Path(self._slot_path))
            if spec is not None:
                del spec.nameParent.nameChildren[spec.name]


def _default_runtime_factory(kind: SensorKind, generation_seed: int) -> SensorRuntime:
    if SensorKind(kind) is SensorKind.RTX_LIDAR:
        return RtxLidarRuntime(generation_seed=generation_seed, stage_metadata=_current_stage_metadata())
    return DepthSensorRuntime(
        generation_seed=generation_seed, stage_metadata=_current_stage_metadata(),
    )


def _current_stage_metadata() -> StageMetadata:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("a USD stage is required for sensor frame provenance")
    return StageMetadata(
        str(UsdGeom.GetStageUpAxis(stage)), float(UsdGeom.GetStageMetersPerUnit(stage)),
    )


def _default_slot_factory(generation: int, pose_world: Pose) -> RuntimeSlot:
    return IsaacSensorRuntimeSlot(generation, pose_world)


class SensorCoordinator:
    """Run exactly one native sensor while preserving atomic frame publication."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory = _default_runtime_factory,
        slot_factory: SlotFactory = _default_slot_factory,
        dispatcher: ScanDispatcher | None = None,
        selection_clear: SelectionClear | None = None,
        camera_pose_world: Pose = Pose(),
        follows_camera: bool = True,
        depth_diagnostics: DepthSensorDiagnostics = DepthSensorDiagnostics(),
        update_wait: UpdateWait = _default_update_wait,
        object_values: Callable[[], Sequence[SceneObject]] | None = None,
    ) -> None:
        self.dispatcher = dispatcher or ScanDispatcher()
        self._runtime_factory = runtime_factory
        self._slot_factory = slot_factory
        self._selection_clear = selection_clear
        self._runtime: SensorRuntime | None = None
        self._slot: RuntimeSlot | None = None
        self._generation_cursor = 0
        self._operation_lock = asyncio.Lock()
        self._object_refresh_lock = asyncio.Lock()
        self._object_values = object_values
        self._notified_object_values: dict = {}
        self._completed_object_values: tuple[SceneObject, ...] | None = None
        self._pending_object_sequence: int | None = None
        self._pending_selection_revision: int | None = None
        self._pending_selection_cleared = False
        self._phase_history: list[SensorControlPhase] = [SensorControlPhase.IDLE]
        self._depth_diagnostics = depth_diagnostics
        self._update_wait = update_wait
        pose = ScannerPoseState(camera_pose_world, camera_pose_world, bool(follows_camera), 0)
        self._snapshot = SensorControlSnapshot(pose=pose)

    @property
    def snapshot(self) -> SensorControlSnapshot:
        return self._snapshot

    @property
    def phase_history(self) -> tuple[SensorControlPhase, ...]:
        return tuple(self._phase_history)

    @property
    def active_runtime(self) -> SensorRuntime | None:
        return self._runtime

    def list_profiles(self) -> tuple[SensorProfileDescriptor, ...]:
        return tuple(profile.descriptor for profile in (*DEPTH_PROFILES, *SCANNER_LIDAR_PROFILES))

    def latest_frame(self) -> ScanFrame | None:
        publication = self.dispatcher.snapshot()
        return None if publication is None else publication.frame

    @property
    def has_pending_object_change(self) -> bool:
        return self._pending_object_sequence is not None

    def active_depth_intrinsics(self) -> DepthIntrinsics | None:
        """Return the calibration attached to the active D455 render product."""

        runtime = self._runtime
        if not isinstance(runtime, DepthSensorRuntime):
            return None
        return runtime.telemetry().last_intrinsics

    @property
    def depth_diagnostics(self) -> DepthSensorDiagnostics:
        return self._depth_diagnostics

    def configure_depth_diagnostics(self, diagnostics: DepthSensorDiagnostics) -> None:
        """Apply shared axial error and DepthSensor-only native noise controls."""

        self._depth_diagnostics = diagnostics
        runtime = self._runtime
        if isinstance(runtime, (DepthSensorRuntime, RtxLidarRuntime)):
            runtime.configure_diagnostics(diagnostics)

    async def start(
        self,
        selection: SensorSelection,
        *,
        disabled: bool = False,
        maximum_warmup_updates: int = 600,
    ) -> SensorControlSnapshot:
        if self._runtime is not None:
            raise RuntimeError("sensor coordinator is already started")
        return await self._switch(selection, disabled=disabled, maximum_warmup_updates=maximum_warmup_updates)

    async def switch(
        self,
        selection: SensorSelection,
        *,
        maximum_warmup_updates: int = 600,
    ) -> SensorControlSnapshot:
        if self._runtime is None:
            return await self.start(selection, maximum_warmup_updates=maximum_warmup_updates)
        disabled = self._active_lifecycle() is SensorLifecycle.DISABLED
        return await self._switch(selection, disabled=disabled, maximum_warmup_updates=maximum_warmup_updates)

    async def _switch(
        self,
        selection: SensorSelection,
        *,
        disabled: bool,
        maximum_warmup_updates: int,
    ) -> SensorControlSnapshot:
        if maximum_warmup_updates <= 0:
            raise ValueError("maximum_warmup_updates must be positive")
        async with self._operation_lock:
            self._require_open()
            old_runtime, old_slot = self._runtime, self._slot
            old_selection = self._snapshot.active
            self._generation_cursor += 1
            target_generation = self._generation_cursor
            attempts = self._snapshot.switch_attempts + 1
            self._set_phase(SensorControlPhase.CREATING, pending=selection, error=None, switch_attempts=attempts)
            candidate: SensorRuntime | None = None
            candidate_slot: RuntimeSlot | None = None
            try:
                candidate_slot = self._slot_factory(target_generation, self._snapshot.pose.sensor_pose_world)
                candidate = self._runtime_factory(selection.sensor_kind, target_generation - 1)
                self._validate_candidate(candidate, selection)
                if isinstance(candidate, (DepthSensorRuntime, RtxLidarRuntime)):
                    candidate.configure_diagnostics(self._depth_diagnostics)
            except Exception as error:
                teardown_errors = self._close_resources(candidate, candidate_slot, owner="candidate")
                message = self._error_with_teardown(error, teardown_errors)
                rollback = 1 if old_runtime is not None else 0
                phase = self._phase_for_runtime(old_runtime) if old_runtime is not None else SensorControlPhase.ERROR
                self._set_phase(
                    phase,
                    pending=None,
                    error=message,
                    rollbacks=self._snapshot.rollbacks + rollback,
                )
                return self._snapshot

            if old_runtime is not None:
                self._set_phase(SensorControlPhase.STOPPING, pending=selection)
                try:
                    old_runtime.provider.isolate_dispatcher()
                    old_runtime.suspend_for_switch()
                except Exception as error:
                    teardown_errors = self._close_resources(candidate, candidate_slot, owner="candidate")
                    message = self._error_with_teardown(error, teardown_errors)
                    if old_selection is not None and old_selection.sensor_kind is SensorKind.RTX_LIDAR:
                        return await self._recreate_previous_lidar(
                            old_runtime, old_slot, old_selection, disabled=disabled,
                            switch_error=f"could not isolate previous sensor: {message}",
                            maximum_warmup_updates=maximum_warmup_updates,
                        )
                    try:
                        old_runtime.resume_after_failed_switch()
                        old_runtime.provider.restore_dispatcher(self.dispatcher)
                    except Exception as rollback_error:
                        return self._fail_retained_restore(
                            old_runtime, old_slot, old_selection,
                            switch_error=f"could not isolate previous sensor: {message}",
                            rollback_error=rollback_error,
                        )
                    self._set_phase(
                        self._phase_for_runtime(old_runtime),
                        pending=None,
                        error=f"could not isolate previous sensor: {message}",
                        rollbacks=self._snapshot.rollbacks + 1,
                    )
                    return self._snapshot
                self._runtime, self._slot = None, None

            try:
                self._set_phase(SensorControlPhase.WARMING, pending=selection)
                status = await candidate.start(
                    selection.profile_id,
                    self._snapshot.pose.sensor_pose_world,
                    candidate_slot.runtime_path,
                    maximum_warmup_updates=maximum_warmup_updates,
                )
                if status.lifecycle is not SensorLifecycle.RUNNING:
                    raise RuntimeError(status.error or "candidate sensor did not reach RUNNING")
                frame = candidate.latest_frame()
                no_returns = candidate.has_no_returns()
                if frame is None and not no_returns:
                    raise RuntimeError("candidate sensor produced no current valid frame")
                if frame is not None and frame.generation != target_generation:
                    raise RuntimeError("candidate sensor produced a stale frame")
                if disabled:
                    candidate.disable()
                candidate.provider.adopt_dispatcher(self.dispatcher)
            except Exception as error:
                teardown_errors = self._close_resources(candidate, candidate_slot, owner="candidate")
                message = self._error_with_teardown(error, teardown_errors)
                if old_runtime is not None and old_selection is not None:
                    await self._settle_native_teardown()
                    return await self._restore_previous(
                        old_runtime,
                        old_slot,
                        old_selection,
                        disabled=disabled,
                        switch_error=message,
                        maximum_warmup_updates=maximum_warmup_updates,
                    )
                self._set_phase(
                    SensorControlPhase.ERROR,
                    active=None,
                    pending=None,
                    error=message,
                )
                return self._snapshot

            teardown_errors = self._close_resources(old_runtime, old_slot, owner="previous sensor")
            if teardown_errors:
                candidate_errors = self._close_resources(candidate, candidate_slot, owner="candidate")
                self._set_phase(
                    SensorControlPhase.ERROR,
                    active=None,
                    pending=None,
                    error="; ".join((*teardown_errors, *candidate_errors)),
                    rollbacks=self._snapshot.rollbacks + 1,
                )
                return self._snapshot
            self._runtime, self._slot = candidate, candidate_slot
            phase = SensorControlPhase.DISABLED if disabled else SensorControlPhase.RUNNING
            self._set_phase(
                phase,
                active_generation=target_generation,
                active=selection,
                pending=None,
                latest_frame_identity=None if frame is None else frame.identity,
                no_returns=no_returns,
                error=None,
                switch_completions=self._snapshot.switch_completions + 1,
            )
            return self._snapshot

    async def _restore_previous(
        self,
        runtime: SensorRuntime,
        slot: RuntimeSlot | None,
        selection: SensorSelection,
        *,
        disabled: bool,
        switch_error: str,
        maximum_warmup_updates: int,
    ) -> SensorControlSnapshot:
        """Resume the retained DepthSensor and restore its public dispatcher."""

        if selection.sensor_kind is SensorKind.RTX_LIDAR:
            return await self._recreate_previous_lidar(
                runtime, slot, selection, disabled=disabled, switch_error=switch_error,
                maximum_warmup_updates=maximum_warmup_updates,
            )
        try:
            runtime.resume_after_failed_switch()
            runtime.provider.restore_dispatcher(self.dispatcher)
        except Exception as rollback_error:
            return self._fail_retained_restore(
                runtime, slot, selection,
                switch_error=switch_error,
                rollback_error=rollback_error,
            )

        self._runtime, self._slot = runtime, slot
        frame = runtime.latest_frame()
        no_returns = runtime.has_no_returns()
        phase = SensorControlPhase.DISABLED if disabled else SensorControlPhase.RUNNING
        self._set_phase(
            phase,
            active_generation=runtime.latest_status().generation,
            active=selection,
            pending=None,
            latest_frame_identity=None if frame is None else frame.identity,
            no_returns=no_returns,
            error=switch_error,
            rollbacks=self._snapshot.rollbacks + 1,
        )
        return self._snapshot

    async def _recreate_previous_lidar(
        self,
        runtime: SensorRuntime,
        slot: RuntimeSlot | None,
        selection: SensorSelection,
        *,
        disabled: bool,
        switch_error: str,
        maximum_warmup_updates: int,
    ) -> SensorControlSnapshot:
        """Rebuild a quiesced LiDAR because disabling its Hydra updates retires it."""

        self._runtime, self._slot = None, None
        teardown_errors = self._close_resources(runtime, slot, owner="retired previous LiDAR")
        if teardown_errors:
            self.dispatcher.clear()
            self._set_phase(
                SensorControlPhase.ERROR,
                active=None,
                pending=None,
                latest_frame_identity=None,
                no_returns=False,
                error="; ".join((switch_error, *teardown_errors)),
                rollbacks=self._snapshot.rollbacks + 1,
            )
            return self._snapshot

        await self._settle_native_teardown()
        self._generation_cursor += 1
        rollback_generation = self._generation_cursor
        replacement: SensorRuntime | None = None
        replacement_slot: RuntimeSlot | None = None
        try:
            self._set_phase(SensorControlPhase.CREATING, pending=selection)
            replacement_slot = self._slot_factory(
                rollback_generation,
                self._snapshot.pose.sensor_pose_world,
            )
            replacement = self._runtime_factory(
                SensorKind.RTX_LIDAR,
                rollback_generation - 1,
            )
            self._validate_candidate(replacement, selection)
            self._set_phase(SensorControlPhase.WARMING, pending=selection)
            status = await replacement.start(
                selection.profile_id,
                self._snapshot.pose.sensor_pose_world,
                replacement_slot.runtime_path,
                maximum_warmup_updates=maximum_warmup_updates,
            )
            if status.lifecycle is not SensorLifecycle.RUNNING:
                raise RuntimeError(status.error or "previous LiDAR recreation did not reach RUNNING")
            frame = replacement.latest_frame()
            no_returns = replacement.has_no_returns()
            if frame is None and not no_returns:
                raise RuntimeError("previous LiDAR recreation produced no current valid scan")
            if frame is not None and frame.generation != rollback_generation:
                raise RuntimeError("previous LiDAR recreation produced a stale frame")
            if disabled:
                replacement.disable()
            replacement.provider.adopt_dispatcher(self.dispatcher)
        except Exception as rollback_error:
            replacement_errors = self._close_resources(
                replacement,
                replacement_slot,
                owner="replacement previous LiDAR",
            )
            self.dispatcher.clear()
            message = self._error_with_teardown(rollback_error, replacement_errors)
            self._set_phase(
                SensorControlPhase.ERROR,
                active=None,
                pending=None,
                latest_frame_identity=None,
                no_returns=False,
                error=(
                    f"{switch_error}; restoring {selection.profile_id} with a fresh RTX backend "
                    f"failed: {message}"
                ),
                rollbacks=self._snapshot.rollbacks + 1,
            )
            return self._snapshot

        self._runtime, self._slot = replacement, replacement_slot
        phase = SensorControlPhase.DISABLED if disabled else SensorControlPhase.RUNNING
        self._set_phase(
            phase,
            active_generation=rollback_generation,
            active=selection,
            pending=None,
            latest_frame_identity=None if frame is None else frame.identity,
            no_returns=no_returns,
            error=(
                f"{switch_error}; restored {selection.profile_id} with a fresh RTX backend"
            ),
            rollbacks=self._snapshot.rollbacks + 1,
        )
        return self._snapshot

    def _fail_retained_restore(
        self,
        runtime: SensorRuntime,
        slot: RuntimeSlot | None,
        selection: SensorSelection | None,
        *,
        switch_error: str,
        rollback_error: Exception,
    ) -> SensorControlSnapshot:
        generation = runtime.latest_status().generation
        self._runtime, self._slot = None, None
        teardown_errors = self._close_resources(runtime, slot, owner="retained sensor")
        self.dispatcher.clear_generation(generation)
        rollback_message = self._error_with_teardown(rollback_error, teardown_errors)
        profile_id = "previous sensor" if selection is None else selection.profile_id
        self._set_phase(
            SensorControlPhase.ERROR,
            active=None,
            pending=None,
            latest_frame_identity=None,
            no_returns=False,
            error=f"{switch_error}; restoring {profile_id} failed: {rollback_message}",
            rollbacks=self._snapshot.rollbacks + 1,
        )
        return self._snapshot

    def poll(self) -> ScanFrame | None:
        runtime = self._require_runtime()
        frame = runtime.poll()
        latest = runtime.latest_frame()
        identity = None if latest is None else latest.identity
        no_returns = runtime.has_no_returns()
        if (
            identity != self._snapshot.latest_frame_identity
            or no_returns != self._snapshot.no_returns
        ):
            self._snapshot = replace(
                self._snapshot,
                latest_frame_identity=identity,
                no_returns=no_returns,
            )
        return frame

    def disable(self) -> SensorControlSnapshot:
        runtime = self._require_runtime()
        runtime.disable()
        self._set_phase(SensorControlPhase.DISABLED, error=None)
        return self._snapshot

    async def enable(self, *, maximum_warmup_updates: int = 600) -> SensorControlSnapshot:
        async with self._operation_lock:
            runtime = self._require_runtime()
            self._set_phase(SensorControlPhase.WARMING, error=None)
            status = await runtime.enable(maximum_warmup_updates=maximum_warmup_updates)
            if status.lifecycle is SensorLifecycle.RUNNING:
                frame = runtime.latest_frame()
                self._set_phase(
                    SensorControlPhase.RUNNING,
                    latest_frame_identity=None if frame is None else frame.identity,
                    no_returns=runtime.has_no_returns(),
                )
            else:
                self._set_phase(SensorControlPhase.ERROR, error=status.error or "sensor enable failed")
            return self._snapshot

    async def scan_once(self, *, maximum_warmup_updates: int = 600) -> ScanFrame | None:
        async with self._operation_lock:
            runtime = self._require_runtime()
            if self._active_lifecycle() is not SensorLifecycle.DISABLED:
                raise RuntimeError("scan_once requires a disabled sensor")
            frame = await runtime.scan_once(maximum_warmup_updates=maximum_warmup_updates)
            if frame is None:
                if runtime.has_no_returns():
                    self._set_phase(
                        SensorControlPhase.DISABLED,
                        latest_frame_identity=None,
                        no_returns=True,
                        error=None,
                    )
                    return None
                self._set_phase(SensorControlPhase.ERROR, error="one-shot scan produced no valid frame")
                return None
            self._set_phase(
                SensorControlPhase.DISABLED,
                latest_frame_identity=frame.identity,
                no_returns=False,
                error=None,
            )
            return frame

    def set_camera_pose(self, pose_world: Pose, *, invalidate_frame: bool = True) -> ScannerPoseState:
        self._require_open()
        pose = self._snapshot.pose
        if pose_world == pose.camera_pose_world:
            return pose
        sensor_pose = pose_world if pose.follows_camera else pose.sensor_pose_world
        updated = ScannerPoseState(pose_world, sensor_pose, pose.follows_camera, pose.revision + 1)
        self._snapshot = replace(self._snapshot, pose=updated)
        if pose.follows_camera and self._slot is not None:
            self._slot.set_pose(sensor_pose)
            if invalidate_frame:
                self._invalidate_runtime_pose()
        return updated

    def freeze_sensor(self) -> ScannerPoseState:
        self._require_open()
        pose = self._snapshot.pose
        if not pose.follows_camera:
            return pose
        updated = replace(pose, follows_camera=False, revision=pose.revision + 1)
        self._snapshot = replace(self._snapshot, pose=updated)
        return updated

    def follow_sensor(self) -> ScannerPoseState:
        self._require_open()
        pose = self._snapshot.pose
        if pose.follows_camera:
            return pose
        updated = ScannerPoseState(pose.camera_pose_world, pose.camera_pose_world, True, pose.revision + 1)
        self._snapshot = replace(self._snapshot, pose=updated)
        if self._slot is not None:
            self._slot.set_pose(updated.sensor_pose_world)
            self._invalidate_runtime_pose()
        return updated

    def notify_object_change(self, event: ObjectChangeEvent) -> bool:
        self._require_open()
        sequence = event.scan_intents[0].sequence
        if self._pending_object_sequence is not None and sequence <= self._pending_object_sequence:
            return False
        self._pending_object_sequence = sequence
        self._notified_object_values[event.after.geometry_type] = event.after
        self._pending_selection_revision = event.selection_revision
        self._pending_selection_cleared = False
        self._snapshot = replace(self._snapshot, object_events=self._snapshot.object_events + 1)
        return True

    async def flush_object_changes(self, *, maximum_warmup_updates: int = 600) -> ScanFrame | None:
        """Keep a single job slot until its input values equal current values."""
        async with self._object_refresh_lock:
            frame = None
            while self.has_pending_object_change:
                frame = await self._scan_object_values(maximum_warmup_updates=maximum_warmup_updates)
                if frame is None and not self._snapshot.no_returns:
                    break
            return frame

    def _current_object_values(self) -> tuple[SceneObject, ...]:
        values = self._object_values() if self._object_values is not None else self._notified_object_values.values()
        return tuple(sorted(values, key=lambda item: item.geometry_type.value))

    @property
    def completed_object_values(self) -> tuple[SceneObject, ...] | None:
        """Input values of the most recently completed object rescan request."""
        return self._completed_object_values

    async def _scan_object_values(self, *, maximum_warmup_updates: int) -> ScanFrame | None:
        sequence = self._pending_object_sequence
        if sequence is None:
            return None
        requested_values = self._current_object_values()
        revision = self._pending_selection_revision
        if (
            not self._pending_selection_cleared
            and self._selection_clear is not None
            and revision is not None
        ):
            self._selection_clear(revision)
            self._pending_selection_cleared = True
        frame = None
        if self._active_lifecycle() is SensorLifecycle.DISABLED:
            frame = await self.scan_once(maximum_warmup_updates=maximum_warmup_updates)
            if frame is None and not self._snapshot.no_returns:
                return None
            self._snapshot = replace(self._snapshot, object_rescans=self._snapshot.object_rescans + 1)
        else:
            # Live sensors stay live, but do not finish the trailing request
            # until a post-edit publication has arrived.
            runtime = self._require_runtime()
            invalidate = getattr(runtime, 'invalidate_scene', None)
            if invalidate is not None:
                invalidate()
            before = runtime.latest_frame()
            for _ in range(maximum_warmup_updates):
                await self._update_wait()
                runtime.poll()
                frame = runtime.latest_frame()
                if frame is not None and frame is not before:
                    break
            else:
                raise RuntimeError('No fresh sensor frame after final object edit')
            self._snapshot = replace(self._snapshot, object_rescans=self._snapshot.object_rescans + 1)
        self._completed_object_values = requested_values
        # Compare actual immutable inputs, not an event counter. This also
        # catches a current-value change that did not emit another scan event.
        # On mismatch the one slot immediately retries with the latest values.
        if tuple(item.sensor_signature for item in self._current_object_values()) == tuple(
            item.sensor_signature for item in requested_values
        ):
            self._pending_object_sequence = None
            self._pending_selection_revision = None
            self._pending_selection_cleared = False
        return frame

    def close(self) -> None:
        if self._snapshot.phase is SensorControlPhase.CLOSED:
            return
        self._set_phase(SensorControlPhase.STOPPING, pending=None)
        runtime, self._runtime = self._runtime, None
        slot, self._slot = self._slot, None
        teardown_errors = self._close_resources(runtime, slot, owner="sensor")
        self._set_phase(
            SensorControlPhase.CLOSED,
            active=None,
            pending=None,
            no_returns=False,
            error=None if not teardown_errors else "; ".join(teardown_errors),
        )

    def _active_lifecycle(self) -> SensorLifecycle:
        return self._require_runtime().latest_status().lifecycle

    def _invalidate_runtime_pose(self) -> None:
        runtime = self._runtime
        invalidate = None if runtime is None else getattr(runtime, "invalidate_pose", None)
        if invalidate is not None:
            invalidate()

    def _validate_candidate(self, runtime: SensorRuntime, selection: SensorSelection) -> None:
        if isinstance(runtime, (DepthSensorRuntime, RtxLidarRuntime)):
            runtime.configure_diagnostics(self._depth_diagnostics)
        profiles = {profile.profile_id: profile for profile in runtime.list_profiles()}
        profile = profiles.get(selection.profile_id)
        if profile is None or profile.sensor_kind is not selection.sensor_kind:
            raise ValueError(f"profile does not belong to {selection.sensor_kind.value}: {selection.profile_id}")

    async def _settle_native_teardown(self) -> None:
        # RTX render-product destruction is consumed on Kit updates.  Do not
        # create the replacement native sensor until those opinions are gone.
        await self._update_wait()
        await self._update_wait()

    @staticmethod
    def _phase_for_runtime(runtime: SensorRuntime) -> SensorControlPhase:
        return (
            SensorControlPhase.DISABLED
            if runtime.latest_status().lifecycle is SensorLifecycle.DISABLED
            else SensorControlPhase.RUNNING
        )

    @staticmethod
    def _error_with_teardown(error: Exception, teardown_errors: Sequence[str]) -> str:
        return "; ".join((str(error), *teardown_errors))

    @staticmethod
    def _close_resources(
        runtime: SensorRuntime | None,
        slot: RuntimeSlot | None,
        *,
        owner: str,
    ) -> list[str]:
        errors: list[str] = []
        for label, resource in (("runtime", runtime), ("slot", slot)):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as error:
                errors.append(f"{owner} {label} teardown failed: {error}")
        return errors

    def _require_runtime(self) -> SensorRuntime:
        self._require_open()
        if self._runtime is None:
            raise RuntimeError("sensor coordinator is not started")
        return self._runtime

    def _require_open(self) -> None:
        if self._snapshot.phase is SensorControlPhase.CLOSED:
            raise RuntimeError("sensor coordinator is closed")

    def _set_phase(self, phase: SensorControlPhase, **changes) -> None:
        self._phase_history.append(phase)
        self._snapshot = replace(self._snapshot, phase=phase, **changes)


class ScannerControlFacade:
    """UI-neutral Scanner property facade with separate modal presentation."""

    def __init__(self, coordinator: SensorCoordinator) -> None:
        self._coordinator = coordinator

    @property
    def snapshot(self) -> SensorControlSnapshot:
        return self._coordinator.snapshot

    @property
    def profiles(self) -> tuple[SensorProfileDescriptor, ...]:
        return self._coordinator.list_profiles()

    @property
    def can_scan_once(self) -> bool:
        return self.snapshot.phase is SensorControlPhase.DISABLED

    @property
    def busy(self) -> bool:
        return self.snapshot.phase in {
            SensorControlPhase.CREATING,
            SensorControlPhase.WARMING,
            SensorControlPhase.STOPPING,
        }

    async def select(self, sensor_kind: SensorKind, profile_id: str, *, maximum_warmup_updates: int = 600):
        return await self._coordinator.switch(
            SensorSelection(sensor_kind, profile_id),
            maximum_warmup_updates=maximum_warmup_updates,
        )


__all__ = [
    "IsaacSensorRuntimeSlot",
    "MemoryRuntimeSlot",
    "RuntimeSlot",
    "ScannerControlFacade",
    "ScannerPoseState",
    "SensorControlPhase",
    "SensorControlSnapshot",
    "SensorCoordinator",
    "SensorRuntime",
    "SensorSelection",
]
