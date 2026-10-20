"""Framework-free provider lifecycle for DepthSensor adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Sequence

from .domain import (
    UINT64_MAX,
    Pose,
    ScanFrame,
    SensorKind,
    SensorLifecycle,
    SensorProfileDescriptor,
    SensorStatus,
    StageMetadata,
)
from .scan import NativePointPayload, ScanDispatcher, normalize_native_payload


@dataclass(frozen=True)
class SensorOperationToken:
    generation: int
    serial: int


@dataclass(frozen=True)
class ProviderTelemetry:
    native_payloads: int = 0
    published_frames: int = 0
    rejected_payloads: int = 0
    empty_payloads: int = 0
    stale_callbacks: int = 0
    late_callbacks: int = 0
    invalid_points: int = 0
    out_of_range_points: int = 0
    one_shot_requests: int = 0
    one_shot_completions: int = 0


class SensorProviderStateMachine:
    """Resource-generation gate; concrete adapters own only native API calls."""

    def __init__(
        self,
        *,
        adapter_name: str,
        profiles: Sequence[SensorProfileDescriptor],
        dispatcher: ScanDispatcher | None = None,
        stage_metadata: StageMetadata = StageMetadata(),
        generation_seed: int = 0,
    ) -> None:
        profiles = tuple(profiles)
        if not adapter_name:
            raise ValueError("adapter_name cannot be empty")
        if not profiles:
            raise ValueError("at least one profile is required")
        if len({profile.profile_id for profile in profiles}) != len(profiles):
            raise ValueError("profile ids must be unique")
        if len({profile.sensor_kind for profile in profiles}) != 1:
            raise ValueError("one provider cannot mix sensor families")
        self._adapter_name = adapter_name
        self._profiles = profiles
        self._profiles_by_id = {profile.profile_id: profile for profile in profiles}
        self._dispatcher = dispatcher or ScanDispatcher()
        self._stage_metadata = stage_metadata
        self._lock = threading.RLock()
        self._ingest_lock = threading.Lock()
        self._status = SensorStatus(generation=generation_seed)
        self._profile: SensorProfileDescriptor | None = None
        self._pose_world = Pose()
        self._sequence = 0
        self._operation_serial = 0
        self._one_shot: SensorOperationToken | None = None
        self._telemetry = ProviderTelemetry()

    @property
    def dispatcher(self) -> ScanDispatcher:
        return self._dispatcher

    @property
    def sensor_kind(self) -> SensorKind:
        return self._profiles[0].sensor_kind

    def list_profiles(self) -> Sequence[SensorProfileDescriptor]:
        return self._profiles

    def adopt_dispatcher(self, dispatcher: ScanDispatcher) -> ScanFrame | None:
        """Promote a warmed provider to a stable downstream dispatcher."""

        with self._ingest_lock:
            with self._lock:
                if dispatcher is self._dispatcher:
                    frame = self.latest_frame()
                    if frame is None and self._status.lifecycle not in {
                        SensorLifecycle.RUNNING,
                        SensorLifecycle.DISABLED,
                    }:
                        raise RuntimeError("provider has no frame to adopt")
                    return frame
                publication = self._dispatcher.snapshot()
                if publication is None:
                    if self._status.lifecycle not in {
                        SensorLifecycle.RUNNING,
                        SensorLifecycle.DISABLED,
                    }:
                        raise RuntimeError("only a ready provider can be adopted without a frame")
                    frame = None
                    dispatcher.activate_empty_generation(self._status.generation)
                else:
                    if publication.frame.generation != self._status.generation:
                        raise RuntimeError("only a provider with a current valid frame can be adopted")
                    frame = publication.frame
                    dispatcher.activate_and_publish(frame)
                self._dispatcher = dispatcher
                return frame

    def isolate_dispatcher(self) -> None:
        """Detach from the public dispatcher before retiring native resources.

        The public dispatcher deliberately keeps its last publication while this
        provider is closed.  Any late native callback is instead confined to the
        replacement private dispatcher and is rejected by the provider generation
        gate once ``close`` advances the lifecycle.
        """

        with self._ingest_lock:
            with self._lock:
                publication = self._dispatcher.snapshot()
                isolated = ScanDispatcher()
                if (
                    publication is not None
                    and publication.frame.generation == self._status.generation
                ):
                    isolated.activate_and_publish(publication.frame)
                else:
                    isolated.activate_generation(self._status.generation)
                self._dispatcher = isolated

    def restore_dispatcher(self, dispatcher: ScanDispatcher) -> ScanFrame:
        """Reattach a retained provider to its unchanged public generation."""

        with self._ingest_lock:
            with self._lock:
                publication = dispatcher.snapshot()
                if (
                    publication is None
                    or publication.frame.generation != self._status.generation
                ):
                    raise RuntimeError("public dispatcher no longer holds the retained generation")
                self._dispatcher = dispatcher
                return publication.frame

    def start(self, profile: SensorProfileDescriptor, pose_world: Pose) -> int:
        with self._lock:
            if self._status.lifecycle is SensorLifecycle.CLOSED:
                raise RuntimeError("closed provider cannot be restarted")
            if self._status.lifecycle not in {SensorLifecycle.STOPPED, SensorLifecycle.ERROR}:
                raise RuntimeError(f"cannot start provider from {self._status.lifecycle.value}")
            selected = self._select_profile(profile)
            generation = self._next_generation()
            self._dispatcher.activate_generation(generation)
            self._profile = selected
            self._pose_world = pose_world
            self._sequence = 0
            self._one_shot = None
            self._status = SensorStatus(SensorLifecycle.CREATING, generation, selected.profile_id)
            return generation

    def replace(self, profile: SensorProfileDescriptor) -> int:
        with self._lock:
            if self._status.lifecycle not in {
                SensorLifecycle.WARMING,
                SensorLifecycle.RUNNING,
                SensorLifecycle.DISABLED,
                SensorLifecycle.ERROR,
            }:
                raise RuntimeError(f"cannot replace provider from {self._status.lifecycle.value}")
            selected = self._select_profile(profile)
            generation = self._next_generation()
            self._dispatcher.activate_generation(generation)
            self._profile = selected
            self._sequence = 0
            self._one_shot = None
            self._status = SensorStatus(SensorLifecycle.STOPPING, generation, selected.profile_id)
            return generation

    def resource_stopped(self, generation: int) -> bool:
        return self._transition(generation, SensorLifecycle.STOPPING, SensorLifecycle.CREATING)

    def resource_created(self, generation: int) -> bool:
        return self._transition(generation, SensorLifecycle.CREATING, SensorLifecycle.WARMING)

    def disable(self) -> None:
        with self._lock:
            if self._status.lifecycle not in {SensorLifecycle.WARMING, SensorLifecycle.RUNNING}:
                raise RuntimeError(f"cannot disable provider from {self._status.lifecycle.value}")
            self._one_shot = None
            self._status = SensorStatus(
                SensorLifecycle.DISABLED,
                self._status.generation,
                self._status.profile_id,
            )

    def enable(self) -> None:
        with self._lock:
            if self._status.lifecycle is not SensorLifecycle.DISABLED:
                raise RuntimeError(f"cannot enable provider from {self._status.lifecycle.value}")
            self._one_shot = None
            self._status = SensorStatus(
                SensorLifecycle.WARMING,
                self._status.generation,
                self._status.profile_id,
            )

    def scan_once(self) -> SensorOperationToken:
        with self._lock:
            if self._status.lifecycle is not SensorLifecycle.DISABLED:
                raise RuntimeError("scan_once requires a disabled provider")
            self._operation_serial += 1
            token = SensorOperationToken(self._status.generation, self._operation_serial)
            self._one_shot = token
            self._telemetry = replace(
                self._telemetry,
                one_shot_requests=self._telemetry.one_shot_requests + 1,
            )
            return token

    def ingest_native(
        self,
        payload: NativePointPayload,
        *,
        generation: int,
        operation: SensorOperationToken | None = None,
        sensor_pose_world: Pose | None = None,
    ) -> ScanFrame | None:
        with self._ingest_lock:
            with self._lock:
                self._telemetry = replace(
                    self._telemetry,
                    native_payloads=self._telemetry.native_payloads + 1,
                )
                if generation != self._status.generation:
                    self._telemetry = replace(
                        self._telemetry,
                        stale_callbacks=self._telemetry.stale_callbacks + 1,
                    )
                    return None
                lifecycle = self._status.lifecycle
                one_shot = (
                    lifecycle is SensorLifecycle.DISABLED
                    and operation is not None
                    and operation == self._one_shot
                )
                if lifecycle not in {SensorLifecycle.WARMING, SensorLifecycle.RUNNING} and not one_shot:
                    self._telemetry = replace(
                        self._telemetry,
                        late_callbacks=self._telemetry.late_callbacks + 1,
                    )
                    return None
                profile = self._profile
                pose_world = sensor_pose_world or self._pose_world
                sequence = self._sequence
            assert profile is not None
            try:
                normalized = normalize_native_payload(
                    adapter=self._adapter_name,
                    profile=profile,
                    generation=generation,
                    sequence=sequence,
                    payload=payload,
                    sensor_pose_world=pose_world,
                    stage_metadata=self._stage_metadata,
                )
            except (TypeError, ValueError):
                with self._lock:
                    self._telemetry = replace(
                        self._telemetry,
                        rejected_payloads=self._telemetry.rejected_payloads + 1,
                    )
                return None
            if normalized.frame.compacted_count == 0:
                with self._lock:
                    if generation != self._status.generation:
                        self._telemetry = replace(
                            self._telemetry,
                            stale_callbacks=self._telemetry.stale_callbacks + 1,
                        )
                        return None
                    lifecycle = self._status.lifecycle
                    one_shot = (
                        lifecycle is SensorLifecycle.DISABLED
                        and operation is not None
                        and operation == self._one_shot
                    )
                    if lifecycle not in {SensorLifecycle.WARMING, SensorLifecycle.RUNNING} and not one_shot:
                        self._telemetry = replace(
                            self._telemetry,
                            late_callbacks=self._telemetry.late_callbacks + 1,
                        )
                        return None
                    self._telemetry = replace(
                        self._telemetry,
                        empty_payloads=self._telemetry.empty_payloads + 1,
                        invalid_points=(
                            self._telemetry.invalid_points + normalized.invalid_point_count
                        ),
                        out_of_range_points=(
                            self._telemetry.out_of_range_points
                            + normalized.out_of_range_point_count
                        ),
                    )
                    # A complete scan with no returns supersedes the previous
                    # cloud. Missing/malformed native data does not.
                    self._dispatcher.clear_generation(generation)
                    self._sequence += 1
                    self._pose_world = pose_world
                    if one_shot:
                        self._one_shot = None
                        self._telemetry = replace(
                            self._telemetry,
                            one_shot_completions=self._telemetry.one_shot_completions + 1,
                        )
                    elif lifecycle is SensorLifecycle.WARMING:
                        self._status = SensorStatus(
                            SensorLifecycle.RUNNING, generation, profile.profile_id,
                        )
                return None
            with self._lock:
                if generation != self._status.generation:
                    self._telemetry = replace(
                        self._telemetry,
                        stale_callbacks=self._telemetry.stale_callbacks + 1,
                    )
                    return None
                lifecycle = self._status.lifecycle
                one_shot = (
                    lifecycle is SensorLifecycle.DISABLED
                    and operation is not None
                    and operation == self._one_shot
                )
                if lifecycle not in {SensorLifecycle.WARMING, SensorLifecycle.RUNNING} and not one_shot:
                    self._telemetry = replace(
                        self._telemetry,
                        late_callbacks=self._telemetry.late_callbacks + 1,
                    )
                    return None
                publication = self._dispatcher.publish(normalized.frame)
                if publication is None:
                    self._telemetry = replace(
                        self._telemetry,
                        stale_callbacks=self._telemetry.stale_callbacks + 1,
                    )
                    return None
                self._sequence += 1
                self._pose_world = pose_world
                self._telemetry = replace(
                    self._telemetry,
                    published_frames=self._telemetry.published_frames + 1,
                    invalid_points=self._telemetry.invalid_points + normalized.invalid_point_count,
                    out_of_range_points=(
                        self._telemetry.out_of_range_points + normalized.out_of_range_point_count
                    ),
                    one_shot_completions=(
                        self._telemetry.one_shot_completions + (1 if one_shot else 0)
                    ),
                )
                if one_shot:
                    self._one_shot = None
                elif lifecycle is SensorLifecycle.WARMING:
                    self._status = SensorStatus(
                        SensorLifecycle.RUNNING,
                        generation,
                        profile.profile_id,
                    )
                return normalized.frame

    def accept_empty_scan(
        self,
        *,
        generation: int,
        operation: SensorOperationToken | None = None,
    ) -> bool:
        """Accept a valid sensor scan with no returns without inventing a frame."""

        with self._ingest_lock:
            with self._lock:
                if generation != self._status.generation:
                    self._telemetry = replace(
                        self._telemetry,
                        stale_callbacks=self._telemetry.stale_callbacks + 1,
                    )
                    return False
                lifecycle = self._status.lifecycle
                one_shot = (
                    lifecycle is SensorLifecycle.DISABLED
                    and operation is not None
                    and operation == self._one_shot
                )
                if lifecycle not in {SensorLifecycle.WARMING, SensorLifecycle.RUNNING} and not one_shot:
                    self._telemetry = replace(
                        self._telemetry,
                        late_callbacks=self._telemetry.late_callbacks + 1,
                    )
                    return False
                self._dispatcher.clear_generation(generation)
                if one_shot:
                    self._one_shot = None
                    self._telemetry = replace(
                        self._telemetry,
                        one_shot_completions=self._telemetry.one_shot_completions + 1,
                    )
                elif lifecycle is SensorLifecycle.WARMING:
                    assert self._profile is not None
                    self._status = SensorStatus(
                        SensorLifecycle.RUNNING,
                        generation,
                        self._profile.profile_id,
                    )
                return True

    def latest_frame(self) -> ScanFrame | None:
        publication = self._dispatcher.snapshot()
        return None if publication is None else publication.frame

    def latest_status(self) -> SensorStatus:
        with self._lock:
            return self._status

    def telemetry(self) -> ProviderTelemetry:
        with self._lock:
            return self._telemetry

    def fail(self, generation: int, message: str) -> bool:
        if not message:
            raise ValueError("error message cannot be empty")
        with self._lock:
            if generation != self._status.generation or self._status.lifecycle is SensorLifecycle.CLOSED:
                return False
            self._one_shot = None
            self._status = SensorStatus(
                SensorLifecycle.ERROR,
                generation,
                self._status.profile_id,
                message,
            )
            return True

    def close(self) -> None:
        with self._lock:
            if self._status.lifecycle is SensorLifecycle.CLOSED:
                return
            owned_generation = self._status.generation
            generation = self._next_generation()
            self._one_shot = None
            self._status = SensorStatus(SensorLifecycle.CLOSED, generation, None)
            self._dispatcher.clear_generation(owned_generation)

    def _select_profile(self, profile: SensorProfileDescriptor) -> SensorProfileDescriptor:
        selected = self._profiles_by_id.get(profile.profile_id)
        if selected is None or selected != profile:
            raise ValueError(f"profile does not belong to provider: {profile.profile_id}")
        return selected

    def _next_generation(self) -> int:
        if self._status.generation >= UINT64_MAX:
            raise OverflowError("sensor generation exhausted")
        return self._status.generation + 1

    def _transition(
        self,
        generation: int,
        expected: SensorLifecycle,
        target: SensorLifecycle,
    ) -> bool:
        with self._lock:
            if generation != self._status.generation or self._status.lifecycle is not expected:
                return False
            self._status = SensorStatus(target, generation, self._status.profile_id)
            return True


class FakePointCloudProvider(SensorProviderStateMachine):
    """Component-test harness for DepthSensor profiles."""

    @classmethod
    def for_kind(
        cls,
        kind: SensorKind,
        *,
        profile_id: str | None = None,
        dispatcher: ScanDispatcher | None = None,
        generation_seed: int = 0,
    ) -> "FakePointCloudProvider":
        kind = SensorKind(kind)
        profile = SensorProfileDescriptor(
            profile_id=profile_id or f"fake-{kind.value.lower()}",
            sensor_kind=kind,
            model={"fixture": True},
            calibration_provenance={"source": "synthetic provider"},
        )
        return cls(
            adapter_name=f"fake-{kind.value.lower()}",
            profiles=(profile,),
            dispatcher=dispatcher,
            generation_seed=generation_seed,
        )


__all__ = [
    "FakePointCloudProvider",
    "ProviderTelemetry",
    "SensorOperationToken",
    "SensorProviderStateMachine",
]
