import asyncio
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "source" / "extensions" / "com.curvsurf.findsurface_isaac_sim"
sys.path.insert(0, str(EXTENSION_ROOT))

from findsurface_isaac_sim.config import DEFAULT_APP_CONFIG  # noqa: E402
from findsurface_isaac_sim.domain import (  # noqa: E402
    GeometryMode,
    Pose,
    SensorKind,
    SensorLifecycle,
    SensorProfileDescriptor,
)
from findsurface_isaac_sim.scan import NativePointPayload  # noqa: E402
from findsurface_isaac_sim.scene import SceneCatalog, SceneState  # noqa: E402
from findsurface_isaac_sim.sensor_control import (  # noqa: E402
    MemoryRuntimeSlot,
    SensorControlPhase,
    SensorCoordinator,
    SensorSelection,
)
from findsurface_isaac_sim.sensors import SensorProviderStateMachine  # noqa: E402


def payload(value, serial):
    return NativePointPayload(
        points_native=((float(value), 0.0, 0.0),),
        timestamp_s=float(serial),
        native_frame_id=f"fake-{serial}",
        native_units="metres",
        native_axes="sensor XYZ",
    )


class FakeRuntime:
    def __init__(self, kind, generation_seed, pose_source, *, fail=False, empty=False, start_gate=None):
        self.kind = kind
        self.profiles = tuple(
            SensorProfileDescriptor(f"{kind.value.lower()}-{index}", kind)
            for index in (1, 2)
        )
        self.provider = SensorProviderStateMachine(
            adapter_name=f"coordinator-{kind.value.lower()}",
            profiles=self.profiles,
            generation_seed=generation_seed,
        )
        self.pose_source = pose_source
        self.fail = fail
        self.empty = empty
        self.start_gate = start_gate
        self.closed = 0
        self.one_shots = 0
        self.fail_one_shots = 0
        self.pose_invalidations = 0
        self.switch_suspends = 0
        self.switch_resumes = 0
        self.serial = 0

    @property
    def dispatcher(self):
        return self.provider.dispatcher

    def list_profiles(self):
        return self.profiles

    async def start(self, profile_id, pose_world, runtime_path, *, maximum_warmup_updates):
        profile = next(profile for profile in self.profiles if profile.profile_id == profile_id)
        generation = self.provider.start(profile, pose_world)
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.fail:
            self.provider.fail(generation, "candidate unavailable")
            return self.latest_status()
        self.provider.resource_created(generation)
        if self.empty:
            self.provider.accept_empty_scan(generation=generation)
        else:
            self._publish()
        return self.latest_status()

    def latest_status(self):
        return self.provider.latest_status()

    def latest_frame(self):
        return self.provider.latest_frame()

    def has_no_returns(self):
        return self.empty and self.provider.latest_frame() is None

    def poll(self):
        if self.latest_status().lifecycle is not SensorLifecycle.RUNNING:
            return None
        return self._publish()

    def disable(self):
        self.provider.disable()
        return self.latest_status()

    def suspend_for_switch(self):
        self.switch_suspends += 1

    def resume_after_failed_switch(self):
        self.switch_resumes += 1

    async def enable(self, *, maximum_warmup_updates):
        self.provider.enable()
        self._publish()
        return self.latest_status()

    async def scan_once(self, *, maximum_warmup_updates):
        token = self.provider.scan_once()
        self.one_shots += 1
        if self.fail_one_shots:
            self.fail_one_shots -= 1
            return None
        return self._publish(operation=token)

    def invalidate_pose(self):
        self.pose_invalidations += 1

    def close(self):
        self.closed += 1
        self.provider.close()

    def _publish(self, operation=None):
        self.serial += 1
        status = self.latest_status()
        value = 1 if self.kind is SensorKind.DEPTH else 2
        frame = self.provider.ingest_native(
            payload(value, self.serial),
            generation=status.generation,
            operation=operation,
            sensor_pose_world=self.pose_source(),
        )
        if frame is not None:
            self.empty = False
        return frame


class Harness:
    def __init__(self):
        self.slots = []
        self.runtimes = []
        self.fail_next = False
        self.failures_remaining = 0
        self.empty_next = False
        self.gate_next = None

    def slot_factory(self, generation, pose):
        slot = MemoryRuntimeSlot(generation, pose)
        self.slots.append(slot)
        return slot

    def runtime_factory(self, kind, seed):
        slot = self.slots[-1]
        fail = self.fail_next or self.failures_remaining > 0
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
        runtime = FakeRuntime(
            kind,
            seed,
            lambda: slot.pose_world,
            fail=fail,
            empty=self.empty_next,
            start_gate=self.gate_next,
        )
        self.fail_next = False
        self.empty_next = False
        self.gate_next = None
        self.runtimes.append(runtime)
        return runtime

    def coordinator(self, **kwargs):
        return SensorCoordinator(
            runtime_factory=self.runtime_factory,
            slot_factory=self.slot_factory,
            **kwargs,
        )


class SensorCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_depth_resume_clears_public_frame_and_closes_retained_runtime(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        retained = coordinator.active_runtime
        def fail_resume():
            raise RuntimeError("resume unavailable")
        retained.resume_after_failed_switch = fail_resume
        harness.fail_next = True
        snapshot = await coordinator.switch(SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1"))
        self.assertIs(snapshot.phase, SensorControlPhase.ERROR)
        self.assertIsNone(snapshot.active)
        self.assertIsNone(coordinator.latest_frame())
        self.assertIsNone(coordinator.active_runtime)
        self.assertEqual(retained.closed, 1)
        self.assertIn("resume unavailable", snapshot.error)

    async def test_failed_depth_isolation_and_resume_closes_both_resources(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        retained = coordinator.active_runtime
        def fail():
            raise RuntimeError("native lifecycle unavailable")
        retained.suspend_for_switch = fail
        retained.resume_after_failed_switch = fail
        snapshot = await coordinator.switch(SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1"))
        self.assertIs(snapshot.phase, SensorControlPhase.ERROR)
        self.assertIsNone(coordinator.latest_frame())
        self.assertEqual(retained.closed, 1)
        self.assertEqual(harness.runtimes[-1].closed, 1)
        self.assertTrue(all(slot.closed for slot in harness.slots))

    async def test_ready_candidate_with_no_returns_commits_and_clears_old_cloud(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        old_frame = coordinator.latest_frame()
        harness.empty_next = True

        snapshot = await coordinator.switch(
            SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1")
        )

        self.assertEqual(SensorControlPhase.RUNNING, snapshot.phase)
        self.assertEqual(SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1"), snapshot.active)
        self.assertTrue(snapshot.no_returns)
        self.assertIsNone(snapshot.error)
        self.assertIsNone(snapshot.latest_frame_identity)
        self.assertIsNone(coordinator.latest_frame())
        self.assertIsNot(old_frame, coordinator.latest_frame())
        self.assertEqual(0, snapshot.rollbacks)
        self.assertEqual(1, harness.runtimes[0].closed)

    async def test_continuous_camera_pose_updates_do_not_invalidate_every_sensor_frame(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))

        for x in (1.0, 2.0, 3.0):
            coordinator.set_camera_pose(Pose((x, 0.0, 0.0)), invalidate_frame=False)
            frame = coordinator.poll()
            self.assertEqual((x, 0.0, 0.0), frame.sensor_pose_world.position_m)

        self.assertEqual(0, harness.runtimes[-1].pose_invalidations)
        coordinator.set_camera_pose(Pose((4.0, 0.0, 0.0)))
        self.assertEqual(1, harness.runtimes[-1].pose_invalidations)

    async def test_warmed_candidate_replaces_frame_atomically_and_old_close_cannot_clear_it(self):
        harness = Harness()
        coordinator = harness.coordinator()
        renderer = coordinator.dispatcher.subscribe("renderer")
        picker = coordinator.dispatcher.subscribe("picker")
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        old_frame = coordinator.latest_frame()
        renderer.take_latest()
        picker.take_latest()

        gate = asyncio.Event()
        harness.gate_next = gate
        task = asyncio.create_task(
            coordinator.switch(SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1"))
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if coordinator.snapshot.phase is SensorControlPhase.WARMING:
                break
        self.assertIs(coordinator.latest_frame(), old_frame)
        self.assertEqual(SensorControlPhase.WARMING, coordinator.snapshot.phase)
        self.assertEqual(0, harness.runtimes[0].closed)
        self.assertFalse(harness.slots[0].closed)
        self.assertEqual(1, harness.runtimes[0].switch_suspends)
        gate.set()
        snapshot = await task

        new_frame = coordinator.latest_frame()
        self.assertIsNot(new_frame, old_frame)
        self.assertEqual(SensorKind.RTX_LIDAR, new_frame.sensor_kind)
        self.assertEqual(2, new_frame.generation)
        self.assertEqual(SensorControlPhase.RUNNING, snapshot.phase)
        deliveries = renderer.take_latest(), picker.take_latest()
        self.assertIs(deliveries[0], deliveries[1])
        self.assertIs(new_frame, deliveries[0].frame)
        self.assertEqual(1, harness.runtimes[0].closed)
        self.assertIs(new_frame, coordinator.latest_frame())
        self.assertTrue(harness.slots[0].closed)

    async def test_failed_candidate_resumes_retained_previous_runtime_and_mode(self):
        harness = Harness()
        coordinator = harness.coordinator()
        previous = SensorSelection(SensorKind.DEPTH, "depth-1")
        await coordinator.start(
            previous,
            disabled=True,
        )
        old_runtime = coordinator.active_runtime
        old_frame = coordinator.latest_frame()
        harness.fail_next = True

        snapshot = await coordinator.switch(
            SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1")
        )

        self.assertEqual(SensorControlPhase.DISABLED, snapshot.phase)
        self.assertIn("unavailable", snapshot.error)
        self.assertIs(old_runtime, coordinator.active_runtime)
        self.assertEqual(previous, snapshot.active)
        self.assertEqual(1, snapshot.active_generation)
        self.assertIs(old_frame, coordinator.latest_frame())
        self.assertEqual(SensorKind.DEPTH, coordinator.latest_frame().sensor_kind)
        self.assertEqual(1, coordinator.latest_frame().generation)
        self.assertEqual(SensorLifecycle.DISABLED, coordinator.active_runtime.latest_status().lifecycle)
        self.assertEqual(1, snapshot.rollbacks)
        self.assertEqual(0, old_runtime.closed)
        self.assertFalse(harness.slots[0].closed)
        self.assertEqual(1, old_runtime.switch_suspends)
        self.assertEqual(1, old_runtime.switch_resumes)
        self.assertEqual(1, harness.runtimes[1].closed)
        self.assertTrue(harness.slots[1].closed)
        self.assertEqual(2, len(harness.runtimes))

    async def test_failed_candidate_restores_running_profile_and_polling(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        harness.fail_next = True

        snapshot = await coordinator.switch(
            SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1")
        )

        self.assertEqual(SensorControlPhase.RUNNING, snapshot.phase)
        self.assertIsNotNone(snapshot.error)
        self.assertEqual(1, harness.runtimes[0].switch_suspends)
        self.assertEqual(1, harness.runtimes[0].switch_resumes)
        restored_frame = coordinator.latest_frame()
        self.assertIsNotNone(coordinator.poll())
        self.assertIsNot(restored_frame, coordinator.latest_frame())

    async def test_failed_candidate_recreates_previous_lidar_instead_of_resuming_it(self):
        harness = Harness()
        coordinator = harness.coordinator()
        previous = SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1")
        await coordinator.start(previous, disabled=True)
        old_runtime = coordinator.active_runtime
        old_slot = harness.slots[0]
        old_frame = coordinator.latest_frame()
        harness.fail_next = True

        snapshot = await coordinator.switch(
            SensorSelection(SensorKind.DEPTH, "depth-1")
        )

        self.assertEqual(SensorControlPhase.DISABLED, snapshot.phase)
        self.assertEqual(previous, snapshot.active)
        self.assertEqual(3, snapshot.active_generation)
        self.assertIn("fresh RTX backend", snapshot.error)
        self.assertIsNot(old_runtime, coordinator.active_runtime)
        self.assertEqual(SensorKind.RTX_LIDAR, coordinator.latest_frame().sensor_kind)
        self.assertEqual(3, coordinator.latest_frame().generation)
        self.assertIsNot(old_frame, coordinator.latest_frame())
        self.assertEqual(1, old_runtime.closed)
        self.assertEqual(0, old_runtime.switch_resumes)
        self.assertTrue(old_slot.closed)
        self.assertEqual(1, harness.runtimes[1].closed)
        self.assertTrue(harness.slots[1].closed)
        self.assertEqual(0, harness.runtimes[2].closed)
        self.assertFalse(harness.slots[2].closed)
        self.assertEqual(3, len(harness.runtimes))
        self.assertEqual(1, snapshot.rollbacks)

    async def test_failed_lidar_recreation_clears_stale_public_frame(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(
            SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1"),
            disabled=True,
        )
        self.assertIsNotNone(coordinator.latest_frame())
        harness.failures_remaining = 2

        snapshot = await coordinator.switch(
            SensorSelection(SensorKind.DEPTH, "depth-1")
        )

        self.assertEqual(SensorControlPhase.ERROR, snapshot.phase)
        self.assertIsNone(snapshot.active)
        self.assertIsNone(snapshot.latest_frame_identity)
        self.assertIsNone(coordinator.active_runtime)
        self.assertIsNone(coordinator.latest_frame())
        self.assertIn("fresh RTX backend failed", snapshot.error)
        self.assertEqual(1, snapshot.rollbacks)
        self.assertTrue(all(runtime.closed == 1 for runtime in harness.runtimes))
        self.assertTrue(all(slot.closed for slot in harness.slots))

    async def test_invalid_profile_preflight_leaves_previous_runtime_untouched(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        old_runtime = coordinator.active_runtime
        old_slot = harness.slots[0]
        old_frame = coordinator.latest_frame()

        snapshot = await coordinator.switch(
            SensorSelection(SensorKind.RTX_LIDAR, "missing-profile")
        )

        self.assertEqual(SensorControlPhase.RUNNING, snapshot.phase)
        self.assertIn("does not belong", snapshot.error)
        self.assertIs(old_runtime, coordinator.active_runtime)
        self.assertEqual(0, old_runtime.closed)
        self.assertFalse(old_slot.closed)
        self.assertIs(old_frame, coordinator.latest_frame())
        self.assertEqual(1, harness.runtimes[1].closed)
        self.assertTrue(harness.slots[1].closed)

    async def test_disable_one_shot_and_object_burst_publish_exactly_once(self):
        harness = Harness()
        cleared = []
        coordinator = harness.coordinator(selection_clear=cleared.append)
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"))
        coordinator.disable()
        held = coordinator.latest_frame()

        catalog = SceneCatalog.from_config(DEFAULT_APP_CONFIG["objects"])
        state = SceneState(catalog, event_sink=coordinator.notify_object_change)
        sphere = state.catalog.object(GeometryMode.SPHERE)
        state.replace_object(replace(sphere, pose_world_m=Pose((0.0, 0.1, 0.8))))
        sphere = state.catalog.object(GeometryMode.SPHERE)
        state.replace_object(replace(sphere, pose_world_m=Pose((0.0, 0.2, 0.8))))

        self.assertIs(coordinator.latest_frame(), held)
        frame = await coordinator.flush_object_changes()
        self.assertIsNot(frame, held)
        self.assertEqual(1, harness.runtimes[0].one_shots)
        self.assertEqual([2], cleared)
        self.assertEqual(2, coordinator.snapshot.object_events)
        self.assertEqual(1, coordinator.snapshot.object_rescans)
        self.assertEqual(SensorControlPhase.DISABLED, coordinator.snapshot.phase)
        self.assertIsNone(await coordinator.flush_object_changes())

    async def test_object_retry_does_not_clear_selection_twice(self):
        harness = Harness()
        cleared = []
        coordinator = harness.coordinator(selection_clear=cleared.append)
        await coordinator.start(SensorSelection(SensorKind.DEPTH, "depth-1"), disabled=True)
        state = SceneState(SceneCatalog.from_config(DEFAULT_APP_CONFIG["objects"]))
        sphere = state.catalog.object(GeometryMode.SPHERE)
        event = state.replace_object(replace(sphere, pose_world_m=Pose((0.0, 0.1, 0.8))))
        coordinator.notify_object_change(event)
        harness.runtimes[0].fail_one_shots = 1

        self.assertIsNone(await coordinator.flush_object_changes())
        self.assertEqual([1], cleared)
        self.assertIsNotNone(await coordinator.flush_object_changes())
        self.assertEqual([1], cleared)
        self.assertEqual(2, harness.runtimes[0].one_shots)

    async def test_edit_during_one_shot_rescans_until_values_match(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, 'depth-1'), disabled=True)
        state = SceneState(SceneCatalog.from_config(DEFAULT_APP_CONFIG['objects']), event_sink=coordinator.notify_object_change)
        sphere = state.catalog.object(GeometryMode.SPHERE)
        state.replace_object(replace(sphere, parameters={'radius_m': 0.8}))
        runtime = harness.runtimes[0]
        original = runtime.scan_once
        async def edit_during_scan(**kwargs):
            result = await original(**kwargs)
            if runtime.one_shots == 1:
                state.replace_object(replace(sphere, parameters={'radius_m': 1.2}))
            return result
        runtime.scan_once = edit_during_scan
        await coordinator.flush_object_changes()
        self.assertFalse(coordinator.has_pending_object_change)
        self.assertEqual(2, runtime.one_shots)
        self.assertEqual(1.2, coordinator.completed_object_values[0].parameter_dict['radius_m'])
        self.assertEqual(SensorControlPhase.DISABLED, coordinator.snapshot.phase)

    async def test_final_edit_refresh_keeps_live_sensor_running(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, 'depth-1'))
        state = SceneState(SceneCatalog.from_config(DEFAULT_APP_CONFIG['objects']), event_sink=coordinator.notify_object_change)
        sphere = state.catalog.object(GeometryMode.SPHERE)
        state.replace_object(replace(sphere, parameters={'radius_m': 1.2}))
        previous = coordinator.latest_frame()
        await coordinator.flush_object_changes()
        self.assertIsNot(previous, coordinator.latest_frame())
        self.assertEqual(SensorControlPhase.RUNNING, coordinator.snapshot.phase)
        self.assertEqual(0, harness.runtimes[0].one_shots)

    async def test_snapshot_comparison_catches_changes_without_another_event(self):
        harness = Harness()
        catalog = SceneCatalog.from_config(DEFAULT_APP_CONFIG['objects'])
        current = list(catalog.objects)
        coordinator = harness.coordinator(object_values=lambda: current)
        await coordinator.start(SensorSelection(SensorKind.DEPTH, 'depth-1'), disabled=True)
        state = SceneState(catalog, event_sink=coordinator.notify_object_change)
        sphere = catalog.object(GeometryMode.SPHERE)
        index = next(i for i, item in enumerate(current) if item.geometry_type is GeometryMode.SPHERE)
        current[index] = replace(sphere, parameters={'radius_m': 0.8})
        state.replace_object(current[index])
        runtime = harness.runtimes[0]
        original = runtime.scan_once
        seen = []
        async def change_again(**kwargs):
            seen.append(current[index].parameter_dict['radius_m'])
            result = await original(**kwargs)
            if len(seen) == 1:
                current[index] = replace(sphere, parameters={'radius_m': 1.2})
            elif len(seen) == 2:
                current[index] = replace(sphere, parameters={'radius_m': 0.6})
            return result
        runtime.scan_once = change_again
        await coordinator.flush_object_changes()
        self.assertEqual([0.8, 1.2, 0.6], seen)
        self.assertEqual(tuple(sorted(current, key=lambda o: o.geometry_type.value)), coordinator.completed_object_values)
        self.assertFalse(coordinator.has_pending_object_change)

    async def test_equal_final_values_do_not_queue_redundant_scan(self):
        harness = Harness()
        coordinator = harness.coordinator()
        await coordinator.start(SensorSelection(SensorKind.DEPTH, 'depth-1'), disabled=True)
        state = SceneState(SceneCatalog.from_config(DEFAULT_APP_CONFIG['objects']), event_sink=coordinator.notify_object_change)
        sphere = state.catalog.object(GeometryMode.SPHERE)
        requested = replace(sphere, parameters={'radius_m': 0.8})
        state.replace_object(requested)
        runtime = harness.runtimes[0]
        original = runtime.scan_once
        async def aba(**kwargs):
            result = await original(**kwargs)
            state.replace_object(replace(sphere, parameters={'radius_m': 1.2}))
            state.replace_object(requested)
            return result
        runtime.scan_once = aba
        await coordinator.flush_object_changes()
        self.assertEqual(1, runtime.one_shots)
        self.assertFalse(coordinator.has_pending_object_change)

    async def test_display_only_edit_during_scan_does_not_queue_another_scan(self):
        harness = Harness()
        catalog = SceneCatalog.from_config(DEFAULT_APP_CONFIG['objects'])
        current = list(catalog.objects)
        coordinator = harness.coordinator(object_values=lambda: current)
        await coordinator.start(SensorSelection(SensorKind.DEPTH, 'depth-1'), disabled=True)
        state = SceneState(catalog, event_sink=coordinator.notify_object_change)
        sphere = catalog.object(GeometryMode.SPHERE)
        index = next(i for i, item in enumerate(current) if item.geometry_type is GeometryMode.SPHERE)
        current[index] = replace(sphere, parameters={'radius_m': 0.8})
        state.replace_object(current[index])
        runtime = harness.runtimes[0]
        original = runtime.scan_once

        async def change_display_only(**kwargs):
            result = await original(**kwargs)
            current[index] = replace(
                current[index],
                display_subdivision=48,
                face_color_rgba=(0.1, 0.2, 0.3, 0.4),
            )
            return result

        runtime.scan_once = change_display_only
        await coordinator.flush_object_changes()
        self.assertEqual(1, runtime.one_shots)
        self.assertFalse(coordinator.has_pending_object_change)

    async def test_follow_freeze_is_independent_from_disable_and_preserves_cloud(self):
        harness = Harness()
        coordinator = harness.coordinator(camera_pose_world=Pose((0.0, 0.0, 3.0)))
        await coordinator.start(
            SensorSelection(SensorKind.RTX_LIDAR, "rtx_lidar-1"),
            disabled=True,
        )
        held = coordinator.latest_frame()
        moved = Pose((1.0, 2.0, 4.0))
        state = coordinator.set_camera_pose(moved)
        self.assertEqual(moved, state.sensor_pose_world)
        self.assertEqual(moved, harness.slots[0].pose_world)
        self.assertIs(held, coordinator.latest_frame())

        frozen = coordinator.freeze_sensor()
        coordinator.set_camera_pose(Pose((9.0, 9.0, 9.0)))
        self.assertFalse(frozen.follows_camera)
        self.assertEqual(moved, harness.slots[0].pose_world)
        followed = coordinator.follow_sensor()
        self.assertTrue(followed.follows_camera)
        self.assertEqual(Pose((9.0, 9.0, 9.0)), harness.slots[0].pose_world)
        frame = await coordinator.scan_once()
        self.assertEqual(Pose((9.0, 9.0, 9.0)), frame.sensor_pose_world)
        self.assertIs(held, held)
        self.assertEqual(2, harness.runtimes[0].pose_invalidations)

    async def test_frozen_start_ignores_camera_until_follow_resynchronizes_exact_pose(self):
        harness = Harness()
        initial = Pose((0.0, 0.0, 3.0))
        coordinator = harness.coordinator(
            camera_pose_world=initial,
            follows_camera=False,
        )
        await coordinator.start(
            SensorSelection(SensorKind.DEPTH, "depth-1"),
            disabled=True,
        )

        moved_while_disabled = Pose(
            (1.25, -0.5, 2.75),
            (0.0, 0.3826834324, 0.0, 0.9238795325),
        )
        coordinator.set_camera_pose(moved_while_disabled, invalidate_frame=False)
        self.assertEqual(initial, coordinator.snapshot.pose.sensor_pose_world)
        self.assertEqual(initial, harness.slots[0].pose_world)

        await coordinator.enable()
        fixed_frame = coordinator.poll()
        self.assertEqual(initial, fixed_frame.sensor_pose_world)
        self.assertFalse(coordinator.snapshot.pose.follows_camera)

        followed = coordinator.follow_sensor()
        self.assertTrue(followed.follows_camera)
        self.assertEqual(moved_while_disabled, followed.sensor_pose_world)
        self.assertEqual(moved_while_disabled, harness.slots[0].pose_world)
        followed_frame = coordinator.poll()
        self.assertEqual(moved_while_disabled, followed_frame.sensor_pose_world)

    async def test_configuration_contains_no_frame_and_close_is_idempotent(self):
        harness = Harness()
        coordinator = harness.coordinator()
        selection = SensorSelection(SensorKind.DEPTH, "depth-2")
        await coordinator.start(selection)
        self.assertEqual(selection, coordinator.snapshot.active)
        self.assertFalse(hasattr(selection, "frame"))
        coordinator.close()
        coordinator.close()
        self.assertEqual(SensorControlPhase.CLOSED, coordinator.snapshot.phase)
        self.assertEqual(1, harness.runtimes[0].closed)
        self.assertTrue(harness.slots[0].closed)
        self.assertIsNone(coordinator.latest_frame())
        self.assertIn(SensorControlPhase.CREATING, coordinator.phase_history)
        self.assertIn(SensorControlPhase.WARMING, coordinator.phase_history)
        self.assertIn(SensorControlPhase.STOPPING, coordinator.phase_history)


if __name__ == "__main__":
    unittest.main()
