import asyncio
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
EXTENSION = os.path.join(ROOT, "source", "extensions", "com.curvsurf.findsurface_isaac_sim")
if EXTENSION not in sys.path:
    sys.path.insert(0, EXTENSION)

from findsurface_isaac_sim.domain import Pose, SensorKind, SensorLifecycle
from findsurface_isaac_sim.lidar_sensor import (
    DEFAULT_LIDAR_PROFILE_ID,
    IsaacRtxLidarBackend,
    LIDAR_PROFILES,
    LIDAR_TO_CAMERA_MOUNT_ORIENTATION_XYZW,
    LIDAR_UNAVAILABLE_PROFILE_IDS,
    OFFICIAL_NON_NVIDIA_CONFIG_COUNT,
    OFFICIAL_NON_NVIDIA_PROFILE_COUNT,
    SCANNER_EXCLUDED_PROFILE_COUNT,
    SCANNER_EXCLUDED_PROFILE_IDS,
    SCANNER_LIDAR_PROFILE_COUNT,
    SCANNER_LIDAR_PROFILES,
    LidarNativeScan,
    LidarProfile,
    RtxLidarRuntime,
    lidar_profile,
    lidar_profile_description_rows,
    lidar_profile_display_label,
    lidar_profile_is_available,
)
from findsurface_isaac_sim.sensor_diagnostics import DepthSensorDiagnostics


class FakeLidarBackend:
    def __init__(self, profile, _path, scans=None):
        self.profile = profile
        self.scans = list(scans or [])
        self.started = False
        self.paused = False
        self.native_quiesces = 0
        self.native_resumes = 0
        self.closed = False

    async def start(self):
        self.started = True

    def read_scan(self):
        if self.paused or not self.scans:
            return None
        return self.scans.pop(0)

    def pose_world_m(self):
        return Pose((1.0, 2.0, 3.0))

    def range_bounds_m(self):
        return self.profile.minimum_range_m, self.profile.maximum_range_m

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def quiesce_native_updates(self):
        self.native_quiesces += 1

    def resume_native_updates(self):
        self.native_resumes += 1

    def close(self):
        self.closed = True


def scan(timestamp, ranges=(2.0, 3.0, 4.0, 5.0)):
    count = len(ranges)
    return LidarNativeScan(
        np.array([0.0, 90.0, 180.0, -90.0][:count]),
        np.zeros(count),
        np.asarray(ranges),
        timestamp,
        16_384,
        intensity=np.arange(count),
        material_id=np.arange(count),
        time_offset_ns=np.arange(count),
    )


class LidarProfileTests(unittest.TestCase):
    def test_catalog_contains_every_official_non_nvidia_profile(self):
        self.assertEqual(len(LIDAR_PROFILES), OFFICIAL_NON_NVIDIA_PROFILE_COUNT)
        self.assertEqual(len({profile.config for profile in LIDAR_PROFILES}), OFFICIAL_NON_NVIDIA_CONFIG_COUNT)
        self.assertFalse(any("NVIDIA" in profile.config.upper() for profile in LIDAR_PROFILES))
        self.assertEqual(lidar_profile(DEFAULT_LIDAR_PROFILE_ID).emitted_rays_per_scan, 65_536)
        self.assertEqual(lidar_profile(DEFAULT_LIDAR_PROFILE_ID).variant, "OS0_REV6_128ch10hz512res")
        self.assertTrue(all(profile.descriptor.sensor_kind is SensorKind.RTX_LIDAR for profile in LIDAR_PROFILES))

        expected_counts = {
            "HESAI_XT32_SD10": 1, "OS0": 10, "OS1": 15, "OS2": 10,
            "Ouster_VLS_128": 1, "SICK_LMS4000": 3, "SICK_LMS5xx": 61,
            "SICK_LRS4000": 18, "SICK_microScan3": 6, "SICK_MRS1000": 3,
            "SICK_multiScan100": 4, "SICK_nanoScan3": 1,
            "SICK_picoScan100": 26, "SICK_TIM781": 1,
            "Slamtec_RPLIDAR_S2E": 1, "ZVISION_ML30S": 1, "ZVISION_MLXS": 1,
        }
        actual_counts = {
            config: sum(profile.config == config for profile in LIDAR_PROFILES)
            for config in {profile.config for profile in LIDAR_PROFILES}
        }
        self.assertEqual(actual_counts, expected_counts)

    def test_catalog_exposes_rotary_solid_state_and_ordered_nested_variants(self):
        self.assertEqual(
            {profile.scan_type for profile in LIDAR_PROFILES},
            {"rotary_2d", "rotary_3d", "solid_state"},
        )
        self.assertEqual(
            {profile.sensor_model for profile in LIDAR_PROFILES if profile.scan_type == "solid_state"},
            {
                "LMS4000", "LMS5xx", "LRS4000", "microScan3", "MRS1000",
                "multiScan100", "picoScan100", "ML-30s+", "ML-Xs",
            },
        )
        nested = next(profile for profile in LIDAR_PROFILES if profile.config == "SICK_LMS4000")
        self.assertEqual(tuple(nested.variant_argument), ("Product", "Profile"))
        self.assertEqual(tuple(name for name, _value in nested.description_rows), (
            "Name", "Type", "Function", "Profile", "Range", "Features",
        ))
        self.assertEqual("concrete", lidar_profile(DEFAULT_LIDAR_PROFILE_ID).recommended_material.base)
        self.assertEqual(
            "steel",
            next(profile for profile in LIDAR_PROFILES if profile.manufacturer == "SICK").recommended_material.base,
        )
        self.assertEqual(
            "aluminum",
            next(profile for profile in LIDAR_PROFILES if profile.manufacturer == "ZVISION").recommended_material.base,
        )

    def test_scanner_catalog_contains_only_os0_rev6_and_rev7(self):
        all_ids = {profile.profile_id for profile in LIDAR_PROFILES}
        selectable_ids = {profile.profile_id for profile in SCANNER_LIDAR_PROFILES}

        self.assertEqual(len(SCANNER_LIDAR_PROFILES), SCANNER_LIDAR_PROFILE_COUNT)
        self.assertEqual(
            len(SCANNER_EXCLUDED_PROFILE_IDS),
            SCANNER_EXCLUDED_PROFILE_COUNT,
        )
        self.assertEqual(
            all_ids,
            selectable_ids | set(SCANNER_EXCLUDED_PROFILE_IDS),
        )
        self.assertFalse(selectable_ids & set(SCANNER_EXCLUDED_PROFILE_IDS))
        self.assertTrue(all(lidar_profile_is_available(profile.profile_id)
                            for profile in SCANNER_LIDAR_PROFILES))
        self.assertTrue(all(not profile.is_single_scan_line for profile in SCANNER_LIDAR_PROFILES))
        self.assertTrue(all(profile.scan_type != "rotary_3d" or profile.channels >= 64
                            for profile in SCANNER_LIDAR_PROFILES))

        for excluded_id in (
            "sick-lms4000-lms4111r-profile01-3p0m-0p0833deg",
            "sick-nanoscan3",
            "sick-tim781",
            "slamtec-rplidar-s2e",
        ):
            self.assertIn(excluded_id, SCANNER_EXCLUDED_PROFILE_IDS)

        self.assertNotIn("hesai-xt32-sd10", selectable_ids)
        self.assertFalse(any(profile.config == "SICK_MRS1000" for profile in SCANNER_LIDAR_PROFILES))
        self.assertFalse(any(profile.config == "SICK_multiScan100" for profile in SCANNER_LIDAR_PROFILES))
        self.assertEqual(len(selectable_ids), 10)
        self.assertEqual({profile.config for profile in SCANNER_LIDAR_PROFILES}, {"OS0"})
        self.assertEqual({profile.sensor_model for profile in SCANNER_LIDAR_PROFILES}, {"OS0 REV6", "OS0 REV7"})
        for revision in ("REV6", "REV7"):
            self.assertEqual(
                {(profile.scan_rate_hz, profile.horizontal_resolution)
                 for profile in SCANNER_LIDAR_PROFILES if profile.sensor_model == f"OS0 {revision}"},
                {(10, 512), (10, 1024), (10, 2048), (20, 512), (20, 1024)},
            )
        self.assertIn(DEFAULT_LIDAR_PROFILE_ID, selectable_ids)
        self.assertNotIn("zvision-ml30s", selectable_ids)

    def test_profile_validation_and_lookup(self):
        with self.assertRaises(ValueError):
            LidarProfile("bad", "Bad", "variant", 0, 512, 20.0)
        with self.assertRaises(ValueError):
            LidarProfile("bad", "Bad", (), 1, 1, 10.0)
        with self.assertRaises(KeyError):
            lidar_profile("missing")

    def test_baseline_unavailable_profiles_are_visible_and_unambiguous(self):
        self.assertEqual({
            "sick-lms5xx-lms5xxextendedrange-profile16-25hz-0p042deg",
            "sick-lms5xx-lms5xxprohr-profile16-25hz-0p042deg",
            "sick-lms5xx-lms5xxprosr-profile16-25hz-0p042deg",
            "zvision-ml30s",
        }, set(LIDAR_UNAVAILABLE_PROFILE_IDS))
        self.assertTrue(LIDAR_UNAVAILABLE_PROFILE_IDS < {
            profile.profile_id for profile in LIDAR_PROFILES
        })
        self.assertTrue(lidar_profile_is_available(DEFAULT_LIDAR_PROFILE_ID))
        self.assertFalse(lidar_profile_is_available("zvision-ml30s"))
        self.assertFalse(lidar_profile_is_available(
            "sick-lms5xx-lms5xxextendedrange-profile16-25hz-0p042deg"
        ))
        self.assertTrue(lidar_profile_is_available(
            "sick-lms5xx-lms5xxextendedrange-profile15-25hz-0p083deg"
        ))
        with self.assertRaises(KeyError):
            lidar_profile_is_available("missing")

        for profile in LIDAR_PROFILES:
            expected_available = profile.profile_id not in LIDAR_UNAVAILABLE_PROFILE_IDS
            self.assertEqual(expected_available, lidar_profile_is_available(profile.profile_id))
            self.assertEqual(
                not expected_available,
                lidar_profile_display_label(profile).endswith(" (N/A)"),
            )
            rows = dict(lidar_profile_description_rows(profile))
            self.assertEqual(
                not expected_available,
                "N/A in the Isaac Sim 6.0.1 baseline" in rows["Features"],
            )

    def test_native_scan_is_immutable_and_converts_spherical_axes(self):
        native = scan(10)
        expected = np.array([[2, 0, 0], [0, 3, 0], [-4, 0, 0], [0, -5, 0]], dtype=float)
        np.testing.assert_allclose(native.points_sensor_m(), expected, atol=1.0e-6)
        self.assertFalse(native.range_m.flags.writeable)
        self.assertEqual(native.returned_count, 4)

    def test_native_aux_lengths_must_match(self):
        with self.assertRaises(ValueError):
            LidarNativeScan([0], [0], [1], 0, 1, material_id=[1, 2])

    def test_physical_mount_maps_os1_axes_to_scanner_camera_axes(self):
        x, y, z, w = LIDAR_TO_CAMERA_MOUNT_ORIENTATION_XYZW
        vector = np.asarray((x, y, z), dtype=float)
        skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        rotation = (w * w - np.dot(vector, vector)) * np.identity(3) + 2.0 * np.outer(vector, vector) + 2.0 * w * skew

        np.testing.assert_allclose(rotation @ (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), atol=1.0e-12)
        np.testing.assert_allclose(rotation @ (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(rotation @ (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), atol=1.0e-12)


class IsaacRtxLidarBackendLifecycleTests(unittest.TestCase):
    def test_pause_resume_retains_sensor_and_reattaches_writer(self):
        class FakeHydraTexture:
            def __init__(self):
                self.enabled = []

            def set_updates_enabled(self, enabled):
                self.enabled.append(enabled)

        class FakeRenderProduct:
            @staticmethod
            def GetPath():
                return "/Render/Product"

        class FakeSensor:
            render_product = FakeRenderProduct()

            def __init__(self):
                self._hydra_texture = FakeHydraTexture()

        class FakeWriter:
            def __init__(self):
                self.latest = object()
                self.detaches = 0
                self.attachments = []

            def detach(self):
                self.detaches += 1

            def attach(self, paths):
                self.attachments.append(paths)

        backend = IsaacRtxLidarBackend.__new__(IsaacRtxLidarBackend)
        backend._lidar = object()
        backend._sensor = FakeSensor()
        backend._writer = FakeWriter()
        backend._writer_attached = True
        sensor = backend._sensor

        backend.pause()
        backend.resume()

        self.assertIs(sensor, backend._sensor)
        self.assertEqual(1, backend._writer.detaches)
        self.assertEqual([["/Render/Product"]], backend._writer.attachments)
        self.assertTrue(backend._writer_attached)
        self.assertEqual([], sensor._hydra_texture.enabled)

        backend.quiesce_native_updates()
        backend.resume_native_updates()

        self.assertEqual([False, True], sensor._hydra_texture.enabled)


class LidarRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backends = []
        self.next_timestamp = 100

        def factory(profile, path):
            values = [scan(self.next_timestamp + offset) for offset in range(8)]
            self.next_timestamp += 10
            backend = FakeLidarBackend(profile, path, values)
            self.backends.append(backend)
            return backend

        async def next_update():
            await asyncio.sleep(0)

        self.runtime = RtxLidarRuntime(backend_factory=factory, next_update=next_update)

    async def test_profile_switch_quiescence_requires_backend_recreation(self):
        await self.runtime.start(DEFAULT_LIDAR_PROFILE_ID, Pose(), "/World/Rig/Lidar")
        backend = self.backends[0]

        self.runtime.suspend_for_switch()
        with self.assertRaisesRegex(RuntimeError, "must be recreated"):
            self.runtime.resume_after_failed_switch()
        self.assertTrue(backend.paused)
        self.assertEqual((1, 0), (backend.native_quiesces, backend.native_resumes))

    async def test_start_publishes_common_world_metre_frame_and_retains_native(self):
        renderer = self.runtime.dispatcher.subscribe("renderer")
        picker = self.runtime.dispatcher.subscribe("picker")
        status = await self.runtime.start(DEFAULT_LIDAR_PROFILE_ID, Pose(), "/World/Rig/Lidar")
        self.assertIs(status.lifecycle, SensorLifecycle.RUNNING)
        frame = self.runtime.latest_frame()
        self.assertIs(frame.sensor_kind, SensorKind.RTX_LIDAR)
        np.testing.assert_allclose(frame.points_world_m[0], (3.0, 2.0, 3.0), atol=1e-6)
        self.assertIs(renderer.take_latest().frame, picker.take_latest().frame)
        self.assertIsNotNone(self.runtime.latest_native_scan().material_id)
        self.assertEqual(self.runtime.telemetry().emitted_rays, 16_384)

    async def test_writer_attach_discards_partial_priming_sweep(self):
        backend = FakeLidarBackend(
            lidar_profile(DEFAULT_LIDAR_PROFILE_ID),
            "ignored",
            [
                scan(10, (2.0,)),
                scan(11, (2.0, 3.0, 4.0, 5.0)),
                scan(12, (2.0,)),
                scan(13, (2.0, 3.0, 4.0, 5.0)),
                scan(14, (2.0,)),
                scan(15, (2.0, 3.0, 4.0, 5.0)),
            ],
        )
        runtime = RtxLidarRuntime(
            backend_factory=lambda *_: backend,
            next_update=lambda: asyncio.sleep(0),
        )

        await runtime.start(DEFAULT_LIDAR_PROFILE_ID, Pose(), "/World/Rig/Lidar")
        self.assertEqual(11, runtime.latest_native_scan().timestamp_ns)
        self.assertEqual(4, runtime.latest_frame().compacted_count)

        runtime.disable()
        await runtime.enable()
        self.assertEqual(13, runtime.latest_native_scan().timestamp_ns)
        self.assertEqual(4, runtime.latest_frame().compacted_count)

        runtime.disable()
        one_shot = await runtime.scan_once()
        self.assertEqual(15, runtime.latest_native_scan().timestamp_ns)
        self.assertEqual(4, one_shot.compacted_count)
        self.assertEqual(3, runtime.telemetry().priming_discarded_scans)

    async def test_application_axial_error_is_applied_to_lidar_payloads(self):
        backend = FakeLidarBackend(LIDAR_PROFILES[0], "ignored", [scan(9), scan(10)])
        diagnostics = DepthSensorDiagnostics(axial_error_enabled=True, axial_error_sigma_m=0.01)
        runtime = RtxLidarRuntime(
            backend_factory=lambda *_: backend,
            next_update=lambda: asyncio.sleep(0),
            diagnostics=diagnostics,
        )
        await runtime.start(LIDAR_PROFILES[0].profile_id, Pose(), "/World/Rig/Lidar")
        frame = runtime.latest_frame()
        provenance = dict(frame.native_payload_provenance.details)
        self.assertEqual(
            0.01,
            provenance["application_axial_error_sigma_m"],
        )
        self.assertTrue(
            dict(provenance["payload"])["application_axial_error_enabled"]
        )

    async def test_valid_empty_scan_marks_ready_and_clears_previous_cloud(self):
        empty = LidarNativeScan([], [], [], 1, 100)
        later_empty = LidarNativeScan([], [], [], 3, 100)
        backend = FakeLidarBackend(
            LIDAR_PROFILES[0],
            "ignored",
            [scan(0), empty, scan(2), scan(2), later_empty, scan(4)],
        )
        runtime = RtxLidarRuntime(backend_factory=lambda *_: backend, next_update=lambda: asyncio.sleep(0))
        status = await runtime.start(LIDAR_PROFILES[0].profile_id, Pose(), "/World/Rig/Lidar")
        self.assertIs(status.lifecycle, SensorLifecycle.RUNNING)
        self.assertTrue(runtime.has_no_returns())
        self.assertIsNone(runtime.latest_frame())

        first = runtime.poll(timestamp_s=1.0)
        self.assertIsNotNone(first)
        self.assertFalse(runtime.has_no_returns())
        self.assertIs(first, runtime.latest_frame())
        self.assertIsNone(runtime.poll(timestamp_s=1.0))
        self.assertIsNone(runtime.poll(timestamp_s=2.0))
        self.assertTrue(runtime.has_no_returns())
        self.assertIsNone(runtime.latest_frame())
        self.assertIsNotNone(runtime.poll(timestamp_s=3.0))
        self.assertFalse(runtime.has_no_returns())
        telemetry = runtime.telemetry()
        self.assertEqual(telemetry.empty_complete_scans, 2)
        self.assertEqual(telemetry.stale_scans, 1)
        self.assertEqual(telemetry.complete_scans, 4)

    async def test_empty_one_shot_is_successful_without_inventing_a_frame(self):
        empty = LidarNativeScan([], [], [], 2, 100)
        backend = FakeLidarBackend(
            LIDAR_PROFILES[0],
            "ignored",
            [scan(0), scan(1), scan(2), empty],
        )
        runtime = RtxLidarRuntime(backend_factory=lambda *_: backend, next_update=lambda: asyncio.sleep(0))
        await runtime.start(LIDAR_PROFILES[0].profile_id, Pose(), "/World/Rig/Lidar")
        runtime.disable()

        self.assertIsNone(await runtime.scan_once(maximum_warmup_updates=2))
        self.assertTrue(runtime.has_no_returns())
        self.assertIsNone(runtime.latest_frame())
        self.assertIs(runtime.latest_status().lifecycle, SensorLifecycle.DISABLED)
        self.assertEqual(1, runtime.provider_telemetry().one_shot_completions)

    async def test_disable_one_shot_recreate_and_close_are_generation_safe(self):
        await self.runtime.start(LIDAR_PROFILES[0].profile_id, Pose(), "/World/Rig/Lidar")
        first = self.runtime.latest_frame()
        self.runtime.disable()
        self.assertIs(self.runtime.latest_frame(), first)
        one_shot = await self.runtime.scan_once()
        self.assertEqual(one_shot.sequence, 1)
        self.assertIs(self.runtime.latest_status().lifecycle, SensorLifecycle.DISABLED)
        status = await self.runtime.replace(LIDAR_PROFILES[1].profile_id)
        self.assertIs(status.lifecycle, SensorLifecycle.DISABLED)
        self.assertEqual(status.generation, 2)
        self.assertTrue(self.backends[0].closed)
        self.runtime.close()
        self.assertIs(self.runtime.latest_status().lifecycle, SensorLifecycle.CLOSED)
        self.assertEqual(self.runtime.telemetry().backend_creations, 2)
        self.assertEqual(self.runtime.telemetry().backend_closes, 2)

    async def test_capability_failure_is_local_to_lidar_runtime(self):
        class FailingBackend(FakeLidarBackend):
            async def start(self):
                raise RuntimeError("Lidar API unavailable")

        runtime = RtxLidarRuntime(backend_factory=lambda profile, path: FailingBackend(profile, path))
        status = await runtime.start(LIDAR_PROFILES[0].profile_id, Pose(), "/World/Rig/Lidar", maximum_warmup_updates=1)
        self.assertIs(status.lifecycle, SensorLifecycle.ERROR)
        self.assertIn("unavailable", status.error)


if __name__ == "__main__":
    unittest.main()
