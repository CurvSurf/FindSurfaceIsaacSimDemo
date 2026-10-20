import asyncio
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/extensions/com.curvsurf.findsurface_isaac_sim'))
from findsurface_isaac_sim.command_facade import session_from_config
from findsurface_isaac_sim.config import load_app_config
from findsurface_isaac_sim.domain import Pose, SensorKind
from findsurface_isaac_sim.lidar_sensor import (
    DEFAULT_LIDAR_PROFILE_ID, LIDAR_PROFILES, SCANNER_EXCLUDED_PROFILE_IDS, RtxLidarRuntime,
)
from findsurface_isaac_sim.sensor_diagnostics import DepthSensorDiagnostics
from findsurface_isaac_sim.user_preferences import load_user_preferences, depth_diagnostics_from_config, encode_user_preferences
from test_lidar_sensor import FakeLidarBackend, scan


class LidarDistanceErrorTests(unittest.IsolatedAsyncioTestCase):
    async def make_frame(self, enabled=True, a=.005, b=.01):
        backend = FakeLidarBackend(LIDAR_PROFILES[0], 'unused', [scan(1), scan(2)])
        diagnostics = DepthSensorDiagnostics(axial_error_enabled=enabled, axial_error_model='distance',
            axial_error_slope_m_per_m=a, axial_error_intercept_m=b)
        runtime = RtxLidarRuntime(backend_factory=lambda *_: backend,
            next_update=lambda: asyncio.sleep(0), diagnostics=diagnostics)
        await runtime.start(LIDAR_PROFILES[0].profile_id, Pose(), '/World/Test/Lidar')
        return runtime, backend, runtime.latest_frame()

    async def test_distance_model_matches_same_normal_draw_and_preserves_rays(self):
        runtime, backend, frame = await self.make_frame()
        try:
            native = runtime.latest_native_scan().points_sensor_m()
            ranges = np.linalg.norm(native, axis=1)
            offset = np.random.Generator(np.random.PCG64((1 << 32) | 1)).normal(0, .005*ranges+.01)
            expected = native + native/ranges[:, None]*offset[:, None] + (1, 2, 3)
            np.testing.assert_allclose(frame.points_world_m, expected, atol=1e-6)
            details = dict(frame.native_payload_provenance.details)
            self.assertEqual(details['application_axial_error_intercept_m'], .01)
            self.assertEqual(details['application_axial_error_distance_formula'], 'max(0, a * range_m + b)')
        finally:
            runtime.close()

    async def test_disable_bypasses_both_models(self):
        runtime, backend, frame = await self.make_frame(enabled=False)
        try:
            np.testing.assert_allclose(frame.points_world_m, runtime.latest_native_scan().points_sensor_m()+(1,2,3), atol=1e-6)
        finally:
            runtime.close()

    async def test_negative_intercept_clamps_sigma_not_points(self):
        runtime, backend, frame = await self.make_frame(a=.005, b=-.1)
        try:
            np.testing.assert_allclose(frame.points_world_m, runtime.latest_native_scan().points_sensor_m()+(1,2,3), atol=1e-6)
        finally:
            runtime.close()

    async def test_pose_change_clears_old_cloud_and_discards_old_sweep(self):
        runtime, backend, frame = await self.make_frame()
        try:
            backend.scans.extend([scan(3), scan(4)])
            runtime.invalidate_pose()
            self.assertIsNone(runtime.latest_frame())
            self.assertIsNone(runtime.poll())
            self.assertIsNotNone(runtime.poll())
        finally:
            runtime.close()


class LidarPreferencesTests(unittest.TestCase):
    def test_legacy_one_metre_reference_is_migrated_for_lidar(self):
        defaults = load_app_config().config
        result = load_user_preferences(defaults, json.dumps({'schema_version': 1, 'sensor': {
            'source': 'RTX', 'profile': DEFAULT_LIDAR_PROFILE_ID,
            'axial_error_model': 'distance', 'axial_error_slope_m_per_m': .005,
            'axial_error_sigma_at_1m_m': .015}}))
        self.assertTrue(result.loaded, result.diagnostic)
        self.assertAlmostEqual(result.config['sensor']['axial_error_intercept_m'], .01)
        self.assertIs(session_from_config(result.config).sensor.sensor_kind, SensorKind.RTX_LIDAR)

    def test_default_lidar_alias_and_retired_profile_handling(self):
        defaults = load_app_config().config
        result = load_user_preferences(defaults, json.dumps({'schema_version': 1, 'sensor': {
            'source': 'RTX', 'profile': 'default'}}))
        self.assertTrue(result.loaded, result.diagnostic)
        self.assertIs(session_from_config(result.config).sensor.sensor_kind, SensorKind.RTX_LIDAR)
        for profile_id in sorted(SCANNER_EXCLUDED_PROFILE_IDS):
            with self.subTest(profile=profile_id):
                retired = load_user_preferences(defaults, json.dumps({'schema_version': 1, 'sensor': {
                    'source': 'RTX', 'profile': profile_id}}))
                self.assertTrue(retired.loaded, retired.diagnostic)
                self.assertIs(session_from_config(retired.config).sensor.sensor_kind, SensorKind.DEPTH)

    def test_lidar_selection_and_coefficients_roundtrip(self):
        defaults = load_app_config().config
        serialized = json.dumps({'schema_version': 1, 'sensor': {'source': 'RTX',
            'profile': DEFAULT_LIDAR_PROFILE_ID, 'axial_error_model': 'distance',
            'axial_error_slope_m_per_m': .005, 'axial_error_intercept_m': .01}})
        result = load_user_preferences(defaults, serialized)
        self.assertTrue(result.loaded, result.diagnostic)
        snapshot = session_from_config(result.config)
        self.assertIs(snapshot.sensor.sensor_kind, SensorKind.RTX_LIDAR)
        encoded = encode_user_preferences(snapshot, depth_diagnostics_from_config(result.config))
        second = load_user_preferences(defaults, encoded)
        self.assertTrue(second.loaded, second.diagnostic)
        self.assertEqual(second.config['sensor']['source'], 'RTX')
        self.assertEqual(second.config['sensor']['axial_error_intercept_m'], .01)


if __name__ == '__main__':
    unittest.main()
