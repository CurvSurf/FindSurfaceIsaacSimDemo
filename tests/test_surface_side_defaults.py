import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/extensions/com.curvsurf.findsurface_isaac_sim'))

from findsurface_isaac_sim.analysis import build_signed_heatmap, build_signed_histogram_heatmap
from findsurface_isaac_sim.analysis_pipeline import SurfaceSideScale, SurfaceSideScaleMode
from findsurface_isaac_sim.command_facade import session_from_config
from findsurface_isaac_sim.config import DEFAULT_APP_CONFIG, load_app_config
from findsurface_isaac_sim.lidar_sensor import DEFAULT_LIDAR_PROFILE_ID
from findsurface_isaac_sim.user_preferences import load_user_preferences


class SurfaceSideDefaultsTests(unittest.TestCase):
    def test_code_and_packaged_defaults_use_three_sigma(self):
        for config in (DEFAULT_APP_CONFIG, load_app_config().config):
            with self.subTest(config=config['heatmap']):
                self.assertEqual(config['heatmap']['surface_side_scale_mode'], 'sigma')
                self.assertEqual(config['heatmap']['surface_side_sigma_multiplier'], 3)
                interaction = session_from_config(config).interaction
                self.assertEqual(interaction.surface_side_scale_mode, 'sigma')
                self.assertEqual(interaction.surface_side_sigma_multiplier, 3)
        self.assertIs(SurfaceSideScale().mode, SurfaceSideScaleMode.SIGMA)
        self.assertEqual(SurfaceSideScale().sigma_multiplier, 3)

    def test_sensor_selection_preserves_common_three_sigma_default(self):
        defaults = load_app_config().config
        for source, profile in (('DepthSensor', 'd455-16x9-50k'), ('RTX', DEFAULT_LIDAR_PROFILE_ID)):
            with self.subTest(source=source):
                loaded = load_user_preferences(defaults, json.dumps({
                    'schema_version': 1, 'sensor': {'source': source, 'profile': profile}}))
                self.assertTrue(loaded.loaded, loaded.diagnostic)
                scale = session_from_config(loaded.config).interaction.surface_side_scale
                self.assertIs(scale.mode, SurfaceSideScaleMode.SIGMA)
                self.assertEqual(scale.sigma_multiplier, 3)

    def test_three_sigma_defines_color_endpoints(self):
        values = (-0.004, -0.002, 0.0, 0.002, 0.004)
        histogram, _ = build_signed_histogram_heatmap(values)
        scale = SurfaceSideScale().resolve_scale_m(values, histogram)
        self.assertAlmostEqual(scale, 3 * histogram.source_sigma_m)
        heatmap = build_signed_heatmap((-scale, 0, scale), scale_m=scale)
        self.assertEqual(heatmap.coordinates, (0.0, 0.5, 1.0))
        self.assertAlmostEqual(histogram.range_min_m, -scale)
        self.assertAlmostEqual(histogram.range_max_m, scale)


if __name__ == '__main__':
    unittest.main()
