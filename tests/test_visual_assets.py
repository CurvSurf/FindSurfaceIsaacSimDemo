import ast
import json
from pathlib import Path
import struct
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / 'source/extensions/com.curvsurf.findsurface_isaac_sim'


class VisualAssetTests(unittest.TestCase):
    def test_axis_preset_faces_match_app_and_manual(self):
        for name, color in (('view_front', '#ff4040'), ('view_side', '#40d060'), ('view_top', '#4080ff')):
            for folder in (EXTENSION / 'data/icons/findsurface', ROOT / 'docs/images/manual-icons/findsurface'):
                with self.subTest(icon=name, folder=folder):
                    paths = ET.parse(folder / f'{name}.svg').getroot().iter('{http://www.w3.org/2000/svg}path')
                    self.assertEqual(next(paths).get('fill'), color)

    def test_ruler_exponent_is_a_separate_raised_label(self):
        # Inspect the Kit-only presentation without requiring a GUI during host tests.
        source = EXTENSION / 'findsurface_isaac_sim/classic_ui.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        labels = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Attribute) and target.attr == '_ruler_exponent'
                          for target in node.targets)]
        self.assertEqual(len(labels), 1)
        keywords = {keyword.arg: keyword.value for keyword in labels[0].keywords}
        self.assertEqual(keywords['height'].value, 12)
        self.assertEqual(keywords['alignment'].attr, 'LEFT_TOP')

    def test_ruler_marks_have_distinct_sizes_and_heavier_boundaries(self):
        source = EXTENSION / 'findsurface_isaac_sim/classic_ui.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names = ('RULER_TICK_HEIGHTS', 'RULER_TICK_WIDTHS',
                 'RULER_BOUNDARY_HEIGHT', 'RULER_BOUNDARY_THICKNESS', 'RULER_PIP_SCALE')
        values = {target.id: ast.literal_eval(node.value)
                  for node in tree.body if isinstance(node, ast.Assign)
                  for target in node.targets if isinstance(target, ast.Name) and target.id in names}
        self.assertEqual(values['RULER_TICK_HEIGHTS'], (16, 12, 8))
        self.assertEqual(values['RULER_TICK_WIDTHS'], (2, 1, 1))
        self.assertEqual(values['RULER_BOUNDARY_HEIGHT'], 20)
        self.assertEqual(values['RULER_BOUNDARY_THICKNESS'], 4)
        self.assertEqual(values['RULER_PIP_SCALE'], .75)
        signatures = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'tick_signature'
                              for target in node.targets)]
        self.assertEqual(len(signatures), 1)
        self.assertIn('primary', [node.id for node in signatures[0].elts if isinstance(node, ast.Name)])

    def test_comparison_crops_have_matching_native_dimensions(self):
        for kind, expected in (('surface-side', (1120, 1120)), ('histograms', (680, 410))):
            for sensor in ('depth', 'lidar'):
                with self.subTest(kind=kind, sensor=sensor):
                    path = ROOT / f'docs/images/comparison/{sensor}-{kind}.png'
                    with path.open('rb') as stream:
                        header = stream.read(24)
                    self.assertEqual(header[:8], b'\x89PNG\r\n\x1a\n')
                    self.assertEqual(struct.unpack('>II', header[16:24]), expected)
                    if kind == 'histograms':
                        self.assertGreater(path.stat().st_size, 10_000,
                                           'Histogram capture is unexpectedly empty')

    def test_comparison_capture_projection_and_viewport_match(self):
        record = json.loads((ROOT / 'docs/data/readme-comparison-captures.json').read_text(encoding='utf-8'))
        depth, lidar = (record['captures'][key] for key in ('d455-16x9-50k', 'os0-rev7-128ch-10hz-512'))
        for key in ('display_pose', 'display_intrinsics', 'viewport', 'scale_mode', 'sigma_multiplier'):
            self.assertEqual(depth[key], lidar[key], key)
        for actual, requested in zip(depth['sensor_pose']['position_m'], depth['scan_rig']['position_m']):
            self.assertAlmostEqual(actual, requested, places=6)
        actual = depth['sensor_pose']['orientation_xyzw']
        requested = depth['scan_rig']['orientation_xyzw']
        sign = 1 if sum(a * b for a, b in zip(actual, requested)) >= 0 else -1
        for a, b in zip(actual, requested):
            self.assertAlmostEqual(a, sign * b, places=6)
        self.assertEqual(record['source_frame_size_px'], [2532, 1383])
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        for sensor in ('depth', 'lidar'):
            for kind in ('surface-side', 'histograms'):
                self.assertIn(f'](docs/images/comparison/{sensor}-{kind}.png)', readme)

    def test_readme_uses_single_combined_images_for_equal_sensor_panel_sizes(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        section = readme.split('## DepthSensor and LiDAR, side by side', 1)[1].split('\n## ', 1)[0]
        for kind, dimensions in (('surface-side', (2262, 1215)), ('histograms', (1374, 468))):
            path = ROOT / f'docs/images/comparison/sensors-{kind}.png'
            with path.open('rb') as stream:
                header = stream.read(24)
            self.assertEqual(header[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('>II', header[16:24]), dimensions)
            self.assertIn(f'<img src="docs/images/comparison/sensors-{kind}.png" width="860"', section)
        self.assertEqual(section.count('<img '), 2)
        self.assertNotIn('|---|', section)


if __name__ == '__main__':
    unittest.main()
