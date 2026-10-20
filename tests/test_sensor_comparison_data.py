import csv
import json
import math
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
SENSOR_LABELS = {
    'd455-16x9-50k': 'D455',
    'os0-rev6-128ch-10hz-512': 'OS0 REV6',
    'os0-rev7-128ch-10hz-512': 'OS0 REV7',
}


class SensorComparisonDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / 'docs/data/sensor-comparison.csv').open(encoding='utf-8', newline='') as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.conditions = json.loads((ROOT / 'docs/data/sensor-comparison-conditions.json').read_text(encoding='utf-8'))
        cls.report = (ROOT / 'docs/SENSOR_COMPARISON.md').read_text(encoding='utf-8')

    def test_complete_repeated_design_and_unique_scans(self):
        self.assertEqual(len(self.rows), 216)
        self.assertEqual(len({(r['condition'], r['generation'], r['sequence']) for r in self.rows}), 72)
        keys = [(r['condition'], r['profile'], r['additional_error'], r['repeat'], r['shape']) for r in self.rows]
        self.assertEqual(len(set(keys)), 216)
        for condition in ('A', 'B'):
            for profile in SENSOR_LABELS:
                for error in ('False', 'True'):
                    for shape in ('Plane', 'Sphere', 'Cylinder'):
                        group = [r for r in self.rows if (r['condition'], r['profile'], r['additional_error'], r['shape'])
                                 == (condition, profile, error, shape)]
                        self.assertEqual({r['repeat'] for r in group}, set(map(str, range(1, 7))))

    def test_d455_optical_pose_matches_both_requested_poses(self):
        for condition, expected in zip(self.conditions['conditions'], ([0., 0., 8.], [.5, .25, 8.5])):
            self.assertEqual(condition['rig_pose']['position_m'], expected)
            actual = condition['measurement_poses']['d455-16x9-50k']
            for a, b in zip(actual['position_m'], expected):
                self.assertAlmostEqual(a, b, places=6)
            self.assertEqual(len(condition['depth_optical_mount_checks']), 12)
            for check in condition['depth_optical_mount_checks']:
                self.assertLess(check['position_error_m'], 1e-6)
                self.assertLess(check['quaternion_error'], 1e-6)

    def test_population_statistics_and_published_success_count(self):
        good = [r for r in self.rows if r['success'] == 'True']
        self.assertIn(f'detection attempts, with {len(good)} successes in this run.', self.report)
        for row in good:
            for key, value in row.items():
                if value and key not in {'condition', 'profile', 'additional_error', 'shape', 'success'}:
                    self.assertTrue(math.isfinite(float(value)), (key, value))
            self.assertTrue(math.isclose(float(row['match_rms_m']) ** 2,
                                        float(row['side_mean_m']) ** 2 + float(row['side_std_m']) ** 2,
                                        abs_tol=1e-12))
            self.assertEqual(int(row['left_count']) + int(row['right_count']), int(row['inliers']))
            for key in ('original_histogram_used', 'side_histogram_used', 'detected_histogram_used'):
                self.assertTrue(0 <= int(row[key]) <= int(row['inliers']))

    def test_report_sphere_tables_match_csv(self):
        for condition in ('A', 'B'):
            for profile, label in SENSOR_LABELS.items():
                for state, error_label in (('False', 'Off'), ('True', 'Distance')):
                    group = [r for r in self.rows if (r['condition'], r['profile'], r['additional_error'], r['shape'])
                             == (condition, profile, state, 'Sphere')]
                    good = [r for r in group if r['success'] == 'True']
                    mean = lambda key: statistics.fmean(float(r[key]) for r in good)
                    with self.subTest(condition=condition, profile=profile, error=state):
                        expected = (f'| {condition} | {label} | {error_label} | {len(good)}/{len(group)} | '
                                    f'{mean("fit_rms_m"):.4f} | {mean("original_std_m"):.4f} | '
                                    f'{mean("match_rms_m"):.4f} |')
                        self.assertIn(expected, self.report)
                        signs = sum(float(r['left_mean_m']) > 0 > float(r['right_mean_m']) for r in good)
                        expected = (f'| {condition} | {label} | {error_label} | {mean("left_mean_m"):+.4f} | '
                                    f'{mean("right_mean_m"):+.4f} | {signs}/{len(good)} |')
                        self.assertIn(expected, self.report)

    def test_native_plane_histogram_exclusions_are_retained(self):
        group = [r for r in self.rows if r['profile'] == 'd455-16x9-50k'
                 and r['shape'] == 'Plane' and r['additional_error'] == 'False' and r['success'] == 'True']
        self.assertEqual(len(group), 12)
        self.assertTrue(all(int(r['original_histogram_used']) == int(r['side_histogram_used']) == 0 for r in group))


if __name__ == '__main__':
    unittest.main()
