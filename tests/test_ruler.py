import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/extensions/com.curvsurf.findsurface_isaac_sim'))
from findsurface_isaac_sim.orthographic import MovableRuler, ruler_length_m, ruler_scale
from findsurface_isaac_sim.command_facade import InteractionSettings, session_from_config
from findsurface_isaac_sim.config import load_app_config
from findsurface_isaac_sim.user_preferences import load_user_preferences, encode_user_preferences
from findsurface_isaac_sim.sensor_diagnostics import DepthSensorDiagnostics


class RulerTests(unittest.TestCase):
    def test_whole_metre_ticks_leave_fractional_remainder(self):
        for span, viewport_height in ((90, 200), (180, 400)):
            with self.subTest(span=span):
                scale = ruler_scale(ruler_length_m(10, viewport_height, span), span)
                offsets = tuple(tick.offset_px for tick in scale.ticks)
                self.assertEqual((scale.significand, scale.exponent), ('4.5', 0))
                self.assertEqual(len(offsets), 4)
                step = viewport_height / 10
                for offset, expected in zip(offsets, (i * step for i in range(1, 5))):
                    self.assertAlmostEqual(offset, expected)
                self.assertAlmostEqual(span - offsets[-1], step / 2)

    def test_tick_scale_follows_zoom_and_resize_not_rounded_label(self):
        self.assertEqual(tuple(tick.offset_px for tick in ruler_scale(4, 80).ticks), (20, 40, 60))
        self.assertEqual(ruler_scale(.75, 15).exponent, -1)
        self.assertEqual(len(ruler_scale(.75, 15).ticks), 7)
        # Rounded label is 4.0, but the 4-unit mark must not exceed the span.
        self.assertEqual(len(ruler_scale(3.99, 79.8).ticks), 3)
        for tick, expected in zip(ruler_scale(6, 120).ticks[:3], (20, 40, 60)):
            self.assertAlmostEqual(tick.offset_px, expected)

    def test_subdivision_density_uses_screen_gap_and_three_lengths(self):
        for length in (1.3e-6, 1.3, 1.3e6):
            with self.subTest(length=length):
                short = ruler_scale(length, 90).ticks
                medium = ruler_scale(length, 180).ticks
                long = ruler_scale(length, 600).ticks
                self.assertEqual([t.level for t in short], [0])
                self.assertEqual([t.level for t in medium], [1, 0])
                self.assertEqual(len(long), 12)
                for index, tick in enumerate(long, 1):
                    self.assertAlmostEqual(tick.offset_px, 600 * index / 13)
                    self.assertEqual(tick.level, 0 if index == 10 else 1 if index == 5 else 2)

    def test_density_thresholds_and_integer_endpoints_have_no_duplicate_marks(self):
        for span, levels in ((99.9, {0}), (100, {0, 1}), (199.9, {0, 1}), (200, {0, 1, 2})):
            # Two major intervals: screen gap is half of span.
            scale = ruler_scale(2, span * 2)
            self.assertEqual({t.level for t in scale.ticks}, levels)
            self.assertTrue(all(0 < t.offset_px < span * 2 for t in scale.ticks))
            self.assertEqual(len({t.offset_px for t in scale.ticks}), len(scale.ticks))

    def test_far_zoom_out_bounds_tick_widgets(self):
        for length in (1e-100, .00045, .999, 1, 4.5, 9.96, 45, 4.5e100):
            with self.subTest(length=length):
                scale = ruler_scale(length, 180)
                offsets = tuple(tick.offset_px for tick in scale.ticks)
                self.assertLessEqual(len(offsets), 99)
                self.assertTrue(all(0 < x < 180 for x in offsets))
                self.assertEqual(tuple(sorted(set(offsets))), offsets)
                self.assertAlmostEqual(float(scale.significand) * 10 ** scale.exponent / length, 1, delta=.05)
        for bad in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ruler_scale(bad, 180)

    def test_retired_ruler_unit_does_not_invalidate_other_preferences(self):
        defaults = load_app_config().config
        baseline = session_from_config(defaults)
        loaded = load_user_preferences(defaults, json.dumps({'schema_version': 1, 'display': {
            'ruler_unit': 'mm', 'point_size': 7.5, 'ruler_main_opacity': .65}}))
        self.assertTrue(loaded.loaded)
        state = session_from_config(loaded.config)
        self.assertFalse(hasattr(state.interaction, 'ruler_unit'))
        self.assertEqual(state.interaction.point_rendering_size, 7.5)
        self.assertEqual(state.interaction.ruler_main_opacity, .65)
        self.assertEqual(state.objects, baseline.objects)
        self.assertEqual(state.detection, baseline.detection)
        encoded = json.loads(encode_user_preferences(state, DepthSensorDiagnostics()))
        self.assertNotIn('ruler_unit', encoded['display'])

    def test_pip_is_half_length_and_independent_of_main_resize(self):
        ruler = MovableRuler()
        self.assertEqual(ruler.layout((400, 300), primary=False)[2], 90)
        self.assertEqual(ruler.layout((800, 600))[2], 180)
        self.assertEqual(ruler.layout((180, 120), primary=False)[2], 74)
        ruler.main_span = 320
        self.assertEqual(ruler.layout((400, 300), primary=False)[2], 90)

    def test_body_drag_preserves_span_and_grab_offset(self):
        ruler = MovableRuler()
        left, top, span = ruler.layout((800, 600))
        ruler.pointer('begin', left + 80, top + 20, 0)
        ruler.pointer('move', left + 140, top - 10, 0)
        self.assertEqual(ruler.layout((800, 600)), (left + 60, top - 30, span))
        ruler.pointer('end', left + 140, top - 10, 0)
        self.assertIsNone(ruler.grab)

    def test_edges_resize_without_vertical_motion_and_keep_other_end_fixed(self):
        ruler = MovableRuler()
        left, top, span = ruler.layout((800, 600))
        right = left + span + 16
        ruler.pointer('begin', right - 2, top + 20, 0)
        ruler.pointer('move', right + 98, top - 100, 0)
        ruler.pointer('end', right + 98, top - 100, 0)
        self.assertEqual(ruler.layout((800, 600)), (left, top, span + 100))
        ruler.pointer('begin', left + 2, top + 20, 0)
        ruler.pointer('move', left - 8, top - 100, 0)
        ruler.pointer('end', left - 8, top - 100, 0)
        self.assertEqual(ruler.layout((800, 600)), (left - 10, top, span + 110))
        self.assertAlmostEqual(ruler_length_m(10, 600, span + 100), 10 / 600 * (span + 100))

    def test_resize_clamps_and_cancel_stops_drag(self):
        ruler = MovableRuler()
        left, top, span = ruler.layout((800, 600))
        ruler.pointer('begin', left + span + 15, top + 20, 0)
        ruler.pointer('move', -1000, top, 0)
        self.assertEqual(ruler.layout((800, 600))[2], 48)
        ruler.pointer('move', 2000, top, 0)
        x, _, span = ruler.layout((800, 600))
        self.assertLessEqual(x + span + 16, 800)
        ruler.cancel_drag()
        self.assertIsNone(ruler.resize_grab)
        self.assertLessEqual(ruler.layout((200, 200))[2], 184)

    def test_opacity_validation_and_preference_roundtrip(self):
        for bad in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                InteractionSettings(ruler_pip_opacity=bad)
        defaults = load_app_config().config
        loaded = load_user_preferences(defaults, json.dumps({'schema_version': 1, 'display': {
            'ruler_main_opacity': .65, 'ruler_pip_opacity': .2}}))
        self.assertTrue(loaded.loaded)
        session = session_from_config(loaded.config)
        self.assertEqual(session.interaction.ruler_main_opacity, .65)
        self.assertEqual(session.interaction.ruler_pip_opacity, .2)
        serialized = json.loads(encode_user_preferences(session, DepthSensorDiagnostics()))
        self.assertEqual(serialized['display']['ruler_main_opacity'], .65)
        self.assertEqual(serialized['display']['ruler_pip_opacity'], .2)

    def test_approved_opacities_are_packaged_defaults(self):
        config = json.loads((ROOT / 'source/extensions/com.curvsurf.findsurface_isaac_sim/config/app_defaults.json')
                            .read_text(encoding='utf-8'))
        self.assertEqual(config['display']['ruler_main_opacity'], .8)
        self.assertEqual(config['display']['ruler_pip_opacity'], .35)
        settings = session_from_config(load_app_config().config).interaction
        self.assertEqual(settings.ruler_main_opacity, .8)
        self.assertEqual(settings.ruler_pip_opacity, .35)


if __name__ == '__main__':
    unittest.main()
